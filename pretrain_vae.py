import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import set_seed, HSI_LazyProcessing, create_patches_from_processed_data, TrainDS
from autoencoder_kl import AutoencoderKL, vae_loss


# ---------------------------------------------------------------------------
#  Memory-mapped Dataset — patches on disk, loaded on demand
# ---------------------------------------------------------------------------

class MmapTrainDS(torch.utils.data.Dataset):
    """
    Like TrainDS but data stays on disk via np.memmap.
    Patches stored as (N, H, W, C) on disk; converted to (C, H, W) on-the-fly.
    """
    def __init__(self, x_path, y_path):
        self.x_data = np.load(x_path, mmap_mode='r')   # (N, H, W, C)
        self.y_data = np.load(y_path, mmap_mode='r')   # (N,)
        self.len = self.x_data.shape[0]
        self.is_fp16 = (self.x_data.dtype == np.float16)

    def __getitem__(self, index):
        patch = self.x_data[index].copy()              # mmap is read-only, need copy
        if self.is_fp16:
            patch = patch.astype(np.float32)
        x = torch.from_numpy(patch).permute(2, 0, 1)   # (H,W,C) → (C,H,W)
        y = torch.tensor(self.y_data[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return self.len


def _save_npy_chunked(arr, path, desc="Saving"):
    """Save numpy array with tqdm progress bar."""
    from tqdm.auto import tqdm
    header = np.lib.format.header_data_from_array_1_0(arr)
    with open(path, 'wb') as f:
        np.lib.format.write_array_header_1_0(f, header)
        chunk = 100000
        flat = arr.ravel()
        for i in tqdm(range(0, len(flat), chunk), desc=desc, unit='MB',
                       unit_scale=chunk * arr.dtype.itemsize / 1e6):
            flat[i:i+chunk].tofile(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain VAE on HSI dataset")

    # ===== Dataset =====
    parser.add_argument("--dataset", "-d", type=str, default="LongKou",
                        help="Dataset name (default: LongKou)")
    parser.add_argument("--n-pc", type=int, default=16,
                        help="Number of PCA components (default: 16)")
    parser.add_argument("--patch-size", type=int, default=32,
                        help="Patch size (default: 32)")
    parser.add_argument("--label-condition", type=str, default="all",
                        help="Label condition: 'all', '!=0', '==0', or int (default: all)")
    parser.add_argument("--no-processing", action="store_true",
                        help="Skip PCA processing")
    parser.add_argument("--whiten", action="store_true", default=True,
                        help="Apply PCA whitening (default: True)")

    # ===== DataLoader =====
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader num_workers (default: 0)")
    parser.add_argument("--patches-cache", type=str, default=None,
                        help="Cache dir for pre-computed patches. Saves/loads .npy via mmap, "
                             "dramatically reduces RAM. (default: None = old behavior)")
    parser.add_argument("--fp16-cache", action="store_true",
                        help="Save patches as float16 (halves disk usage, minor precision loss)")

    # ===== Model Architecture =====
    parser.add_argument("--latent-channels", type=int, default=64,
                        help="Latent space channels (default: 64)")
    parser.add_argument("--block-out-channels", type=str, default="32,64",
                        help="Block output channels, comma-separated (default: 32,64)")
    parser.add_argument("--layers-per-block", type=int, default=4,
                        help="ResNet layers per block (default: 4)")
    parser.add_argument("--num-down-blocks", type=int, default=2,
                        help="Number of down/up blocks (default: 2)")
    parser.add_argument("--norm-num-groups", type=int, default=16,
                        help="Number of groups for GroupNorm (default: 16)")
    parser.add_argument("--no-mid-attention", action="store_true",
                        help="Disable attention in mid block")
    parser.add_argument("--act-fn", type=str, default="silu",
                        help="Activation function (default: silu)")
    parser.add_argument("--scaling-factor", type=float, default=1.0,
                        help="Latent scaling factor (default: 1.0)")

    # ===== Loss Weights =====
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="MSE loss weight (default: 1.0)")
    parser.add_argument("--beta", type=float, default=0.001,
                        help="KL loss weight (default: 0.001)")
    parser.add_argument("--gamma", type=float, default=1.5,
                        help="SAM loss weight (default: 1.5)")

    # ===== Training =====
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda:N' or 'cpu' (default: auto-detect)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                        help="ReduceLROnPlateau factor (default: 0.5)")
    parser.add_argument("--scheduler-patience", type=int, default=5,
                        help="ReduceLROnPlateau patience (default: 5)")

    # ===== Saving =====
    parser.add_argument("--output-dir", type=str, default="pretrain_vae",
                        help="Output directory (default: pretrain_vae)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save model every N epochs (default: 10)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)

    # ===== Device =====
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ===== Load Dataset =====
    print(f"\n{'='*60}")
    print(f"Loading dataset: {args.dataset}")
    print(f"{'='*60}")
    X, Y, [row, col, band] = HSI_LazyProcessing(
        dataset_name=args.dataset,
        n_pc=args.n_pc,
        no_processing=args.no_processing,
        whiten=args.whiten,
    )
    print(f"Data shape: {X.shape}, Labels: {Y.shape}")
    print(f"Spatial size: {row} x {col}, Bands: {band}")

    # Cast to float32 ASAP — PCA outputs float64, which doubles all downstream memory
    X = X.astype(np.float32)

    # ===== Create or Load Patches (with optional disk cache) =====
    if args.patches_cache:
        os.makedirs(args.patches_cache, exist_ok=True)
        x_path = os.path.join(args.patches_cache, f"{args.dataset}_X_p{args.patch_size}_{args.label_condition}.npy")
        y_path = os.path.join(args.patches_cache, f"{args.dataset}_Y_p{args.patch_size}_{args.label_condition}.npy")

        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"\n📂 Loading patches from cache (mmap, near-zero RAM):")
            print(f"   {x_path}")
            trainset = MmapTrainDS(x_path, y_path)
        else:
            print(f"\n🔨 Creating patches (size={args.patch_size}) → saving to cache ...")
            save_dtype = np.float16 if args.fp16_cache else np.float32

            # === Chunked creation + write — never holds full array in RAM ===
            row, col, band = X.shape
            Y_2d = Y.reshape(row, col)
            margin = args.patch_size // 2
            padded_X = np.pad(X.astype(save_dtype, copy=False),
                              ((margin, margin), (margin, margin), (0, 0)),
                              mode='constant')

            # Determine which positions to include
            Y_flat = Y.flatten()
            lc = args.label_condition
            if lc == "all":
                positions = np.arange(row * col)
            elif lc == "!=0":
                positions = np.where(Y_flat != 0)[0]
            elif lc == "==0":
                positions = np.where(Y_flat == 0)[0]
            elif isinstance(lc, int):
                positions = np.where(Y_flat == lc)[0]
            elif isinstance(lc, (list, tuple, set)):
                positions = np.where(np.isin(Y_flat, list(lc)))[0]
            else:
                raise TypeError(f"Invalid label_condition: {lc}")

            N = len(positions)
            ps = args.patch_size
            chunk = 5000  # ~320 MB per chunk

            # === Stream-write: create chunks, write to disk, discard ===
            import sys
            print(f"   Writing {N:,} patches to disk in chunks of {chunk} "
                  f"({N * ps * ps * band * save_dtype().nbytes / 1e9:.1f} GB total) ...")
            sys.stdout.flush()

            Y_out = np.zeros(N, dtype=np.int64)
            with open(x_path, 'wb') as f:
                # Write .npy header for shape (N, ps, ps, band)
                header = np.lib.format.header_data_from_array_1_0(
                    np.empty((N, ps, ps, band), dtype=save_dtype))
                np.lib.format.write_array_header_1_0(f, header)

                for start in tqdm(range(0, N, chunk), desc="   Writing patches"):
                    end = min(start + chunk, N)
                    batch_pos = positions[start:end]
                    m = end - start
                    batch = np.empty((m, ps, ps, band), dtype=save_dtype)
                    for j, pos in enumerate(batch_pos):
                        r, c = pos // col, pos % col
                        batch[j] = padded_X[r:r+ps, c:c+ps, :]
                    batch.tofile(f)
                    Y_out[start:end] = Y_2d[batch_pos // col, batch_pos % col]
                    del batch

            np.save(y_path, Y_out)
            print(f"   ✅ Saved: {x_path}  ({os.path.getsize(x_path)/1e9:.2f} GB)")
            print(f"   ✅ Saved: {y_path}")
            del Y_out, padded_X, positions
            trainset = MmapTrainDS(x_path, y_path)
    else:
        # Original behavior — all patches in RAM
        print(f"\nCreating patches (size={args.patch_size})...")
        X_patches, Y_patches = create_patches_from_processed_data(
            X, Y, [row, col, band],
            patch_size=args.patch_size,
            label_condition=args.label_condition,
        )
        print(f"Patches: {len(X_patches)} samples")
        X_patches = X_patches.transpose((0, 3, 1, 2))
        trainset = TrainDS(X_patches, Y_patches)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Batch size: {args.batch_size}, Total batches: {len(train_loader)}")

    # Verify DataLoader
    test_batch = next(iter(train_loader))
    print(f"Batch shape: {test_batch[0].shape}")

    # ===== Parse block_out_channels =====
    block_out_channels = tuple(int(x) for x in args.block_out_channels.split(","))
    down_block_types = ("DownEncoderBlock2D",) * args.num_down_blocks
    up_block_types   = ("UpDecoderBlock2D",)   * args.num_down_blocks

    # ===== Build Model =====
    model = AutoencoderKL(
        in_channels=band,
        out_channels=band,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=args.layers_per_block,
        act_fn=args.act_fn,
        latent_channels=args.latent_channels,
        norm_num_groups=args.norm_num_groups,
        sample_size=args.patch_size,
        scaling_factor=args.scaling_factor,
        force_upcast=True,
        use_quant_conv=True,
        use_post_quant_conv=True,
        mid_block_add_attention=not args.no_mid_attention,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ===== Optimizer & Scheduler =====
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=True,
    )

    # ===== Output Paths =====
    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.dataset}_vae-final.pth")

    # ===== Training History =====
    history = {"epoch": [], "loss": [], "mse": [], "kl": [], "sam": [], "lr": []}

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Start Training VAE")
    print(f"{'='*60}")
    print(f"Dataset:     {args.dataset}")
    print(f"Samples:     {len(trainset)}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Epochs:      {args.epochs}")
    print(f"LR:          {args.lr}")
    print(f"Loss (α,β,γ) = ({args.alpha}, {args.beta}, {args.gamma})")
    print(f"Device:      {device}")
    print(f"Save to:     {save_dir}")
    print(f"{'='*60}\n")

    # ===== Training Loop =====
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = epoch_mse = epoch_kl = epoch_sam = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (data, _) in enumerate(pbar):
            x = data.to(device)  # (B, C, H, W)

            # Encode
            output = model.encode(x)
            posterior = output.latent_dist

            mu = posterior.mean
            std = posterior.std
            logvar = 2 * torch.log(std)

            # Reparameterization trick
            z = mu + std * torch.randn_like(std)
            recon = model.decode(z).sample

            # Loss (pass logvar, not std)
            loss, mse, kl, sam = vae_loss(
                recon, x, mu, logvar,
                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            # NaN check
            if torch.isnan(loss):
                print(f"\n⚠️  NaN detected! Epoch {epoch+1}, Batch {batch_idx}, skipping...")
                continue

            epoch_loss += loss.item()
            epoch_mse  += mse.item()
            epoch_kl   += kl.item()
            epoch_sam  += sam.item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MSE":  f"{mse.item():.4f}",
                "KL":   f"{kl.item():.4f}",
            })

        # ===== End of Epoch =====
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_mse  = epoch_mse  / n_batches
        avg_kl   = epoch_kl   / n_batches
        avg_sam  = epoch_sam  / n_batches
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["mse"].append(avg_mse)
        history["kl"].append(avg_kl)
        history["sam"].append(avg_sam)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | "
              f"KL: {avg_kl:.4f} | SAM: {avg_sam:.4f} | LR: {current_lr:.6f}")

        scheduler.step(avg_loss)

        # ===== Save Checkpoints =====
        if (epoch + 1) % args.save_interval == 0:
            if avg_loss < best_loss and not np.isnan(avg_loss):
                best_loss = avg_loss
                best_epoch = epoch + 1
                best_state = model.state_dict().copy()

            # Save checkpoint
            ckpt_path = save_path.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Checkpoint saved: {ckpt_path}")

            # Save history (npy + txt)
            np.save(ckpt_path.replace(".pth", "_history.npy"), history)
            with open(ckpt_path.replace(".pth", "_history.txt"), "w") as f:
                f.write("Epoch\tLoss\t\tMSE\t\tKL\t\tSAM\t\tLR\n")
                f.write("-" * 80 + "\n")
                for i in range(len(history["epoch"])):
                    f.write(f"{history['epoch'][i]}\t"
                            f"{history['loss'][i]:.6f}\t"
                            f"{history['mse'][i]:.6f}\t"
                            f"{history['kl'][i]:.6f}\t"
                            f"{history['sam'][i]:.6f}\t"
                            f"{history['lr'][i]:.8f}\n")

    # ===== Finalize =====
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Final Loss: {history['loss'][-1]:.4f}")
    print(f"Final MSE:  {history['mse'][-1]:.4f}")
    print(f"Final KL:   {history['kl'][-1]:.4f}")
    print(f"Final SAM:  {history['sam'][-1]:.4f}")

    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Final history
    np.save(save_path.replace(".pth", "_history.npy"), history)
    with open(save_path.replace(".pth", "_history.txt"), "w") as f:
        f.write("Epoch\tLoss\t\tMSE\t\tKL\t\tSAM\t\tLR\n")
        f.write("-" * 80 + "\n")
        for i in range(len(history["epoch"])):
            f.write(f"{history['epoch'][i]}\t"
                    f"{history['loss'][i]:.6f}\t"
                    f"{history['mse'][i]:.6f}\t"
                    f"{history['kl'][i]:.6f}\t"
                    f"{history['sam'][i]:.6f}\t"
                    f"{history['lr'][i]:.8f}\n")

    # Report
    report_path = save_path.replace(".pth", "_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("VAE Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset:      {args.dataset}\n")
        f.write(f"Samples:      {len(trainset)}\n")
        f.write(f"Batch size:   {args.batch_size}\n")
        f.write(f"Epochs:       {args.epochs}\n")
        f.write(f"LR:           {args.lr}\n")
        f.write(f"Loss (α,β,γ) = ({args.alpha}, {args.beta}, {args.gamma})\n")
        f.write(f"Device:       {device}\n\n")
        f.write(f"Total time:   {total_time:.2f}s ({total_time/60:.2f}min)\n")
        f.write(f"Best epoch:   {best_epoch}\n")
        f.write(f"Best loss:    {best_loss:.4f}\n")
        f.write(f"Final loss:   {history['loss'][-1]:.4f}\n")
        f.write(f"Final MSE:    {history['mse'][-1]:.4f}\n")
        f.write(f"Final KL:     {history['kl'][-1]:.4f}\n")
        f.write(f"Final SAM:    {history['sam'][-1]:.4f}\n\n")
        f.write("Files:\n")
        f.write(f"  - Final model:   {save_path}\n")
        f.write(f"  - Best model:    {save_path.replace('.pth', '_best.pth')}\n")
        f.write(f"  - History:       {save_path.replace('.pth', '_history.npy')}\n")
        f.write(f"  - Report:        {report_path}\n")
        f.write("=" * 60 + "\n")

    # ===== Compute & save scaling factor into checkpoint =====
    print("\nComputing VAE scaling factor from training data...")
    model.eval()
    with torch.no_grad():
        all_latents = []
        count = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            dist = model.encode(batch_x).latent_dist
            z = dist.sample()
            all_latents.append(z.cpu())
            count += batch_x.shape[0]
            if count >= 1000:
                break
        all_latents = torch.cat(all_latents, dim=0)
        latent_std = all_latents.std().item()
        latent_mean = all_latents.mean().item()
        scaling_factor_val = 1.0 / latent_std

    print(f"Computed: mean={latent_mean:.4f}, std={latent_std:.4f}")
    print(f"Recommended scaling_factor={scaling_factor_val:.4f}")

    # Inject latent_std into the saved state_dict for auto-loading later
    model.latent_std = latent_std
    model.scaling_factor = scaling_factor_val

    # Re-save final & best with embedded scaling info
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_std": latent_std,
        "scaling_factor": scaling_factor_val,
    }, save_path)
    print(f"Final model re-saved with scaling_factor={scaling_factor_val:.4f} embedded")

    if best_state is not None:
        best_path = save_path.replace(".pth", "_best.pth")
        torch.save({
            "model_state_dict": best_state,
            "latent_std": latent_std,
            "scaling_factor": scaling_factor_val,
        }, best_path)
        print(f"Best model re-saved with scaling_factor={scaling_factor_val:.4f} embedded")

    # Append to report
    with open(report_path, "a") as f:
        f.write(f"\nLatent stats:\n")
        f.write(f"  Mean:           {latent_mean:.6f}\n")
        f.write(f"  Std:            {latent_std:.6f}\n")
        f.write(f"  Scaling Factor: {scaling_factor_val:.6f}\n")
        f.write(f"  (based on {min(count, 1000)} samples)\n")

    print(f"✅ Report saved: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
