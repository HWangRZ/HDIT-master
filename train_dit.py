# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified for HSI latent diffusion — no EMA, gradient clipping, scheduler, mmap patches.

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL

from data import set_seed, HSI_LazyProcessing
from models import DiT_models, DiT
from diffusion import create_diffusion


# ---------------------------------------------------------------------------
#  Memory-mapped Dataset — same as pretrain_vae.py
# ---------------------------------------------------------------------------

class MmapTrainDS(torch.utils.data.Dataset):
    """Patches on disk, loaded on demand via mmap."""
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


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT on HSI latent space")

    # Dataset
    parser.add_argument("--dataset", "-d", type=str, default="PU")
    parser.add_argument("--n-pc", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--label-condition", type=str, default="all")
    parser.add_argument("--no-processing", action="store_true")
    parser.add_argument("--whiten", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patches-cache", type=str, default="cache_patches",
                        help="Dir for pre-saved mmap patches (same as VAE's --patches-cache)")

    # VAE
    parser.add_argument("--vae-checkpoint", type=str, default=None,
                        help="VAE checkpoint. Default: pretrain_vae/{dataset}/{dataset}_vae-final_best.pth")
    parser.add_argument("--vae-latent-channels", type=int, default=64)
    parser.add_argument("--vae-block-out", type=str, default="32,64")
    parser.add_argument("--vae-layers-per-block", type=int, default=4)
    parser.add_argument("--vae-num-down", type=int, default=2)
    parser.add_argument("--vae-norm-groups", type=int, default=16)
    parser.add_argument("--vae-scaling-factor", type=float, default=1.0,
                        help="VAE scaling factor. Loaded from checkpoint if embedded.")
    parser.add_argument("--vae-no-mid-attention", action="store_true")

    # DiT Model
    parser.add_argument("--model", type=str, default="DiT-S/2",
                        help="Predefined variant (DiT-S/2, DiT-B/2) or 'custom' to use --dit-*")
    parser.add_argument("--dit-depth", type=int, default=4)
    parser.add_argument("--dit-hidden-size", type=int, default=256)
    parser.add_argument("--dit-num-heads", type=int, default=4)
    parser.add_argument("--dit-patch-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--timestep-respacing", type=str, default="500")
    parser.add_argument("--learn-sigma", action="store_true",
                        help="If set, DiT predicts variance too (doubles output). Off by default for HSI.")

    # Training
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler-patience", type=int, default=5,
                        help="ReduceLROnPlateau patience (default: 5)")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                        help="ReduceLROnPlateau factor (default: 0.5)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-tf32", action="store_true")

    # Saving
    parser.add_argument("--results-dir", type=str, default="results_latent")

    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # TF32
    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Device
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    set_seed(args.seed)

    # ==================================================================
    #  Dataset
    # ==================================================================
    print(f"\nLoading dataset: {args.dataset}")
    X, Y, [row, col, band] = HSI_LazyProcessing(
        dataset_name=args.dataset, n_pc=args.n_pc,
        no_processing=args.no_processing, whiten=args.whiten,
    )
    X = X.astype(np.float32)          # PCA outputs float64, cast early
    print(f"Data: {X.shape}, Labels: {Y.shape}")

    # Use mmap cache (same files as pretrain_vae.py)
    x_path = os.path.join(args.patches_cache, f"{args.dataset}_X_p{args.patch_size}_{args.label_condition}.npy")
    y_path = os.path.join(args.patches_cache, f"{args.dataset}_Y_p{args.patch_size}_{args.label_condition}.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"📂 Loading patches from cache: {x_path}")
        trainset = MmapTrainDS(x_path, y_path)
    else:
        print(f"❌ Patch cache not found! Run pretrain_vae.py with --patches-cache first.")
        sys.exit(1)

    print(f"Trainset: {len(trainset)} samples")

    # ==================================================================
    #  VAE
    # ==================================================================
    vae_block_out = tuple(int(x) for x in args.vae_block_out.split(","))
    vae_down_up = ("DownEncoderBlock2D",) * args.vae_num_down
    vae_up = ("UpDecoderBlock2D",) * args.vae_num_down

    vae = AutoencoderKL(
        in_channels=band, out_channels=band,
        down_block_types=vae_down_up, up_block_types=vae_up,
        block_out_channels=vae_block_out, layers_per_block=args.vae_layers_per_block,
        act_fn="silu", latent_channels=args.vae_latent_channels,
        norm_num_groups=args.vae_norm_groups, sample_size=args.patch_size,
        scaling_factor=args.vae_scaling_factor, force_upcast=True,
        use_quant_conv=True, use_post_quant_conv=True,
        mid_block_add_attention=not args.vae_no_mid_attention,
    ).to(device)

    if args.vae_checkpoint is None:
        args.vae_checkpoint = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pretrain_vae",
            args.dataset, f"{args.dataset}_vae-final_best.pth")

    print(f"\n📦 Loading VAE from: {args.vae_checkpoint}")
    if os.path.exists(args.vae_checkpoint):
        ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            vae.load_state_dict(ckpt["model_state_dict"])
            if "scaling_factor" in ckpt:
                vae.scaling_factor = ckpt["scaling_factor"]
            print(f"✅ VAE loaded | scaling_factor={vae.scaling_factor:.4f} | "
                  f"latent_std={ckpt.get('latent_std', 'N/A')}")
        else:
            vae.load_state_dict(ckpt)
            print(f"✅ VAE loaded (plain state_dict)")
    else:
        print(f"❌ VAE checkpoint NOT FOUND — using random weights!")
    vae.eval()
    print(f"   VAE scaling_factor = {vae.scaling_factor:.4f}")

    # ==================================================================
    #  DataLoader
    # ==================================================================
    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=(args.num_workers > 0))
    print(f"DataLoader: batch_size={args.batch_size}, batches={len(loader)}")

    # ==================================================================
    #  DiT Model
    # ==================================================================
    if args.model == "custom":
        model = DiT(
            input_size=args.latent_size, patch_size=args.dit_patch_size,
            in_channels=args.vae_latent_channels, hidden_size=args.dit_hidden_size,
            depth=args.dit_depth, num_heads=args.dit_num_heads,
            num_classes=args.num_classes, learn_sigma=args.learn_sigma,
            class_dropout_prob=0.0,
        ).to(device)
    else:
        model = DiT_models[args.model](
            input_size=args.latent_size, num_classes=args.num_classes,
            learn_sigma=args.learn_sigma, class_dropout_prob=0.0,
        ).to(device)

    diffusion = create_diffusion(timestep_respacing=args.timestep_respacing,
                                   learn_sigma=args.learn_sigma)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    if args.model == "custom":
        print(f"  depth={args.dit_depth}  hidden={args.dit_hidden_size}  heads={args.dit_num_heads}  patch={args.dit_patch_size}")
    print(f"  learn_sigma={args.learn_sigma}")
    print(f"Params: {total:,} ({total/1e6:.2f}M)  trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  timestep_respacing: {args.timestep_respacing}")
    print(f"{'='*60}\n")

    # ==================================================================
    #  Optimizer & Scheduler
    # ==================================================================
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience, verbose=True,
    )

    # ==================================================================
    #  Training Loop
    # ==================================================================
    model.train()
    train_steps = 0
    running_loss = 0.0
    loss_history = {"epoch": [], "epoch_avg_loss": [], "step": [], "step_loss": []}
    start_time = time.time()

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                latents = vae.encode(x).latent_dist.sample().mul_(vae.scaling_factor)
            t = torch.randint(0, diffusion.num_timesteps, (latents.shape[0],), device=device)
            y = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            loss_dict = diffusion.training_losses(model, latents, t, dict(y=y))
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss += loss_val
            epoch_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                elapsed = time.time() - start_time
                avg = running_loss / args.log_every
                print(f"Epoch [{epoch+1}/{args.epochs}] Step {train_steps} "
                      f"Loss: {avg:.4f} ({train_steps/elapsed:.2f} steps/sec)")
                loss_history["step"].append(train_steps)
                loss_history["step_loss"].append(avg)
                running_loss = 0.0
                start_time = time.time()

        epoch_avg = epoch_loss / epoch_steps if epoch_steps else 0.0
        loss_history["epoch"].append(epoch + 1)
        loss_history["epoch_avg_loss"].append(epoch_avg)
        current_lr = opt.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{args.epochs}]  Avg Loss: {epoch_avg:.4f}  "
              f"Steps: {epoch_steps}  Time: {epoch_time:.2f}s  LR: {current_lr:.2e}")
        print(f"{'='*80}\n")

        scheduler.step(epoch_avg)

    # ==================================================================
    #  Save
    # ==================================================================
    save_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "loss_history.npy"), loss_history)
    with open(os.path.join(save_dir, "loss_history.txt"), "w") as f:
        f.write("Epoch\tAvg_Loss\n")
        for ep, ls in zip(loss_history["epoch"], loss_history["epoch_avg_loss"]):
            f.write(f"{ep}\t{ls:.6f}\n")
        f.write("\nStep\tLoss\n")
        for st, ls in zip(loss_history["step"], loss_history["step_loss"]):
            f.write(f"{st}\t{ls:.6f}\n")

    init_loss = loss_history["epoch_avg_loss"][0] if loss_history["epoch_avg_loss"] else 0
    final_loss = loss_history["epoch_avg_loss"][-1] if loss_history["epoch_avg_loss"] else 0
    print(f"\nTraining stats:")
    print(f"  Initial Loss: {init_loss:.4f}")
    print(f"  Final Loss:   {final_loss:.4f}")
    print(f"  Drop:         {init_loss - final_loss:.4f}")

    ckpt_path = os.path.join(save_dir, f"{args.dataset}-dit_latent-{args.timestep_respacing}.pt")
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "args": vars(args),
        "loss_history": loss_history,
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")
    print(f"Training complete.")


if __name__ == "__main__":
    main()
