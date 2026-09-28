import os
import sys
import time # ✅ 新增：用于计时
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
from datetime import datetime
import argparse
import itertools
import random
import gc
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === 自定义模块 ===
from data import HSI_LazyProcessing
from autoencoder_kl import AutoencoderKL
from models import DiT_models, DiT
from diffusion import create_diffusion


# ---------------------------------------------------------------------------
#  Memory-mapped Dataset — same as pretrain_vae.py & train_dit.py
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


# ==========================================
# 1. 全局配置
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Feature search & classifier evaluation for HSI")

    parser.add_argument("--dataset", "-d", type=str, default="PU",
                        help="Dataset name (default: PU)")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patches-cache", type=str, default="cache_patches",
                        help="Dir for mmap patch cache (same as training)")
    parser.add_argument("--label-condition", type=str, default="all")

    # VAE / DiT 权重路径 (默认自动从数据集名拼接)
    parser.add_argument("--vae-path", type=str, default=None,
                        help="VAE checkpoint path. Auto-resolved if not set.")
    parser.add_argument("--dit-path", type=str, default=None,
                        help="DiT checkpoint path. Auto-resolved if not set.")

    # DiT architecture (must match training)
    parser.add_argument("--model", type=str, default="DiT-S/2")
    parser.add_argument("--learn-sigma", action="store_true",
                        help="Whether DiT was trained with learn_sigma")
    parser.add_argument("--dit-depth", type=int, default=6)
    parser.add_argument("--dit-hidden-size", type=int, default=384)
    parser.add_argument("--dit-num-heads", type=int, default=6)
    parser.add_argument("--dit-patch-size", type=int, default=2)

    # 自动时间步筛选
    parser.add_argument("--test-steps", type=str, default="0-500",
                        help="Timestep range for CH evaluation, e.g. '0-599' (default: 0-599)")
    parser.add_argument("--target-layer", type=int, default=5,
                        help="DiT block layer for CH evaluation (default: 5)")
    parser.add_argument("--num-probes", type=int, default=100,
                        help="Number of probe samples for CH eval (default: 200)")
    parser.add_argument("--n-clusters", type=int, default=15,
                        help="K-Means clusters for CH score (default: 9)")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Length of the contiguous high-CH interval (default: 151)")
    parser.add_argument("--smooth-window", type=int, default=11,
                        help="Moving-average window used to smooth CH scores (default: 11)")
    parser.add_argument("--pick-every", type=int, default=15,
                        help="Sample every K timesteps from the selected peak interval (default: 25)")

    # 手动指定时间步 (跳过自动筛选)
    parser.add_argument("--pool-steps", type=str, default=None,
                        help="Comma-sep timesteps, e.g. '120,180,160'. Skips CH eval if set.")

    # 搜索空间
    parser.add_argument("--pool-layers", type=str, default="3,5,4",
                        help="DiT layers for feature extraction, comma-sep (default: 3,5,4)")
    parser.add_argument("--min-combo-t", type=int, default=6,
                        help="Min steps per combo (default: 5)")
    parser.add_argument("--max-combo-t", type=int, default=6,
                        help="Max steps per combo (default: 5)")
    parser.add_argument("--min-combo-l", type=int, default=1,
                        help="Min layers per combo (default: 1)")
    parser.add_argument("--max-combo-l", type=int, default=1,
                        help="Max layers per combo (default: 1)")
    parser.add_argument("--search-limit", type=int, default=2000,
                        help="Max configs to search (default: 2000)")

    # 评估
    parser.add_argument("--seeds", type=str, default="1220,1224,1226,1229,1233,1235,1236,1330,1336,1337",    #1220,1224,1226,1229,1233,1235,1236,1330,1336,1337
                        help="Random seeds for evaluation, comma-sep")
    parser.add_argument("--samples-per-class", type=int, default=5,
                        help="Training samples per class (default: 5)")

    return parser.parse_args()


def build_args():
    args = parse_args()

    # Auto-resolve VAE / DiT paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.vae_path is None:
        args.vae_path = os.path.join(
            base_dir, "pretrain_vae", args.dataset,
            f"{args.dataset}_vae-final_best.pth"
        )
    if args.dit_path is None:
        args.dit_path = os.path.join(
            base_dir, "results_latent", args.dataset,
            f"{args.dataset}-dit_latent-500.pt"
        )

    # Parse ranges and lists
    def parse_range(s):
        parts = s.split("-")
        if len(parts) == 2:
            return list(range(int(parts[0]), int(parts[1]) + 1))
        return [int(s)]

    test_steps = parse_range(args.test_steps)

    pool_steps = [int(x) for x in args.pool_steps.split(",")] if args.pool_steps else None
    pool_layers = [int(x) for x in args.pool_layers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    ARGS = {
        "dataset": args.dataset,
        "patch_size": args.patch_size,
        "vae_path": args.vae_path,
        "dit_path": args.dit_path,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "patches_cache": args.patches_cache,
        "label_condition": args.label_condition,
        "model": args.model,
        "learn_sigma": args.learn_sigma,
        "dit_depth": args.dit_depth,
        "dit_hidden_size": args.dit_hidden_size,
        "dit_num_heads": args.dit_num_heads,
        "dit_patch_size": args.dit_patch_size,

        "test_steps": test_steps,
        "target_layer": args.target_layer,
        "num_probes": args.num_probes,
        "n_clusters": args.n_clusters,
        "top_n_timesteps": args.top_n,
        "smooth_window": args.smooth_window,
        "pick_every": args.pick_every,

        "pool_steps": pool_steps,
        "pool_layers": pool_layers,

        "min_combo_t": args.min_combo_t,
        "max_combo_t": args.max_combo_t,
        "min_combo_l": args.min_combo_l,
        "max_combo_l": args.max_combo_l,
        "search_limit": args.search_limit,

        "seeds": seeds,
        "samples_per_class": args.samples_per_class,
    }
    return ARGS


ARGS = build_args()
device = torch.device(ARGS["device"] if torch.cuda.is_available() else "cpu")

# --device 只决定张量放在哪个 GPU，不会改变 CUDA 的 current device。
# 这里显式设置 current device，避免 torch.cuda 的无 device 参数 API 触碰 cuda:0。
if device.type == 'cuda':
    torch.cuda.set_device(device)


def set_seed(seed=42):
    """只初始化 CPU 和本进程目标 GPU 的 RNG，不使用 manual_seed_all。"""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed() 在部分 PyTorch 版本中会调用 manual_seed_all，
    # 从而为所有 GPU（包括 cuda:0）建立 CUDA context。
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(ARGS["seed"])
  
# ==========================================
# 2. 辅助功能 & 性能监控工具 (✅ 新增监控工具)
# ==========================================
def calculate_vae_scaling_factor(vae, dataset, device, num_samples=1000):
    """
    专门用来计算 VAE Latent 标准差的函数
    """
    print(f"正在计算 VAE Scaling Factor,基于 {num_samples} 个样本...")
    
    # 创建一个临时的 DataLoader
    temp_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    collected_latents = []
    count = 0
    
    vae.eval()
    with torch.no_grad():
        for x, _ in temp_loader:
            x = x.to(device)
            # 编码得到分布
            dist = vae.encode(x).latent_dist
            # 采样 (sample)
            latents = dist.sample()
            collected_latents.append(latents.cpu())
            
            count += x.shape[0]
            if count >= num_samples:
                break
    
    # 拼接所有 latents
    all_latents = torch.cat(collected_latents, dim=0)
    
    # 计算标准差
    std = all_latents.std().item()
    mean = all_latents.mean().item()
    
    # Scaling Factor 是标准差的倒数
    scaling_factor = 1.0 / std
    
    print(f"统计结果 -> Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"✅ 建议使用的 Scaling Factor: {scaling_factor:.4f}")
    
    return scaling_factor


def auto_select_timesteps(model, diffusion, vae, loader, config):
    """
    无监督时间步自动筛选：
    1. 对每个待测时间步提取探针特征
    2. PCA降维 → K-Means聚类 → 计算CH风格得分（簇间散度/簇内散度）
    3. 取Top N个高分时间步，每隔K步等距选一个作为 pool_steps
    """
    print(f"\n{'='*60}")
    print("🔍 自动筛选最佳时间步 (无监督 CH 聚类质量评估)")
    print(f"{'='*60}")
    print(f"探针数: {config['num_probes']}, 评估层: block {config['target_layer']}")
    print(f"时间步范围: {min(config['test_steps'])} ~ {max(config['test_steps'])}")

    device = next(model.parameters()).device
    layer_idx = config['target_layer']

    # 采样探针
    probe_x = []
    count = 0
    for x, _ in loader:
        probe_x.append(x)
        count += x.shape[0]
        if count >= config['num_probes']:
            break
    probe_x = torch.cat(probe_x, dim=0)[:config['num_probes']].to(device)
    print(f"探针样本: {probe_x.shape[0]}")

    # Hook
    feature_buffer = {}
    def hook_fn(m, i, o):
        feature_buffer['feat'] = o.detach()
    handle = model.blocks[layer_idx].register_forward_hook(hook_fn)

    t_features = {}
    with torch.no_grad():
        dist = vae.encode(probe_x).latent_dist
        z_raw = dist.sample()
        z_scaled = z_raw * vae.scaling_factor

        for t_val in tqdm(config['test_steps'], desc="CH评估各时间步"):
            t_tensor = torch.tensor([t_val] * probe_x.shape[0], device=device, dtype=torch.long)
            noise = torch.randn_like(z_scaled)
            z_noisy = diffusion.q_sample(z_scaled, t_tensor, noise=noise)
            _ = model(z_noisy, t_tensor,
                      y=torch.zeros(probe_x.shape[0], dtype=torch.long, device=device))

            feat = feature_buffer['feat']
            B, N, D = feat.shape
            Hf = int(N ** 0.5)
            feat_img = feat.transpose(1, 2).view(B, D, Hf, Hf)
            ch = cw = Hf // 2
            fc = feat_img[:, :, max(0,ch-1):min(Hf,ch+1), max(0,cw-1):min(Hf,cw+1)].mean(dim=(2,3))
            t_features[t_val] = torch.cat([feat.mean(dim=1), fc], dim=1).cpu().numpy()

    handle.remove()

    # 计算 CH 聚类质量得分
    n_clusters = config['n_clusters']
    scores = {}
    for t, feat in tqdm(t_features.items(), desc="计算CH得分"):
        norm_feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-6)
        pca = PCA(n_components=min(32, norm_feat.shape[1]), random_state=42)
        pca_feat = pca.fit_transform(norm_feat)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_pred = km.fit_predict(pca_feat)

        v_within = 0.0
        for c in range(n_clusters):
            mask = labels_pred == c
            nc = mask.sum()
            if nc > 1:
                center = pca_feat[mask].mean(axis=0)
                v_within += np.sum((pca_feat[mask] - center) ** 2)

        global_mean = pca_feat.mean(axis=0)
        v_between = 0.0
        for c in range(n_clusters):
            mask = labels_pred == c
            nc = mask.sum()
            if nc > 0:
                center = pca_feat[mask].mean(axis=0)
                v_between += nc * np.sum((center - global_mean) ** 2)

        scores[t] = v_between / (v_within + 1e-6)

    # ------------------------------------------------------------------
    # 选择连续的高质量时间区间，而不是选择若干个孤立的最高分时间步。
    # CH 分数可能因噪声/聚类初始化而上下震荡，因此先对按时间排序后的
    # 分数做移动平均，再寻找平均分最高的固定长度区间。
    # ------------------------------------------------------------------
    ordered_steps = sorted(scores)
    raw_scores = np.asarray([scores[t] for t in ordered_steps], dtype=np.float64)

    smooth_window = max(1, int(config.get('smooth_window', 11)))
    # 使用边界复制，避免 np.convolve(mode='same') 在两端补 0 造成假低谷。
    smooth_window = min(smooth_window, len(raw_scores))
    padded = np.pad(
        raw_scores,
        (smooth_window // 2, smooth_window - 1 - smooth_window // 2),
        mode='edge',
    )
    smoothed_scores = np.convolve(
        padded, np.ones(smooth_window, dtype=np.float64) / smooth_window, mode='valid'
    )

    interval_len = max(1, int(config['top_n_timesteps']))
    interval_len = min(interval_len, len(ordered_steps))
    pick_every = max(1, int(config['pick_every']))

    # 在所有连续候选区间中，选择平滑 CH 平均值最高的一段。
    # 这里的“连续”指 test_steps 排序后的连续位置，通常即连续 timestep。
    cumsum = np.concatenate(([0.0], np.cumsum(smoothed_scores)))
    all_interval_means = (
        cumsum[interval_len:] - cumsum[:-interval_len]
    ) / interval_len

    # 只允许真正相邻的 timestep 组成区间（例如 100,101,102），避免
    # test_steps 中存在跳号时把两个不连续区段错误拼接起来。
    candidate_starts = [
        start for start in range(len(ordered_steps) - interval_len + 1)
        if interval_len == 1 or np.all(
            np.diff(ordered_steps[start:start + interval_len]) == 1
        )
    ]
    if not candidate_starts:
        # 当用户通过 --test-steps 传入了稀疏时间点，仍然提供可用结果，
        # 但明确提示此时只能按输入顺序选取区间。
        candidate_starts = list(range(len(all_interval_means)))
        print("⚠️ test_steps 不包含足够长的连续 timestep，将按排序后的输入位置选择峰值区间。")

    best_start = max(candidate_starts, key=lambda i: all_interval_means[i])
    interval_means = all_interval_means
    peak_steps = ordered_steps[best_start:best_start + interval_len]
    selected = peak_steps[::pick_every]

    # 保证至少抽取区间的末端时间步，避免区间长度不是 pick_every 的整数倍时
    # 丢掉峰值段尾部的信息。
    if selected[-1] != peak_steps[-1]:
        selected.append(peak_steps[-1])

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    raw_peak = max(raw_scores)
    smooth_peak = max(smoothed_scores)
    print(
        f"\n🏆 连续峰值区间: t={peak_steps[0]} ~ t={peak_steps[-1]} "
        f"({len(peak_steps)} 个时间步)"
    )
    print(
        f"   区间平滑 CH 平均值: {interval_means[best_start]:.4f}; "
        f"全局原始峰值: {raw_peak:.4f}; 全局平滑峰值: {smooth_peak:.4f}"
    )
    print(
        f"   平滑窗口: {smooth_window}; 区间内最高 CH: "
        f"t={peak_steps[int(np.argmax(raw_scores[best_start:best_start + interval_len]))]}"
    )
    print(f"📌 自动选出的 pool_steps ({len(selected)}个): {selected}")
    print(f"{'='*60}\n")

    return selected


def count_parameters(model):
    """计算模型参数量并返回百万(M)为单位的数值"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def reset_vram(device):
    """重置 GPU 峰值显存记录"""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def get_peak_vram(device):
    """获取当前 GPU 峰值显存 (MB)"""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0

def sync_time(device=None):
    """只同步目标 GPU；不指定 device 时不要默认同步 cuda:0。"""
    if device is None:
        device = globals().get('device')
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.time()

def generate_auto_configs(steps, layers, min_t=5, max_t=5, min_l=1, max_l=1, limit=200):
    steps = sorted(list(set(steps)))
    layers = sorted(list(set(layers)))
    configs = []
    
    step_combos = []
    for r in range(min_t, max_t + 1):
        step_combos.extend(list(itertools.combinations(steps, r)))
    
    layer_combos = []
    for r in range(min_l, max_l + 1):
        layer_combos.extend(list(itertools.combinations(layers, r)))
        
    for s in step_combos:
        for l in layer_combos:
            configs.append({
                "name": f"t{list(s)}_l{list(l)}".replace(" ", ""),
                "steps": list(s),
                "layers": list(l)
            })
            
    if len(configs) > limit:
        random.seed(42)
        random.shuffle(configs)
        configs = configs[:limit]
    return configs

class FeaturePoolExtractor:
    def __init__(self, model, diffusion, vae, device):
        self.model = model
        self.diffusion = diffusion
        self.vae = vae
        self.device = device
        self.features_buffer = {}
    def _hook_fn(self, name):
        def hook(model, input, output): self.features_buffer[name] = output.detach()
        return hook

    def extract_pool(self, loader, t_list, layer_list):
        unique_layers = sorted(list(set(layer_list)))
        unique_steps = sorted(list(set(t_list)))
        
        for i in unique_layers: 
            self.model.blocks[i].register_forward_hook(self._hook_fn(f'block_{i}'))
            
        print(f"🚀 [Feature Pool] Building pool for Steps={unique_steps}, Layers={unique_layers}...")
        pool_dict = {(t, l): [] for t in unique_steps for l in unique_layers}
        vae_features = []
        all_labels = []
        
        # ✅ 性能监控：开始提取特征
        reset_vram(self.device)
        start_time = sync_time(self.device)
        
        # ✅ DiT推理时间统计
        dit_total_time = 0.0
        dit_call_count = 0
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Extracting")):
                x = x.to(self.device)
                bs = x.shape[0]
                
                if batch_idx == 0:
                    print(f"\n{'='*60}")
                    print(f"📊 特征提取形状追踪 (第一个batch)")
                    print(f"{'='*60}")
                    print(f"输入 x: {x.shape}")
                
                # ==========================================
                # ✅ 修复 1: 强制固定当前 batch 的 VAE 采样种子
                # 确保无论跑几个时间步，同一个 batch 生成的 VAE latent 永远一致
                # ==========================================
                torch.random.default_generator.manual_seed(42 + batch_idx)
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(42 + batch_idx)

                # VAE
                dist = self.vae.encode(x).latent_dist
                z_raw = dist.sample()
                z_scaled = z_raw * self.vae.scaling_factor
                
                if batch_idx == 0:
                    print(f"\n--- VAE ---")
                    print(f"z_raw (VAE latent): {z_raw.shape}")
                    print(f"z_scaled (z_raw * {self.vae.scaling_factor}): {z_scaled.shape}")
                
                B, C, H, W = z_raw.shape
                center_h, center_w = H // 2, W // 2
                h_start = max(0, center_h - 1)
                h_end = min(H, center_h + 1)
                w_start = max(0, center_w - 1)
                w_end = min(W, center_w + 1)
                vae_center = z_raw[:, :, h_start:h_end, w_start:w_end].mean(dim=(2, 3))
                
                if batch_idx == 0:
                    print(f"VAE latent空间: {H}×{W}")
                    print(f"中心区域: [{h_start}:{h_end}, {w_start}:{w_end}] = {(h_end-h_start)}×{(w_end-w_start)}")
                    print(f"vae_center (中心区域平均): {vae_center.shape}")
                
                vae_features.append(vae_center.cpu())
                all_labels.append(y)
                
                # DiT
                for t_val in unique_steps:
                    t = torch.tensor([t_val] * bs, device=self.device, dtype=torch.long)
                    
                    # ==========================================
                    # ✅ 修复 2: 强制绑定 batch_idx 和 t_val 的噪声生成种子
                    # 确保特定图片、特定时间步加的噪声永远不变，不受时间步列表长度影响
                    # ==========================================
                    seed_val = 42 + batch_idx * 1000 + int(t_val)
                    torch.random.default_generator.manual_seed(seed_val)
                    if self.device.type == 'cuda':
                        with torch.cuda.device(self.device):
                            torch.cuda.manual_seed(seed_val)
                        
                    noise = torch.randn_like(z_scaled) 
                    z_noisy = self.diffusion.q_sample(z_scaled, t, noise=noise)
                    
                    if batch_idx == 0:
                        print(f"\n--- DiT (t={t_val}) ---")
                        print(f"z_noisy: {z_noisy.shape}")
                    
                    # ✅ 计算DiT推理时间
                    dit_start = sync_time(self.device)
                    _ = self.model(z_noisy, t, y=torch.zeros(bs, dtype=torch.long, device=self.device))
                    dit_end = sync_time(self.device)
                    dit_total_time += (dit_end - dit_start)
                    dit_call_count += 1
                    
                    for layer_idx in unique_layers:
                        feat = self.features_buffer[f'block_{layer_idx}']
                        B, N, D = feat.shape
                        H_feat = int(N ** 0.5)
                        feat_img = feat.transpose(1, 2).view(B, D, H_feat, H_feat)
                        
                        center_h, center_w = H_feat // 2, H_feat // 2
                        h_start = max(0, center_h - 1)
                        h_end = min(H_feat, center_h + 1)
                        w_start = max(0, center_w - 1)
                        w_end = min(H_feat, center_w + 1)
                        feat_center = feat_img[:, :, h_start:h_end, w_start:w_end].mean(dim=(2, 3))

                        feat_cat = torch.cat([feat.mean(dim=1), feat_center], dim=1)
                        pool_dict[(t_val, layer_idx)].append(feat_cat.cpu())
                        
                        if batch_idx == 0:
                            print(f"  Block {layer_idx}: feat={feat.shape} [B,N,D] → feat_img={feat_img.shape} → 中心[{h_start}:{h_end},{w_start}:{w_end}] → global_avg={feat.mean(dim=1).shape} + center_avg={feat_center.shape} → cat={feat_cat.shape}")
                
                if batch_idx == 0:
                    print(f"\n{'='*60}")
                    print(f"📊 特征维度总结")
                    print(f"{'='*60}")
                    print(f"VAE特征: {vae_center.shape[1]}维")
                    print(f"DiT特征 (每个t,l组合): {feat_cat.shape[1]}维")
                    print(f"pool_steps={unique_steps}, pool_layers={unique_layers}")
                    print(f"DiT组合数: {len(unique_steps)} × {len(unique_layers)} = {len(unique_steps)*len(unique_layers)}")
                    total_dim = vae_center.shape[1] + len(unique_steps) * len(unique_layers) * feat_cat.shape[1]
                    print(f"总特征维度 (VAE + 所有DiT): {total_dim}")
                    print(f"{'='*60}\n")

        # ✅ 性能监控：结束提取特征
        extract_time = sync_time(self.device) - start_time
        peak_vram = get_peak_vram(self.device)
        
        # ✅ DiT推理时间统计
        dit_avg_time = dit_total_time / dit_call_count if dit_call_count > 0 else 0
        
        print(f"⏱️ Feature Extraction Time: {extract_time:.2f} s")
        print(f"💾 Peak VRAM (Extraction): {peak_vram:.2f} MB")
        print(f"🔧 DiT Inference Time: {dit_total_time:.4f} s (Total), {dit_avg_time*1000:.2f} ms (Avg per call), {dit_call_count} calls")
        
        return {k: torch.cat(v, dim=0).numpy() for k, v in pool_dict.items()}, torch.cat(vae_features, dim=0).numpy(), torch.cat(all_labels).numpy(), extract_time, peak_vram, dit_total_time, dit_avg_time

class SimpleClassifier(nn.Module):
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.6),###########
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

    
    

DATASET_CLASS_NAMES = {
    "PU": ["Asphalt", "Meadows", "Gravel", "Trees",
           "Painted metal sheets", "Bare Soil", "Bitumen",
           "Self-Blocking Bricks", "Shadows"],
    "Houston": ["Healthy Grass", "Stressed Grass", "Trees", "Water",
                "Residential Buildings", "Non-Residential Buildings", "Roads",
                "Sidewalks", "Crosswalks", "Major Thoroughfares", "Highways",
                "Railway", "Paved Parking Lots", "Unpaved Parking Lots", "Cars"],
    "Salinas": ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
                "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
                "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"],
    "LongKou": ["Corn", "Cotton", "Sesame", "Broad-leaf soybean", "Narrow-leaf soybean",
                "Rice", "Water", "Roads and houses", "Mixed weed"],
}

# ==========================================
# 3. 主流程
# ==========================================
if __name__ == "__main__":
    print("Loading Data & Models...")
    X, Y, [row, col, band] = HSI_LazyProcessing(ARGS['dataset'], n_pc=16, whiten=True)
    X = X.astype(np.float32)

    # Use mmap patch cache (always load 'all', filter on-the-fly if needed)
    x_all = os.path.join(ARGS['patches_cache'], f"{ARGS['dataset']}_X_p{ARGS['patch_size']}_all.npy")
    y_all = os.path.join(ARGS['patches_cache'], f"{ARGS['dataset']}_Y_p{ARGS['patch_size']}_all.npy")
    if not os.path.exists(x_all):
        print(f"❌ Patch cache not found: {x_all}")
        print(f"   Run pretrain_vae.py with --patches-cache first.")
        sys.exit(1)

    print(f"📂 Loading patches from cache: {x_all}")
    full_dataset = MmapTrainDS(x_all, y_all)

    # Filter on-the-fly by label_condition — no extra files needed
    lc = ARGS['label_condition']
    y_mmap = np.load(y_all, mmap_mode='r')
    if lc == "all":
        dataset = full_dataset
    elif lc == "!=0":
        indices = np.where(y_mmap != 0)[0]
        dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"   Filtered: {len(indices):,} / {len(y_mmap):,} patches (label != 0)")
    elif lc == "==0":
        indices = np.where(y_mmap == 0)[0]
        dataset = torch.utils.data.Subset(full_dataset, indices)
    elif isinstance(lc, int):
        indices = np.where(y_mmap == lc)[0]
        dataset = torch.utils.data.Subset(full_dataset, indices)
    elif isinstance(lc, (list, tuple, set)):
        indices = np.where(np.isin(y_mmap, list(lc)))[0]
        dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        raise TypeError(f"Invalid label_condition: {lc}")

    loader = DataLoader(dataset, batch_size=ARGS['batch_size'], shuffle=False, num_workers=4)
    
    vae = AutoencoderKL(in_channels=band, out_channels=band, down_block_types=("DownEncoderBlock2D", )*2,
                        up_block_types=("UpDecoderBlock2D", )*2, block_out_channels=(32,64), layers_per_block=4,
                        latent_channels=64, norm_num_groups=16, sample_size=32)
    ckpt = torch.load(ARGS['vae_path'], map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        vae.load_state_dict(ckpt['model_state_dict'])
        vae.latent_std = ckpt.get('latent_std', None)
        vae.scaling_factor = ckpt.get('scaling_factor', 1.0)
    else:
        vae.load_state_dict(ckpt)
    vae = vae.to(device).eval()
    if vae.scaling_factor is None or vae.scaling_factor == 1.0:
        vae.scaling_factor = calculate_vae_scaling_factor(vae, dataset, device)
    print(f"   VAE scaling_factor = {vae.scaling_factor:.4f}")
    ##############################
    
    # Build DiT matching training config
    if ARGS['model'] == "custom":
        model = DiT(
            input_size=16, patch_size=ARGS['dit_patch_size'],
            in_channels=64, hidden_size=ARGS['dit_hidden_size'],
            depth=ARGS['dit_depth'], num_heads=ARGS['dit_num_heads'],
            num_classes=1, learn_sigma=ARGS['learn_sigma'],
            class_dropout_prob=0.0,
        )
    else:
        model = DiT_models[ARGS['model']](
            input_size=16, num_classes=1,
            learn_sigma=ARGS['learn_sigma'], class_dropout_prob=0.0,
        )
    ckpt = torch.load(ARGS['dit_path'], map_location='cpu')
    state_dict = {k: v for k, v in ckpt['model'].items() if not k.startswith('total_') and '.total_' not in k}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    diffusion = create_diffusion(timestep_respacing="", learn_sigma=ARGS['learn_sigma'])
    
    CLASS_NAMES = DATASET_CLASS_NAMES.get(ARGS['dataset'], [f"Class_{i}" for i in range(100)])

    # ✅ 记录大模型的参数量
    vae_params = count_parameters(vae)
    dit_params = count_parameters(model)
    print(f"📦 VAE Parameters: {vae_params:.2f} M")
    print(f"📦 DiT Parameters: {dit_params:.2f} M")
    
    # ✅ 计算FLOPs
    from thop import profile
    try:
        # VAE FLOPs
        dummy_vae_input = torch.randn(1, band, 32, 32).to(device)
        vae_flops, _ = profile(vae, inputs=(dummy_vae_input,), verbose=False)
        print(f"🔧 VAE FLOPs: {vae_flops:,} ({vae_flops/1e9:.2f}G)")
    except Exception as e:
        vae_flops = 0
        print(f"⚠️ VAE FLOPs计算失败: {e}")
    
    try:
        # DiT FLOPs
        dummy_dit_input = torch.randn(1, 64, 16, 16).to(device)
        dummy_t = torch.tensor([0], dtype=torch.long, device=device)
        dummy_y = torch.tensor([0], dtype=torch.long, device=device)
        dit_flops, _ = profile(model, inputs=(dummy_dit_input, dummy_t, dummy_y), verbose=False)
        print(f"🔧 DiT FLOPs: {dit_flops:,} ({dit_flops/1e9:.2f}G)")
    except Exception as e:
        dit_flops = 0
        print(f"⚠️ DiT FLOPs计算失败: {e}")
    
    # ===== 自动筛选时间步 (若未手动指定) =====
    if ARGS['pool_steps'] is None:
        ARGS['pool_steps'] = auto_select_timesteps(
            model, diffusion, vae, loader, ARGS
        )
        # 重新创建 loader（shuffle=True 保证提取时多样性）
        loader = DataLoader(dataset, batch_size=ARGS['batch_size'], shuffle=False, num_workers=4)

    # 提取特征池
    extractor = FeaturePoolExtractor(model, diffusion, vae, device)
    feature_pool, vae_feat, labels, extract_time, extract_vram, dit_total_time, dit_avg_time = extractor.extract_pool(loader, ARGS['pool_steps'], ARGS['pool_layers'])
    
    # ✅ 打印DiT推理时间统计
    print(f"\n{'='*60}")
    print(f"🔧 DiT推理时间统计")
    print(f"{'='*60}")
    print(f"总推理时间: {dit_total_time:.4f} 秒")
    print(f"平均每次推理: {dit_avg_time*1000:.2f} 毫秒")
    print(f"推理次数: {len(ARGS['pool_steps'])} timesteps × {len(loader)} batches = {len(ARGS['pool_steps']) * len(loader)} calls")
    print(f"{'='*60}\n")
    
    del model, vae, extractor
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    
    scaler_vae = StandardScaler()
    vae_scaled = scaler_vae.fit_transform(vae_feat)
    
    unique_labels = np.unique(labels)
    label_map = {val: i for i, val in enumerate(unique_labels)}
    y_mapped = np.array([label_map[l] for l in labels])
    num_classes = len(unique_labels)
    
    configs = generate_auto_configs(
        ARGS['pool_steps'], ARGS['pool_layers'],
        min_t=ARGS['min_combo_t'], max_t=ARGS['max_combo_t'],
        min_l=ARGS['min_combo_l'], max_l=ARGS['max_combo_l'],
        limit=ARGS['search_limit'],
    )
    
    print(f"\n⚡ Starting Search: {len(configs)} configs x {len(ARGS['seeds'])} seeds")
    print("="*60)
    
    results = []
    
    for config in tqdm(configs, desc="Searching"):
        cfg_name = config["name"]
        
        try:
            #blocks = [vae_scaled]
            blocks = []
            for t in config["steps"]:
                for l in config["layers"]:
                    raw = feature_pool[(t, l)]
                    norm = (raw - raw.mean()) / (raw.std() + 1e-6)
                    blocks.append(norm)
            X_full = np.concatenate(blocks, axis=1)
        except KeyError: continue
        
        seed_metrics = {"oa": [], "aa": [], "kappa": [], "class_acc": [], "inf_time_ms": []}
        best_seed_oa_for_cfg = -1.0
        best_seed_report_str = ""
        cfg_cls_params = 0
        cfg_peak_vram = 0
        
        for seed in ARGS['seeds']:
            tqdm.write(f"  [{cfg_name}] seed={seed} ({ARGS['seeds'].index(seed)+1}/{len(ARGS['seeds'])}) training ...")
            set_seed(seed)
            
            # 数据划分
            train_idx, test_idx = [], []
            for c in range(num_classes):
                c_idx = np.where(y_mapped == c)[0]
                np.random.shuffle(c_idx)
                if len(c_idx) >= ARGS['samples_per_class']:
                    train_idx.append(c_idx[:ARGS['samples_per_class']])
                    test_idx.append(c_idx[ARGS['samples_per_class']:])
                else:
                    train_idx.append(c_idx)
            
            idx_tr_flat = np.concatenate(train_idx)
            idx_te_flat = np.concatenate(test_idx)
            
            X_tr = torch.FloatTensor(X_full[idx_tr_flat]).to(device)
            y_tr = torch.LongTensor(y_mapped[idx_tr_flat]).to(device)
            X_te = torch.FloatTensor(X_full[idx_te_flat]).to(device)
            y_te = torch.LongTensor(y_mapped[idx_te_flat]).to(device)
            
            net = SimpleClassifier(X_full.shape[1], num_classes).to(device)
            cfg_cls_params = count_parameters(net) # 记录分类器参数量
            
            opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-3)
            crit = nn.CrossEntropyLoss()
            
            loader_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
            
            best_preds_this_seed = None
            best_oa_this_seed = 0.0
            
            reset_vram(device)
            
            patience = 10
            no_improve_count = 0
            
            for ep in range(100): 
                net.train()
                for bx, by in loader_tr:
                    opt.zero_grad(); crit(net(bx), by).backward(); opt.step()
                
                if (ep+1) % 5 == 0:
                    net.eval()
                    with torch.no_grad():
                        t0 = sync_time(device)
                        logits = net(X_te)
                        preds = logits.argmax(1)
                        t1 = sync_time(device)
                        
                        acc = (preds == y_te).float().mean().item()
                        tqdm.write(f"    seed={seed} epoch={ep+1} acc={acc:.4f}")
                        if acc > best_oa_this_seed: 
                            best_oa_this_seed = acc
                            best_preds_this_seed = preds.cpu().numpy()
                            best_inf_time = (t1 - t0) * 1000
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        
                        if no_improve_count >= patience:
                            break
            
            # 记录此 Config 当前 Seed 的训练推理最高显存
            current_vram = get_peak_vram(device)
            if current_vram > cfg_peak_vram:
                cfg_peak_vram = current_vram

            y_true_cpu = y_mapped[idx_te_flat]
            oa = accuracy_score(y_true_cpu, best_preds_this_seed)
            cm = confusion_matrix(y_true_cpu, best_preds_this_seed)
            per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
            aa = np.nanmean(per_class_acc)
            kappa = cohen_kappa_score(y_true_cpu, best_preds_this_seed)
            
            seed_metrics["oa"].append(oa)
            seed_metrics["aa"].append(aa)
            seed_metrics["kappa"].append(kappa)
            seed_metrics["class_acc"].append(per_class_acc)
            seed_metrics["inf_time_ms"].append(best_inf_time)
            
            if oa > best_seed_oa_for_cfg:
                best_seed_oa_for_cfg = oa
                best_seed_report_str = classification_report(
                    y_true_cpu, best_preds_this_seed, target_names=CLASS_NAMES, digits=4
                )
            
    


        results.append({
            "name": cfg_name,
            "mean_oa": np.mean(seed_metrics["oa"]),
            "std_oa": np.std(seed_metrics["oa"]),
            "mean_aa": np.mean(seed_metrics["aa"]),
            "std_aa": np.std(seed_metrics["aa"]),
            "mean_kappa": np.mean(seed_metrics["kappa"]),
            "std_kappa": np.std(seed_metrics["kappa"]),
            "class_acc_matrix": np.array(seed_metrics["class_acc"]),
            "best_report": best_seed_report_str,
            "seed_oas": seed_metrics["oa"],
            # ✅ 新增硬件统计数据
            "cls_params": cfg_cls_params,
            "mean_inf_time": np.mean(seed_metrics["inf_time_ms"]),
            "peak_vram_cls": cfg_peak_vram
        })
        
    # --- 4. 结果保存 ---
    results.sort(key=lambda x: x['mean_oa'], reverse=True)
    best_res = results[0] 
    
    print("\n" + "="*60)
    print(f"🏆 Best Configuration: {best_res['name']}")
    print(f"   OA: {best_res['mean_oa']*100:.2f}% (±{best_res['std_oa']*100:.2f}%)")
    
    # 确保文件夹存在
    save_dir = f"results_auto_best/{ARGS['dataset']}"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/{timestamp}_{ARGS['dataset']}_AutoSearch_BestReport.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        # === Part 1: 最佳组合的统计详情 ===
        f.write("="*60 + "\n")
        f.write(f"🏆 WINNER CONFIGURATION: {best_res['name']}\n")
        f.write("="*60 + "\n\n")
        
        # ✅ 新增报告模块：硬件与性能指标
        f.write("--- 0. Hardware & Performance Metrics ---\n")
        f.write(f"[Parameters]\n")
        f.write(f"  - VAE Parameters         : {vae_params:.2f} M\n")
        f.write(f"  - DiT Parameters         : {dit_params:.2f} M\n")
        f.write(f"  - Classifier Parameters  : {best_res['cls_params']:.4f} M\n")
        f.write(f"[FLOPs (单次前向传播)]\n")
        f.write(f"  - VAE FLOPs              : {vae_flops:,} ({vae_flops/1e9:.2f}G)\n")
        f.write(f"  - DiT FLOPs              : {dit_flops:,} ({dit_flops/1e9:.2f}G)\n")
        f.write(f"[Time]\n")
        f.write(f"  - Feature Extract Time   : {extract_time:.2f} Seconds\n")
        f.write(f"  - DiT Inference Time     : {dit_total_time:.4f} Seconds (Total), {dit_avg_time*1000:.2f} ms (Avg per call)\n")
        f.write(f"  - Classifier Inference   : {best_res['mean_inf_time']:.2f} ms (Avg per Test Set)\n")
        f.write(f"[GPU Memory]\n")
        f.write(f"  - Peak VRAM (Extraction) : {extract_vram:.2f} MB\n")
        f.write(f"  - Peak VRAM (Train/Test) : {best_res['peak_vram_cls']:.2f} MB\n\n")

        f.write("--- 1. Overall Metrics (Mean ± Std over 10 seeds) ---\n")
        f.write(f"OA:    {best_res['mean_oa']*100:.4f}% ± {best_res['std_oa']*100:.4f}%\n")
        f.write(f"AA:    {best_res['mean_aa']*100:.4f}% ± {best_res['std_aa']*100:.4f}%\n")
        f.write(f"Kappa: {best_res['mean_kappa']:.4f}  ± {best_res['std_kappa']:.4f}\n\n")
        
        f.write("--- 2. Per-Class Accuracy (Mean ± Std) ---\n")
        class_means = np.mean(best_res['class_acc_matrix'], axis=0)
        class_stds = np.std(best_res['class_acc_matrix'], axis=0)
        
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{name:<25}: {class_means[i]*100:.2f}% ± {class_stds[i]*100:.2f}%\n")
            
        f.write("\n")
        f.write("--- 3. Best Seed Detailed Report (Highest OA run) ---\n")
        f.write(best_res['best_report']) 
        f.write("\n\n")
        
        # === Part 2: 所有组合的排行榜 ===
        f.write("="*60 + "\n")
        f.write("🚀 LEADERBOARD (All Configs)\n")
        f.write("="*60 + "\n")
        f.write(f"{'Rank':<4} | {'Config':<25} | {'Mean OA':<10} | {'Std Dev':<10} | {'Cls Params(M)':<15} | {'Runs (OA)'}\n")
        f.write("-" * 95 + "\n")
        
        for i, res in enumerate(results):
            runs_str = ",".join([f"{x*100:.1f}" for x in res['seed_oas']])
            f.write(f"{i+1:<4} | {res['name']:<25} | {res['mean_oa']*100:.2f}%    | {res['std_oa']*100:.2f}%    | {res['cls_params']:<15.4f} | [{runs_str}]\n")
            
    print(f"✅ Full Report Saved: {filename}")
 