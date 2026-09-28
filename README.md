# HDiT — HyperSpectral Latent Diffusion for Feature Extraction & Classification

A two-stage latent-diffusion pipeline for **hyperspectral image (HSI) classification**.
A VAE compresses HSI patches into a compact latent space, a **DiT (Diffusion Transformer)**
is trained in that latent space, and the **intermediate denoising features** of the DiT are
automatically searched and used as transferable representations for a lightweight classifier.

> VAE (spectral reconstruction) → DiT (latent diffusion) → auto-search of (timestep, layer) features → classifier

---

## Pipeline

| Stage | Script | What it does | Main output |
|-------|--------|--------------|-------------|
| 1. VAE pretraining | `pretrain_vae.py` | Learns a spatial–spectral latent space with `α·MSE + β·KL + γ·SAM` | `pretrain_vae/<ds>/<ds>_vae-final_best.pth` |
| 2. DiT training | `train_dit.py` | Trains a DiT on frozen VAE latents (`timestep_respacing=500`) | `results_latent/<ds>/<ds>-dit_latent-500.pt` |
| 3. Feature search & classification | `extraction_classifier.py` | Scores timesteps/layers, searches feature combinations, trains a linear/MLP classifier, reports OA/AA/Kappa over multiple seeds | `results_auto_best/<ds>/<timestamp>_<ds>_AutoSearch_BestReport.txt` |

**Loss used for the VAE**

```
Loss = α · MSE(recon, x) + β · KL(q(z|x) ‖ N(0,I)) + γ · SAM(recon, x)
```

`SAM` is the *spectral angle mapper*, which is particularly important for HSI because it
preserves the shape of the spectral curve rather than only its magnitude.

Stage 3 first computes a **CH (cluster-homogeneity) score** for every timestep to find a
contiguous high-information interval, then exhaustively evaluates combinations of
`(timesteps × DiT layers)` and keeps the best configuration by mean overall accuracy.

---

## Repository layout

```
.
├── data.py                    # dataset registry, PCA preprocessing, patch creation / mmap cache
├── models.py                  # DiT backbone (adapted from facebookresearch/DiT)
├── autoencoder_kl.py          # VAE + MSE / KL / SAM loss
├── pretrain_vae.py            # Stage 1 entry point
├── train_dit.py               # Stage 2 entry point
├── extraction_classifier.py   # Stage 3 entry point
├── diffusion/                 # Gaussian diffusion (adapted from OpenAI GLIDE/ADM/IDDPM)
│   ├── __init__.py
│   ├── gaussian_diffusion.py
│   ├── respace.py
│   ├── timestep_sampler.py
│   └── diffusion_utils.py
├── docs/
│   └── commands.md            # ready-to-run command reference (per dataset)
├── results_auto_best/         # example auto-search reports (per dataset)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone <your-repo-url>
cd HDiT

conda create -n hdit python=3.10 -y
conda activate hdit
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended (the examples assume `cuda:0`).

---

## Dataset preparation

All datasets are read from a single root directory controlled by the **`HSI_DATASET_ROOT`**
environment variable (default: `./datasets`).

```bash
export HSI_DATASET_ROOT=/path/to/your/datasets
```

Expected layout (file names must match exactly):

```
$HSI_DATASET_ROOT/
├── PU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── Houston/
│   ├── Houston.mat
│   └── Houston_gt.mat
├── LongKou/
│   ├── WHU_Hi_LongKou.mat
│   └── WHU_Hi_LongKou_gt.mat
├── Salinas/
│   ├── Salinas_corrected.mat
│   └── Salinas_gt.mat
├── IP/  (Indian Pines)
├── KSC/
├── Botswana/
├── HongHu/
└── ...
```

> The datasets themselves are **not** included in this repository — please download them
> from their original providers and respect their individual licenses/terms of use.

---

## Quick start (PaviaU example)

### Stage 1 — Pretrain the VAE

```bash
python pretrain_vae.py -d PU --n-pc 16 --patch-size 32 --batch-size 1024 \
  --latent-channels 64 --block-out-channels 32,64 --layers-per-block 4 \
  --num-down-blocks 2 --alpha 1.0 --beta 0.01 --gamma 1.0 \
  -e 50 --lr 1e-3 --device cuda:0 --num-workers 4 \
  --patches-cache cache_patches
```

This also builds the memory-mapped patch cache used by the following stages.

### Stage 2 — Train the DiT in latent space

```bash
python train_dit.py -d PU -e 50 --batch-size 128 --lr 5e-4 \
  --device cuda:0 --num-workers 2
```

### Stage 3 — Feature extraction, search and classification

```bash
python extraction_classifier.py -d PU --label-condition "!=0" --batch-size 128 \
  --pool-layers "4" --min-combo-t 5 --max-combo-t 5 \
  --min-combo-l 1 --max-combo-l 1 --device cuda:0
```

The stage-3 script auto-resolves the VAE/DiT checkpoints from the outputs of stages 1–2
(override with `--vae-path` / `--dit-path`).

More per-dataset commands and the full parameter tables are in
[`docs/commands.md`](docs/commands.md).

---

## Runtime directory layout

These directories are created automatically and are **git-ignored** (they can be very
large — the patch cache alone can reach tens of GB for big scenes):

```
cache_patches/        # mmap patch cache  <ds>_X_p<patch>_<cond>.npy / <ds>_Y_...
save/pca_result/      # PCA-whitened full images (reused across runs)
pretrain_vae/         # VAE checkpoints + training history/report
results_latent/       # DiT latents + checkpoints
results_auto_best/    # auto-search reports (kept in the repo)
```

If you move a run to another machine, only the source code + datasets are required;
everything under the four generated directories above can be regenerated.

---

## Command-line reference

All parameters are documented in the module docstrings and in
[`docs/commands.md`](docs/commands.md). Highlights:

- `data.py`
  - `--n-pc` — number of PCA components
  - `--patch-size` — spatial patch size
  - `--label-condition` — `all` / `!=0` / `==0` / class id
- `pretrain_vae.py`
  - `--alpha` / `--beta` / `--gamma` — MSE / KL / SAM loss weights
  - `--latent-channels`, `--block-out-channels`, `--num-down-blocks`
- `train_dit.py`
  - `--model` — `DiT-S/2`, `DiT-B/2`, … or `custom` with `--dit-*`
  - `--timestep-respacing` — diffusion steps used at inference time
- `extraction_classifier.py`
  - `--pool-steps` / `--pool-layers` — manual (timestep, layer) selection
  - `--test-steps`, `--top-n`, `--smooth-window` — automatic timestep search
  - `--seeds`, `--samples-per-class` — evaluation protocol

---


## Acknowledgements

This project builds on several excellent open-source works:

- **DiT** — Scalable Diffusion Models with Transformers, `facebookresearch/DiT` (CC BY-NC 4.0)
- **guided-diffusion / GLIDE / improved-diffusion** — OpenAI (MIT)
- **diffusers** — HuggingFace, for the `AutoencoderKL` VAE components (Apache-2.0)
- **timm** — HuggingFace, for ViT building blocks (Apache-2.0)

Please review [`LICENSE`](LICENSE) for the third-party notices. **Note that the DiT
backbone is licensed for non-commercial use only.**

---

## 中文说明

**HDiT** 是一个面向高光谱图像（HSI）分类的**潜在扩散**两阶段框架：

1. **VAE 预训练**（`pretrain_vae.py`）：用 `MSE + KL + SAM` 损失把高光谱 patch 压缩到潜空间，
   其中 SAM（光谱角）损失保证光谱曲线形状。
2. **DiT 训练**（`train_dit.py`）：在冻结的 VAE 潜空间中训练扩散 Transformer。
3. **特征提取与分类**（`extraction_classifier.py`）：用 CH 分数自动挑选信息量高的时间步区间，
   再穷举 `(时间步 × DiT 层)` 组合，训练轻量分类器，并在多个随机种子上汇报 OA/AA/Kappa。

**数据集路径**：通过环境变量 `HSI_DATASET_ROOT` 指定，默认 `./datasets`，目录结构见上文。
仓库内**不含**任何数据集、权重和缓存文件。

**快速开始**：依次运行上面 Stage 1/2/3 三条命令即可；更多按数据集的完整命令见
[`docs/commands.md`](docs/commands.md)。

生成的中间目录（`cache_patches/`、`save/`、`results_latent/`、`pretrain_vae/`）都已被
`.gitignore` 忽略，不会误传到大文件平台；只有体积很小的实验报告 `results_auto_best/` 会随仓库提交。
