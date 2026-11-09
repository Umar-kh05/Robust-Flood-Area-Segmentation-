# Cross-Dataset Federated Learning for Floodwater Segmentation

This repository contains the implementation of the federated learning framework for binary floodwater segmentation across heterogeneous Sentinel-1 synthetic aperture radar (SAR) datasets: **ETCI 2021**, **MMFlood**, and **Kuro Siwo**.

The experimental design investigates four federated optimization algorithms (**FedAvg**, **FedProx**, **FedNova**, and **SCAFFOLD**) alongside localized and centralized reference baselines under controlled architecture, loss, and evaluation protocols.

---

## Repository Layout

| Path | Contents |
|---|---|
| `flood_federated_learning.ipynb` | **v1** self-contained Kaggle notebook (original protocol) |
| `flood_federated_learning_v2.ipynb` | **v2** improved notebook — longer training, momentum + cosine LR, threshold tuning, TTA flag, stronger augmentation |
| `nb_core.py` / `nb_core_v2.py` | Embedded training/FL/plotting code used to build the notebooks |
| `build_kaggle_notebook*.py` | Notebook assembly scripts |
| `src/` | Modular Python package (models, losses, metrics, data loaders, federated client/server, baselines) |
| `scripts/` | CLI runners for the `src/` package |
| `configs/` | Hyperparameter configuration (`default.yaml`) |
| `tests/` | Unit and smoke tests |
| `data/federated_kaggle.zip` | Kaggle-ready dataset archive (600 tiles, 3 clients, train/val/test) — stored via **Git LFS** |
| `checkpoints/` | Trained model weights (seed 42, all methods) — stored via **Git LFS** |
| `results/` | Result figures and metrics output |
| `figures/` | Paper figures |
| `report.tex` / `report.pdf` | Paper report and LaTeX sources |

---

## Methodological Summary

### 1. Data Clients & Preprocessing
Each client retains its dataset locally, simulating decentralized flood mapping institutions:
- **Client 1 (ETCI 2021)**: Post-event VV/VH imagery. Masks decoded to binary (`0` non-flood, `1` flood, `255` ignore). Partitioned into Nebraska (train), North Alabama (validation), and Bangladesh (test).
- **Client 2 (MMFlood)**: Matched `s1_raw`, `mask`, and `hydro`. Hydrography used for conflict resolution (flood/water conflicts set to `255` ignore). EMS activation codes split 60/20/20 (train/val/test) deterministically.
- **Client 3 (Kuro Siwo)**: Post-event GRD VV/VH. Class 2 mapped to `1`, classes 0 & 1 mapped to `0`, invalid pixels mapped to `255`. Test partition preserves the official 10-event holdout; remaining events split 80/20 into train/validation.

**Tiling & Normalization**:
- Non-overlapping $256 \times 256$ tiles. Incomplete edge tiles are zero-padded bottom/right, with padded mask locations assigned `255`.
- Per client and channel, 1st and 99th percentiles are calculated exclusively on training events, clipped to $[p_1, p_{99}]$, and mapped to $[-1, 1]$.
- Spatial augmentations on training tiles: horizontal flips ($p=0.5$), vertical flips ($p=0.5$), and rotations from $\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$.

### 2. Common Segmentation Architecture
- Standard U-Net tailored for 2-channel Sentinel-1 SAR (VV, VH):
  - Encoder: 4 stages (32, 64, 128, 256 channels) with $2 \times 2$ max pooling.
  - Bottleneck: 512 channels with $0.1$ dropout.
  - Decoder: 4 stages (256, 128, 64, 32 channels) with bilinear upsampling ($2\times$, `align_corners=False`) and skip concatenation.
  - Normalization: Group Normalization (8 groups, $\varepsilon = 10^{-5}$) throughout the network to avoid client batch-statistic discrepancies.
  - Output: $1 \times 1$ convolution producing single unthresholded logit per pixel.
  - Common seed initialization across all methods.

### 3. Loss & Optimization
- **Loss**: $0.5 \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{Dice}}$ over valid pixels ($y \neq 255$).
  - If a mini-batch contains no positive flood pixels, the Dice term is omitted.
  - All-invalid batches are discarded without advancing optimizer step counters.
- **v1 local optimizer**: SGD without momentum, $\eta = 0.01$, weight decay $10^{-4}$, grad-norm clip $5.0$.
- **v2 local optimizer**: SGD **momentum 0.9 + Nesterov**, $\eta = 0.01$, weight decay $10^{-4}$, grad-norm clip $5.0$, 5-step warmup + cosine decay. Optional `pos_weight` for class imbalance.
- **v2 evaluation**: decision threshold tuned per run on validation IoU (sweep 0.2–0.8); optional 8-view flip/rotation TTA.
- **Federated budget**:
  - **v1**: 10 local steps/client/round, up to 15 rounds (reduced for Kaggle).
  - **v2**: 30 local steps/client/round, up to 60 rounds; centralized reference matched at ~5,400 steps, local-only at ~1,800 steps.
  - **Unequal computation**: rotated local-step allocations across clients.
  - **Validation & Early Stopping**: best checkpoint by highest macro-mean validation flood IoU; early stop after 15 non-improving rounds/checks.

### 4. Reference Baselines
- **Local-Only**: Individual U-Net models trained per client on local data.
- **Centralized**: Single U-Net trained on pooled training events across all three clients.

---

## Running the Experiments

### Option A — Kaggle notebook (recommended, GPU)
1. Create a Kaggle Notebook, set **Accelerator = GPU**.
2. Attach the dataset (`data/federated_kaggle.zip`, already uploaded as a Kaggle Dataset) or upload the extracted `federated_kaggle/` folder.
3. Open `flood_federated_learning_v2.ipynb` → **Run All**.
4. Edit the **Section 3** config cell to scale rounds/steps/seeds to your time budget.

### Option B — Modular `src/` package
```bash
pip install -r requirements.txt

# Generate verification data
python scripts/generate_mock_data.py --output_dir data/mock --tiles_per_split 16

# Federated training (main & unequal experiments)
python scripts/run_federated.py --algorithm fedavg --experiment main --seed 42 \
    --manifest data/mock/manifest.json --stats data/mock/norm_stats.json

# Reference baselines
python scripts/run_local.py --client all --seed 42 \
    --manifest data/mock/manifest.json --stats data/mock/norm_stats.json
python scripts/run_centralized.py --seed 42 \
    --manifest data/mock/manifest.json --stats data/mock/norm_stats.json

# Evaluate checkpoints
python scripts/evaluate.py --checkpoint checkpoints/federated/main_fedavg/fedavg_seed42_best.pt \
    --manifest data/mock/manifest.json --stats data/mock/norm_stats.json
```

Supported federated algorithms: `fedavg`, `fedprox`, `fednova`, `scaffold`.

### Tests
```bash
python tests/test_unet.py
python tests/test_loss.py
python tests/test_metrics.py
python tests/test_federated.py
```

---

## Results

Trained checkpoints for all methods (seed 42) are in `checkpoints/`; figures and metrics are in `results/` and `figures/`; the compiled paper is `report.pdf`.

> Raw full datasets (`data/federated_ready/`) are **not** included — they exceed repository size limits. Use the download/formatting scripts (`download_datasets.py`, `format_federated_datasets.py`, `format_mmlflood.py`, `prepare_kaggle_dataset.py`) to rebuild them, or use the included `federated_kaggle.zip`.
