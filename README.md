# Cross-Dataset Federated Learning for Floodwater Segmentation

This repository contains the implementation of the federated learning framework for binary floodwater segmentation across heterogeneous Sentinel-1 synthetic aperture radar (SAR) datasets: **ETCI 2021**, **MMFlood**, and **Kuro Siwo**.

The experimental design investigates four federated optimization algorithms (**FedAvg**, **FedProx**, **FedNova**, and **SCAFFOLD**) alongside localized and centralized reference baselines under controlled architecture, loss, and evaluation protocols.

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
- **Local Optimizer**: SGD without momentum, learning rate $\eta = 0.01$, weight decay $10^{-4}$, gradient norm clipping at $5.0$. Batch size: 8 tiles.
- **Federated Budget**:
  - **Main Experiment**: 20 local updates per client per round (all 3 clients participating each round), up to 100 rounds.
  - **Unequal Computation Experiment**: Rotated allocations of $(10, 20, 30)$ local steps across clients over seeds 42, 43, 44 (total 60 steps/round).
  - **Validation & Early Stopping**: Global model evaluated every round on each client's validation set; best checkpoint selected by highest macro-mean validation flood IoU. Early stopping triggers after 15 consecutive rounds without improvement.

### 4. Reference Baselines
- **Local-Only**: Individual U-Net models trained per client on local data (max 2,000 steps, evaluated every 20 steps, patience 15).
- **Centralized**: Single U-Net trained on pooled training events across all three clients (max 6,000 steps, evaluated every 60 steps, patience 15).

---

## Repository Structure

```
Robust-Flood-Area-Segmentation/
├── configs/
│   └── default.yaml          # Hyperparameters and experiment configurations
├── src/
│   ├── models/
│   │   └── unet.py           # Common U-Net with GroupNorm and dropout
│   ├── losses/
│   │   └── flood_loss.py     # Masked BCE + Soft Dice loss
│   ├── metrics/
│   │   └── flood_metrics.py  # Confusion matrix accumulator and evaluation metrics
│   ├── data/
│   │   ├── transforms.py     # SAR normalization, padding, and augmentations
│   │   ├── etci.py           # ETCI 2021 decoding and split rules
│   │   ├── mmflood.py        # MMFlood decoding and EMS split rules
│   │   ├── kuro_siwo.py      # Kuro Siwo decoding and event split rules
│   │   ├── dataset.py        # FloodDataset and DataLoader utilities
│   │   └── preprocess.py     # Data audit, manifest builder, and mock generator
│   ├── federated/
│   │   ├── client.py         # Client training (FedAvg, FedProx, FedNova, SCAFFOLD)
│   │   └── server.py         # Server aggregation, validation, and evaluation
│   ├── baselines/
│   │   ├── local.py          # Local-only reference training
│   │   └── centralized.py    # Pooled centralized reference training
│   └── utils/
│       └── seed.py           # Deterministic seeding helper
├── scripts/
│   ├── generate_mock_data.py # Generates synthetic tiles for pipeline validation
│   ├── run_federated.py      # Federated experiment runner (main & unequal)
│   ├── run_local.py          # Local reference runner
│   ├── run_centralized.py    # Centralized reference runner
│   └── evaluate.py           # Checkpoint evaluation script
├── tests/
│   ├── test_unet.py          # U-Net architecture and seed tests
│   ├── test_loss.py          # Masked loss edge case tests
│   ├── test_metrics.py       # Metrics accumulation tests
│   └── test_federated.py     # End-to-end smoke tests for all 4 FL algorithms
├── requirements.txt
└── .gitignore
```

---

## Setup & Quickstart

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Generate Verification Data
Before downloading large satellite archives, verify the complete pipeline using the mock data generator:
```bash
python scripts/generate_mock_data.py --output_dir data/mock --tiles_per_split 16
```
This produces `data/mock/manifest.json` and `data/mock/norm_stats.json`.

### 3. Run Federated Training
**Main experiment (equal local steps: 20 per client/round):**
```bash
python scripts/run_federated.py --algorithm fedavg --experiment main --seed 42 --manifest data/mock/manifest.json --stats data/mock/norm_stats.json
```
Supported algorithms: `fedavg`, `fedprox`, `fednova`, `scaffold`.

**Unequal computation experiment (10, 20, 30 steps/round):**
```bash
python scripts/run_federated.py --algorithm fednova --experiment unequal --seed 42 --manifest data/mock/manifest.json --stats data/mock/norm_stats.json
```

### 4. Run Reference Baselines
**Local-only models:**
```bash
python scripts/run_local.py --client all --seed 42 --manifest data/mock/manifest.json --stats data/mock/norm_stats.json
```

**Pooled centralized model:**
```bash
python scripts/run_centralized.py --seed 42 --manifest data/mock/manifest.json --stats data/mock/norm_stats.json
```

### 5. Evaluate Checkpoints
```bash
python scripts/evaluate.py --checkpoint checkpoints/federated/main_fedavg/fedavg_seed42_best.pt --manifest data/mock/manifest.json --stats data/mock/norm_stats.json
```

### 6. Run Unit Tests
```bash
python tests/test_unet.py
python tests/test_loss.py
python tests/test_metrics.py
python tests/test_federated.py
```
