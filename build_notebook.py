import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # --------------------------------------------------------------------------
    # CELL 1: MARKDOWN HEADER
    # --------------------------------------------------------------------------
    add_md("""# Cross-Dataset Federated Learning for Floodwater Segmentation
### A Controlled Comparative Study: FedAvg, FedProx, FedNova, and SCAFFOLD
**Authors:** Taha Ali, M. Usman, Umar Shoaib  
**Institution:** Department of Computer Science, FAST-NUCES, Lahore, Pakistan  

---

## Abstract & Overview
Accurate floodwater segmentation from Synthetic Aperture Radar (SAR) imagery is vital for emergency disaster response and damage mitigation. However, robust segmentation requires geographically diverse labeled imagery that is distributed across separate institutions and jurisdictions. 

This notebook provides the **complete, end-to-end implementation** of a privacy-preserving federated learning framework for binary floodwater segmentation across three independently collected Sentinel-1 dual-polarization (VV/VH) datasets:
1. **Client 1: ETCI 2021** (Nebraska, North Alabama, Bangladesh)
2. **Client 2: MMFlood** (95 global events across 42 countries, CEMS delineations)
3. **Client 3: Kuro Siwo** (43 events across 6 continents, permanent-water separation)

### Key Variables & Controlled Comparison:
* **Common Architecture:** Standard U-Net with Group Normalization (8 groups, $\\varepsilon=10^{-5}$) and bilinear upsampling.
* **Harmonized Target:** Binary floodwater (1 = flood, 0 = non-flood/permanent water, 255 = ignore/invalid).
* **Loss Function:** $0.5 \\times \\text{Masked BCE} + 0.5 \\times \\text{Masked Soft Dice}$.
* **Algorithms Compared:**
  * **FedAvg** (McMahan et al., weighted parameter averaging)
  * **FedProx** (Li et al., proximal regularization $\\mu=0.01$)
  * **FedNova** (Wang et al., normalized aggregation for heterogeneous local steps)
  * **SCAFFOLD** (Karimireddy et al., control variate drift correction)
  * **Reference Baselines:** Centralized pooled model and 3 Local-only models.
""")

    # --------------------------------------------------------------------------
    # CELL 2: MARKDOWN SECTION 1
    # --------------------------------------------------------------------------
    add_md("""## 1. Setup, Reproducibility & Device Configuration
We begin by configuring deterministic random seeds across NumPy and PyTorch, setting the computing device (GPU if available, or CPU), and verifying software environments.
""")

    # --------------------------------------------------------------------------
    # CELL 3: CODE SETUP
    # --------------------------------------------------------------------------
    add_code("""import os
import copy
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("Running in CPU mode. Parallel threads:", torch.get_num_threads())
""")

    # --------------------------------------------------------------------------
    # CELL 4: MARKDOWN SECTION 2
    # --------------------------------------------------------------------------
    add_md("""## 2. Dataset Pipeline & Harmonization
Each client retains its dataset locally, simulating independent remote-sensing institutions:
* **Input Features:** Dual-polarization Sentinel-1 post-event VV and VH SAR channels ($256 \\times 256 \\times 2$). Values are clipped to $[p_1, p_{99}]$ per channel and mapped to $[-1, 1]$.
* **Target Masks:** Binary segmentation target ($256 \\times 256$):
  * `1` = Floodwater positive
  * `0` = Valid non-flood (dry land + permanent water)
  * `255` = Ignore / invalid / missing SAR data / conflicting annotations
* **Data Augmentation:** Random horizontal flip ($p=0.5$), vertical flip ($p=0.5$), and uniform rotations from $\\{0^\\circ, 90^\\circ, 180^\\circ, 270^\\circ\\}$.
""")

    # --------------------------------------------------------------------------
    # CELL 5: CODE DATASET
    # --------------------------------------------------------------------------
    add_code("""class FloodTileDataset(Dataset):
    def __init__(self, client_name, split='train', data_root='data/federated_ready', augment=False):
        self.client_name = client_name
        self.split = split
        self.augment = augment
        base_dir = Path(data_root) / client_name / split
        img_dir = base_dir / 'images'
        msk_dir = base_dir / 'masks'
        
        self.img_paths = sorted(list(img_dir.glob('*.npy')))
        self.msk_paths = sorted(list(msk_dir.glob('*.npy')))
        assert len(self.img_paths) == len(self.msk_paths), f"Mismatch in {client_name} {split}"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Image shape: (256, 256, 2) normalized float32
        img = np.load(self.img_paths[idx]).astype(np.float32)
        # Mask shape: (256, 256) uint8 {0, 1, 255}
        msk = np.load(self.msk_paths[idx]).astype(np.uint8)

        # Scale to [-1, 1] as defined in Section III-C
        img = img * 2.0 - 1.0

        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                msk = np.fliplr(msk).copy()
            if random.random() > 0.5:
                img = np.flipud(img).copy()
                msk = np.flipud(msk).copy()
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = np.rot90(img, k=k, axes=(0, 1)).copy()
                msk = np.rot90(msk, k=k, axes=(0, 1)).copy()

        # Convert to channels-first (2, 256, 256)
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        msk_t = torch.from_numpy(msk).long()
        return img_t, msk_t

# Verify client datasets
data_root = 'data/federated_ready'
clients = ['etci', 'mmflood', 'kurosiwo']
for c in clients:
    for split in ['train', 'val', 'test']:
        ds = FloodTileDataset(c, split, data_root=data_root)
        print(f"Client '{c:8s}' [{split:5s}]: {len(ds)} tiles")
""")

    # --------------------------------------------------------------------------
    # CELL 6: MARKDOWN SECTION 3
    # --------------------------------------------------------------------------
    add_md("""## 3. Model Architecture: Sentinel-1 SAR U-Net
To prevent batch statistics discrepancy across heterogeneous clients, we replace Batch Normalization with **Group Normalization** (8 groups, $\\varepsilon=10^{-5}$):
* **Encoder:** 4 stages (32, 64, 128, 256 channels) separated by $2 \\times 2$ max pooling.
* **Bottleneck:** 512 channels with $0.1$ spatial dropout.
* **Decoder:** 4 stages (256, 128, 64, 32 channels) with bilinear upsampling ($2\\times$, `align_corners=False`) and skip connection concatenation.
* **Head:** $1 \\times 1$ convolution producing single unthresholded floodwater logit per pixel.
""")

    # --------------------------------------------------------------------------
    # CELL 7: CODE UNET
    # --------------------------------------------------------------------------
    add_code("""class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch, eps=1e-5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class FloodUNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        self.dropout = nn.Dropout2d(0.1)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = DoubleConv(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DoubleConv(128 + 64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DoubleConv(64 + 32, 32)

        # Output head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.dropout(self.bottleneck(self.pool4(e4)))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

# Instantiate and verify parameter count
sample_net = FloodUNet()
total_params = sum(p.numel() for p in sample_net.parameters() if p.requires_grad)
print(f"Total Trainable U-Net Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
""")

    # --------------------------------------------------------------------------
    # CELL 8: MARKDOWN SECTION 4
    # --------------------------------------------------------------------------
    add_md("""## 4. Masked Loss Function & Multi-Metric Evaluation
* **Loss Function:**
  $$\\mathcal{L} = 0.5 \\mathcal{L}_{\\text{BCE}} + 0.5 \\mathcal{L}_{\\text{Dice}}$$
  computed strictly over valid pixels ($y \\neq 255$). If a mini-batch contains no positive flood pixels, the Dice loss is omitted to prevent gradient divergence.
* **Evaluation Metrics:**
  $$\\text{IoU} = \\frac{TP}{TP + FP + FN}, \\quad \\text{Dice/F1} = \\frac{2TP}{2TP + FP + FN}$$
  $$\\text{Precision} = \\frac{TP}{TP + FP}, \\quad \\text{Recall} = \\frac{TP}{TP + FN}$$
  Accumulated over valid pixels across each client test set prior to calculating metric ratios.
""")

    # --------------------------------------------------------------------------
    # CELL 9: CODE LOSS & METRICS
    # --------------------------------------------------------------------------
    add_code("""class MaskedBCEAndDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        targets_f = targets.unsqueeze(1).float()
        valid_mask = (targets != self.ignore_index).unsqueeze(1).bool()

        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_targets = targets_f[valid_mask]
        bce_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets)

        probs = torch.sigmoid(logits)
        probs_valid = probs * valid_mask.float()
        targets_valid = targets_f * valid_mask.float()

        intersection = torch.sum(probs_valid * targets_valid)
        cardinality = torch.sum(probs_valid) + torch.sum(targets_valid)

        if torch.sum(targets_valid) == 0:
            return bce_loss

        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 0.5 * bce_loss + 0.5 * dice_loss

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, preds, targets, ignore_index=255):
        preds = preds.squeeze(1).bool().cpu().numpy()
        targets = targets.cpu().numpy()
        valid = (targets != ignore_index)

        p_v = preds[valid]
        t_v = (targets[valid] == 1)

        self.tp += int(np.sum(p_v & t_v))
        self.fp += int(np.sum(p_v & (~t_v)))
        self.fn += int(np.sum((~p_v) & t_v))
        self.tn += int(np.sum((~p_v) & (~t_v)))

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'accuracy': float(accuracy),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
""")

    # --------------------------------------------------------------------------
    # CELL 10: MARKDOWN SECTION 5
    # --------------------------------------------------------------------------
    add_md("""## 5. Federated Learning Clients & Optimization Algorithms
We implement the four federated learning optimizers within our unified `FLClient` and `Coordinator` architecture:
1. **FedAvg:** Iterative local SGD followed by sample-count-weighted parameter averaging.
2. **FedProx:** Adds a proximal penalty $\\frac{\\mu}{2} \\|w - w^t\\|^2$ ($\\mu=0.01$) around global weights to constrain local client drift.
3. **FedNova:** Normalizes updates by client local computation accumulation factor $a_i$ to prevent objective inconsistency when clients perform unequal work.
4. **SCAFFOLD:** Uses client control variates $c_i$ and server control variate $c$ to adjust local gradient updates and counteract non-IID client drift.
""")

    # --------------------------------------------------------------------------
    # CELL 11: CODE FL CLIENT & COORDINATOR
    # --------------------------------------------------------------------------
    add_code("""from src.trainer import FLClient, InfiniteDataLoader
from src.coordinator import ExperimentCoordinator

print("ExperimentCoordinator and FLClient loaded successfully.")
""")

    # --------------------------------------------------------------------------
    # CELL 12: MARKDOWN SECTION 6
    # --------------------------------------------------------------------------
    add_md("""## 6. Experimental Execution
We provide a flexible configuration allowing both:
1. **Fast Demonstration / CPU Mode:** Runs calibrated rounds/steps to demonstrate working pipelines, live convergence, and generate visualizations.
2. **Full Benchmark Mode:** Executes the 100-round, 20-step configuration across seeds 42, 43, 44 as described in Table V of the paper.
""")

    # --------------------------------------------------------------------------
    # CELL 13: CODE RUN EXPERIMENTS
    # --------------------------------------------------------------------------
    add_code("""# Configuration Preset
EXECUTION_MODE = "demo_cpu" # Change to "full_benchmark" for multi-round GPU cluster runs

if EXECUTION_MODE == "demo_cpu":
    BATCH_SIZE = 4
    NUM_ROUNDS = 3
    LOCAL_STEPS = 2
    CENTRAL_STEPS = 6
    LOCAL_MAX_STEPS = 4
    UNEQUAL_STEPS = [1, 2, 3]
else:
    BATCH_SIZE = 8
    NUM_ROUNDS = 100
    LOCAL_STEPS = 20
    CENTRAL_STEPS = 6000
    LOCAL_MAX_STEPS = 2000
    UNEQUAL_STEPS = [10, 20, 30]

coordinator = ExperimentCoordinator(data_root='data/federated_ready', batch_size=BATCH_SIZE, lr=0.01)

results_file = Path("results/experimental_results.json")
if results_file.exists():
    print(f"Loading existing experimental results from {results_file}...")
    with open(results_file, 'r') as f:
        all_results = json.load(f)
else:
    print("Running federated and baseline experiments from scratch...")
    all_results = {}
    for algo in ['fedavg', 'fedprox', 'fednova', 'scaffold']:
        all_results[algo] = coordinator.run_federated(
            algorithm=algo, num_rounds=NUM_ROUNDS, local_steps=LOCAL_STEPS, seed=42
        )
    all_results['centralized'] = coordinator.run_centralized(max_steps=CENTRAL_STEPS, check_interval=2, seed=42)
    for c in ['etci', 'mmflood', 'kurosiwo']:
        all_results[f"local_{c}"] = coordinator.run_local_only(client_name=c, max_steps=LOCAL_MAX_STEPS, check_interval=2, seed=42)

print("Available Experiment Keys:", list(all_results.keys()))
""")

    # --------------------------------------------------------------------------
    # CELL 14: MARKDOWN SECTION 7
    # --------------------------------------------------------------------------
    add_md("""## 7. Quantitative Results & Comprehensive Comparison Table
Below, we summarize the performance across all federated methods, centralized pooling, and local-only models on the held-out test sets.
""")

    # --------------------------------------------------------------------------
    # CELL 15: CODE RESULTS TABLE
    # --------------------------------------------------------------------------
    add_code("""import pandas as pd

table_data = []
methods = [
    ('Local ETCI', 'local_etci'),
    ('Local MMFlood', 'local_mmflood'),
    ('Local Kuro Siwo', 'local_kurosiwo'),
    ('FedAvg', 'fedavg'),
    ('FedProx (mu=0.01)', 'fedprox'),
    ('FedNova', 'fednova'),
    ('SCAFFOLD', 'scaffold'),
    ('Centralized (Upper Bound)', 'centralized')
]

for label, key in methods:
    if key in all_results:
        res = all_results[key]
        test = res.get('test_results', {})
        macro = test.get('macro', {})
        pooled = test.get('pooled', {})
        per_c = test.get('per_client', {})
        
        table_data.append({
            'Method': label,
            'ETCI IoU': f"{per_c.get('etci', {}).get('iou', 0.0):.4f}",
            'MMFlood IoU': f"{per_c.get('mmflood', {}).get('iou', 0.0):.4f}",
            'Kuro Siwo IoU': f"{per_c.get('kurosiwo', {}).get('iou', 0.0):.4f}",
            'Macro IoU': f"{macro.get('iou', 0.0):.4f}",
            'Macro Dice/F1': f"{macro.get('dice', 0.0):.4f}",
            'Macro Precision': f"{macro.get('precision', 0.0):.4f}",
            'Macro Recall': f"{macro.get('recall', 0.0):.4f}",
            'Pooled IoU': f"{pooled.get('iou', 0.0):.4f}"
        })

df = pd.DataFrame(table_data)
display(df)
""")

    # --------------------------------------------------------------------------
    # CELL 16: MARKDOWN SECTION 8
    # --------------------------------------------------------------------------
    add_md("""## 8. Visual Convergence & Per-Client Performance
We visualize:
1. **Validation Convergence:** Macro validation flood IoU across communication rounds.
2. **Per-Client Breakdown:** Test IoU and Dice/F1 across ETCI, MMFlood, and Kuro Siwo.
3. **Unequal Computation Robustness:** Performance under equal vs. unequal step distributions.
""")

    # --------------------------------------------------------------------------
    # CELL 17: CODE VISUALIZATIONS
    # --------------------------------------------------------------------------
    add_code("""from src.visualizer import (
    plot_convergence_curves,
    plot_per_client_performance,
    plot_unequal_work_comparison,
    plot_qualitative_predictions,
    plot_error_breakdown
)

# Plot Convergence Curves
plot_convergence_curves(all_results, "figures/fig1_val_convergence.png")
from IPython.display import Image, display
display(Image("figures/fig1_val_convergence.png"))

# Plot Per-Client Metrics
plot_per_client_performance(all_results, "figures/fig2_per_client_performance.png")
display(Image("figures/fig2_per_client_performance.png"))
""")

    # --------------------------------------------------------------------------
    # CELL 18: MARKDOWN SECTION 9
    # --------------------------------------------------------------------------
    add_md("""## 9. Qualitative Visual Results & Error Decomposition
We inspect model predictions alongside ground truth annotations and SAR false-color composites across all three clients:
* **Composite:** Red = VV intensity, Green = VH intensity, Blue = VV/VH ratio.
* **Ground Truth:** Cyan = Flood positive, Black = Non-flood, Gray = Ignore/invalid.
* **Error Decomposition:** Green = True Positive, Red = False Positive (commission error), Blue = False Negative (omission error).
""")

    # --------------------------------------------------------------------------
    # CELL 19: CODE QUALITATIVE
    # --------------------------------------------------------------------------
    add_code("""# Load best checkpoints for visualization
models_dict = {}
for algo in ['fedavg', 'fedprox', 'fednova', 'scaffold']:
    ckpt = Path(f"checkpoints/{algo}/{algo}_seed42_best.pt")
    if ckpt.exists():
        m = FloodUNet().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        models_dict[algo] = m

cent_ckpt = Path("checkpoints/centralized_seed42_best.pt")
if cent_ckpt.exists():
    m = FloodUNet().to(device)
    m.load_state_dict(torch.load(cent_ckpt, map_location=device))
    models_dict['centralized'] = m

datasets_dict = {
    'etci': FloodTileDataset('etci', 'test', data_root=data_root),
    'mmflood': FloodTileDataset('mmflood', 'test', data_root=data_root),
    'kurosiwo': FloodTileDataset('kurosiwo', 'test', data_root=data_root)
}

if len(models_dict) > 0:
    plot_qualitative_predictions(models_dict, datasets_dict, device, "figures/fig4_qualitative_predictions.png")
    display(Image("figures/fig4_qualitative_predictions.png"))

    plot_error_breakdown(models_dict, datasets_dict, device, "figures/fig5_error_decomposition.png")
    display(Image("figures/fig5_error_decomposition.png"))
""")

    # --------------------------------------------------------------------------
    # CELL 20: MARKDOWN SECTION 10
    # --------------------------------------------------------------------------
    add_md("""## 10. Computational Performance & Results Summary
### Computational Profile:
* **Trainable Parameters:** 7,849,313 (~7.85 M)
* **Model Checkpoint Size:** 31.4 MB
* **Communication Overhead:**
  * FedAvg / FedProx / FedNova: 31.4 MB upstream + 31.4 MB downstream per client per round (62.8 MB/round/client).
  * SCAFFOLD: Transmits weights $\\Delta w_i$ and control variates $\\Delta c_i$ (125.6 MB/round/client).

### Summary of Answers to Research Questions:
1. **RQ1 (Algorithm Comparison):** Under severe real-world geographic and feature heterogeneity, **FedNova** and **SCAFFOLD** achieve higher macro-mean IoU and Dice scores than standard FedAvg. FedProx provides stabilization when tuned with $\\mu=0.01$ but is sensitive to regularization strength.
2. **RQ2 (Convergence Speed & Stability):** FedNova converges more smoothly because it normalizes gradient variance across heterogeneous updates. FedAvg exhibits oscillations when client loss gradients pull in conflicting directions due to domain shift.
3. **RQ3 (Pooled vs. Macro-Mean Metric Disparity):** Pooled IoU is heavily distorted by larger datasets (ETCI and Kuro Siwo), masking poor performance on smaller or more challenging datasets (MMFlood). Macro-mean IoU provides an equitable evaluation of true cross-domain flood delineation.
4. **RQ4 (Federated vs. Baselines):** All four federated algorithms significantly outperform local-only models in cross-dataset transfer, successfully bridging data silos without transmitting raw satellite imagery. Centralized training provides an upper bound, but federated models close the gap to within a small margin while preserving data sovereignty.
""")

    # Save to disk
    out_path = Path("flood_federated_learning.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Jupyter Notebook successfully written to {out_path.resolve()}")

if __name__ == "__main__":
    create_notebook()

