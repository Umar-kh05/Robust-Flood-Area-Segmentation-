"""Embedded code for the self-contained Kaggle notebook.

Two string constants:
    CODE_CORE : dataset, model, loss, metrics, FL client + coordinator, aggregation
    VIZ_CORE  : all plotting helpers (convergence, per-client, unequal, confusion
                matrix, qualitative predictions, error decomposition)

These strings are written verbatim into notebook code cells by
build_kaggle_notebook.py. The notebook therefore runs with NO external `src`
package -- it is fully self-contained for a Kaggle session.
"""

CODE_CORE = r'''
import os, copy, time, json, random, glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def _show(df):
    """Render a DataFrame in-notebook when IPython is available, else print."""
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Data path auto-detection (Kaggle mount or local folder)
# ---------------------------------------------------------------------------
def resolve_data_root(dataset_name='flood-federated-kaggle', folder='federated_kaggle'):
    candidates = ['data/' + folder]
    input_root = Path('/kaggle/input')
    if input_root.exists():
        for sub in sorted(input_root.iterdir()):
            candidates.append(str(sub / folder))
            candidates.append(str(sub))
            for deeper in sorted(sub.iterdir()):
                candidates.append(str(deeper / folder))
                candidates.append(str(deeper))
    for path in candidates:
        if Path(path, 'etci', 'train', 'images').exists():
            print('[Data] Using data root:', path)
            return path
    raise FileNotFoundError('Could not locate federated dataset (looked for ' + folder + ')')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Dataset & augmentation (Section III-C)
# ---------------------------------------------------------------------------
class FloodTileDataset(Dataset):
    def __init__(self, client_name, split='train', data_root='federated_kaggle', augment=False):
        self.augment = augment
        base = Path(data_root) / client_name / split
        self.img_paths = sorted(base.joinpath('images').glob('*.npy'))
        self.msk_paths = sorted(base.joinpath('masks').glob('*.npy'))
        assert len(self.img_paths) == len(self.msk_paths) and len(self.img_paths) > 0, \
            f'Image/mask mismatch or empty for {client_name}/{split}'

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)   # (256,256,2) in [0,1]
        msk = np.load(self.msk_paths[idx]).astype(np.uint8)     # (256,256) in {0,1,255}
        img = img * 2.0 - 1.0                                    # map to [-1,1]
        if self.augment:
            if random.random() > 0.5:
                img, msk = np.fliplr(img).copy(), np.fliplr(msk).copy()
            if random.random() > 0.5:
                img, msk = np.flipud(img).copy(), np.flipud(msk).copy()
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = np.rot90(img, k=k, axes=(0, 1)).copy()
                msk = np.rot90(msk, k=k, axes=(0, 1)).copy()
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        msk_t = torch.from_numpy(msk).long()
        return img_t, msk_t


# ---------------------------------------------------------------------------
# U-Net with GroupNorm + bilinear upsampling + bottleneck dropout (Section III-B)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch, eps=1e-5), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch, eps=1e-5), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FloodUNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64);           self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128);          self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(128, 256);         self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.dropout = nn.Dropout2d(0.1)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = DoubleConv(512 + 256, 256)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DoubleConv(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DoubleConv(64 + 32, 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.dropout(self.bottleneck(self.pool4(e4)))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Masked 0.5 BCE + 0.5 Soft-Dice loss (Eq. 1)
# ---------------------------------------------------------------------------
class MaskedBCEAndDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        targets_f = targets.unsqueeze(1).float()
        valid = (targets != self.ignore_index).unsqueeze(1).bool()
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        vl, vt = logits[valid], targets_f[valid]
        bce = F.binary_cross_entropy_with_logits(vl, vt)
        probs = torch.sigmoid(logits)
        pv = probs * valid.float()
        tv = targets_f * valid.float()
        inter = torch.sum(pv * tv)
        card = torch.sum(pv) + torch.sum(tv)
        if torch.sum(tv) == 0:
            return bce
        dice = 1.0 - (2.0 * inter + self.eps) / (card + self.eps)
        return 0.5 * bce + 0.5 * dice


# ---------------------------------------------------------------------------
# Pixel-level metric accumulation (Eq. 3-4)
# ---------------------------------------------------------------------------
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, preds, targets, ignore_index=255):
        preds = preds.squeeze(1).bool().cpu().numpy()
        targets = targets.cpu().numpy()
        valid = targets != ignore_index
        p_v, t_v = preds[valid], (targets[valid] == 1)
        self.tp += int(np.sum(p_v & t_v))
        self.fn += int(np.sum((~p_v) & t_v))
        self.fp += int(np.sum(p_v & (~t_v)))
        self.tn += int(np.sum((~p_v) & (~t_v)))

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn
        return {
            'iou': float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0,
            'dice': float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0,
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'accuracy': float((tp + tn) / total) if total > 0 else 0.0,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }


def get_params(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def set_params(model, params):
    model.load_state_dict(params)


def fed_average(client_params, weights):
    out = {}
    for k in client_params[0]:
        out[k] = sum(client_params[i][k] * weights[i] for i in range(len(client_params)))
    return out


class InfiniteDataLoader:
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def next_batch(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


# ---------------------------------------------------------------------------
# Federated client with FedAvg / FedProx / SCAFFOLD local updates (Section III-D)
# ---------------------------------------------------------------------------
class FLClient:
    def __init__(self, name, data_root, batch_size=8, lr=0.01, weight_decay=1e-4, grad_clip=5.0):
        self.name = name
        self.batch_size, self.lr, self.weight_decay, self.grad_clip = batch_size, lr, weight_decay, grad_clip
        self.loss_fn = MaskedBCEAndDiceLoss()
        self.train_ds = FloodTileDataset(name, 'train', data_root, augment=True)
        self.val_ds = FloodTileDataset(name, 'val', data_root, augment=False)
        self.test_ds = FloodTileDataset(name, 'test', data_root, augment=False)
        self.num_train = len(self.train_ds)
        pin = DEVICE.type == 'cuda'
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=pin)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        self.train_iter = InfiniteDataLoader(self.train_loader)
        self.c_i = None

    def init_scaffold(self, model):
        self.c_i = {k: torch.zeros_like(v, device=DEVICE) for k, v in model.named_parameters()}

    def _train_step(self, model, optimizer, imgs, masks):
        imgs, masks = imgs.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        loss = self.loss_fn(model(imgs), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        return loss

    def train_fedavg(self, model, num_steps):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        tot, done = 0.0, 0
        while done < num_steps:
            imgs, masks = self.train_iter.next_batch()
            if not (masks != 255).any():
                continue
            tot += self._train_step(model, opt, imgs, masks).item()
            done += 1
        return tot / max(1, done)

    def train_fedprox(self, model, global_params, num_steps, mu=0.01):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gp = {k: v.to(DEVICE) for k, v in global_params.items()}
        tot, done = 0.0, 0
        while done < num_steps:
            imgs, masks = self.train_iter.next_batch()
            if not (masks != 255).any():
                continue
            model.train()
            opt.zero_grad()
            logits = model(imgs.to(DEVICE, non_blocking=True))
            base = self.loss_fn(logits, masks.to(DEVICE, non_blocking=True))
            prox = sum(((p - gp[n]) ** 2).sum() for n, p in model.named_parameters() if n in gp)
            loss = base + 0.5 * mu * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            opt.step()
            tot += loss.item()
            done += 1
        return tot / max(1, done)

    def train_scaffold(self, model, c_global, num_steps):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.c_i is None:
            self.init_scaffold(model)
        init = {k: v.clone().detach() for k, v in model.named_parameters()}
        tot, done = 0.0, 0
        while done < num_steps:
            imgs, masks = self.train_iter.next_batch()
            if not (masks != 255).any():
                continue
            loss = self._train_step(model, opt, imgs, masks)  # fills .grad
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.add_(-self.c_i[n] + c_global[n])
            opt.step()
            tot += loss.item()
            done += 1
        delta_c = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                new_c = self.c_i[n] - c_global[n] + (1.0 / (num_steps * self.lr)) * (init[n] - p.data)
                delta_c[n] = new_c - self.c_i[n]
                self.c_i[n] = new_c
        return tot / max(1, done), delta_c


class Coordinator:
    def __init__(self, data_root, clients=('etci', 'mmflood', 'kurosiwo'), batch_size=8, lr=0.01, fedprox_mu=0.01):
        self.data_root = data_root
        self.trained = {}
        self.client_names = list(clients)
        self.batch_size, self.lr, self.mu = batch_size, lr, fedprox_mu
        self.clients = {n: FLClient(n, data_root, batch_size=batch_size, lr=lr) for n in self.client_names}
        total = sum(c.num_train for c in self.clients.values())
        self.weights = [self.clients[n].num_train / total for n in self.client_names]
        print('[Coordinator] clients =', self.client_names,
              '| tiles =', [self.clients[n].num_train for n in self.client_names],
              '| weights =', [round(w, 4) for w in self.weights])

    def _eval(self, model, loader, max_batches=None):
        model.eval()
        tr = MetricTracker()
        with torch.no_grad():
            for i, (imgs, tg) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                logits = model(imgs.to(DEVICE))
                tr.update((torch.sigmoid(logits) >= 0.5), tg)
        return tr.compute()

    def eval_val(self, model):
        per = {n: self._eval(model, self.clients[n].val_loader)['iou'] for n in self.client_names}
        return float(np.mean(list(per.values()))), per

    def eval_test(self, model):
        per = {n: self._eval(model, self.clients[n].test_loader) for n in self.client_names}
        pooled = MetricTracker()
        pooled.tp = sum(per[n]['tp'] for n in per)
        pooled.fp = sum(per[n]['fp'] for n in per)
        pooled.fn = sum(per[n]['fn'] for n in per)
        pooled.tn = sum(per[n]['tn'] for n in per)
        macro = {m: float(np.mean([per[n][m] for n in self.client_names]))
                 for m in ('iou', 'dice', 'precision', 'recall', 'accuracy')}
        return {'per_client': per, 'macro': macro, 'pooled': pooled.compute()}

    def run_federated(self, algorithm, num_rounds=15, local_steps=10, seed=42, patience=15):
        set_seed(seed)
        g = FloodUNet().to(DEVICE)
        c_global = None
        if algorithm == 'scaffold':
            c_global = {k: torch.zeros_like(v, device=DEVICE) for k, v in g.named_parameters()}
            for cl in self.clients.values():
                cl.init_scaffold(g)
        steps_list = local_steps if isinstance(local_steps, list) else [local_steps] * len(self.client_names)

        best_iou, best_round, best_state, no_imp = -1.0, 0, copy.deepcopy(g.state_dict()), 0
        history = []
        t0 = time.time()
        print(f'\n--- {algorithm.upper()} (seed={seed}, rounds<={num_rounds}, steps={steps_list}) ---', flush=True)
        for r in range(1, num_rounds + 1):
            states, losses, deltas = [], {}, []
            for i, name in enumerate(self.client_names):
                cl = self.clients[name]
                lm = copy.deepcopy(g)
                if algorithm in ('fedavg', 'fednova'):
                    losses[name] = cl.train_fedavg(lm, steps_list[i])
                elif algorithm == 'fedprox':
                    losses[name] = cl.train_fedprox(lm, g.state_dict(), steps_list[i], mu=self.mu)
                elif algorithm == 'scaffold':
                    losses[name], dc = cl.train_scaffold(lm, c_global, steps_list[i])
                    deltas.append(dc)
                else:
                    raise ValueError(algorithm)
                states.append(get_params(lm))

            if algorithm in ('fedavg', 'fedprox'):
                set_params(g, fed_average(states, self.weights))
            elif algorithm == 'fednova':
                a_eff = sum(self.weights[i] * steps_list[i] for i in range(len(steps_list)))
                gs = g.state_dict()
                newp = {}
                for k in gs:
                    if gs[k].dtype.is_floating_point:
                        d = sum(self.weights[i] * (gs[k].cpu() - states[i][k]) / float(steps_list[i])
                                for i in range(len(states)))
                        newp[k] = gs[k].cpu() - a_eff * d
                    else:
                        newp[k] = states[0][k]
                set_params(g, newp)
            elif algorithm == 'scaffold':
                set_params(g, fed_average(states, self.weights))
                with torch.no_grad():
                    for k in c_global:
                        c_global[k].add_(sum(deltas[i][k] for i in range(len(deltas))) / len(deltas))

            macro_iou, per_iou = self.eval_val(g)
            history.append({'round': r, 'macro_val_iou': macro_iou, 'per_val': per_iou,
                            'train_loss': losses, 'round_time': time.time() - t0})
            print(f'Round {r:02d}: Macro Val IoU={macro_iou:.4f} | ' +
                  ', '.join(f'{n}:{per_iou[n]:.4f}' for n in self.client_names), flush=True)
            if macro_iou > best_iou:
                best_iou, best_round, no_imp = macro_iou, r, 0
                best_state = copy.deepcopy(g.state_dict())
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f'Early stop @ round {r}', flush=True)
                    break

        total_time = time.time() - t0
        g.load_state_dict(best_state)
        test = self.eval_test(g)
        self.trained[(algorithm, seed)] = copy.deepcopy(g.state_dict())
        return {'algorithm': algorithm, 'seed': seed, 'best_round': best_round,
                'best_val_iou': best_iou, 'test': test, 'history': history,
                'time': total_time, 'rounds_run': len(history)}

    def run_centralized(self, max_steps=300, check_interval=30, seed=42, patience=15):
        set_seed(seed)
        m = FloodUNet().to(DEVICE)
        pooled = ConcatDataset([self.clients[n].train_ds for n in self.client_names])
        it = InfiniteDataLoader(DataLoader(pooled, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0))
        opt = torch.optim.SGD(m.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = MaskedBCEAndDiceLoss()
        best, bstep, bstate, no_imp, hist = -1.0, 0, copy.deepcopy(m.state_dict()), 0, []
        t0 = time.time()
        print(f'\n--- CENTRALIZED (seed={seed}, max_steps={max_steps}) ---', flush=True)
        for s in range(1, max_steps + 1):
            imgs, masks = it.next_batch()
            if not (masks != 255).any():
                continue
            m.train()
            opt.zero_grad()
            l = loss_fn(m(imgs.to(DEVICE)), masks.to(DEVICE))
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            if s % check_interval == 0:
                mi, per = self.eval_val(m)
                hist.append({'step': s, 'round': s / check_interval, 'macro_val_iou': mi, 'per_val': per})
                print(f'Step {s:03d}: Macro Val IoU={mi:.4f}', flush=True)
                if mi > best:
                    best, bstep, no_imp = mi, s, 0
                    bstate = copy.deepcopy(m.state_dict())
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        break
        tt = time.time() - t0
        m.load_state_dict(bstate)
        test = self.eval_test(m)
        self.trained[('centralized', seed)] = copy.deepcopy(m.state_dict())
        return {'algorithm': 'centralized', 'seed': seed, 'best_step': bstep,
                'best_val_iou': best, 'test': test, 'history': hist, 'time': tt,
                'rounds_run': len(hist)}

    def run_local(self, name, max_steps=150, check_interval=30, seed=42, patience=15):
        set_seed(seed)
        m = FloodUNet().to(DEVICE)
        cl = self.clients[name]
        it = cl.train_iter
        opt = torch.optim.SGD(m.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = MaskedBCEAndDiceLoss()
        best, bstep, bstate, no_imp, hist = -1.0, 0, copy.deepcopy(m.state_dict()), 0, []
        t0 = time.time()
        print(f'\n--- LOCAL-ONLY [{name.upper()}] (seed={seed}) ---', flush=True)
        for s in range(1, max_steps + 1):
            imgs, masks = it.next_batch()
            if not (masks != 255).any():
                continue
            m.train()
            opt.zero_grad()
            l = loss_fn(m(imgs.to(DEVICE)), masks.to(DEVICE))
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            if s % check_interval == 0:
                vi = self._eval(m, cl.val_loader)['iou']
                hist.append({'step': s, 'round': s / check_interval, 'val_iou': vi})
                if vi > best:
                    best, bstep, no_imp = vi, s, 0
                    bstate = copy.deepcopy(m.state_dict())
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        break
        tt = time.time() - t0
        m.load_state_dict(bstate)
        test = self.eval_test(m)
        self.trained[(f'local_{name}', seed)] = copy.deepcopy(m.state_dict())
        return {'algorithm': f'local_{name}', 'seed': seed, 'client': name, 'best_step': bstep,
                'best_val_iou': best, 'test': test, 'history': hist, 'time': tt,
                'rounds_run': len(hist)}


# ---------------------------------------------------------------------------
# Result aggregation across seeds -> mean (+/- std) tables
# ---------------------------------------------------------------------------
FL_METHODS = [('FedAvg', 'fedavg'), ('FedProx', 'fedprox'), ('FedNova', 'fednova'),
              ('SCAFFOLD', 'scaffold'), ('Centralized', 'centralized'),
              ('Local ETCI', 'local_etci'), ('Local MMFlood', 'local_mmflood'),
              ('Local Kuro Siwo', 'local_kurosiwo')]


def _collect(runs, key):
    return [r['test']['per_client'][key]['iou'] for r in runs if key in r['test']['per_client']]


def summarize_federated(results):
    rows = []
    for label, key in FL_METHODS:
        if key not in results:
            continue
        runs = results[key]
        macro = [r['test']['macro'] for r in runs]
        per = {n: _collect(runs, n) for n in ('etci', 'mmflood', 'kurosiwo')}
        rows.append({
            'Method': label,
            'ETCI IoU': np.mean(per['etci']),
            'MMFlood IoU': np.mean(per['mmflood']),
            'KuroSiwo IoU': np.mean(per['kurosiwo']),
            'Macro IoU': np.mean([m['iou'] for m in macro]),
            'Macro IoU std': np.std([m['iou'] for m in macro]),
            'Macro Dice': np.mean([m['dice'] for m in macro]),
            'Macro Precision': np.mean([m['precision'] for m in macro]),
            'Macro Recall': np.mean([m['recall'] for m in macro]),
            'Pooled IoU': np.mean([r['test']['pooled']['iou'] for r in runs]),
            'Seeds': len(runs),
        })
    return pd.DataFrame(rows)


def fmt(df, cols, prec=3):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f'{x:.{prec}f}')
    return out


def cross_domain_table(results):
    rows = []
    for label, key in FL_METHODS:
        if not key.startswith('local_'):
            continue
        runs = results[key]
        home = key.replace('local_', '')
        vals = {}
        for n in ('etci', 'mmflood', 'kurosiwo'):
            vals[n] = np.mean(_collect(runs, n))
        rows.append({'Model (trained on)': label,
                     'In-domain IoU': vals[home],
                     'ETCI IoU': vals['etci'], 'MMFlood IoU': vals['mmflood'],
                     'KuroSiwo IoU': vals['kurosiwo'],
                     'Macro Cross-domain IoU': np.mean([v for k, v in vals.items() if k != home])})
    return pd.DataFrame(rows)


def compute_profile(coordinator):
    m = FloodUNet().to(DEVICE)
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    ckpt_mb = sum(p.numel() * p.element_size() for p in m.parameters()) / (1024 ** 2)
    n_clients = len(coordinator.client_names)
    comm_up_mb = ckpt_mb * n_clients
    comm_round_mb = ckpt_mb * 2 * n_clients
    scaffold_round_mb = ckpt_mb * 4 * n_clients
    test = coordinator.clients[coordinator.client_names[0]].test_loader
    imgs = next(iter(test))[0].to(DEVICE)
    m.eval()
    with torch.no_grad():
        m(imgs)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        bs = imgs.shape[0]
        reps = 20
        for _ in range(reps):
            m(imgs)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        infer_ms = (time.time() - t0) / reps / bs * 1000
    return {'trainable_params': params, 'ckpt_mb': ckpt_mb, 'n_clients': n_clients,
            'comm_up_mb': comm_up_mb, 'comm_round_mb': comm_round_mb,
            'scaffold_round_mb': scaffold_round_mb, 'infer_ms_per_tile': infer_ms}
'''.strip('\n')


VIZ_CORE = r'''
# ---------------------------------------------------------------------------
# Visualization helpers (all figures written to the notebook working dir)
# ---------------------------------------------------------------------------
FIG_COLORS = {'fedavg': '#1f77b4', 'fedprox': '#ff7f0e', 'fednova': '#2ca02c',
              'scaffold': '#d62728', 'centralized': '#9467bd'}
FIG_LABELS = {'fedavg': 'FedAvg', 'fedprox': 'FedProx ($\\mu$=0.01)', 'fednova': 'FedNova',
              'scaffold': 'SCAFFOLD', 'centralized': 'Centralized (pooled upper bound)'}


def _mean_curve(runs):
    buckets = {}
    for r in runs:
        for h in r['history']:
            rd = int(round(h['round']))
            buckets.setdefault(rd, []).append(h['macro_val_iou'])
    xs = sorted(buckets)
    return xs, [float(np.mean(buckets[x])) for x in xs]


def plot_convergence(results, save_path='fig1_val_convergence.png'):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    order = ['fedavg', 'fedprox', 'fednova', 'scaffold', 'centralized']
    for key in order:
        if key not in results:
            continue
        xs, ys = _mean_curve(results[key])
        style = '--' if key == 'centralized' else '-'
        ax.plot(xs, ys, style, marker='o', ms=4, lw=2, color=FIG_COLORS[key], label=FIG_LABELS[key])
    ax.set_xlabel('Communication Rounds (centralized: equivalent work)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Macro Validation Flood IoU', fontsize=11, fontweight='bold')
    ax.set_title('Fig. 1 - Validation Convergence (mean over seeds)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    ax.set_ylim(0, 1.0)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()
    return save_path


def plot_per_client(results, save_path='fig2_per_client_performance.png'):
    clients = ['etci', 'mmflood', 'kurosiwo']
    names = ['ETCI 2021', 'MMFlood', 'Kuro Siwo']
    methods = ['fedavg', 'fedprox', 'fednova', 'scaffold', 'centralized']
    x = np.arange(len(clients)); w = 0.15
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    for i, m in enumerate(methods):
        if m not in results:
            continue
        ious = [np.mean(_collect(results[m], c)) for c in clients]
        dics = [np.mean([r['test']['per_client'][c]['dice'] for r in results[m]]) for c in clients]
        off = (i - len(methods) / 2 + 0.5) * w
        a1.bar(x + off, ious, w, label=FIG_LABELS[m], color=FIG_COLORS[m], edgecolor='black', lw=0.4)
        a2.bar(x + off, dics, w, label=FIG_LABELS[m], color=FIG_COLORS[m], edgecolor='black', lw=0.4)
    for ax, ttl, ylab in ((a1, 'Test Flood IoU', 'IoU'), (a2, 'Test Flood Dice / F1', 'Dice')):
        ax.set_title(ttl, fontsize=11, fontweight='bold'); ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, fontweight='bold'); ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.0); ax.grid(axis='y', ls='--', alpha=0.6)
    a2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.suptitle('Fig. 2 - Per-Client Test Performance (mean over seeds)', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()
    return save_path


def plot_unequal(equal_res, unequal_res, save_path='fig3_unequal_computation.png'):
    algos = ['fedavg', 'fedprox', 'fednova', 'scaffold']
    eq = [np.mean([r['test']['macro']['iou'] for r in equal_res.get(a, [])]) for a in algos]
    un = [np.mean([r['test']['macro']['iou'] for r in unequal_res.get(a, [])]) for a in algos]
    x = np.arange(len(algos)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.bar(x - w / 2, eq, w, label='Equal steps (10,10,10)', color='#3470a3', edgecolor='black', lw=0.5)
    ax.bar(x + w / 2, un, w, label='Unequal steps (5,10,15)', color='#d95f02', edgecolor='black', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([FIG_LABELS[a] for a in algos], fontsize=10, fontweight='bold')
    ax.set_ylabel('Macro Test Flood IoU', fontsize=11, fontweight='bold'); ax.set_ylim(0, 1.0)
    ax.set_title('Fig. 3 - Robustness to Unequal Local Computation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', ls='--', alpha=0.6)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()
    return save_path


def plot_confusion_matrices(results, save_path='fig4_confusion_matrices.png'):
    methods = ['fedavg', 'fedprox', 'fednova', 'scaffold', 'centralized']
    methods = [m for m in methods if m in results]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.1 * len(methods), 3.4), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        tp = np.mean([r['test']['pooled']['tp'] for r in results[m]])
        fp = np.mean([r['test']['pooled']['fp'] for r in results[m]])
        fn = np.mean([r['test']['pooled']['fn'] for r in results[m]])
        tn = np.mean([r['test']['pooled']['tn'] for r in results[m]])
        conf = np.array([[tn, fp], [fn, tp]], dtype=float)
        rowsum = conf.sum(1, keepdims=True); rowsum[rowsum == 0] = 1
        pct = conf / rowsum * 100.0
        im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Flood', 'Flood'], fontsize=8); ax.set_yticklabels(['Non-Flood', 'Flood'], fontsize=8)
        ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('Ground Truth', fontsize=9)
        ax.set_title(FIG_LABELS[m].split(' (')[0], fontsize=10, fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{pct[i, j]:.1f}%', ha='center', va='center',
                        color='white' if pct[i, j] > 55 else 'black', fontsize=9)
    fig.suptitle('Fig. 4 - Pixel-level Confusion Matrices (row-normalized, pooled test)', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()
    return save_path


def load_model(coordinator, key, seed=42):
    m = FloodUNet().to(DEVICE)
    m.load_state_dict(coordinator.trained[(key, seed)])
    m.eval()
    return m


def _pick_positive(ds, min_px=200):
    for i in range(len(ds)):
        _, mt = ds[i]
        if (mt == 1).sum() > min_px:
            return i
    return 0


def _sar_rgb(img_t):
    vv = np.clip((img_t[0].numpy() + 1.0) / 2.0, 0, 1)
    vh = np.clip((img_t[1].numpy() + 1.0) / 2.0, 0, 1)
    return np.stack([vv, vh, np.clip(vv / (vh + 1e-4), 0, 1)], axis=-1)


def _mask_disp(msk_np, pred=None):
    disp = np.zeros((msk_np.shape[0], msk_np.shape[1], 3))
    if pred is None:
        disp[msk_np == 0] = [0.1, 0.1, 0.1]; disp[msk_np == 1] = [0.0, 0.8, 0.9]
        disp[msk_np == 255] = [0.5, 0.5, 0.5]
    else:
        disp[~pred] = [0.1, 0.1, 0.1]; disp[pred] = [0.0, 0.8, 0.9]
        disp[msk_np == 255] = [0.5, 0.5, 0.5]
    return disp


def plot_qualitative(coordinator, data_root, save_path='fig5_qualitative.png', seed=42):
    clients = ['etci', 'mmflood', 'kurosiwo']
    titles = ['ETCI 2021', 'MMFlood', 'Kuro Siwo']
    algos = [a for a in ['fedavg', 'fedprox', 'fednova', 'scaffold', 'centralized']
             if (a, seed) in coordinator.trained]
    headers = ['SAR (VV/VH/ratio)', 'Ground Truth'] + [FIG_LABELS[a].split(' (')[0] for a in algos]
    ncol = len(headers)
    fig, axes = plt.subplots(len(clients), ncol, figsize=(2.4 * ncol, 7.4), dpi=150)
    axes = np.atleast_2d(axes)
    for j, h in enumerate(headers):
        axes[0, j].set_title(h, fontsize=10, fontweight='bold', pad=6)
    models = {a: load_model(coordinator, a, seed) for a in algos}
    for ri, c in enumerate(clients):
        ds = FloodTileDataset(c, 'test', data_root)
        idx = _pick_positive(ds)
        img_t, msk_t = ds[idx]; msk_np = msk_t.numpy()
        axes[ri, 0].imshow(_sar_rgb(img_t)); axes[ri, 0].set_ylabel(titles[ri], fontsize=10, fontweight='bold')
        axes[ri, 0].axis('off')
        axes[ri, 1].imshow(_mask_disp(msk_np)); axes[ri, 1].axis('off')
        inp = img_t.unsqueeze(0).to(DEVICE)
        for ci, a in enumerate(algos):
            with torch.no_grad():
                pred = (torch.sigmoid(models[a](inp))[0, 0].cpu().numpy() >= 0.5)
            axes[ri, ci + 2].imshow(_mask_disp(msk_np, pred)); axes[ri, ci + 2].axis('off')
    fig.suptitle('Fig. 5 - Qualitative Flood Predictions (cyan = detected flood)', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()
    return save_path


def plot_error_breakdown(coordinator, data_root, save_path='fig6_error_breakdown.png', seed=42):
    clients = ['etci', 'mmflood', 'kurosiwo']
    key = 'fednova' if ('fednova', seed) in coordinator.trained else next(
        (a for a, s in coordinator.trained if s == seed), 'centralized')
    model = load_model(coordinator, key, seed)
    headers = ['SAR Composite', 'Ground Truth', f'{FIG_LABELS.get(key, key).split(" (")[0]} Pred', 'Error Map']
    fig, axes = plt.subplots(len(clients), 4, figsize=(11, 8.4), dpi=150)
    for j, h in enumerate(headers):
        axes[0, j].set_title(h, fontsize=10, fontweight='bold', pad=6)
    for ri, c in enumerate(clients):
        ds = FloodTileDataset(c, 'test', data_root)
        idx = _pick_positive(ds, 250)
        img_t, msk_t = ds[idx]; msk_np = msk_t.numpy()
        axes[ri, 0].imshow(_sar_rgb(img_t)); axes[ri, 0].set_ylabel(c.upper(), fontsize=10, fontweight='bold'); axes[ri, 0].axis('off')
        axes[ri, 1].imshow(_mask_disp(msk_np)); axes[ri, 1].axis('off')
        inp = img_t.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = (torch.sigmoid(model(inp))[0, 0].cpu().numpy() >= 0.5)
        axes[ri, 2].imshow(_mask_disp(msk_np, pred)); axes[ri, 2].axis('off')
        err = np.full((*msk_np.shape, 3), 0.15)
        valid = msk_np != 255
        err[pred & (msk_np == 1) & valid] = [0.0, 0.85, 0.2]
        err[pred & (msk_np == 0) & valid] = [0.95, 0.15, 0.15]
        err[(~pred) & (msk_np == 1) & valid] = [0.15, 0.35, 0.95]
        err[msk_np == 255] = [0.5, 0.5, 0.5]
        axes[ri, 3].imshow(err); axes[ri, 3].axis('off')
    p = [mpatches.Patch(color=[0, .85, .2], label='TP (detected)'),
         mpatches.Patch(color=[.95, .15, .15], label='FP (false alarm)'),
         mpatches.Patch(color=[.15, .35, .95], label='FN (missed flood)'),
         mpatches.Patch(color=[.15, .15, .15], label='TN (background)')]
    fig.legend(handles=p, loc='lower center', ncol=4, fontsize=9, frameon=True)
    fig.suptitle('Fig. 6 - Error Decomposition Overlay', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.subplots_adjust(bottom=0.07); plt.savefig(save_path, bbox_inches='tight'); plt.close()
    return save_path
'''.strip('\n')
