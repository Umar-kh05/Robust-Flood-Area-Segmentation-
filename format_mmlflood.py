"""
format_mmlflood.py -- convert extracted MMFlood GeoTIFFs into the same
federated_ready/<client>/<split>/ layout as ETCI / Kuro Siwo.

Run from the project folder:
    python format_mmlflood.py
    python format_mmlflood.py --allow-missing-hydro
    python format_mmlflood.py --allow-missing-hydro --group-by aoi --overwrite

Needs:  pip install numpy rasterio     (or: pip install tifffile imagecodecs)
"""
import argparse, json, shutil, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TILE, SEED = 256, 42
SPLIT_FRAC = {"train": 0.6, "val": 0.2, "test": 0.2}


# ----------------------------------------------------------------- reading
def read_tif(path):
    """Return (array[bands,H,W], nodata)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(), src.nodata
    except ImportError:
        pass
    import tifffile                          # needs `imagecodecs` for compressed tiffs
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3 and arr.shape[-1] <= 4 and arr.shape[0] > 4:
        arr = np.transpose(arr, (2, 0, 1))
    return arr, None


# ------------------------------------------------- match existing layout
def detect_layout(fed: Path):
    layout = {"mode": "dirs", "chw": True}
    ref = next((fed / c / "train" for c in ("etci", "kurosiwo", "kuro_siwo")
                if (fed / c / "train").exists()), None)
    if ref is None:
        print("[layout] no existing client found; using images/ + masks/ .npy, channels-first")
        return layout
    try:
        if (ref / "images").is_dir():
            img = np.load(next((ref / "images").glob("*.np*")))
            if isinstance(img, np.lib.npyio.NpzFile):
                img = img[img.files[0]]
        else:
            layout["mode"] = "npz"
            img = np.load(next(ref.glob("*.npz")))["image"]
        layout["chw"] = (img.shape[0] == 2)
        print(f"[layout] matched {ref}: mode={layout['mode']}, shape={img.shape}, "
              f"min={img.min():.3g}, max={img.max():.3g}")
    except StopIteration:
        print(f"[layout] {ref} is empty; using default layout")
    return layout


def save_tile(out: Path, split, name, img, lab, layout):
    if not layout["chw"]:
        img = np.transpose(img, (1, 2, 0))
    if layout["mode"] == "dirs":
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)
        np.save(out / split / "images" / f"{name}.npy", img)
        np.save(out / split / "masks" / f"{name}.npy", lab)
    else:
        (out / split).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / split / f"{name}.npz", image=img, mask=lab)


# ---------------------------------------------------- scene -> img + label
def find_sibling(s1: Path, folder: str):
    base = s1.parent.parent
    for d in base.iterdir():
        if d.is_dir() and d.name.lower() == folder:
            for f in d.iterdir():
                if f.stem == s1.stem:
                    return f
    return None


def read_hydro_aligned(hyp, ref_path):
    """Read hydro and resample it (nearest) onto the grid of ref_path (the mask)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(ref_path) as ref, rasterio.open(hyp) as src:
        if src.crs is None or ref.crs is None:
            raise ValueError("missing CRS, cannot align hydro")
        if src.shape == ref.shape and src.transform == ref.transform and src.crs == ref.crs:
            return src.read(1)
        dst = np.zeros(ref.shape, dtype=src.dtypes[0])   # outside hydro extent -> 0
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref.transform, dst_crs=ref.crs,
                  resampling=Resampling.nearest)
    return dst


def load_scene(s1p, mkp, hyp):
    s1, nod = read_tif(s1p)
    if s1.shape[0] < 2:
        raise ValueError("s1_raw has fewer than 2 bands")
    s1 = s1[:2].astype(np.float32)
    mk = read_tif(mkp)[0][0]
    if mk.shape != s1.shape[1:]:
        raise ValueError(f"s1 {s1.shape[1:]} vs mask {mk.shape} mismatch")
    hy = read_hydro_aligned(hyp, mkp) if hyp is not None else None

    valid = np.isfinite(s1).all(0)
    if nod is not None and np.isfinite(nod):
        valid &= ~(s1 == nod).any(0)
    valid &= (mk == 0) | (mk == 1)             # 255 / undocumented codes -> ignore
    F = mk == 1

    H = np.zeros(mk.shape, bool)
    if hy is not None:
        u = set(np.unique(hy).tolist())
        if u <= {0, 255} and 255 in u:         # hydro stored as {0,255}
            H = hy == 255
        else:                                   # hydro stored as {0,1}
            H = hy == 1
            valid &= (hy == 0) | (hy == 1)

    lab = np.full(mk.shape, 255, np.uint8)
    lab[valid & ~F] = 0
    lab[valid & F & ~H] = 1                     # flood AND hydro = conflict -> stays 255

    s1 = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
    s1[:, ~valid] = 0.0

    ph, pw = (-mk.shape[0]) % TILE, (-mk.shape[1]) % TILE
    s1 = np.pad(s1, ((0, 0), (0, ph), (0, pw)))
    lab = np.pad(lab, ((0, ph), (0, pw)), constant_values=255)
    return s1, lab


def tile_coords(lab):
    return [(y, x) for y in range(0, lab.shape[0], TILE)
            for x in range(0, lab.shape[1], TILE)
            if (lab[y:y + TILE, x:x + TILE] != 255).any()]


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"data\mmlflood")
    ap.add_argument("--out", default=r"data\federated_ready")
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-missing-hydro", action="store_true",
                    help="keep scenes without hydro (treated as no permanent water)")
    ap.add_argument("--group-by", choices=["activation", "aoi"], default="activation",
                    help="split unit: EMSR activation (report) or sub-region folder")
    a = ap.parse_args()

    src, fed = Path(a.src), Path(a.out)
    out = fed / "mmflood"
    if not src.exists():
        sys.exit(f"Source folder not found: {src.resolve()}")
    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            sys.exit(f"{out} already has files. Re-run with --overwrite to replace it.")
        shutil.rmtree(out)

    layout = detect_layout(fed)

    # --- find scenes -------------------------------------------------------
    all_tifs = [p for p in src.rglob("*") if p.suffix.lower() in (".tif", ".tiff")]
    s1_files = sorted(p for p in all_tifs if p.parent.name.lower() == "s1_raw")
    print(f"[scan] {len(all_tifs)} tif files, {len(s1_files)} s1_raw scenes under {src.resolve()}")
    if not s1_files:
        sys.exit("No tif files inside an 's1_raw' folder.")

    scenes, no_mask, no_hydro = [], 0, 0
    for s1 in s1_files:
        mk, hy = find_sibling(s1, "mask"), find_sibling(s1, "hydro")
        if mk is None:
            no_mask += 1
            continue
        if hy is None:
            no_hydro += 1
            if not a.allow_missing_hydro:
                continue
        group = s1.name.split("-")[0] if a.group_by == "activation" else s1.parent.parent.name
        scenes.append((group, s1, mk, hy))
    print(f"[scenes] {len(s1_files)} total | {no_mask} without mask | {no_hydro} without hydro "
          f"-> {len(scenes)} usable ({'hydro optional' if a.allow_missing_hydro else 'hydro required'})")
    if not scenes:
        sys.exit("No usable scenes. Your extracted files have no hydro folders.\n"
                 "Either extract the hydro files too, or run with --allow-missing-hydro "
                 "(a deviation from the report; state it as a limitation).")

    # --- group-level split (60/20/20) ----------------------------------------
    by_group = defaultdict(list)
    for g, s1, mk, hy in scenes:
        by_group[g].append((s1, mk, hy))
    groups = sorted(by_group)
    G = len(groups)
    print(f"[groups] {G} ({a.group_by}): {groups[:12]}{' ...' if G > 12 else ''}")
    if G < 3:
        sys.exit(f"Only {G} group(s); need at least 3. Try --group-by aoi or extract more files.")
    groups = [groups[i] for i in np.random.default_rng(SEED).permutation(G)]
    n = max(1, G // 5)                                       # max(1, floor(0.2*G))
    split_groups = {"test": groups[:n], "val": groups[n:2 * n], "train": groups[2 * n:]}
    print("[split] " + ", ".join(f"{k}: {len(v)} groups" for k, v in split_groups.items()))
    if not split_groups["train"]:
        sys.exit("No groups left for training; extract more files.")

    # --- sample tiles per split, keep them in memory -------------------------
    rng = np.random.default_rng(SEED)
    tiles = {}                                               # split -> [(name, img, lab)]
    for split, gs in split_groups.items():
        cap = round(a.max_samples * SPLIT_FRAC[split])
        cands = []
        for g in gs:
            for sc in by_group[g]:
                try:
                    _, lab = load_scene(*sc)
                except Exception as e:
                    print(f"  skip {sc[0].name}: {e!r}")
                    continue
                cands += [(sc, y, x) for y, x in tile_coords(lab)]
        if len(cands) > cap:
            cands = [cands[i] for i in rng.choice(len(cands), cap, replace=False)]
        by_scene = defaultdict(list)
        for sc, y, x in cands:
            by_scene[sc].append((y, x))

        tiles[split] = []
        for sc, coords in by_scene.items():
            img, lab = load_scene(*sc)
            for y, x in coords:
                tiles[split].append((img[:, y:y + TILE, x:x + TILE].copy(),
                                     lab[y:y + TILE, x:x + TILE].copy()))
        print(f"[{split}] {len(tiles[split])} tiles selected")

    if not tiles["train"]:
        sys.exit("No valid training tiles.")

    # --- normalisation: 1st/99th percentile of valid TRAIN pixels -> [0, 1] ---
    lo, hi = [], []
    for c in range(2):
        samples = []
        for img, lab in tiles["train"]:
            v = img[c][lab != 255]
            if v.size:
                samples.append(v if v.size <= 20000 else rng.choice(v, 20000, replace=False))
        p = np.percentile(np.concatenate(samples), [1, 99])
        lo.append(float(p[0])); hi.append(float(p[1]))
    print(f"[norm] p1={lo}, p99={hi}")

    # --- write ----------------------------------------------------------------
    for split, items in tiles.items():
        pos = tot = 0
        for i, (img, lab) in enumerate(items):
            norm = np.empty_like(img, dtype=np.float32)
            for c in range(2):
                norm[c] = np.clip((img[c] - lo[c]) / max(hi[c] - lo[c], 1e-8), 0, 1)
            norm[:, lab == 255] = 0.0
            save_tile(out, split, f"mmflood_{split}_{i:05d}", norm, lab, layout)
            pos += int((lab == 1).sum()); tot += int((lab != 255).sum())
        print(f"[{split}] wrote {len(items)} tiles | flood pixel fraction {pos / max(tot, 1):.3f}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "normalization.json").write_text(json.dumps(
        {"p1": lo, "p99": hi, "group_by": a.group_by,
         "allow_missing_hydro": a.allow_missing_hydro,
         "split_groups": split_groups}, indent=2))
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()