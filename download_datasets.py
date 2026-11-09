"""
Download up to 1,000 MMFlood scenes and up to 1,000 Kuro Siwo GRD samples.
ETCI is not modified.

MMFlood:
  Downloads the official 11.3 GB ZIP once (resumable), selects <=1000
  image/mask pairs across activations, and uses 7-Zip to extract only
  selected files. The ZIP is deleted only after successful verification.

Kuro Siwo:
  Downloads official labelled GRD TAR shards from Hugging Face, extracts
  complete samples, and stops at 800 train + 200 test samples.
  Each selected sample retains all available fields (VV/VH pre/post,
  DEM, mask, valid_mask, info).

Install:
  pip install requests tqdm huggingface_hub
  Install 7-Zip on Windows and ensure 7z.exe is on PATH.
"""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import requests
from huggingface_hub import HfApi, hf_hub_url, hf_hub_download

# ------------------------- PATHS / LIMITS -------------------------

DATA_ROOT = Path(r"C:\Users\umars\OneDrive\Desktop\FloodArea\data")

MM_ROOT = DATA_ROOT / "mmflood" / "mmflood"
MM_ZIP = DATA_ROOT / "mmflood_download.zip"
MM_URL = "https://zenodo.org/records/6534637/files/mmflood.zip"
MM_MD5 = "2817eb94107f1c96ef68d843010bdf97"
MM_LIMIT = 1000

KURO_ROOT = DATA_ROOT / "kuro_siwo"
KURO_REPO = "orion-ai-lab/Kuro-Siwo-Webdataset"
KURO_REVISION = "e8e61b7b1254b04bfa2b5e26d43db90cb86b5004"
KURO_TRAIN_LIMIT = 800
KURO_TEST_LIMIT = 200

RETRIES = 3
CHUNK = 8 * 1024 * 1024
RANDOM_SEED = 2026

# If the RemoteZip attempt already downloaded MMFlood successfully,
# set RUN_MMFLOOD = False so this file downloads only Kuro Siwo.
RUN_MMFLOOD = True
RUN_KURO = True

# Kuro GRD fields documented by the dataset authors.
KURO_FIELDS = [
    "flood_vv.npy", "flood_vh.npy",
    "sec1_vv.npy", "sec1_vh.npy",
    "sec2_vv.npy", "sec2_vh.npy",
    "dem.npy", "mask.npy", "valid_mask.npy", "info.json"
]
KURO_REQUIRED = {"flood_vv.npy", "flood_vh.npy", "mask.npy", "valid_mask.npy"}


# ------------------------- COMMON HELPERS -------------------------

def find_7z():
    candidates = [
        shutil.which("7z"),
        shutil.which("7zz"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def download_resumable(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, RETRIES + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                if offset and response.status_code != 206:
                    # Server did not honour Range; restart cleanly.
                    offset = 0
                    part.unlink(missing_ok=True)
                    response.close()
                    with requests.get(url, stream=True, timeout=(30, 180)) as response2:
                        response2.raise_for_status()
                        total = int(response2.headers.get("content-length", 0))
                        mode = "wb"
                        written = 0
                        with open(part, mode) as f:
                            for chunk in response2.iter_content(CHUNK):
                                if chunk:
                                    f.write(chunk)
                                    written += len(chunk)
                                    if written % (256 * 1024 * 1024) < CHUNK:
                                        print(f"[MMFlood] Downloaded {written / (1024**3):.2f} GB"
                                              + (f" / {total / (1024**3):.2f} GB" if total else ""))
                else:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0)) + offset
                    mode = "ab" if offset else "wb"
                    written = offset
                    with open(part, mode) as f:
                        for chunk in response.iter_content(CHUNK):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if written % (256 * 1024 * 1024) < CHUNK:
                                    print(f"[MMFlood] Downloaded {written / (1024**3):.2f} GB"
                                          + (f" / {total / (1024**3):.2f} GB" if total else ""))
            os.replace(part, destination)
            return
        except Exception as exc:
            print(f"[MMFlood] Download attempt {attempt}/{RETRIES} failed: {exc}")
            if attempt < RETRIES:
                time.sleep(5)
    raise RuntimeError("Could not download MMFlood ZIP. Partial .part file is retained for resume.")


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ------------------------- MMFLOOD -------------------------

def mm_activation(parts):
    for part in parts:
        if re.fullmatch(r"EMSR\d+-\d+", part):
            return part
    return None


def select_mmflood_members(zip_path):
    print("[MMFlood] Reading ZIP central directory...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    scenes = {}
    for member in names:
        p = PurePosixPath(member)
        parts = list(p.parts)
        if p.suffix.lower() not in (".tif", ".tiff") or "s1_raw" not in parts:
            continue
        activation = mm_activation(parts)
        if not activation:
            continue
        idx = parts.index("s1_raw")
        mask_parts = parts.copy()
        mask_parts[idx] = "mask"
        mask_member = str(PurePosixPath(*mask_parts))
        if mask_member not in names:
            continue

        key = (activation, p.name)
        if key not in scenes:
            scenes[key] = {"activation": activation, "image": member, "mask": mask_member,
                           "optional": []}

        # Include matching DEM/hydro files when they exist.
        for modality in ("DEM", "hydro"):
            other = parts.copy()
            other[idx] = modality
            candidate = str(PurePosixPath(*other))
            if candidate in names and candidate not in scenes[key]["optional"]:
                scenes[key]["optional"].append(candidate)

    grouped = defaultdict(list)
    for scene in scenes.values():
        grouped[scene["activation"]].append(scene)
    for activation in grouped:
        grouped[activation].sort(key=lambda x: x["image"])

    activations = sorted(grouped)
    chosen = []
    cursor = 0
    while len(chosen) < MM_LIMIT:
        added = False
        for activation in activations:
            if cursor < len(grouped[activation]):
                chosen.append(grouped[activation][cursor])
                added = True
                if len(chosen) >= MM_LIMIT:
                    break
        if not added:
            break
        cursor += 1

    members = []
    for scene in chosen:
        members.extend([scene["image"], scene["mask"]])
        members.extend(scene["optional"])

    # Preserve archive paths. They will be extracted under MM_ROOT.
    print(f"[MMFlood] Found {len(scenes)} usable pairs across {len(activations)} activations.")
    print(f"[MMFlood] Selected {len(chosen)} pairs across "
          f"{len(set(s['activation'] for s in chosen))} activations.")
    print(f"[MMFlood] Selected {len(members)} files including optional modalities.")
    return chosen, sorted(set(members))


def extract_selected_with_7zip(zip_path, members, output_root):
    seven_zip = find_7z()
    if not seven_zip:
        raise RuntimeError(
            "7-Zip was not found. Install 7-Zip from https://www.7-zip.org/ "
            "and rerun. The downloaded ZIP is kept."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    list_file = DATA_ROOT / "mmflood_selected_members.txt"

    # 7-Zip listfile uses one archive member per line.
    with open(list_file, "w", encoding="utf-8") as f:
        for member in members:
            f.write(member.replace("/", "\\") + "\n")

    command = [
        seven_zip, "x", str(zip_path),
        f"-o{output_root}", "-y", "-bso1", "-bsp1",
        "-scsUTF-8", f"@{list_file}"
    ]
    print("[MMFlood] Extracting selected files with 7-Zip...")
    print("[MMFlood] This may take time; extraction progress is shown below.")
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(
            f"7-Zip extraction failed with exit code {result.returncode}. "
            "ZIP is retained; rerun after fixing the issue."
        )


def run_mmflood():
    print("\n" + "=" * 65)
    print("MMFLOOD DOWNLOAD")
    print("=" * 65)
    MM_ROOT.mkdir(parents=True, exist_ok=True)

    if not MM_ZIP.exists():
        print("[MMFlood] Downloading official archive (Zenodo lists it as 11.3 GB).")
        download_resumable(MM_URL, MM_ZIP)
    else:
        print(f"[MMFlood] Reusing existing archive: {MM_ZIP}")

    print("[MMFlood] Checking archive MD5...")
    actual_md5 = md5_file(MM_ZIP)
    if actual_md5.lower() != MM_MD5.lower():
        raise RuntimeError(
            f"MMFlood ZIP MD5 mismatch. Expected {MM_MD5}, got {actual_md5}. "
            "Do not extract this archive. Delete the ZIP and rerun to download again."
        )
    print("[MMFlood] Archive checksum verified.")

    chosen, members = select_mmflood_members(MM_ZIP)

    if not chosen:
        raise RuntimeError("No valid MMFlood image/mask pairs found in ZIP.")

    extract_selected_with_7zip(MM_ZIP, members, MM_ROOT)

    # Verify every selected SAR image and mask was extracted.
    missing = []
    for scene in chosen:
        for member in (scene["image"], scene["mask"]):
            # Strip the leading activations/ directory if present, but
            # first test the archive-relative path exactly.
            candidate = MM_ROOT / Path(*PurePosixPath(member).parts)
            if not candidate.is_file() or candidate.stat().st_size == 0:
                # Some archives may put files directly below activation.
                parts = list(PurePosixPath(member).parts)
                if parts and parts[0].lower() == "activations":
                    candidate2 = MM_ROOT / Path(*parts[1:])
                    if candidate2.is_file() and candidate2.stat().st_size > 0:
                        continue
                missing.append(member)

    if missing:
        print(f"[MMFlood] {len(missing)} required files are missing.")
        print("[MMFlood] Keeping ZIP so extraction can be retried.")
        with open(DATA_ROOT / "mmflood_missing_files.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        return

    print(f"[MMFlood] Verified {len(chosen)} image/mask pairs.")
    print("[MMFlood] Removing temporary ZIP after successful verification.")
    MM_ZIP.unlink(missing_ok=True)
    print("[MMFlood] COMPLETE.")


# ------------------------- KURO SIWO -------------------------

def parse_kuro_member(name):
    basename = PurePosixPath(name).name
    for field in KURO_FIELDS:
        suffix = "." + field
        if basename.endswith(suffix):
            return basename[:-len(suffix)], field
    return None, None


def get_kuro_shards(api, split):
    files = api.list_repo_files(
        repo_id=KURO_REPO, repo_type="dataset", revision=KURO_REVISION
    )
    return sorted(
        x for x in files
        if x.startswith(split + "/") and x.lower().endswith(".tar")
    )


def extract_kuro_tar(tar_path, output_split, limit):
    """
    Extract up to `limit` complete samples from a downloaded TAR.
    All known fields for each selected sample are saved.
    """
    groups = defaultdict(dict)
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key, field = parse_kuro_member(member.name)
            if key is not None:
                groups[key][field] = member.name

        complete_keys = [
            key for key, fields in groups.items()
            if KURO_REQUIRED.issubset(fields.keys())
        ]
        complete_keys.sort()
        selected_keys = complete_keys[:limit]

        extracted = 0
        for key in selected_keys:
            sample_dir = output_split / (Path(tar_path).stem + "__" + re.sub(r"[^A-Za-z0-9_.-]", "_", key))
            sample_dir.mkdir(parents=True, exist_ok=True)
            fields = groups[key]

            for field, member_name in fields.items():
                target = sample_dir / field
                if target.exists() and target.stat().st_size > 0:
                    continue
                info = tf.getmember(member_name)
                source = tf.extractfile(info)
                if source is None:
                    continue
                temp = target.with_suffix(target.suffix + ".part")
                with source, open(temp, "wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                os.replace(temp, target)

            if all((sample_dir / field).is_file() and (sample_dir / field).stat().st_size > 0
                   for field in KURO_REQUIRED):
                extracted += 1

    return extracted


def run_kuro_split(api, split, target):
    print(f"\n[Kuro] Processing {split}; target={target}")
    shards = get_kuro_shards(api, split)
    if not shards:
        raise RuntimeError(f"No shards found under {split}/ at pinned revision.")

    # Shuffle deterministically so selection is not always from the first shard.
    rng = random.Random(RANDOM_SEED + (1 if split == "test_GRD" else 0))
    rng.shuffle(shards)

    split_root = KURO_ROOT / split
    split_root.mkdir(parents=True, exist_ok=True)

    # Resume: count complete sample folders already extracted.
    def count_complete():
        count = 0
        for folder in split_root.iterdir():
            if folder.is_dir() and all(
                (folder / f).is_file() and (folder / f).stat().st_size > 0
                for f in KURO_REQUIRED
            ):
                count += 1
        return count

    complete = count_complete()
    print(f"[Kuro] Existing complete samples: {complete}/{target}")

    for shard in shards:
        if complete >= target:
            break

        print(f"[Kuro] Downloading shard: {shard}")
        local_tar = hf_hub_download(
            repo_id=KURO_REPO,
            filename=shard,
            repo_type="dataset",
            revision=KURO_REVISION,
            local_dir=str(KURO_ROOT / "_shards")
        )

        remaining = target - complete
        newly_complete = extract_kuro_tar(
            local_tar, split_root, remaining
        )
        complete = count_complete()
        print(f"[Kuro] Complete samples: {complete}/{target}")

        # Remove local TAR after successful extraction to save disk space.
        try:
            Path(local_tar).unlink(missing_ok=True)
        except Exception:
            pass

    if complete < target:
        print(f"[Kuro] WARNING: only {complete} complete samples available/downloaded.")
    else:
        print(f"[Kuro] Target reached: {complete} samples.")


def run_kuro():
    print("\n" + "=" * 65)
    print("KURO SIWO DOWNLOAD")
    print("=" * 65)
    KURO_ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    run_kuro_split(api, "train_GRD", KURO_TRAIN_LIMIT)
    run_kuro_split(api, "test_GRD", KURO_TEST_LIMIT)
    print("[Kuro] COMPLETE.")


# ------------------------- MAIN -------------------------

if __name__ == "__main__":
    print("ETCI is already downloaded and will NOT be modified.")
    print("MMFlood limit: 1,000 image/mask pairs.")
    print("Kuro Siwo limit: 800 train + 200 test samples.")
    print("MMFlood archive is retained unless extraction verifies successfully.")

    if RUN_MMFLOOD:
        try:
            run_mmflood()
        except Exception as exc:
            print("\n[MMFlood] STOPPED:", exc)
    else:
        print("\n[MMFlood] Skipped because RUN_MMFLOOD=False.")

    if RUN_KURO:
        try:
            run_kuro()
        except Exception as exc:
            print("\n[Kuro] STOPPED:", exc)

    print("\nDownloader finished. Review the warnings above.")
