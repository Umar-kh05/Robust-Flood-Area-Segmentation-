
import os
import re
import json
import time
import random
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from collections import defaultdict

import fsspec
from remotezip import RemoteZip
from huggingface_hub import HfApi, hf_hub_url


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\umars\OneDrive\Desktop\FloodArea\data"
)

# Existing MMFlood location
MMFLOOD_ROOT = DATA_ROOT / "mmflood" / "mmflood"

# Kuro Siwo output location
KURO_ROOT = DATA_ROOT / "kuro_siwo"

# MMFlood: maximum 1,000 image-mask pairs
MAX_MMFLOOD_SCENES = 1000

# Kuro Siwo: 800 official training samples + 200 test samples
MAX_KURO_TRAIN = 800
MAX_KURO_TEST = 200

# MMFlood official Zenodo archive
MMFLOOD_URL = (
    "https://zenodo.org/records/6534637/files/mmflood.zip"
)

# Kuro Siwo Hugging Face dataset
KURO_REPO = "orion-ai-lab/Kuro-Siwo-Webdataset"

KURO_REVISION = (
    "e8e61b7b1254b04bfa2b5e26d43db90cb86b5004"
)

RETRIES = 3

MMFLOOD_ROOT.mkdir(parents=True, exist_ok=True)
KURO_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def is_valid_file(path):
    return path.exists() and path.is_file() and path.stat().st_size > 0


def copy_stream(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = destination.with_name(
        destination.name + ".part"
    )

    try:
        with open(temp_path, "wb") as output:
            shutil.copyfileobj(
                source,
                output,
                length=1024 * 1024
            )

        if not is_valid_file(temp_path):
            raise RuntimeError("Downloaded file is empty")

        os.replace(temp_path, destination)

    finally:
        if temp_path.exists():
            temp_path.unlink()


# ============================================================
# 3. MMFLOOD
# ============================================================

def mmflood_activation(parts):

    for part in parts:
        if re.fullmatch(r"EMSR\d+-\d+", part):
            return part

    return None


def build_mmflood_scenes(zip_file):

    print("\n[MMFlood] Reading archive structure...")

    names = set(zip_file.namelist())

    scenes = {}

    for member in names:

        path = PurePosixPath(member)

        if path.suffix.lower() not in (".tif", ".tiff"):
            continue

        parts = list(path.parts)

        if "s1_raw" not in parts:
            continue

        activation = mmflood_activation(parts)

        if activation is None:
            continue

        modality_index = parts.index("s1_raw")

        # Match image and mask by identical relative filename
        mask_parts = parts.copy()
        mask_parts[modality_index] = "mask"

        mask_member = str(PurePosixPath(*mask_parts))

        # Both image and ground-truth mask are mandatory
        if mask_member not in names:
            continue

        scene_key = (activation, path.name)

        if scene_key not in scenes:

            scenes[scene_key] = {
                "activation": activation,
                "filename": path.name,
                "files": {
                    "s1_raw": member,
                    "mask": mask_member
                }
            }

        # Download other available modalities too
        for modality in ("DEM", "hydro"):

            optional_parts = parts.copy()
            optional_parts[modality_index] = modality

            optional_member = str(
                PurePosixPath(*optional_parts)
            )

            if optional_member in names:
                scenes[scene_key]["files"][modality] = (
                    optional_member
                )

    return list(scenes.values())


def select_mmflood_scenes(scenes):

    groups = defaultdict(list)

    for scene in scenes:
        groups[scene["activation"]].append(scene)

    activations = sorted(groups.keys())

    for activation in activations:
        groups[activation].sort(
            key=lambda x: x["filename"]
        )

    selected = []

    # Round-robin selection across activations.
    # This avoids selecting all scenes from one flood event.

    index = 0

    while len(selected) < MAX_MMFLOOD_SCENES:

        added = False

        for activation in activations:

            if index < len(groups[activation]):

                selected.append(groups[activation][index])
                added = True

                if len(selected) >= MAX_MMFLOOD_SCENES:
                    break

        if not added:
            break

        index += 1

    return selected


def download_mmflood_file(zip_file, member, destination):

    if is_valid_file(destination):
        return "skipped"

    if destination.exists():
        destination.unlink()

    for attempt in range(1, RETRIES + 1):

        temp_path = destination.with_name(
            destination.name + ".part"
        )

        try:

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with zip_file.open(member) as source:
                copy_stream(source, destination)

            return "downloaded"

        except Exception as error:

            if temp_path.exists():
                temp_path.unlink()

            if destination.exists():
                destination.unlink()

            print(
                f"  [MMFlood] Retry {attempt}/{RETRIES}: "
                f"{Path(member).name}: {error}"
            )

            if attempt < RETRIES:
                time.sleep(3)

    return "failed"


def run_mmflood():

    print("\n" + "=" * 65)
    print("STARTING MMFLOOD DOWNLOAD")
    print("=" * 65)

    downloaded = 0
    skipped = 0
    failed = 0

    try:

        with RemoteZip(MMFLOOD_URL) as zip_file:

            scenes = build_mmflood_scenes(zip_file)

            print(
                f"[MMFlood] Usable image-mask pairs: "
                f"{len(scenes)}"
            )

            if not scenes:
                raise RuntimeError(
                    "No image-mask pairs found in archive."
                )

            selected = select_mmflood_scenes(scenes)

            print(
                f"[MMFlood] Selected scenes: {len(selected)}"
            )

            print(
                f"[MMFlood] Activations represented: "
                f"{len(set(s['activation'] for s in selected))}"
            )

            print(
                "[MMFlood] Downloading s1_raw, mask, "
                "DEM and available hydro files."
            )

            for index, scene in enumerate(selected, start=1):

                activation = scene["activation"]
                filename = scene["filename"]

                for modality, member in scene["files"].items():

                    destination = (
                        MMFLOOD_ROOT /
                        activation /
                        modality /
                        filename
                    )

                    status = download_mmflood_file(
                        zip_file,
                        member,
                        destination
                    )

                    if status == "downloaded":
                        downloaded += 1

                    elif status == "skipped":
                        skipped += 1

                    else:
                        failed += 1

                if (
                    index % 25 == 0
                    or index == len(selected)
                ):

                    print(
                        f"[MMFlood] Scenes {index}/{len(selected)} | "
                        f"Downloaded files: {downloaded} | "
                        f"Skipped existing: {skipped} | "
                        f"Failed files: {failed}"
                    )

        print("\n[MMFlood] FINISHED")
        print("Selected scenes:", len(selected))
        print("Downloaded files:", downloaded)
        print("Skipped existing files:", skipped)
        print("Failed files:", failed)

    except Exception as error:

        print("\n[MMFlood] ERROR:", error)
        print("[MMFlood] You can rerun this script to resume.")


# ============================================================
# 4. KURO SIWO: SAMPLE FIELD DEFINITIONS
# ============================================================

KURO_FIELDS = [
    "flood_vv.npy",
    "flood_vh.npy",

    "sec1_vv.npy",
    "sec1_vh.npy",

    "sec2_vv.npy",
    "sec2_vh.npy",

    "dem.npy",
    "mask.npy",
    "valid_mask.npy",
    "info.json"
]

KURO_REQUIRED = [
    "flood_vv.npy",
    "flood_vh.npy",
    "mask.npy",
    "valid_mask.npy"
]


def parse_kuro_member(member_name):

    filename = PurePosixPath(member_name).name

    for field in KURO_FIELDS:

        suffix = "." + field

        if filename.endswith(suffix):

            sample_key = filename[:-len(suffix)]

            return sample_key, field

    return None, None


# ============================================================
# 5. KURO SIWO: REMOTE TAR ACCESS
# ============================================================

def get_kuro_url(shard_path):

    return hf_hub_url(
        repo_id=KURO_REPO,
        filename=shard_path,
        repo_type="dataset",
        revision=KURO_REVISION
    )


def scan_kuro_shard(shard_path):

    """
    Read TAR headers through HTTP range requests.
    This avoids downloading the entire 10+ GB TAR shard.
    """

    print(f"[Kuro] Scanning {shard_path}")

    url = get_kuro_url(shard_path)

    groups = defaultdict(dict)

    with fsspec.open(
        url,
        mode="rb",
        block_size=64 * 1024,
        cache_type="bytes"
    ) as remote_file:

        remote_file.seek(0, 2)
        archive_size = remote_file.tell()
        remote_file.seek(0)

        if archive_size <= 0:
            raise RuntimeError("Could not determine TAR size")

        with tarfile.open(
            fileobj=remote_file,
            mode="r:"
        ) as archive:

            for member in archive:

                if not member.isfile():
                    continue

                sample_key, field = parse_kuro_member(
                    member.name
                )

                if sample_key is None:
                    continue

                groups[sample_key][field] = member.name

    # Keep only complete segmentation samples
    valid_samples = []

    for sample_key, fields in groups.items():

        if all(
            field in fields
            for field in KURO_REQUIRED
        ):

            valid_samples.append({
                "key": sample_key,
                "members": fields
            })

    print(
        f"[Kuro] {shard_path}: "
        f"{len(valid_samples)} complete samples found"
    )

    return valid_samples


# ============================================================
# 6. KURO SIWO: CREATE BALANCED SELECTION PLAN
# ============================================================

def build_kuro_selection(api):

    print("\n[Kuro] Listing official GRD shards...")

    repo_files = api.list_repo_files(
        repo_id=KURO_REPO,
        repo_type="dataset",
        revision=KURO_REVISION
    )

    split_targets = {
        "train_GRD": MAX_KURO_TRAIN,
        "test_GRD": MAX_KURO_TEST
    }

    plan = {}

    for split, target in split_targets.items():

        shards = sorted([
            filename
            for filename in repo_files
            if filename.startswith(split + "/")
            and filename.endswith(".tar")
        ])

        if not shards:
            raise RuntimeError(
                f"No TAR shards found for {split}"
            )

        print(
            f"\n[Kuro] {split}: {len(shards)} shards"
        )

        candidates_by_shard = []

        # Scan all shards in the selected official split.
        # Only TAR headers are inspected at this stage.

        for shard in shards:

            candidates = scan_kuro_shard(shard)

            candidates_by_shard.append(
                (shard, candidates)
            )

        # Distribute the target evenly across shards
        base_quota = target // len(shards)
        remainder = target % len(shards)

        selected = []
        leftovers = []

        for index, (shard, candidates) in enumerate(
            candidates_by_shard
        ):

            quota = base_quota

            if index < remainder:
                quota += 1

            rng = random.Random(
                2026 + index + (
                    10000 if split == "test_GRD" else 0
                )
            )

            candidates = candidates.copy()
            rng.shuffle(candidates)

            chosen = candidates[:quota]
            remaining = candidates[quota:]

            for item in chosen:

                selected.append({
                    "shard": shard,
                    "key": item["key"],
                    "members": item["members"]
                })

            leftovers.extend([
                {
                    "shard": shard,
                    "key": item["key"],
                    "members": item["members"]
                }
                for item in remaining
            ])

        # Fill any shortage from other shards
        rng = random.Random(2026)
        rng.shuffle(leftovers)

        needed = target - len(selected)

        if needed > 0:
            selected.extend(leftovers[:needed])

        if len(selected) > target:
            selected = selected[:target]

        plan[split] = selected

        print(
            f"[Kuro] {split}: selected "
            f"{len(selected)} samples"
        )

        if len(selected) < target:
            print(
                f"[Kuro] WARNING: Only {len(selected)} "
                f"complete samples were available."
            )

    return plan


# ============================================================
# 7. KURO SIWO: SAVE SELECTION PLAN FOR RESUMING
# ============================================================

def get_kuro_plan(api):

    plan_file = KURO_ROOT / "selection_plan.json"

    if plan_file.exists():

        try:

            with open(plan_file, "r", encoding="utf-8") as file:
                plan = json.load(file)

            if (
                "train_GRD" in plan
                and "test_GRD" in plan
            ):

                print(
                    "\n[Kuro] Resuming from saved selection plan."
                )

                return plan

        except Exception:
            print("[Kuro] Old selection plan is invalid.")

    plan = build_kuro_selection(api)

    with open(plan_file, "w", encoding="utf-8") as file:

        json.dump(
            plan,
            file,
            indent=2
        )

    print("[Kuro] Selection plan saved.")

    return plan


# ============================================================
# 8. KURO SIWO: EXTRACT ONLY SELECTED SAMPLES
# ============================================================

def safe_sample_name(key):

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        key
    )


def get_kuro_sample_folder(split, shard, key):

    shard_name = Path(shard).stem

    sample_name = (
        shard_name + "__" + safe_sample_name(key)
    )

    return KURO_ROOT / split / sample_name


def extract_kuro_shard(shard_path, selected_samples, split):

    """
    Extract only the chosen sample fields.
    Other TAR data is skipped through seek operations.
    """

    if not selected_samples:
        return 0, 0

    wanted_members = {}
    sample_destinations = {}

    complete_before = 0

    for sample in selected_samples:

        folder = get_kuro_sample_folder(
            split,
            shard_path,
            sample["key"]
        )

        folder.mkdir(parents=True, exist_ok=True)

        sample_destinations[sample["key"]] = folder

        missing = []

        for field, member_name in sample["members"].items():

            destination = folder / field

            if is_valid_file(destination):
                continue

            if destination.exists():
                destination.unlink()

            wanted_members[member_name] = destination
            missing.append(field)

        if not missing:
            complete_before += 1

    if not wanted_members:

        print(
            f"[Kuro] {shard_path}: "
            f"all selected samples already exist."
        )

        return complete_before, 0

    url = get_kuro_url(shard_path)

    print(
        f"\n[Kuro] Extracting {len(selected_samples)} "
        f"selected samples from {shard_path}"
    )

    extracted_files = 0

    with fsspec.open(
        url,
        mode="rb",
        block_size=64 * 1024,
        cache_type="bytes"
    ) as remote_file:

        with tarfile.open(
            fileobj=remote_file,
            mode="r:"
        ) as archive:

            for member in archive:

                if member.name not in wanted_members:
                    continue

                destination = wanted_members[member.name]

                try:

                    source = archive.extractfile(member)

                    if source is None:
                        continue

                    with source:
                        copy_stream(source, destination)

                    extracted_files += 1

                except Exception as error:

                    print(
                        f"[Kuro] Failed: "
                        f"{member.name}: {error}"
                    )

    complete_after = 0

    for sample in selected_samples:

        folder = get_kuro_sample_folder(
            split,
            shard_path,
            sample["key"]
        )

        if all(
            is_valid_file(folder / field)
            for field in KURO_REQUIRED
        ):
            complete_after += 1

    failed_samples = len(selected_samples) - complete_after

    print(
        f"[Kuro] {shard_path} finished | "
        f"Complete samples: {complete_after}/{len(selected_samples)} | "
        f"Extracted files: {extracted_files} | "
        f"Incomplete samples: {failed_samples}"
    )

    return complete_after, extracted_files


# ============================================================
# 9. KURO SIWO: MAIN DOWNLOAD
# ============================================================

def run_kuro():

    print("\n" + "=" * 65)
    print("STARTING KURO SIWO DOWNLOAD")
    print("=" * 65)

    try:

        api = HfApi()

        plan = get_kuro_plan(api)

        total_complete = 0
        total_files = 0

        for split in ("train_GRD", "test_GRD"):

            print("\n" + "-" * 55)
            print("[Kuro] Processing:", split)
            print("-" * 55)

            selected = plan[split]

            # Group selected samples by their original shard
            by_shard = defaultdict(list)

            for sample in selected:

                by_shard[sample["shard"]].append(sample)

            for shard, shard_samples in by_shard.items():

                complete, files = extract_kuro_shard(
                    shard,
                    shard_samples,
                    split
                )

                total_complete += complete
                total_files += files

        expected = (
            len(plan["train_GRD"])
            + len(plan["test_GRD"])
        )

        print("\n[Kuro] FINISHED")
        print("Selected samples:", expected)
        print("Complete samples:", total_complete)
        print("Newly extracted files:", total_files)
        print("Incomplete samples:", expected - total_complete)

        print("Output:", KURO_ROOT)

    except Exception as error:

        print("\n[Kuro] ERROR:", error)

        print(
            "[Kuro] You can rerun this script. "
            "The selection plan and completed files are preserved."
        )


# ============================================================
# 10. RUN BOTH DATASETS
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("FLOOD DATASET DOWNLOADER")
    print("=" * 65)

    print("\nETCI: Already downloaded. Will NOT be modified.")
    print("MMFlood: Maximum 1,000 scenes.")
    print("Kuro Siwo: Maximum 1,000 samples.")
    print("Kuro Siwo split: 800 train + 200 test.")

    run_mmflood()

    run_kuro()

    print("\n" + "=" * 65)
    print("ALL DOWNLOAD TASKS FINISHED")
    print("=" * 65)
