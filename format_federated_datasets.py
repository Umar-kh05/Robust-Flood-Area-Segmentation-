import os
import re
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:
    raise ImportError(
        "Please install rasterio first:\n"
        "pip install rasterio"
    )


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\umars\OneDrive\Desktop\FloodArea\data"
)

# ETCI downloaded dataset
ETCI_ROOT = DATA_ROOT / "etci"

# Complete MMFlood ZIP
MMFLOOD_ZIP = DATA_ROOT / "mmflood_download.zip"

# Kuro Siwo downloaded dataset
KURO_ROOT = DATA_ROOT / "kuro_siwo"

# Final federated dataset
OUTPUT_ROOT = DATA_ROOT / "federated_ready"

# Maximum number of samples from each dataset
MAX_SAMPLES = 1000

# Final split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Fixed random seed
SEED = 42

# Output image size
IMAGE_SIZE = 256

# Overwrite old formatted output
CLEAR_OUTPUT = True


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def ensure_folder(path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dataset_folder(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_array(array):
    """
    Normalize image data to float32 between 0 and 1.
    """

    array = array.astype(np.float32)

    valid = np.isfinite(array)

    if not np.any(valid):
        return np.zeros_like(array, dtype=np.float32)

    values = array[valid]

    low = np.percentile(values, 1)
    high = np.percentile(values, 99)

    if high <= low:
        return np.zeros_like(array, dtype=np.float32)

    array = np.clip(array, low, high)
    array = (array - low) / (high - low)

    array[~np.isfinite(array)] = 0

    return array.astype(np.float32)


def resize_image(image, size=IMAGE_SIZE):
    """
    Resize multi-channel image using PIL.
    Input shape: H, W, C
    """

    channels = []

    for channel in range(image.shape[2]):

        data = image[:, :, channel]

        pil_image = Image.fromarray(
            data.astype(np.float32),
            mode="F"
        )

        pil_image = pil_image.resize(
            (size, size),
            Image.Resampling.BILINEAR
        )

        channels.append(
            np.array(pil_image, dtype=np.float32)
        )

    return np.stack(channels, axis=-1)


def resize_mask(mask, size=IMAGE_SIZE):
    """
    Resize segmentation mask using nearest-neighbor.
    """

    mask = mask.astype(np.uint8)

    pil_mask = Image.fromarray(mask)

    pil_mask = pil_mask.resize(
        (size, size),
        Image.Resampling.NEAREST
    )

    mask = np.array(pil_mask)

    return (mask > 0).astype(np.uint8)


def save_sample(dataset_root, split, sample_id, image, mask):
    """
    Save one standardized sample.
    """

    image_dir = dataset_root / split / "images"
    mask_dir = dataset_root / split / "masks"

    ensure_folder(image_dir)
    ensure_folder(mask_dir)

    np.save(
        image_dir / f"{sample_id}.npy",
        image.astype(np.float32)
    )

    np.save(
        mask_dir / f"{sample_id}.npy",
        mask.astype(np.uint8)
    )


def make_splits(samples):
    """
    Randomly split samples into train, validation and test.
    """

    samples = list(samples)

    rng = random.Random(SEED)
    rng.shuffle(samples)

    total = len(samples)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train = samples[:train_count]
    val = samples[
        train_count:train_count + val_count
    ]
    test = samples[
        train_count + val_count:
    ]

    return {
        "train": train,
        "val": val,
        "test": test
    }


def read_png(path):
    """
    Read PNG as numpy array.
    """

    image = Image.open(path)
    image = np.array(image)

    if image.ndim == 3:
        image = image[:, :, 0]

    return image


def count_final_samples(dataset_root):
    result = {}

    for split in ["train", "val", "test"]:

        image_dir = dataset_root / split / "images"
        mask_dir = dataset_root / split / "masks"

        if not image_dir.exists() or not mask_dir.exists():
            result[split] = 0
            continue

        images = {
            p.stem
            for p in image_dir.glob("*.npy")
        }

        masks = {
            p.stem
            for p in mask_dir.glob("*.npy")
        }

        result[split] = len(images & masks)

    return result


# ============================================================
# ETCI
# ============================================================

def remove_vv_suffix(name):
    return re.sub(r"_vv$", "", name)


def find_etci_samples():
    """
    Find valid ETCI samples.

    Required:
        vv image
        vh image
        flood_label mask

    Directory structure:
        region/date/tiles/vv
        region/date/tiles/vh
        region/date/tiles/flood_label
    """

    samples = []

    if not ETCI_ROOT.exists():
        print(f"[ETCI] Folder not found: {ETCI_ROOT}")
        return samples

    vv_files = list(ETCI_ROOT.rglob("*_vv.png"))

    print(
        f"[ETCI] Searching {len(vv_files)} VV images..."
    )

    for vv_path in vv_files:

        try:
            if vv_path.parent.name != "vv":
                continue

            tiles_dir = vv_path.parent.parent

            vh_dir = tiles_dir / "vh"
            flood_dir = tiles_dir / "flood_label"

            base_name = remove_vv_suffix(
                vv_path.stem
            )

            vh_path = vh_dir / (
                base_name + "_vh.png"
            )

            mask_path = flood_dir / (
                base_name + ".png"
            )

            if not vh_path.is_file():
                continue

            if not mask_path.is_file():
                continue

            samples.append({
                "vv": vv_path,
                "vh": vh_path,
                "mask": mask_path,
                "id": (
                    vv_path.parent.parent.parent.name
                    + "_"
                    + base_name
                )
            })

        except Exception:
            continue

    return samples


def format_etci():
    print("\n" + "=" * 60)
    print("FORMATTING ETCI")
    print("=" * 60)

    output = OUTPUT_ROOT / "etci"

    samples = find_etci_samples()

    print(
        f"[ETCI] Valid image-mask pairs found: "
        f"{len(samples)}"
    )

    if not samples:
        print("[ETCI] No valid samples found.")
        return

    rng = random.Random(SEED)
    rng.shuffle(samples)

    samples = samples[:MAX_SAMPLES]

    splits = make_splits(samples)

    processed = 0

    for split, split_samples in splits.items():

        print(
            f"[ETCI] Processing {split}: "
            f"{len(split_samples)} samples"
        )

        for sample in split_samples:

            try:
                vv = read_png(sample["vv"])
                vh = read_png(sample["vh"])
                mask = read_png(sample["mask"])

                vv = normalize_array(vv)
                vh = normalize_array(vh)

                image = np.stack(
                    [vv, vh],
                    axis=-1
                )

                image = resize_image(image)
                mask = resize_mask(mask)

                sample_id = (
                    f"etci_{processed:04d}"
                )

                save_sample(
                    output,
                    split,
                    sample_id,
                    image,
                    mask
                )

                processed += 1

            except Exception as error:
                print(
                    f"[ETCI] Skipped sample: "
                    f"{error}"
                )

    print(
        f"[ETCI] Complete: {processed} samples"
    )


# ============================================================
# MMFLOOD
# ============================================================

def mm_activation(parts):
    for part in parts:
        if re.fullmatch(
            r"EMSR\d+-\d+",
            part
        ):
            return part
    return None


def read_tiff_from_zip(zip_file, member):
    """
    Read TIFF directly from extracted temporary file.
    """

    temp_dir = OUTPUT_ROOT / "_temp_mmflood"
    ensure_folder(temp_dir)

    temp_name = (
        str(abs(hash(member)))
        + "_"
        + Path(member).name
    )

    temp_path = temp_dir / temp_name

    try:

        with zip_file.open(member) as source:
            with open(temp_path, "wb") as destination:
                shutil.copyfileobj(
                    source,
                    destination,
                    length=1024 * 1024
                )

        with rasterio.open(temp_path) as dataset:

            array = dataset.read(1)

        return array

    finally:

        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def find_mmflood_samples():
    """
    Find matching s1_raw and mask TIFF files
    directly inside the complete MMFlood ZIP.
    """

    samples = []

    if not MMFLOOD_ZIP.exists():

        print(
            f"[MMFlood] ZIP not found:\n"
            f"{MMFLOOD_ZIP}"
        )

        return samples

    print(
        "[MMFlood] Reading ZIP directory..."
    )

    with zipfile.ZipFile(
        MMFLOOD_ZIP,
        "r"
    ) as zip_file:

        names = set(zip_file.namelist())

        scenes = defaultdict(list)

        for member in names:

            path = PurePosixPath(member)
            parts = list(path.parts)

            if path.suffix.lower() not in (
                ".tif",
                ".tiff"
            ):
                continue

            if "s1_raw" not in parts:
                continue

            activation = mm_activation(parts)

            if activation is None:
                continue

            index = parts.index("s1_raw")

            mask_parts = parts.copy()
            mask_parts[index] = "mask"

            mask_member = str(
                PurePosixPath(*mask_parts)
            )

            if mask_member not in names:
                continue

            scenes[activation].append({
                "image": member,
                "mask": mask_member,
                "activation": activation
            })

    print(
        f"[MMFlood] Valid activations: "
        f"{len(scenes)}"
    )

    # Round-robin selection for diversity
    for activation in scenes:
        scenes[activation].sort(
            key=lambda x: x["image"]
        )

    activations = sorted(scenes.keys())

    selected = []
    position = 0

    while len(selected) < MAX_SAMPLES:

        added = False

        for activation in activations:

            if position < len(
                scenes[activation]
            ):

                selected.append(
                    scenes[activation][position]
                )

                added = True

                if len(selected) >= MAX_SAMPLES:
                    break

        if not added:
            break

        position += 1

    return selected


def format_mmflood():
    print("\n" + "=" * 60)
    print("FORMATTING MMFLOOD")
    print("=" * 60)

    output = OUTPUT_ROOT / "mmflood"

    samples = find_mmflood_samples()

    print(
        f"[MMFlood] Selected valid pairs: "
        f"{len(samples)}"
    )

    if not samples:
        return

    splits = make_splits(samples)

    processed = 0

    with zipfile.ZipFile(
        MMFLOOD_ZIP,
        "r"
    ) as zip_file:

        for split, split_samples in splits.items():

            print(
                f"[MMFlood] Processing {split}: "
                f"{len(split_samples)} samples"
            )

            for index, sample in enumerate(
                split_samples,
                start=1
            ):

                try:

                    image = read_tiff_from_zip(
                        zip_file,
                        sample["image"]
                    )

                    mask = read_tiff_from_zip(
                        zip_file,
                        sample["mask"]
                    )

                    image = normalize_array(image)

                    # MMFlood input may be single-band.
                    # Duplicate it to create a common
                    # 2-channel input: VV, VH.
                    image = np.stack(
                        [image, image],
                        axis=-1
                    )

                    image = resize_image(image)
                    mask = resize_mask(mask)

                    sample_id = (
                        f"mmflood_{processed:04d}"
                    )

                    save_sample(
                        output,
                        split,
                        sample_id,
                        image,
                        mask
                    )

                    processed += 1

                    if (
                        index % 50 == 0
                        or index == len(
                            split_samples
                        )
                    ):
                        print(
                            f"[MMFlood] "
                            f"{index}/"
                            f"{len(split_samples)}"
                        )

                except Exception as error:

                    print(
                        "[MMFlood] Skipped sample: "
                        f"{Path(sample['image']).name}: "
                        f"{error}"
                    )

    print(
        f"[MMFlood] Complete: "
        f"{processed} samples"
    )


# ============================================================
# KURO SIWO
# ============================================================

def find_kuro_samples():
    """
    Find valid Kuro Siwo sample folders.
    """

    samples = []

    if not KURO_ROOT.exists():

        print(
            f"[Kuro] Folder not found: "
            f"{KURO_ROOT}"
        )

        return samples

    required = [
        "flood_vv.npy",
        "flood_vh.npy",
        "mask.npy"
    ]

    for folder in KURO_ROOT.rglob("*"):

        if not folder.is_dir():
            continue

        if all(
            (folder / file).is_file()
            for file in required
        ):

            samples.append({
                "folder": folder
            })

    return samples


def load_kuro_array(path):

    array = np.load(path)

    array = np.squeeze(array)

    return array


def format_kurosiwo():
    print("\n" + "=" * 60)
    print("FORMATTING KURO SIWO")
    print("=" * 60)

    output = OUTPUT_ROOT / "kurosiwo"

    samples = find_kuro_samples()

    print(
        f"[Kuro] Valid samples found: "
        f"{len(samples)}"
    )

    if not samples:
        return

    rng = random.Random(SEED)
    rng.shuffle(samples)

    samples = samples[:MAX_SAMPLES]

    splits = make_splits(samples)

    processed = 0

    for split, split_samples in splits.items():

        print(
            f"[Kuro] Processing {split}: "
            f"{len(split_samples)} samples"
        )

        for sample in split_samples:

            try:

                folder = sample["folder"]

                vv = load_kuro_array(
                    folder / "flood_vv.npy"
                )

                vh = load_kuro_array(
                    folder / "flood_vh.npy"
                )

                mask = load_kuro_array(
                    folder / "mask.npy"
                )

                vv = normalize_array(vv)
                vh = normalize_array(vh)

                image = np.stack(
                    [vv, vh],
                    axis=-1
                )

                image = resize_image(image)
                mask = resize_mask(mask)

                sample_id = (
                    f"kuro_{processed:04d}"
                )

                save_sample(
                    output,
                    split,
                    sample_id,
                    image,
                    mask
                )

                processed += 1

            except Exception as error:

                print(
                    "[Kuro] Skipped sample: "
                    f"{error}"
                )

    print(
        f"[Kuro] Complete: "
        f"{processed} samples"
    )


# ============================================================
# VERIFY OUTPUT
# ============================================================

def verify_output():

    print("\n" + "=" * 60)
    print("VERIFYING FEDERATED DATASETS")
    print("=" * 60)

    for dataset in [
        "etci",
        "mmflood",
        "kurosiwo"
    ]:

        dataset_root = (
            OUTPUT_ROOT / dataset
        )

        counts = count_final_samples(
            dataset_root
        )

        total = sum(
            counts.values()
        )

        print(f"\n{dataset.upper()}")

        for split in [
            "train",
            "val",
            "test"
        ]:

            print(
                f"  {split}: "
                f"{counts[split]} samples"
            )

        print(
            f"  TOTAL: {total}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FEDERATED FLOOD DATASET FORMATTER")
    print("=" * 60)

    print("\nInput datasets:")
    print("ETCI:", ETCI_ROOT)
    print("MMFlood ZIP:", MMFLOOD_ZIP)
    print("Kuro Siwo:", KURO_ROOT)

    print("\nOutput:")
    print(OUTPUT_ROOT)

    if CLEAR_OUTPUT:

        print(
            "\nPreparing clean output folder..."
        )

        clear_dataset_folder(
            OUTPUT_ROOT
        )

    else:

        ensure_folder(
            OUTPUT_ROOT
        )

    # Format all three federated clients
    format_etci()

    format_mmflood()

    format_kurosiwo()

    # Remove temporary MMFlood files
    temp_folder = (
        OUTPUT_ROOT /
        "_temp_mmflood"
    )

    if temp_folder.exists():

        try:
            shutil.rmtree(
                temp_folder
            )
        except Exception:
            pass

    # Final verification
    verify_output()

    print("\n" + "=" * 60)
    print("FORMATTING COMPLETE")
    print("=" * 60)

    print(
        "\nYour three datasets are now "
        "stored in a common format."
    )

    print(
        "\nEach client contains:"
    )

    print(
        "  train/images + train/masks"
    )

    print(
        "  val/images + val/masks"
    )

    print(
        "  test/images + test/masks"
    )

    print(
        "\nAll model inputs are standardized "
        f"to {IMAGE_SIZE}x{IMAGE_SIZE} "
        "with 2 channels."
    )