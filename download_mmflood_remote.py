"""
Final MMFlood RemoteZip diagnostic/download attempt.
This does NOT delete or alter the full archive.
If the ZIP compression method is unsupported by Python/RemoteZip,
it reports the compression method and exits safely.
"""

import re
import time
from pathlib import Path, PurePosixPath
from collections import defaultdict

from remotezip import RemoteZip

URL = "https://zenodo.org/records/6534637/files/mmflood.zip"
OUT = Path(r"C:\Users\umars\OneDrive\Desktop\FloodArea\data\mmflood\mmflood")
LIMIT = 1000
RETRIES = 2


def activation(parts):
    for p in parts:
        if re.fullmatch(r"EMSR\d+-\d+", p):
            return p
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Opening remote MMFlood ZIP...")

    with RemoteZip(URL) as zf:
        infos = zf.infolist()
        name_to_info = {i.filename: i for i in infos}
        names = set(name_to_info)

        print("Archive entries:", len(infos))
        methods = defaultdict(int)
        for info in infos:
            methods[info.compress_type] += 1
        print("ZIP compression method IDs:", dict(methods))
        print("Common IDs: 0=stored, 8=deflate, 12=bzip2, 14=LZMA, 93=Zstandard")

        scenes = {}
        for name in names:
            p = PurePosixPath(name)
            parts = list(p.parts)
            if p.suffix.lower() not in (".tif", ".tiff") or "s1_raw" not in parts:
                continue
            act = activation(parts)
            if not act:
                continue
            idx = parts.index("s1_raw")
            mask_parts = parts.copy()
            mask_parts[idx] = "mask"
            mask_name = str(PurePosixPath(*mask_parts))
            if mask_name not in names:
                continue
            scenes[(act, p.name)] = (act, name, mask_name)

        grouped = defaultdict(list)
        for scene in scenes.values():
            grouped[scene[0]].append(scene)
        for act in grouped:
            grouped[act].sort(key=lambda x: x[1])

        selected = []
        acts = sorted(grouped)
        pos = 0
        while len(selected) < LIMIT:
            added = False
            for act in acts:
                if pos < len(grouped[act]):
                    selected.append(grouped[act][pos])
                    added = True
                    if len(selected) == LIMIT:
                        break
            if not added:
                break
            pos += 1

        print(f"Selected {len(selected)} image/mask pairs.")
        ok = 0
        failed = 0
        skipped = 0

        for i, (act, image_name, mask_name) in enumerate(selected, 1):
            for modality, member in (("s1_raw", image_name), ("mask", mask_name)):
                filename = PurePosixPath(member).name
                destination = OUT / "activations" / act / modality / filename
                destination.parent.mkdir(parents=True, exist_ok=True)

                if destination.exists() and destination.stat().st_size > 0:
                    skipped += 1
                    continue

                success = False
                for attempt in range(1, RETRIES + 1):
                    try:
                        with zf.open(member) as src, open(destination.with_suffix(destination.suffix + ".part"), "wb") as dst:
                            while True:
                                block = src.read(1024 * 1024)
                                if not block:
                                    break
                                dst.write(block)
                        temp = destination.with_suffix(destination.suffix + ".part")
                        if temp.stat().st_size == 0:
                            raise RuntimeError("Empty extracted file")
                        temp.replace(destination)
                        success = True
                        ok += 1
                        break
                    except Exception as exc:
                        temp = destination.with_suffix(destination.suffix + ".part")
                        temp.unlink(missing_ok=True)
                        print(f"[RemoteZip] {member}: attempt {attempt}/{RETRIES}: {exc}")
                        if "compression method is not supported" in str(exc).lower():
                            raise RuntimeError(
                                "RemoteZip cannot decode this ZIP compression method. "
                                "Stop here and use the full-ZIP + 7-Zip script."
                            ) from exc
                        time.sleep(2)

                if not success:
                    failed += 1

            if i % 25 == 0 or i == len(selected):
                print(f"[RemoteZip] Scenes {i}/{len(selected)} | extracted={ok} skipped={skipped} failed={failed}")

        print("RemoteZip run complete.")
        print("New files:", ok, "Existing files:", skipped, "Failed files:", failed)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nREMOTE ZIP ATTEMPT STOPPED:", e)
        print("Use download_datasets.py for the full archive + 7-Zip method.")
