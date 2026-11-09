from pathlib import Path

DATA_ROOT = Path(r"C:\Users\umars\OneDrive\Desktop\FloodArea\data")

for dataset in ["etci", "mmflood", "kuro_siwo"]:
    path = DATA_ROOT / dataset

    print("\n" + "=" * 60)
    print(dataset.upper())
    print("=" * 60)

    if not path.exists():
        print("Folder not found:", path)
        continue

    for item in sorted(path.rglob("*")):
        if item.is_file():
            relative = item.relative_to(path)
            print(relative)