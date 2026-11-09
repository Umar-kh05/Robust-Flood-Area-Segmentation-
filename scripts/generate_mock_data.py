import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import generate_mock_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tiles for testing pipeline")
    parser.add_argument("--output_dir", type=str, default="data/mock", help="Directory to save mock data")
    parser.add_argument("--tiles_per_split", type=int, default=16, help="Number of tiles per client split")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print(f"Generating mock data in: {out_dir}")
    generate_mock_dataset(out_dir, tiles_per_split=args.tiles_per_split)
    print("Done! Use --manifest data/mock/manifest.json --stats data/mock/norm_stats.json to run experiments.")


if __name__ == "__main__":
    main()
