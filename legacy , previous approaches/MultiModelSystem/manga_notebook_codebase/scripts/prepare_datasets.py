from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset_utils import prepare_datasets

if __name__ == "__main__":
    split = prepare_datasets(ROOT)
    print(f"Prepared {len(split['train'])} train, {len(split['val'])} validation sequences")
