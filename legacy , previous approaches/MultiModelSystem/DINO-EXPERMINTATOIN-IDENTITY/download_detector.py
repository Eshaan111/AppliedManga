"""Download only hysts' YOLOv3 face weights, never HRNet landmarks."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
REPO = 'hysts/anime-face-detector-yolov3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'models/anime-face-detector-yolov3')
    args = parser.parse_args()
    print(snapshot_download(REPO, local_dir=args.destination, cache_dir=ROOT / 'models/.hub',
                            allow_patterns=['model.safetensors', 'README.md'], token=False))
