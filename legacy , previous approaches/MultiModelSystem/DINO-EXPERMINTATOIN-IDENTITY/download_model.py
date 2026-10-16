"""Run this first. Downloads only the Transformers safetensors model."""
import argparse
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default='C:/MODELS/dinov2-small')
    args = parser.parse_args()
    print(snapshot_download('facebook/dinov2-small', local_dir=args.destination,
                            allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md'],
                            token=False))
