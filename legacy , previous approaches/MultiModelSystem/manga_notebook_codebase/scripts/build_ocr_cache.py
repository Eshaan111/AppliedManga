"""Extract raw OCR candidates from three-page sequences without reading labels."""
import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ocr_utils import discover_sequences, extract_sequences, sha256

def package_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'TaskData')
    parser.add_argument('--split', choices=['all', 'development', 'test'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/ocr/paddleocr/candidates.jsonl')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sequence-id', action='append', help='Extract only these manifest/folder IDs; repeatable')
    parser.add_argument('--inspect', action='store_true', help='Validate inputs without OCR or writes')
    parser.add_argument('--model-root', type=Path, default=Path.home() / '.paddlex/official_models',
                        help='Directory containing the two downloaded PaddleOCR models')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    rows = discover_sequences(args.input, args.split)
    if args.sequence_id:
        requested = set(args.sequence_id)
        missing = requested - {row['sequence_id'] for row in rows}
        if missing:
            parser.error(f'Unknown sequence IDs: {sorted(missing)}')
        rows = [row for row in rows if row['sequence_id'] in requested]
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f'Input: {args.input.resolve()}; {len(rows)} sequences / {3 * len(rows)} pages', flush=True)
    if args.inspect:
        return
    if args.output.exists() and not args.output.with_suffix('.metadata.json').exists():
        parser.error('Existing legacy candidates are preserved. Choose a new --output for extraction.')
    model_names = ['PP-OCRv6_medium_det', 'PP-OCRv6_medium_rec']
    model_dirs = [args.model_root.resolve() / name for name in model_names]
    required = [folder / name for folder in model_dirs
                for name in ('inference.json', 'inference.pdiparams', 'inference.yml')]
    if any(not path.is_file() for path in required):
        parser.error('Local model files missing. Download the documented models and set --model-root.')
    from paddleocr import PaddleOCR
    config = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, device='gpu:0', enable_mkldnn=False,
                  cpu_threads=1, text_detection_model_name=model_names[0],
                  text_recognition_model_name=model_names[1])
    ocr = PaddleOCR(**config, text_detection_model_dir=str(model_dirs[0]),
                    text_recognition_model_dir=str(model_dirs[1]))
    runtime = dict(config=config, python=platform.python_version(), platform=platform.platform(),
                   packages = {
                        "paddle": package_version("paddlepaddle-gpu", "paddlepaddle"),
                        "paddleocr": package_version("paddleocr"),
                        "paddlex": package_version("paddlex"),
                    },
                   model_files={f'{p.parent.name}/{p.name}': sha256(p) for p in required},
                   code={p.name: sha256(p) for p in [Path(__file__), ROOT / 'src/ocr_utils.py']})
    extract_sequences(rows, ocr, args.output, runtime)
    print(f'Saved: {args.output.resolve()}')


if __name__ == '__main__':
    main()
