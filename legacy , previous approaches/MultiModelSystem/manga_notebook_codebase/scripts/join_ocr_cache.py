"""Join raw OCR candidates on labelled or unseen data without reading references."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ocr_joining import join_ocr_candidates
from ocr_utils import atomic_write, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.candidates.resolve() == args.output.resolve():
        parser.error('Output must differ from raw candidates')
    records, seen = [], set()
    for line in args.candidates.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sid = record['sequence_id']
        if sid in seen or not isinstance(record['pages'], list) or len(record['pages']) != 3:
            raise ValueError(f'Invalid or duplicate sequence: {sid}')
        seen.add(sid)
        records.append(dict(sequence_id=sid, pages=[join_ocr_candidates(page) for page in record['pages']]))
    atomic_write(args.output, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    atomic_write(args.output.with_suffix('.metadata.json'), json.dumps(dict(
        candidates_sha256=sha256(args.candidates), output_sha256=sha256(args.output),
        code_sha256=sha256(ROOT / 'src/ocr_joining.py'),
        entrypoint_sha256=sha256(Path(__file__)), python=sys.version,
        config=dict(max_gap=1.25, center_tolerance=1.5)), indent=2) + '\n')
    print(f'Joined {len(records)} sequences: {args.output.resolve()}')


if __name__ == '__main__':
    main()
