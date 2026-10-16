"""Compare exactly two crops from a saved embedding run; no model reload."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare_pair(run, sequence, face_a, face_b):
    run = Path(run)
    rows = json.loads((run / 'crop_index.json').read_text(encoding='utf-8'))
    vectors = np.load(run / 'embeddings.npy', allow_pickle=False)
    indices = []
    for crop_id in (face_a, face_b):
        matches = [i for i, row in enumerate(rows) if row['sequence_id'] == sequence
                   and row['crop_id'] == crop_id and row['strategy'] == 'face']
        if len(matches) != 1:
            raise ValueError(f'Expected one face crop for {sequence}/{crop_id}; found {len(matches)}')
        indices.append(matches[0])
    a, b = indices
    return dict(sequence_id=sequence, face_a=face_a, page_a=rows[a]['page_index'],
                face_b=face_b, page_b=rows[b]['page_index'],
                similarity=float(np.clip(vectors[a] @ vectors[b], -1, 1)),
                crop_path_a=rows[a]['crop_path'], crop_path_b=rows[b]['crop_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--face-a', required=True)
    parser.add_argument('--face-b', required=True)
    args = parser.parse_args()
    print(json.dumps(compare_pair(args.run, args.sequence, args.face_a, args.face_b), indent=2))
