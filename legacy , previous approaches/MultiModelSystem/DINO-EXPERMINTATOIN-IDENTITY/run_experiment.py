import argparse
from datetime import datetime
from identity import MODEL_PATH, ROOT, run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINOv2 character identity experiment on manual crops')
    parser.add_argument('--annotations', default=ROOT / 'annotations/characters.json')
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--output', default=ROOT / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    report = run_experiment(args.annotations, args.output, args.model, args.batch_size)
    print(report)
    print(f'Saved results: {args.output}')
