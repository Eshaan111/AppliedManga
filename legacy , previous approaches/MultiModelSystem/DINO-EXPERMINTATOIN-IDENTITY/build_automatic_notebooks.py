"""Generate the two automatic-pipeline notebooks reproducibly."""
import nbformat as nb
from identity import ROOT

SETUP = '''from pathlib import Path
import sys, json
from datetime import datetime
from IPython.display import display, Image as DisplayImage
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if ROOT.name != 'DINO-EXPERMINTATOIN-IDENTITY':
    ROOT = ROOT / 'DINO-EXPERMINTATOIN-IDENTITY'
assert (ROOT / 'detection.py').exists(), ROOT
sys.path.insert(0, str(ROOT))
'''


def write(name, cells):
    notebook = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(text) if kind == 'md'
                                       else nb.v4.new_code_cell(text) for kind, text in cells])
    notebook.metadata = dict(kernelspec=dict(display_name='AppliedTask (CUDA GPU)',
                             language='python', name='venvappliedtask'), language_info=dict(name='python'))
    path = ROOT / 'notebooks' / name
    path.parent.mkdir(exist_ok=True)
    nb.write(notebook, path)
    print(path)


def build():
    write('02_character_detection.ipynb', [
        ('md', '# 1. Detect faces and crop them\n\nOnly hysts YOLOv3 face detection. No landmarks. Run download_detector.py once before this notebook. Manual annotations are preserved.'),
        ('code', SETUP + 'import detection'),
        ('code', "PAGES = 3\nCONFIDENCE = 0.5\npages = detection.select_pages(limit=PAGES)\noutput = ROOT / 'outputs' / f'detection_yolov3_{datetime.now():%Y%m%d_%H%M%S_%f}'\nmanifest = detection.run_detection(pages, output, confidence=CONFIDENCE)\nprint('Use this in notebook 03:', output / 'detections.json')"),
        ('md', 'Inspect faces and missed detections. Red boxes are the exact crops sent to DINO.'),
        ('code', "for page in manifest['pages']:\n    display(DisplayImage(filename=page['overlay_path'], width=850))"),
    ])
    write('03_dino_identity.ipynb', [
        ('md', '# 2. DINO cosine similarity: test pairs\n\nQuestion: does DINO similarity separate same and different manga characters? No clustering. Human labels are needed to answer this.'),
        ('code', SETUP + 'import automatic_identity as auto'),
        ('code', "DETECTIONS = None  # Set Path(r'C:/.../detections.json')\nassert DETECTIONS is not None, 'Paste the manifest path from notebook 02'\nTHRESHOLD = 0.8  # exploratory; not calibrated\noutput = ROOT / 'outputs' / f'pairs_{datetime.now():%Y%m%d_%H%M%S_%f}'\nsummary = auto.run_identity(DETECTIONS, output, threshold=THRESHOLD)\nprint(summary)\nprint(output / 'pair_review.html')"),
        ('md', 'Open pair_review.html for side-by-side crops. Enter same/different in the same_character column of pair_review.csv. Scores are cosine similarities, not probabilities. Review both high and low scores. All cross-page pairs within each sequence are saved; identity labels never drive embeddings. Optional: fill the detector identity_labels.json and pass --labels to the CLI for fixed-threshold metrics.'),
        ('code', "pairs = json.loads((output / 'cross_page_pairs.json').read_text())\nfor pair in pairs[:10]:\n    print(pair['crop1'], 'vs', pair['crop2'], 'similarity =', round(pair['similarity'], 4))\n    display(DisplayImage(filename=pair['crop_path1'], width=160))\n    display(DisplayImage(filename=pair['crop_path2'], width=160))"),
    ])


if __name__ == '__main__':
    build()
