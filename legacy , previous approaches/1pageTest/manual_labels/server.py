"""Local manual labeling tool. Run with Python; no extra packages required."""
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / 'data/intermediate'


def read_stage(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def get_state():
    labels = read_stage('bubble_gt_labels.json')
    bubbles = read_stage('bubbles_clean.json')
    characters = read_stage('characters.json')
    if labels['page'] != bubbles['page'] or labels['page'] != characters['page']:
        raise ValueError('Stage results belong to different pages. Regenerate them first.')
    gt = labels['records']
    sequence = gt['sequence_id']
    number = gt['page_number']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', sequence) or not isinstance(number, int) or number < 1:
        raise ValueError('Invalid sequence or page number.')
    image = ROOT / 'TaskData/development/images' / sequence / f'{number:02d}.png'
    if hashlib.sha256(image.read_bytes()).hexdigest() != labels['page']['sha256']:
        raise ValueError('The image differs from the stage results.')
    appearances = []
    for character in characters['records']:
        # Same detections as the notebooks, with A IDs for manual identity grouping.
        appearance = dict(character)
        appearance['appearance_id'] = character['character_id'].replace('C', 'A', 1)
        appearances.append(appearance)
    by_id = {label['bubble_id']: label for label in gt['bubbles']}
    annotated_bubbles = [dict(bubble, **{
        'speaker': by_id[bubble['bubble_id']]['speaker'],
        'text': by_id[bubble['bubble_id']]['text'],
    }) for bubble in bubbles['records']]
    speakers = sorted({b['speaker'] for b in annotated_bubbles if b['speaker'] and b['speaker'].upper() != 'UNKNOWN'})
    # Also list identities whose dialogue did not receive an OCR match.
    with (ROOT / 'TaskData/development/labels.jsonl').open(encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry['sequence_id'] == sequence:
                speakers = sorted(set(speakers) | {row['speaker'] for row in entry['pages'][number - 1]
                    if row['speaker'] and row['speaker'].upper() != 'UNKNOWN'})
                break
    response_file = HERE / 'responses' / f'{sequence}_page_{number:02d}.json'
    fingerprint = hashlib.sha256((OUT / 'characters.json').read_bytes()).hexdigest()
    saved = json.loads(response_file.read_text(encoding='utf-8')) if response_file.exists() else None
    if saved and (saved['page_sha256'] != labels['page']['sha256'] or saved['detections_sha256'] != fingerprint):
        raise ValueError('Saved appearance IDs refer to older detections. Move the saved response aside before relabeling.')
    overrides = saved.get('bubble_speaker_overrides', {}) if saved else {}
    for bubble in annotated_bubbles:
        bubble['automatic_speaker'] = bubble['speaker']
        bubble['speaker'] = overrides.get(bubble['bubble_id'], bubble['speaker'])
    state = {
        'sequence_id': sequence, 'page_number': number,
        'page': labels['page'], 'detections_sha256': fingerprint,
        'speakers': speakers, 'bubbles': annotated_bubbles,
        'appearances': appearances, 'saved': saved,
        'response_file': str(response_file),
    }
    return state, image, response_file


def parse_answer(answer, rejected, state):
    groups = {}
    aliases = {name.lower(): name for name in state['speakers']}
    for name in state['speakers']:
        aliases[name.lower().replace('character', 'char', 1)] = name
    valid_ids = {a['appearance_id'] for a in state['appearances']}
    used = set()

    def appearance_ids(text):
        values = [value.upper() for value in re.split(r'[,\s]+', text.strip()) if value]
        for value in values:
            if value not in valid_ids:
                raise ValueError(f'Unknown appearance: {value}')
            if value in used:
                raise ValueError(f'{value} is assigned more than once.')
            used.add(value)
        return values

    for line in answer.splitlines():
        if not line.strip():
            continue
        parts = re.split(r'\s*[-:]\s*', line.strip(), maxsplit=1)
        if len(parts) != 2:
            raise ValueError('Use one line per identity: Char1 - A1, A2')
        speaker = aliases.get(parts[0].strip().lower())
        if speaker is None:
            raise ValueError(f'Unknown GT identity: {parts[0]}')
        if speaker in groups:
            raise ValueError(f'{speaker} appears on more than one line.')
        groups[speaker] = appearance_ids(parts[1])
    rejected_ids = appearance_ids(rejected)
    return groups, rejected_ids, sorted(valid_ids - used)


def parse_bubble_overrides(overrides, state):
    if not isinstance(overrides, dict):
        raise ValueError('Bubble corrections must be an object.')
    valid_bubbles = {b['bubble_id'] for b in state['bubbles']}
    valid_speakers = set(state['speakers']) | {'UNKNOWN'}
    for bubble_id, speaker in overrides.items():
        if bubble_id not in valid_bubbles:
            raise ValueError(f'Unknown bubble: {bubble_id}')
        if speaker not in valid_speakers:
            raise ValueError(f'Unknown GT speaker: {speaker}')
    return dict(overrides)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == '/':
                self.reply(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/state':
                self.reply(200, get_state()[0])
            elif path == '/page.png':
                self.reply(200, get_state()[1].read_bytes(), 'image/png')
            else:
                self.reply(404, {'error': 'Not found'})
        except Exception as error:
            self.reply(400, {'error': str(error)})

    def do_POST(self):
        try:
            if self.path != '/api/save':
                self.reply(404, {'error': 'Not found'})
                return
            # Accept changes only from this local page.
            if self.headers.get('Origin') != f'http://{self.headers.get("Host")}':
                raise ValueError('Open the tool through its localhost address to save.')
            size = int(self.headers.get('Content-Length', '0'))
            if size < 1 or size > 100_000:
                raise ValueError('Invalid response size.')
            request = json.loads(self.rfile.read(size))
            state, image, path = get_state()
            if request['page_sha256'] != state['page']['sha256'] or request['detections_sha256'] != state['detections_sha256']:
                raise ValueError('Page or detections changed. Reload the tool.')
            groups, rejected, unassigned = parse_answer(request['answer'], request['rejected'], state)
            previous = state['saved'] or {}
            overrides = parse_bubble_overrides(request.get('bubble_speaker_overrides', previous.get('bubble_speaker_overrides', {})), state)
            response = {
                'sequence_id': state['sequence_id'], 'page_number': state['page_number'],
                'page_sha256': state['page']['sha256'],
                'detections_sha256': state['detections_sha256'],
                'answer': request['answer'], 'rejected_answer': request['rejected'],
                'character_appearances': groups, 'rejected_appearances': rejected,
                'unassigned_appearances': unassigned,
                'bubble_speaker_overrides': overrides,
                'appearance_to_detection': {a['appearance_id']: a['character_id'] for a in state['appearances']},
            }
            # Save only your response. Never write the display annotations or GT overlay.
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temporary.replace(path)
            self.reply(200, {'saved': str(path), 'unassigned': unassigned})
        except Exception as error:
            self.reply(400, {'error': str(error)})


if __name__ == '__main__':
    print('Open http://127.0.0.1:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
