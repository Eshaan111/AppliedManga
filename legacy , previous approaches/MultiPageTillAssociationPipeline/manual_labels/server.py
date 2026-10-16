"""Local multipage labeling server; no model execution."""
import hashlib
import json
import re
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import pipeline_helpers as P

def find_page(key):
    for item in P.pages():
        if key == f"{item['split']}/{item['sequence_id']}/{item['page_number']}":
            return item
    raise ValueError('Page is not in the fixed split manifest.')

def page_list():
    result = []
    for item in P.pages():
        folder = P.page_folder(item)
        ready = all((folder / name).is_file() for name in ['bubbles_clean.json','ocr_clean.json','bubble_gt_labels.json','characters.json'])
        saved_path = P.response_path(item)
        status = 'not prepared'
        if ready:
            status = 'unlabeled'
            if saved_path.exists():
                saved = P.read_json(saved_path)
                speakers = {b['bubble_id']:b['speaker'] for b in P.load_stage(item, 'bubble_gt_labels.json')['bubbles']}
                speakers.update(saved.get('bubble_speaker_overrides', {}))
                ignored = set(saved.get('ignored_bubble_ids', []))
                speakers = {b:s for b,s in speakers.items() if b not in ignored}
                status = 'saved' if not saved['unassigned_appearances'] and all(s is not None for s in speakers.values()) else 'incomplete'
                if saved.get('stage_fingerprint') != P.stage_fingerprint(item):
                    status = 'needs review'
        result.append(dict(item, key=f"{item['split']}/{item['sequence_id']}/{item['page_number']}", ready=ready, status=status))
    return result

def get_state(key):
    item = find_page(key)
    bubbles = P.load_stage(item, 'bubbles_clean.json')
    characters = P.load_stage(item, 'characters.json')
    gt = P.load_stage(item, 'bubble_gt_labels.json')
    info = P.page_info(item)
    path = P.response_path(item)
    saved = P.read_json(path) if path.exists() else None
    fingerprint = P.stage_fingerprint(item)
    if saved and (saved['page_sha256'] != info['sha256'] or saved.get('stage_fingerprint') != fingerprint):
        raise ValueError('Preparation changed. Move this saved response aside and relabel: ' + str(path))
    characters += saved.get('added_appearances', []) if saved else []
    appearances = [dict(c, appearance_id=c['character_id'].replace('C','A',1)) for c in characters]
    labels = {b['bubble_id']:b for b in gt['bubbles']}
    overrides = saved.get('bubble_speaker_overrides', {}) if saved else {}
    ignored = set(saved.get('ignored_bubble_ids', [])) if saved else set()
    annotated = [dict(b, ignored=b['bubble_id'] in ignored, text=labels[b['bubble_id']]['text'],
        automatic_speaker=labels[b['bubble_id']]['speaker'],
        speaker=overrides.get(b['bubble_id'], labels[b['bubble_id']]['speaker'])) for b in bubbles]
    dialogue = P.gt_sequence(item)['pages'][item['page_number'] - 1]
    speakers = sorted({r['speaker'] for r in dialogue if r['speaker'] and r['speaker'].upper() != 'UNKNOWN'})
    if saved:
        speakers = sorted(set(speakers) | {name for name in saved['character_appearances'] if re.fullmatch(r'silent[1-9]\d*', name)})
    state = dict(item, key=key, page=info, stage_fingerprint=fingerprint,
        detections_sha256=fingerprint['characters.json'], speakers=speakers, bubbles=annotated,
        appearances=appearances, saved=saved, response_file=str(path))
    return state, Path(item['image_path']), path

def added_appearances(values, state):
    original = P.load_stage(state, 'characters.json')
    original_ids = {c['character_id'] for c in original}
    result = []
    for value in values:
        cid = value['character_id']
        if not re.fullmatch(r'C[1-9]\d*', cid) or cid in original_ids:
            raise ValueError('Duplicate or invalid added appearance ID.')
        box = value['box']
        if len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError('Box must contain four integer coordinates.')
        x1,y1,x2,y2 = box
        if not (0 <= x1 < x2 <= state['page']['width'] and 0 <= y1 < y2 <= state['page']['height']):
            raise ValueError('Added box must lie inside the image.')
        original_ids.add(cid)
        result.append({'character_id':cid, 'box':box, 'face':None, 'body':None, 'manual':True})
    return result

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
        if speaker is None and re.fullmatch(r'silent[1-9]\d*', parts[0].strip().lower()):
            speaker = parts[0].strip().lower()
        if speaker is None:
            raise ValueError(f'Unknown GT identity: {parts[0]}. Use silent1, silent2, etc. for identities without GT dialogue.')
        if speaker in groups:
            raise ValueError(f'{speaker} appears on more than one line.')
        groups[speaker] = appearance_ids(parts[1])
    rejected_ids = appearance_ids(rejected)
    return groups, rejected_ids, sorted(valid_ids - used)


def add_unassigned_to_silent(groups, unassigned):
    groups = {name:list(ids) for name, ids in groups.items()}
    automatic = {}
    number = 1
    for appearance_id in sorted(unassigned, key=lambda value:int(value[1:])):
        while f"silent{number}" in groups:
            number += 1
        name = f"silent{number}"
        groups[name] = [appearance_id]
        automatic[name] = [appearance_id]
        number += 1
    return groups, automatic


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


def parse_ignored_bubbles(values, state):
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError('Ignored bubble IDs must be a list of strings.')
    valid_ids = {b['bubble_id'] for b in state['bubbles']}
    if len(values) != len(set(values)) or not set(values).issubset(valid_ids):
        raise ValueError('Ignored bubble IDs must be unique and belong to this page.')
    return list(values)


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
            url = urlparse(self.path)
            if url.path == '/':
                self.reply(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif url.path == '/api/pages':
                self.reply(200, page_list())
            elif url.path in ['/api/state', '/page.png']:
                key = parse_qs(url.query)['key'][0]
                state, image, path = get_state(key)
                if url.path == '/api/state':
                    self.reply(200, state)
                else:
                    self.reply(200, image.read_bytes(), 'image/png')
            else:
                self.reply(404, {'error':'Not found'})
        except Exception as error:
            self.reply(400, {'error':str(error)})

    def do_POST(self):
        try:
            if self.path != '/api/save':
                self.reply(404, {'error':'Not found'})
                return
            if self.headers.get('Origin') != f'http://{self.headers.get("Host")}':
                raise ValueError('Open this tool through its localhost address.')
            size = int(self.headers.get('Content-Length', '0'))
            if not 1 <= size <= 500_000:
                raise ValueError('Invalid response size.')
            request = json.loads(self.rfile.read(size))
            state, image, path = get_state(request['key'])
            if request['page_sha256'] != state['page']['sha256'] or request['stage_fingerprint'] != state['stage_fingerprint']:
                raise ValueError('Page changed. Reload before saving.')
            additions = added_appearances(request.get('added_appearances', []), state)
            original = P.load_stage(state, 'characters.json')
            state['appearances'] = [dict(c, appearance_id=c['character_id'].replace('C','A',1)) for c in original + additions]
            groups, rejected, unassigned = parse_answer(request['answer'], request['rejected'], state)
            groups, automatic = add_unassigned_to_silent(groups, unassigned)
            unassigned = []
            answer = '\n'.join(f"{name} - {', '.join(ids)}" for name, ids in groups.items())
            overrides = parse_bubble_overrides(request['bubble_speaker_overrides'], state)
            ignored = parse_ignored_bubbles(request.get('ignored_bubble_ids', []), state)
            response = {k:state[k] for k in ['sequence_id','page_number','split','stage_fingerprint','detections_sha256']}
            response.update(page_sha256=state['page']['sha256'], answer=answer,
                rejected_answer=request['rejected'], character_appearances=groups,
                rejected_appearances=rejected, unassigned_appearances=unassigned,
                bubble_speaker_overrides=overrides, added_appearances=additions, ignored_bubble_ids=ignored,
                appearance_to_detection={a['appearance_id']:a['character_id'] for a in state['appearances']})
            P.write_json(path, response)
            self.reply(200, {'saved':str(path), 'unassigned':unassigned, 'answer':answer, 'automatic_silent_groups':automatic})
        except Exception as error:
            self.reply(400, {'error':str(error)})

if __name__ == '__main__':
    print('Open http://127.0.0.1:8766', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8766), Handler).serve_forever()
