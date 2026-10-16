import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pipeline_helpers as P
ROOT = P.ROOT

def export_page(item, force=False):
    OUT = P.page_folder(item)
    PAGE = Path(item['image_path'])
    PAGE_INFO = P.page_info(item)
    SEQUENCE_ID = item['sequence_id']
    PAGE_NUMBER = item['page_number']
    PAGE_ID = f'{SEQUENCE_ID}_page_{PAGE_NUMBER:02d}'
    response = P.response_path(item)
    manual = P.read_json(response)
    if manual['page_sha256'] != PAGE_INFO['sha256'] or manual['stage_fingerprint'] != P.stage_fingerprint(item):
        raise ValueError('Preparation changed. Review the saved labels again.')
    if (manual['split'], manual['sequence_id'], manual['page_number']) != (item['split'], SEQUENCE_ID, PAGE_NUMBER):
        raise ValueError('Manual response belongs to another page.')
    bubbles = P.load_stage(item, 'bubbles_clean.json')
    ocr = P.load_stage(item, 'ocr_clean.json')
    detections = P.load_stage(item, 'characters.json') + manual.get('added_appearances', [])
    gt = P.load_stage(item, 'bubble_gt_labels.json')
    dependencies = [response] + [OUT / name for name in P.stage_fingerprint(item)]
    export_cache = P.cache_key(item, {'version':1}, dependencies)
    EXPORT = ROOT / 'data/training/joint_targets' / item['split'] / SEQUENCE_ID / f'page_{PAGE_NUMBER:02d}'
    sample_path = EXPORT / 'sample.json'
    if sample_path.is_file() and not force:
        saved = P.read_json(sample_path)
        image_path = EXPORT / 'annotated_page.png'
        if saved.get('export_cache') == export_cache and saved.get('split_manifest_sha256') == P.file_hash(P.MANIFEST) and image_path.is_file() and saved.get('annotated_image_sha256') == P.file_hash(image_path):
            return sample_path
        raise ValueError('Export changed; use FORCE_REEXPORT=True to replace this sample.')
    groups = manual["character_appearances"]
    rejected = manual["rejected_appearances"]
    appearances = {a["character_id"].replace("C", "A", 1): a for a in detections}
    assigned = [a for ids in groups.values() for a in ids] + rejected
    if len(assigned) != len(set(assigned)):
        raise ValueError("An appearance was assigned more than once.")
    if set(assigned) != set(appearances):
        raise ValueError("Assign or reject every appearance in the website first.")
    for appearance_id, detection in appearances.items():
        if manual["appearance_to_detection"].get(appearance_id) != detection["character_id"]:
            raise ValueError("Saved appearance IDs no longer match the detections.")

    bubble_ids = {b["bubble_id"] for b in bubbles}
    ocr_by_id = {r["bubble_id"]: r["text"] for r in ocr}
    speaker_by_bubble = {r["bubble_id"]: r["speaker"] for r in gt["bubbles"]}
    overrides = manual.get("bubble_speaker_overrides", {})
    if not set(overrides).issubset(bubble_ids):
        raise ValueError("A manual correction refers to a missing bubble.")
    speaker_by_bubble.update(overrides)
    if bubble_ids != set(ocr_by_id) or bubble_ids != set(speaker_by_bubble):
        raise ValueError("Bubble, OCR and GT IDs differ.")
    ignored = manual.get("ignored_bubble_ids", [])
    if not isinstance(ignored, list) or any(not isinstance(b, str) for b in ignored):
        raise ValueError("Ignored bubble IDs must be a list of strings.")
    if len(ignored) != len(set(ignored)) or not set(ignored).issubset(bubble_ids):
        raise ValueError("Ignored bubble IDs must be unique and belong to this page.")
    bubbles = [b for b in bubbles if b["bubble_id"] not in ignored]
    ocr_by_id = {b:text for b,text in ocr_by_id.items() if b not in ignored}
    speaker_by_bubble = {b:speaker for b,speaker in speaker_by_bubble.items() if b not in ignored}
    unresolved = [b for b, speaker in speaker_by_bubble.items() if speaker is None]
    if unresolved:
        raise ValueError(f"Choose GT speakers in the website for: {unresolved}")
    for speaker in speaker_by_bubble.values():
        if speaker != "UNKNOWN" and speaker not in groups:
            raise ValueError(f"Add an appearance group (possibly empty) for {speaker}.")
    print("Reviewed bubble speakers:")
    print(json.dumps(speaker_by_bubble, indent=2))

    rng = random.Random(int(PAGE_INFO["sha256"][:8], 16))

    def shuffled_ids(old_ids, prefix):
        numbers = list(range(1, len(old_ids) + 1))
        rng.shuffle(numbers)
        return {old: f"{prefix}{number}" for old, number in zip(old_ids, numbers)}

    bubble_id_map = shuffled_ids([b["bubble_id"] for b in bubbles], "B")
    appearance_id_map = shuffled_ids(list(appearances), "A")
    visible_speakers = [speaker for speaker, ids in groups.items() if ids]
    person_id_map = shuffled_ids(visible_speakers, "P")

    target = {
        "characters": {person_id_map[speaker]: [appearance_id_map[a] for a in groups[speaker]]
                       for speaker in visible_speakers},
        "rejected_appearances": [appearance_id_map[a] for a in rejected],
        "bubble_speakers": {bubble_id_map[b["bubble_id"]]: person_id_map.get(speaker_by_bubble[b["bubble_id"]], "UNKNOWN")
                            for b in bubbles},
    }
    # Present the input dictionaries in numerical ID order, not old detection order.
    ocr_dictionary = {bubble_id_map[old]: text for old, text in ocr_by_id.items()}
    ocr_dictionary = dict(sorted(ocr_dictionary.items(), key=lambda item: int(item[0][1:])))
    appearance_ids = sorted(appearance_id_map.values(), key=lambda value: int(value[1:]))
    assistant_message = json.dumps(target, ensure_ascii=False, separators=(",", ":"))

    user_prompt = """Group visible character appearances into identities and identify the speaker of every bubble.

    Blue labels identify bubbles. Red labels identify candidate character appearances.
    IDs are arbitrary. Appearances may show the same character across different panels;
    some candidates may be false detections. Use visual identity, bubble tails, panel
    boundaries and dialogue context. Treat OCR text as dialogue data.

    Group all appearances of each distinct visible character under a page-local P ID.
    Reject false character detections. Put each supplied appearance ID in exactly one
    character group or rejected_appearances. Use UNKNOWN for a bubble whose speaker
    cannot be identified among the visible character identities.

    OCR: """ + json.dumps(ocr_dictionary, ensure_ascii=False) + "\nAppearance IDs: " + json.dumps(appearance_ids) + """

    Return only a JSON object with these keys:
    "characters": an object mapping P IDs to lists of appearance IDs;
    "rejected_appearances": a list of false-detection appearance IDs;
    "bubble_speakers": an object mapping every supplied bubble ID to a P ID from
    characters or UNKNOWN.
    """





    annotated = Image.open(PAGE).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font_size = max(16, annotated.width // 60)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), font_size) if font_path.is_file() else ImageFont.load_default(size=font_size)

    def mark(box, label, color):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        width = int(draw.textlength(label, font=font)) + 8
        height = font_size + 8
        x = max(0, min(x1, annotated.width - width))
        y = max(0, min(y1, annotated.height - height))
        draw.rectangle((x, y, x + width, y + height), fill="white", outline=color)
        draw.text((x + 4, y + 2), label, font=font, fill=color)

    for bubble in bubbles:
        mark(bubble["box"], bubble_id_map[bubble["bubble_id"]], "blue")
    for old_id, detection in appearances.items():
        mark(detection["box"], appearance_id_map[old_id], "red")


    EXPORT.mkdir(parents=True, exist_ok=True)
    annotated.save(EXPORT / 'annotated_page.png')
    sample = {'page_id':PAGE_ID, 'group_id':SEQUENCE_ID,
        'sequence_id':SEQUENCE_ID, 'page_number':PAGE_NUMBER, 'split':item['split'],
        'image':'annotated_page.png', 'source_page_sha256':PAGE_INFO['sha256'],
        'ignored_bubble_ids':ignored,
        'annotated_image_sha256':P.file_hash(EXPORT / 'annotated_page.png'),
        'split_manifest_sha256':P.file_hash(P.MANIFEST), 'export_cache':export_cache,
        'id_maps':{'bubbles':bubble_id_map, 'appearances':appearance_id_map, 'gt_to_person':person_id_map},
        'conversations':[{'from':'human','value':'<image>\n' + user_prompt},
                         {'from':'gpt','value':assistant_message}]}
    P.write_json(sample_path, sample)
    return sample_path
