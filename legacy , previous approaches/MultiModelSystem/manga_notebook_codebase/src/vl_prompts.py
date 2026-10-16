"""Shared ordered speaker inputs, targets, and strict output parsing."""
import json
import random
import re

OUTPUT_FORMAT = 'ordered_speaker_lines_v1'


def ordered_groups(pages):
    """Flatten pages in their saved order; never sort IDs alphabetically."""
    if len(pages) != 3:
        raise ValueError('Expected exactly three pages')
    groups = []
    for index, page in enumerate(pages):
        if page['page_index'] != index:
            raise ValueError('Pages must be in order 0, 1, 2')
        for group in page['groups']:
            if not group['id'].startswith(f'p{index}_'):
                raise ValueError('Group ID does not match its page')
            groups.append(dict(group_id=group['id'], text=group['text'], box=group['box']))
    validate_groups(groups)
    return groups


def validate_groups(groups):
    ids = [g['group_id'] for g in groups]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate requested group IDs')
    for group in groups:
        if not re.fullmatch(r'p[0-2]_[^:\s]+', group['group_id']):
            raise ValueError('Invalid page-qualified group ID')
        if not isinstance(group['text'], str) or not group['text'].strip():
            raise ValueError('Empty OCR text')
        if len(group['box']) != 4:
            raise ValueError('Expected box [left, top, right, bottom]')


def build_speaker_prompt(requested_groups):
    """Use OCR and boxes only. Reference speakers never enter the prompt."""
    validate_groups(requested_groups)
    lines = [
        'You are given three consecutive manga pages with OCR groups marked on the images.',
        'Images are in page order. p0, p1, and p2 refer to the first, second, and third page.',
        'Boxes use original-image pixel coordinates [left, top, right, bottom].',
        'Your task is only speaker attribution. Use the supplied OCR; do not transcribe it.',
        'For every listed group, output exactly one speaker label.',
        'Output exactly one label per line, in exactly the same order as the groups below.',
        'Do not output group IDs, JSON, explanations, or extra text.',
        'Use char1, char2, char3, ... for characters.',
        'The same character must use the same label across all three pages.',
        'Different characters must use different labels.',
        'Character numbering is arbitrary.',
        'char1 does NOT mean the first character, main character, or most common speaker.',
        'Use NARRATION only for narrator text.',
        'Use UNKNOWN only when the speaker cannot be determined.',
        '', 'Groups:',
    ]
    for group in requested_groups:
        # JSON quoting keeps multiline OCR text on one input line.
        lines.append(f'{group["group_id"]}: text={json.dumps(group["text"], ensure_ascii=False)} '
                     f'box={json.dumps(group["box"])}')
    return '\n'.join(lines)


def valid_speaker(label):
    # Positive numbers only: char0 and char01 are not valid labels.
    return isinstance(label, str) and (
        label in {'UNKNOWN', 'NARRATION'} or re.fullmatch(r'char[1-9][0-9]*', label) is not None)


def build_speaker_target(requested_groups):
    labels = [group['speaker'] for group in requested_groups]
    if not all(valid_speaker(label) for label in labels):
        raise ValueError('Invalid speaker in training target')
    return '\n'.join(labels)


def parse_speaker_output(text, requested_groups):
    """Reject the whole response if its length or any label is invalid."""
    ids = [group['group_id'] for group in requested_groups]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate requested group IDs')
    labels = [line.strip() for line in text.splitlines() if line.strip()]
    if len(labels) != len(requested_groups):
        raise ValueError(f'Expected {len(requested_groups)} labels, got {len(labels)}')
    if not all(valid_speaker(label) for label in labels):
        raise ValueError('Invalid speaker label in response')
    # A missing line could shift every later speaker; never accept a partial list.
    return {group['group_id']: label for group, label in zip(requested_groups, labels)}


def permute_speaker_labels(labels, rng=random):
    """One bijection for the entire sequence; special classes remain fixed."""
    if not all(valid_speaker(label) for label in labels):
        raise ValueError('Cannot permute invalid speaker labels')
    characters = sorted({s for s in labels if s.startswith('char')}, key=lambda s: int(s[4:]))
    shuffled = characters.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(characters, shuffled))
    return [mapping.get(label, label) for label in labels]
