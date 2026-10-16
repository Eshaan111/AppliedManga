"""Deterministic OCR line grouping; references are used only for comparison."""
import re
from difflib import SequenceMatcher
from statistics import median


def join_ocr_candidates(candidates, max_gap=1.25, center_tolerance=1.5):
    """Link nearby aligned lines without relying on OCR candidate ID order.

    Distances are relative to line height. One nearest predecessor/successor
    per fragment prevents branching into multiple neighboring text columns.
    This proposes text regions, not verified speech-bubble boundaries.
    """
    if not candidates:
        return []
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('Candidate IDs must be unique within a page')
    ordered = sorted(candidates, key=lambda c: (c['box'][1], c['box'][0]))
    heights = [max(1, c['box'][3] - c['box'][1]) for c in ordered]
    typical = median(heights)
    edges = []
    for i, a in enumerate(ordered):
        ax0, ay0, ax1, ay1 = a['box']
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            bx0, by0, bx1, by1 = b['box']
            scale = min(heights[i], heights[j], typical * 1.5)
            # Same-row fragments and very different font sizes stay separate.
            if by0 - ay0 < scale * .45 or max(heights[i], heights[j]) / min(heights[i], heights[j]) > 1.8:
                continue
            gap = by0 - ay1
            if gap > max_gap * scale:
                continue
            overlap = min(ax1, bx1) - max(ax0, bx0)
            width = max(1, min(ax1 - ax0, bx1 - bx0))
            center = abs((ax0 + ax1 - bx0 - bx1) / 2)
            if overlap / width < .5 or center > center_tolerance * scale:
                continue
            cost = max(gap, 0) / scale + center / scale
            edges.append((cost, i, j))
    parent = list(range(len(ordered)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    outgoing, incoming = set(), set()
    for _, i, j in sorted(edges):
        if i in outgoing or j in incoming:
            continue
        # Require all lines in the resulting group to share horizontal space.
        members = [k for k in range(len(ordered)) if find(k) in {find(i), find(j)}]
        if max(ordered[k]['box'][0] for k in members) >= min(ordered[k]['box'][2] for k in members):
            continue
        parent[find(j)] = find(i)
        outgoing.add(i)
        incoming.add(j)
    clusters = {}
    for i, candidate in enumerate(ordered):
        clusters.setdefault(find(i), []).append(candidate)
    groups = []
    for lines in clusters.values():
        boxes = [c['box'] for c in lines]
        groups.append(dict(candidate_ids=[c['id'] for c in lines],
                           text=' '.join(c['text'].strip() for c in lines),
                           box=[min(b[0] for b in boxes), min(b[1] for b in boxes),
                                max(b[2] for b in boxes), max(b[3] for b in boxes)]))
    groups.sort(key=lambda g: (g['box'][1], -g['box'][0]))
    for i, group in enumerate(groups):
        group['id'] = f'g{i}'
    return groups


def compare_groups_to_targets(groups, targets, threshold=.5):
    """One-to-one fuzzy allocation for diagnostics; never edits script text.

    Target transcription items may span multiple visible regions. Similarity
    ignores punctuation/spacing to isolate grouping from small OCR errors.
    """
    from eval_utils import official
    def normalized(text):
        return re.sub(r'[^a-z0-9]', '', text.lower())
    weights = []
    for i, target in enumerate(targets):
        for group in groups:
            score = SequenceMatcher(None, normalized(target['text']), normalized(group['text']), autojunk=False).ratio()
            if score >= threshold:
                weights.append((f't{i}', group['id'], score))
    mapping = official.best_mapping(weights)
    scores = {(t, g): s for t, g, s in weights}
    allocated = []
    for group in groups:
        target_id = mapping.get(group['id'])
        index = int(target_id[1:]) if target_id else None
        allocated.append(dict(**group, target_index=index,
                              target_text=targets[index]['text'] if index is not None else None,
                              similarity=scores.get((target_id, group['id']))))
    matched = {g['target_index'] for g in allocated if g['target_index'] is not None}
    return dict(groups=allocated, unmatched_targets=[dict(target_index=i, **t)
                for i, t in enumerate(targets) if i not in matched])


def draw_ocr_groups(image, candidates, groups):
    """Return raw-ID and color-group PIL previews, without modifying the image."""
    from PIL import ImageDraw
    palette = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#008b8b', '#e377c2']
    raw, joined = image.convert('RGB'), image.convert('RGB')
    raw_draw, group_draw = ImageDraw.Draw(raw), ImageDraw.Draw(joined)
    by_id = {c['id']: c for c in candidates}
    for candidate in candidates:
        raw_draw.rectangle(candidate['box'], outline='red', width=2)
        raw_draw.text(candidate['box'][:2], candidate['id'], fill='red', stroke_width=1, stroke_fill='white')
    for i, group in enumerate(groups):
        color = palette[i % len(palette)]
        for cid in group['candidate_ids']:
            group_draw.rectangle(by_id[cid]['box'], outline=color, width=2)
        group_draw.rectangle(group['box'], outline=color, width=4)
        x, y = group['box'][:2]
        group_draw.text((x, max(0, y - 13)), group['id'], fill=color, stroke_width=1, stroke_fill='white')
    return raw, joined
