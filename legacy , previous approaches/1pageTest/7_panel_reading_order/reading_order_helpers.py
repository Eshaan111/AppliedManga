"""Small geometry helpers. Importing this file does not load a detector."""


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def assign_bubbles_to_panels(bubbles, panels, overrides=None):
    """Use center containment; choose the smallest containing panel on overlap."""
    overrides = overrides or {}
    panel_ids = {panel['panel_id'] for panel in panels}
    bubble_ids = {bubble['bubble_id'] for bubble in bubbles}
    if not set(overrides).issubset(bubble_ids):
        raise ValueError('A manual override refers to a missing bubble.')
    assigned = []
    for bubble in bubbles:
        x, y = center(bubble['box'])
        candidates = []
        for panel in panels:
            x1, y1, x2, y2 = panel['box']
            if x1 <= x <= x2 and y1 <= y <= y2:
                candidates.append(panel)
        panel_id = min(candidates, key=lambda panel: area(panel['box']))['panel_id'] if candidates else None
        if bubble['bubble_id'] in overrides:
            panel_id = overrides[bubble['bubble_id']]
            if panel_id is not None and panel_id not in panel_ids:
                raise ValueError(f'Unknown panel ID: {panel_id}')
        assigned.append(dict(bubble, panel_id=panel_id))
    return assigned


def order_bubbles_in_panel(bubbles, row_tolerance=40):
    """Top-to-bottom rows; right-to-left within each approximate row."""
    if row_tolerance < 0:
        raise ValueError('Row tolerance cannot be negative.')
    rows = []
    for bubble in sorted(bubbles, key=lambda bubble: center(bubble['box'])[1]):
        y = center(bubble['box'])[1]
        # Anchor each row at its first center to avoid chaining distant bubbles.
        if not rows or y - rows[-1]['y'] > row_tolerance:
            rows.append({'y': y, 'bubbles': []})
        rows[-1]['bubbles'].append(bubble)
    ordered = []
    for row in rows:
        ordered.extend(sorted(row['bubbles'], key=lambda bubble: -center(bubble['box'])[0]))
    return ordered


def flatten_panel_groups(estimator, panels):
    """Convert the estimator's single panels/groups back to our PN IDs."""
    known_ids = {panel['panel_id'] for panel in panels}
    groups = []
    ambiguous_groups = []
    for box, separable in zip(estimator.ordered_bbs, estimator.bb_estimation_statuses):
        members = [box] if separable else list(box.panels)
        # The estimator does not order members of an inseparable group.
        # A deterministic top/right fallback is explicit in the saved output.
        members = sorted(members, key=lambda box: ((box.ymin + box.ymax) / 2,
                         -(box.xmin + box.xmax) / 2, box.id))
        ids = [box.id for box in members]
        groups.append(ids)
        if not separable:
            ambiguous_groups.append(ids)
    order = [panel_id for group in groups for panel_id in group]
    if len(order) != len(set(order)) or set(order) != known_ids:
        raise ValueError('The estimator did not return every panel exactly once.')
    return order, groups, ambiguous_groups
