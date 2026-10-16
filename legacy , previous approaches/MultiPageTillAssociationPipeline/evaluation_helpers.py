"""Compare grouping and speakers without depending on arbitrary P numbering."""
def metrics(prediction, target):
    if not isinstance(prediction, dict) or set(prediction) != set(target):
        raise ValueError('Invalid target schema.')
    appearance_ids = set(target['rejected_appearances'])
    for group in target['characters'].values():
        appearance_ids.update(group)
    used = []
    for person, group in prediction['characters'].items():
        if not isinstance(group, list) or not group:
            raise ValueError('Character groups must be nonempty lists.')
        used.extend(group)
    used.extend(prediction['rejected_appearances'])
    if len(used) != len(set(used)) or set(used) != appearance_ids:
        raise ValueError('Appearances must form a complete, unique partition.')
    if set(prediction['bubble_speakers']) != set(target['bubble_speakers']):
        raise ValueError('Bubble IDs differ.')
    def speaker_group(value, groups):
        if value == 'UNKNOWN':
            return 'UNKNOWN'
        if value not in groups:
            raise ValueError('Speaker refers to a missing identity.')
        return frozenset(groups[value])
    correct = 0
    unknown_correct = 0
    unknown_count = 0
    for bubble, expected in target['bubble_speakers'].items():
        actual = prediction['bubble_speakers'][bubble]
        equal = speaker_group(actual, prediction['characters']) == speaker_group(expected, target['characters'])
        correct += equal
        if expected == 'UNKNOWN':
            unknown_count += 1
            unknown_correct += equal
    groups_equal = {frozenset(x) for x in prediction['characters'].values()} == {frozenset(x) for x in target['characters'].values()}
    rejected_equal = set(prediction['rejected_appearances']) == set(target['rejected_appearances'])
    return {'speaker_correct':correct, 'bubble_count':len(target['bubble_speakers']),
            'unknown_correct':unknown_correct, 'unknown_count':unknown_count,
            'grouping_correct':groups_equal, 'rejection_correct':rejected_equal,
            'exact_semantic_match':groups_equal and rejected_equal and correct == len(target['bubble_speakers'])}
