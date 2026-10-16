"""Current character cleanup and entity clustering, reused unchanged from full inference."""
import copy
import numpy as np

def remove_duplicate_characters(results, embeddings, min_iou=0.90, min_cosine=0.95):
    """Remove contained, visually similar duplicates; preserve all text indices."""
    assert 0 <= min_iou <= 1 and -1 <= min_cosine <= 1
    assert len(results) == len(embeddings)
    cleaned_pages, cleaned_embeddings, audits = [], [], []
    for page_index, (raw, vectors) in enumerate(zip(results, embeddings)):
        boxes = np.asarray(raw['characters'], dtype=float).reshape(-1, 4)
        assert len(vectors) == len(boxes), 'Character/embedding count mismatch'
        assert len(raw['character_cluster_labels']) == len(boxes)
        areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        # Largest first. Equal areas prefer the earlier original character index.
        retained, representatives, decisions = [], {}, []
        for inner in sorted(range(len(boxes)), key=lambda i: (-areas[i], i)):
            representative = inner
            for outer in retained:
                a, b = boxes[inner], boxes[outer]
                if not np.isfinite(a).all() or not np.isfinite(b).all() or areas[inner] <= 0 or areas[outer] <= 0:
                    continue
                contained = a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]
                iou = float(areas[inner] / areas[outer]) if contained else 0.0
                if not contained or iou < min_iou:
                    continue
                x, y = np.asarray(vectors[inner], dtype=float), np.asarray(vectors[outer], dtype=float)
                assert x.ndim == y.ndim == 1 and x.shape == y.shape
                denominator = np.linalg.norm(x) * np.linalg.norm(y)
                if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(denominator) or denominator <= 0:
                    continue
                cosine = float(np.clip(np.dot(x, y) / denominator, -1, 1))
                if cosine >= min_cosine:
                    representative = outer
                    decisions.append(dict(removed_original_index=inner, retained_original_index=outer,
                                          smaller_box_coverage=1.0, iou=iou, cosine=cosine))
                    break
            representatives[inner] = representative
            if representative == inner:
                retained.append(inner)
        # Keep the original relative order of surviving character appearances.
        kept = sorted(retained)
        new_index = {old: new for new, old in enumerate(kept)}
        old_to_new = [new_index[representatives[i]] for i in range(len(boxes))]
        page = copy.deepcopy(raw)
        for key in ['characters', 'character_cluster_labels', 'character_names']:
            if key in raw:
                assert len(raw[key]) == len(boxes), f'Misaligned {key}'
                page[key] = [raw[key][i] for i in kept]
        links, seen = [], set()
        for text_index, char_index in raw['text_character_associations']:
            assert 0 <= text_index < len(raw['texts']) and 0 <= char_index < len(boxes)
            link = (text_index, old_to_new[char_index])
            if link not in seen:
                seen.add(link)
                links.append(list(link))
        page['text_character_associations'] = links
        assert page['texts'] == raw['texts'] and page['ocr'] == raw['ocr']
        cleaned_pages.append(page)
        cleaned_embeddings.append([vectors[i] for i in kept])
        audits.append(dict(page_index=page_index, kept_original_indices=kept,
                           original_to_cleaned_index=old_to_new, removals=decisions))
    return dict(pages=cleaned_pages, embeddings=cleaned_embeddings, audit=audits)

def page_clusters(result, page_index):
    groups = {}
    for char_index, label in enumerate(result['character_cluster_labels']):
        cluster_id = f'PG{page_index+1}:CL{int(label)+1}' if int(label) >= 0 else f'PG{page_index+1}:C{char_index+1}:unassigned'
        groups.setdefault(cluster_id, []).append(char_index)
    return groups


def clusterize(results, embeddings, threshold=0.50):
    assert -1 <= threshold <= 1, 'Cosine threshold must be between -1 and 1'
    vectors = [np.asarray(page, dtype=float) for page in embeddings]
    assert len(results) == len(vectors)
    for result, page_vectors in zip(results, vectors):
        assert len(result['characters']) == len(page_vectors), 'Embedding count mismatch'
    entities, decisions, char_to_entity = {}, [], {}
    for page_index, result in enumerate(results):
        clusters = page_clusters(result, page_index)
        candidates = []
        for cluster_id, indices in clusters.items():
            for entity_id, entity in entities.items():
                best_score, best_pair = -float('inf'), None
                for current in indices:
                    for member in entity['members']:
                        score = float(np.dot(vectors[page_index][current], vectors[member['page_index']][member['character_index']]))
                        if score > best_score:
                            best_score = score
                            best_pair = dict(current=f'PG{page_index+1}:C{current+1}', existing=member['character_id'])
                if best_score >= threshold:
                    candidates.append(dict(cluster_id=cluster_id, entity_id=entity_id, score=best_score, best_pair=best_pair))
        candidates.sort(key=lambda candidate: candidate['score'], reverse=True)
        accepted, used_entities = {}, set()
        for candidate in candidates:
            if candidate['cluster_id'] not in accepted and candidate['entity_id'] not in used_entities:
                accepted[candidate['cluster_id']] = candidate
                used_entities.add(candidate['entity_id'])
        for cluster_id, indices in clusters.items():
            match = accepted.get(cluster_id)
            entity_id = match['entity_id'] if match else f'E{len(entities)+1}'
            if entity_id not in entities:
                entities[entity_id] = dict(clusters=[], members=[])
            entities[entity_id]['clusters'].append(cluster_id)
            for char_index in indices:
                character_id = f'PG{page_index+1}:C{char_index+1}'
                entities[entity_id]['members'].append(dict(page_index=page_index, character_index=char_index,
                                                         character_id=character_id, cluster_id=cluster_id))
                char_to_entity[character_id] = entity_id
            decisions.append(dict(page_index=page_index, cluster_id=cluster_id, entity_id=entity_id,
                                  reason='matched' if match else 'new_entity',
                                  score=match['score'] if match else None, best_pair=match['best_pair'] if match else None))
    return dict(entities=entities, char_to_entity=char_to_entity, decisions=decisions)
