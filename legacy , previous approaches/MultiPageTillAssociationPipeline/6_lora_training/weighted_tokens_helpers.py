"""Extra functions for joint JSON targets. Importing this module loads no model."""
import json
import math
import re
from pathlib import Path


def load_joint_sample(path):
    path = Path(path)
    sample = json.loads(path.read_text(encoding='utf-8'))
    image = path.parent / sample['image']
    if not image.is_file():
        raise FileNotFoundError(image)
    messages = sample['conversations']
    if len(messages) != 2 or messages[0]['from'] != 'human' or messages[1]['from'] != 'gpt':
        raise ValueError('Expected one human message and one gpt message.')
    return {
        'page_id': sample['page_id'], 'group_id': sample['group_id'],
        'sequence_id': sample['sequence_id'], 'page_number': sample['page_number'], 'split': sample['split'],
        'image': str(image), 'prompt': messages[0]['value'],
        'target': json.loads(messages[1]['value']),
    }


def build_assistant_with_spans(target, include_field_spans=False):
    """Write compact JSON and record ID spans, excluding their quotation marks."""
    if set(target) != {'characters', 'rejected_appearances', 'bubble_speakers'}:
        raise ValueError('Expected the three fields in our joint target schema.')
    text = ''
    spans = []
    field_spans = []

    def fixed(value):
        nonlocal text
        text += value

    def decision(value, pattern):
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f'Invalid decision ID: {value}')
        fixed('"')
        start = len(text)
        fixed(value)
        spans.append((start, len(text)))
        fixed('"')

    def field(name):
        fixed('"')
        start = len(text)
        fixed(name)
        field_spans.append((start, len(text)))
        fixed('"')

    fixed('{')
    field('characters')
    fixed(':{')
    for index, (person, appearances) in enumerate(target['characters'].items()):
        if index:
            fixed(',')
        decision(person, r'P[1-9]\d*')
        fixed(':[')
        for position, appearance in enumerate(appearances):
            if position:
                fixed(',')
            decision(appearance, r'A[1-9]\d*')
        fixed(']')
    fixed('},')
    field('rejected_appearances')
    fixed(':[')
    for index, appearance in enumerate(target['rejected_appearances']):
        if index:
            fixed(',')
        decision(appearance, r'A[1-9]\d*')
    fixed('],')
    field('bubble_speakers')
    fixed(':{')
    for index, (bubble, speaker) in enumerate(target['bubble_speakers'].items()):
        if index:
            fixed(',')
        decision(bubble, r'B[1-9]\d*')
        fixed(':')
        decision(speaker, r'P[1-9]\d*|UNKNOWN')
        if speaker != 'UNKNOWN' and speaker not in target['characters']:
            raise ValueError(f'{bubble} refers to missing identity {speaker}.')
    fixed('}}')
    if json.loads(text) != target:
        raise ValueError('Target serialization changed the labels.')
    if include_field_spans:
        return text, spans, field_spans
    return text, spans


def build_masked_target(tokenizer, target, separator='', loss_config=None):
    """Make labels and positional weights for decisions, structure and response end."""
    if not tokenizer.is_fast:
        raise ValueError('Use the local fast tokenizer for character offsets.')
    config = loss_config or {}
    decision_weight = config.get('decision_weight', 1.0)
    structure_weight = config.get('formatting_weight', 0.1)
    end_weight = config.get('end_weight', structure_weight)
    field_weight = config.get('field_weight', 0.0)
    if not all(math.isfinite(w) for w in [decision_weight, structure_weight, end_weight, field_weight]):
        raise ValueError('Loss weights must be finite.')
    if decision_weight <= 0 or structure_weight < 0 or end_weight < 0 or field_weight < 0:
        raise ValueError('Decision weight must be positive; other weights must be nonnegative.')
    text, spans, field_spans = build_assistant_with_spans(target, include_field_spans=True)
    encoded = tokenizer(text + separator, add_special_tokens=False,
                        return_offsets_mapping=True)
    labels = []
    weights = []
    separator_ids = tokenizer(separator, add_special_tokens=False)['input_ids'] if separator else []
    end_start = len(encoded['input_ids']) - len(separator_ids)
    if separator_ids and encoded['input_ids'][end_start:] != separator_ids:
        raise ValueError('Chat separator token boundary is not aligned.')
    for index, (token_id, (start, end)) in enumerate(zip(encoded['input_ids'], encoded['offset_mapping'])):
        is_decision = any(start < right and end > left for left, right in spans)
        is_field = any(start < right and end > left for left, right in field_spans)
        if separator_ids and index >= end_start:
            weight = end_weight
        elif is_decision:
            weight = decision_weight
        elif is_field:
            weight = field_weight
        elif end > start:
            weight = structure_weight
        else:
            weight = 0.0
        weights.append(weight)
        labels.append(token_id if weight > 0 else -100)
    if not spans:
        raise ValueError('This sample has no supervised decision tokens.')
    return {'text': text, 'spans': spans, 'input_ids': encoded['input_ids'],
            'field_spans': field_spans, 'offsets': encoded['offset_mapping'],
            'labels': labels, 'token_weights': weights}


def collate_weighted_tokens(samples, model, tokenizer, offset_tokenizer, config):
    """Batch size one, matching the existing single-GPU InternVL helpers."""
    import torch
    import lora_helpers as H

    if len(samples) != 1:
        raise ValueError('Use batch_size=1; use accumulation for larger effective batches.')
    sample = samples[0]
    # Reuse our existing image tiling and the downloaded model's chat template.
    pixels = H.tiles(sample, config['max_tiles'], view='annotated')
    template = H.conversation(model)
    template.append_message(template.roles[0], sample['prompt'])
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt()
    if prompt.count('<image>') != 1:
        raise ValueError('Expected exactly one image placeholder in the prompt.')
    image_tokens = '<img>' + '<IMG_CONTEXT>' * (model.num_image_token * len(pixels)) + '</img>'
    prompt = prompt.replace('<image>', image_tokens, 1)
    prefix_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    target = build_masked_target(offset_tokenizer, sample['target'], template.sep, config.get('loss'))
    slow_answer_ids = tokenizer(target['text'] + template.sep, add_special_tokens=False)['input_ids']
    if slow_answer_ids != target['input_ids']:
        raise ValueError('Fast and model tokenizer IDs differ. Do not use a misaligned loss mask.')
    ids = prefix_ids + target['input_ids']
    if len(ids) > config['max_length']:
        raise ValueError(f'{len(ids)} tokens exceed max_length. Reduce tiles or shorten input; no truncation is applied.')
    device = next(model.language_model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    if int((input_ids == model.img_context_token_id).sum()) != model.num_image_token * len(pixels):
        raise ValueError('Image-token count mismatch.')
    # All tokens remain in input_ids as context. Only labels change.
    labels = [-100] * len(prefix_ids) + target['labels']
    weights = [0.0] * len(prefix_ids) + target['token_weights']
    return {
        'input_ids': input_ids,
        'labels': torch.tensor([labels], dtype=torch.long, device=device),
        'token_weights': torch.tensor([weights], dtype=torch.float32, device=device),
        'answer_start': len(prefix_ids),
        'pixel_values': pixels.to(device=device, dtype=next(model.vision_model.parameters()).dtype),
    }


def weighted_cross_entropy(logits, targets, token_weights):
    """Use one weight per target position, not one weight per vocabulary class."""
    import torch
    import torch.nn.functional as F
    if logits.shape[:-1] != targets.shape or token_weights.shape != targets.shape:
        raise ValueError('Logits, targets and token weights are not aligned.')
    if not torch.isfinite(token_weights).all() or (token_weights < 0).any():
        raise ValueError('Token weights must be finite and nonnegative.')
    weights = token_weights.reshape(-1).float() * (targets.reshape(-1) != -100)
    if weights.sum() <= 0:
        raise ValueError('The batch has no positive loss weights.')
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             targets.reshape(-1), reduction='none', ignore_index=-100)
    return (losses * weights).sum() / weights.sum()


def weighted_supervised_loss(model, batch):
    """Same InternVL forward/causal shift as H.supervised_loss; weighted reduction."""
    from contextlib import nullcontext
    import torch
    import lora_helpers as H

    llm = H.base_llm(model)
    visual_trainable = any(p.requires_grad for p in model.vision_model.parameters()) or any(
        p.requires_grad for p in model.mlp1.parameters())
    with nullcontext() if visual_trainable else torch.no_grad():
        visual = model.extract_feature(batch['pixel_values']).reshape(-1, llm.config.hidden_size)
    inputs = llm.get_input_embeddings()(batch['input_ids']).clone()
    selected = batch['input_ids'] == model.img_context_token_id
    if int(selected.sum()) != len(visual):
        raise ValueError('Visual embeddings and image tokens differ.')
    inputs[selected] = visual.to(inputs.dtype)
    hidden = llm.model(inputs_embeds=inputs, attention_mask=torch.ones_like(batch['input_ids']),
                       use_cache=False, return_dict=True).last_hidden_state
    start = batch['answer_start']
    # Hidden state t-1 predicts target t. Weights follow the targets, not the inputs.
    logits = llm.get_output_embeddings()(hidden[:, start - 1:-1, :])
    targets = batch['labels'][:, start:]
    weights = batch['token_weights'][:, start:]
    return weighted_cross_entropy(logits, targets, weights)
