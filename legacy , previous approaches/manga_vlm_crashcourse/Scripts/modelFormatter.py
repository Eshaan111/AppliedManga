"""Lossless transcript targets and a separate text-model JSON formatter.

The VLM emits PAGE headers and tab-separated speaker/text lines. Only the
assistant serialization changes; canonical JSON stays the scoring reference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from Scripts.dataUtils import MangaPromptBuilder, MangaSequenceDataset


def _escape(text):
    return text.replace('\\', '\\\\').replace('\t', '\\t').replace('\r', '\\r').replace('\n', '\\n')


def _unescape(text):
    result = []
    index = 0
    escapes = {'\\': '\\', 't': '\t', 'r': '\r', 'n': '\n'}
    while index < len(text):
        if text[index] == '\\':
            index += 1
            if index >= len(text) or text[index] not in escapes:
                raise ValueError('Invalid text escape; use \\\\, \\t, \\r, or \\n')
            result.append(escapes[text[index]])
        else:
            result.append(text[index])
        index += 1
    return ''.join(result)


def pages_to_transcript(pages):
    if not isinstance(pages, list) or len(pages) != 3:
        raise ValueError('Expected exactly three page arrays')
    lines = []
    for number, page in enumerate(pages, 1):
        if not isinstance(page, list):
            raise ValueError('Each page must be an array')
        lines.append(f'PAGE {number}')
        for item in page:
            if (not isinstance(item, dict) or set(item) != {'speaker', 'text'}
                    or not isinstance(item['text'], str)
                    or not isinstance(item['speaker'], str)
                    or not re.fullmatch(r'character[1-9]\d*|NARRATION|UNKNOWN', item['speaker'])):
                raise ValueError('Invalid dialogue item')
            lines.append(item['speaker'] + '\t' + _escape(item['text']))
    return '\n'.join(lines)


def transcript_to_pages(transcript, *, tolerant=False):
    """Parse supplied content; tolerant mode ignores blank lines/header whitespace.

    Dialogue text is never stripped, corrected or inferred. Missing or reordered
    pages, invalid speakers and invalid escapes remain errors in both modes.
    """
    if not isinstance(transcript, str):
        raise ValueError('Transcript must be a string')
    pages = []
    for number, line in enumerate(transcript.rstrip('\n').split('\n'), 1):
        if tolerant:
            # CRLF is a line separator, not dialogue content. Literal carriage
            # returns in the transcript must still be escaped as \\r.
            line = line.removesuffix('\r')
            if not line.strip():
                continue
            if re.fullmatch(r'PAGE [1-3]', line.strip()):
                line = line.strip()
        if re.fullmatch(r'PAGE [1-3]', line):
            if line != f'PAGE {len(pages) + 1}':
                raise ValueError(f'Line {number}: page headers must be ordered 1, 2, 3')
            pages.append([])
            continue
        if not pages or '\t' not in line:
            raise ValueError(f'Line {number}: expected a page header or speaker TAB text')
        speaker, text = line.split('\t', 1)
        if not re.fullmatch(r'character[1-9]\d*|NARRATION|UNKNOWN', speaker) or '\t' in text:
            raise ValueError(f'Line {number}: invalid speaker or unescaped tab')
        pages[-1].append({'speaker': speaker, 'text': _unescape(text)})
    if len(pages) != 3:
        raise ValueError('Transcript must contain exactly three page headers')
    return pages


def format_transcript(transcript, *, tolerant=False):
    """Deterministic comparison formatter, separate from the model pipeline."""
    return json.dumps(transcript_to_pages(transcript, tolerant=tolerant), ensure_ascii=False)


def validate_formatter_output(transcript, raw):
    """Reject formatter changes to content, identity, page placement or order."""
    try:
        pages = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f'Formatting model returned invalid JSON: {error}') from error
    pages_to_transcript(pages)
    expected = transcript_to_pages(transcript, tolerant=True)
    if pages != expected:
        raise ValueError('Formatting model changed transcript text, speakers, pages or order')
    return raw


class ModelJSONFormatter:
    """Frozen text-only second stage, loaded after releasing the VLM."""

    def __init__(self, model_path, prompt, device='cpu', max_new_tokens=2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True,
            torch_dtype=torch.float32 if device == 'cpu' else torch.bfloat16,
        ).to(device).eval()
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.last_raw_output = None

    def __call__(self, transcript):
        import torch

        self.last_raw_output = None
        if not isinstance(transcript, str):
            raise ValueError('Transcript must be a string')
        # Do not give reference targets or images to the formatting model.
        messages = [
            {'role': 'system', 'content': self.prompt},
            {'role': 'user', 'content': transcript},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self.last_raw_output = raw
        return validate_formatter_output(transcript, raw)


def format_saved_transcripts(transcript_path, formatter, output_dir, minimum_similarity=0.50):
    """Format saved VLM predictions without needing a VLM, processor or images.

    Targets remain scoring metadata and are never passed to the formatter.
    Preserve the VLM's generation diagnostics and retain formatting failures.
    """
    from Scripts.evaluator import (
        calculate_metrics, load_prediction_records, save_per_sequence_reports,
        _generation_diagnostics,
    )

    transcript_path = Path(transcript_path)
    output_dir = Path(output_dir)
    prediction_path = output_dir / 'predictions.jsonl'
    if transcript_path.resolve() == prediction_path.resolve():
        raise ValueError('Save formatted predictions separately from raw transcripts')
    records = load_prediction_records(transcript_path)
    if not records:
        raise ValueError('No saved VLM transcripts to format')
    output_dir.mkdir(parents=True, exist_ok=True)
    with prediction_path.open('w', encoding='utf-8') as file:
        for index, record in enumerate(records, 1):
            transcript = record['prediction']
            record['raw_transcript'] = transcript
            try:
                record['prediction'] = formatter(transcript)
                record['formatter_error'] = None
            except ValueError as error:
                record['prediction'] = ''
                record['formatter_error'] = str(error)
            record['raw_formatter_prediction'] = getattr(formatter, 'last_raw_output', None)
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
            file.flush()
            print(f'Formatted {index}/{len(records)} sequences')
    metrics = calculate_metrics(records, minimum_similarity=minimum_similarity)
    metrics['generation_diagnostics'] = _generation_diagnostics(records)
    (output_dir / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    save_per_sequence_reports(
        prediction_path, output_dir / 'per_sequence_results.jsonl',
        minimum_similarity=minimum_similarity,
    )
    return metrics


class FormatterSequenceDataset(MangaSequenceDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['transcript_target'] = pages_to_transcript(sample['target'])
        return sample


class FormatterPromptBuilder(MangaPromptBuilder):
    def build_training_messages(self, sample):
        return [self.build_user_message(sample), {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': sample['transcript_target']}],
        }]


def export_transcript_dataset(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for sample in dataset:
            if transcript_to_pages(sample['transcript_target']) != sample['target']:
                raise ValueError(f"Lossy target for {sample['sequence_id']}")
            file.write(json.dumps({
                'sequence_id': sample['sequence_id'],
                'image_paths': [p.relative_to(Path(__file__).resolve().parents[1]).as_posix() for p in sample['image_paths']],
                'target': sample['target'],
                'transcript_target': sample['transcript_target'],
            }, ensure_ascii=False) + '\n')
    return path
