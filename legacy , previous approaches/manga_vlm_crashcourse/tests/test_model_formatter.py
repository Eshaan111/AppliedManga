import json
import unittest
import shutil
import uuid
import gc
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from Scripts.modelFormatter import (
    FormatterPromptBuilder, pages_to_transcript, transcript_to_pages,
    validate_formatter_output, ModelJSONFormatter, format_transcript,
    format_saved_transcripts,
)


@contextmanager
def test_directory():
    # Use inherited workspace permissions (TemporaryDirectory uses mode 0o700).
    root = Path(__file__).resolve().parent
    path = root / ('formatter_test_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != root:
            raise RuntimeError('Test cleanup must stay inside the tests folder')
        shutil.rmtree(path)


class FormatterTests(unittest.TestCase):
    def test_tolerant_validation_preserves_dialogue_whitespace_and_escapes(self):
        transcript = '\r\n  PAGE 1 \r\ncharacter1\t  Wait! \\n\\t\\\\  \r\n\r\n PAGE 2\r\n PAGE 3  \r\n'
        pages = [[{'speaker': 'character1', 'text': '  Wait! \n\t\\  '}], [], []]
        raw = json.dumps(pages)
        self.assertEqual(validate_formatter_output(transcript, raw), raw)
        self.assertEqual(json.loads(format_transcript(transcript, tolerant=True)), pages)
        with self.assertRaises(ValueError):
            transcript_to_pages(transcript)
        for changed in ([[], [], []],
                        [[{'speaker': 'character1', 'text': 'Wait!'}], [], []],
                        [pages[0], [], [{'speaker': 'UNKNOWN', 'text': 'invented'}]]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_formatter_output(transcript, json.dumps(changed))

    def test_model_runs_before_transcript_validation_and_retains_failed_response(self):
        pages = [[{'speaker': 'character1', 'text': 'Wait!'}], [], []]
        raw = json.dumps(pages)
        seen = []

        class Inputs(dict):
            def to(self, device):
                return self

        def template(messages, **kwargs):
            seen.append(messages)
            return 'prompt'

        formatter = ModelJSONFormatter.__new__(ModelJSONFormatter)
        formatter.prompt = 'format'
        formatter.max_new_tokens = 20
        formatter.last_raw_output = None
        formatter.tokenizer = SimpleNamespace(
            apply_chat_template=template, eos_token_id=2,
            decode=lambda *args, **kwargs: raw,
        )
        # Special methods are resolved on the class, so use a callable mock.
        from unittest.mock import Mock
        tokenizer = Mock(return_value=Inputs(input_ids=torch.tensor([[1]])))
        tokenizer.apply_chat_template = formatter.tokenizer.apply_chat_template
        tokenizer.eos_token_id = 2
        tokenizer.decode = formatter.tokenizer.decode
        formatter.tokenizer = tokenizer
        generate = Mock(return_value=torch.tensor([[1, 2]]))
        formatter.model = SimpleNamespace(device='cpu', generate=generate)
        transcript = ' PAGE 1\ncharacter1\tWait!\n\n PAGE 2\nPAGE 3'
        self.assertEqual(formatter(transcript), raw)
        self.assertEqual(seen[-1][1]['content'], transcript)
        with self.assertRaisesRegex(ValueError, 'exactly three page headers'):
            formatter('PAGE 1\ncharacter1\tWait!')
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(formatter.last_raw_output, raw)

    def test_notebooks_release_vlm_and_trainer_before_formatter_load(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'notebooks-ModelFormatter').glob('*.ipynb'):
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding='utf-8'))
                source = next(''.join(cell['source']) for cell in notebook['cells']
                              if cell['id'] == 'format-saved-transcripts')
                class Model:
                    pass
                namespace = dict(gc=gc, torch=SimpleNamespace(
                    cuda=SimpleNamespace(is_available=lambda: False)),
                    TRANSCRIPT_PATH=Path('saved.jsonl'), validation_ids=['sample'],
                    FORMATTER_MODEL_PATH=Path('formatter'), FORMATTER_PROMPT='prompt',
                    FORMATTER_DEVICE='cpu', FORMATTER_MAX_NEW_TOKENS=20,
                    MINIMUM_SIMILARITY=0.5, OUTPUT_DIR=Path('output'), EVAL_DIR=Path('output'),
                    load_prediction_records=lambda path: [{'sequence_id': 'sample'}],
                    format_saved_transcripts=lambda *args, **kwargs: {'ok': True})
                namespace['model'] = Model()
                namespace['base_model'] = namespace['model']
                namespace['trainer'] = SimpleNamespace(model=namespace['model'])
                namespace['processor'] = Model()
                namespace['eval_collator'] = SimpleNamespace(processor=namespace['processor'])
                model_ref = weakref.ref(namespace['model'])
                processor_ref = weakref.ref(namespace['processor'])

                def load_formatter(*args, **kwargs):
                    self.assertIsNone(model_ref())
                    self.assertIsNone(processor_ref())
                    self.assertNotIn('trainer', namespace)
                    return Model()

                namespace['ModelJSONFormatter'] = load_formatter
                exec(compile(source, str(path), 'exec'), namespace)
                self.assertNotIn('formatter', namespace)

    def test_saved_transcripts_format_without_vlm_and_keep_failures(self):
        pages = [[{'speaker': 'character1', 'text': 'Wait!'}], [], []]
        transcript = pages_to_transcript(pages)
        diagnostic = {'generated_tokens': 7, 'reached_eos': True,
                      'hit_max_new_tokens': False, 'suspected_repetition_loop': False}
        rows = [dict(sequence_id=name, target=pages, prediction=transcript,
                     image_paths=['1', '2', '3'], **diagnostic)
                for name in ('ok', 'failure')]
        formatter = SimpleNamespace(last_raw_output=None)
        seen = []

        def format_text(raw):
            seen.append(raw)
            formatter.last_raw_output = json.dumps(pages) if len(seen) == 1 else 'bad JSON'
            if len(seen) == 2:
                raise ValueError('Invalid JSON')
            return formatter.last_raw_output

        class Formatter:
            @property
            def last_raw_output(self):
                return formatter.last_raw_output

            def __call__(self, raw):
                return format_text(raw)

        with test_directory() as folder:
            root = Path(folder)
            source = root / 'transcripts.jsonl'
            original = ''.join(json.dumps(row) + '\n' for row in rows)
            source.write_text(original, encoding='utf-8')
            metrics = format_saved_transcripts(source, Formatter(), root / 'formatted')
            saved = [json.loads(line) for line in
                     (root / 'formatted/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(source.read_text(encoding='utf-8'), original)
            self.assertEqual(seen, [transcript, transcript])
            self.assertEqual(metrics['schema_compliance_rate'], 0.5)
            self.assertEqual(saved[0]['raw_transcript'], transcript)
            self.assertEqual(saved[0]['prediction'], json.dumps(pages))
            self.assertEqual(saved[1]['prediction'], '')
            self.assertEqual(saved[1]['formatter_error'], 'Invalid JSON')
            self.assertEqual(saved[1]['raw_formatter_prediction'], 'bad JSON')
            for record in saved:
                self.assertEqual(record['target'], pages)
                for key, value in diagnostic.items():
                    self.assertEqual(record[key], value)
            self.assertTrue((root / 'formatted/metrics.json').is_file())
            self.assertTrue((root / 'formatted/per_sequence_results.jsonl').is_file())

    def test_formatter_must_preserve_content_identity_and_page_order(self):
        pages = [[{'speaker': 'character1', 'text': 'Wait!'}], [],
                 [{'speaker': 'character2', 'text': '?!'}]]
        transcript = pages_to_transcript(pages)
        raw = json.dumps(pages, indent=2)
        self.assertEqual(validate_formatter_output(transcript, raw), raw)
        for changed in (
            [[{'speaker': 'character1', 'text': 'Please wait!'}], [], pages[2]],
            [[{'speaker': 'character2', 'text': 'Wait!'}], [], pages[2]],
            [[], pages[0], pages[2]],
            [pages[2], [], pages[0]],
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_formatter_output(transcript, json.dumps(changed))

    def test_evaluation_formats_raw_transcripts_and_keeps_failures(self):
        from Scripts.evaluator import evaluate_model

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.generation_config = SimpleNamespace(eos_token_id=2)

            def generate(self, **kwargs):
                return torch.tensor([[1, 2]])

        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0),
            batch_decode=lambda *args, **kwargs: ['PAGE 1\nPAGE 2\nPAGE 3'],
        )
        sample = {'sequence_id': 'sample', 'target': [[], [], []], 'image_paths': ['1', '2', '3']}
        collator = lambda batch: ({'input_ids': torch.tensor([[1]])}, [sample])
        with patch('Scripts.evaluator.verify_image_inputs', return_value=[{'verified': True}]):
            with test_directory() as folder:
                with patch('Scripts.evaluator.calculate_metrics') as score:
                    result = evaluate_model(Model(), processor, [sample], collator,
                                            output_dir=folder, generation_only=True)
                    self.assertIsNone(result)
                    score.assert_not_called()
                saved = json.loads((Path(folder) / 'predictions.jsonl').read_text(encoding='utf-8'))
                self.assertEqual(saved['prediction'], 'PAGE 1\nPAGE 2\nPAGE 3')
                self.assertEqual(saved['generated_tokens'], 1)
                self.assertNotIn('raw_formatter_prediction', saved)
            metrics = evaluate_model(Model(), processor, [sample], collator,
                                     prediction_formatter=lambda raw: json.dumps(transcript_to_pages(raw)))
            self.assertEqual(metrics['sequence_exact_match'], 1)

            def fail(raw):
                raise ValueError('Malformed formatter output')

            metrics = evaluate_model(Model(), processor, [sample], collator, prediction_formatter=fail)
            self.assertEqual(metrics['schema_compliance_rate'], 0)

    def test_roundtrip_preserves_empty_pages_and_special_text(self):
        pages = [[{'speaker': 'character1', 'text': 'Quote " | \\ \t\r\nPAGE 2 日本語'}], [],
                 [{'speaker': 'NARRATION', 'text': ''}, {'speaker': 'UNKNOWN', 'text': 'Hi'}]]
        self.assertEqual(transcript_to_pages(pages_to_transcript(pages)), pages)

    def test_malformed_transcripts_are_rejected(self):
        for text in ['PAGE 1\nPAGE 3', 'PAGE 1\nPAGE 2',
                     'PAGE 1\nAlice\tHello\nPAGE 2\nPAGE 3',
                     'PAGE 1\ncharacter1\tbad\\x\nPAGE 2\nPAGE 3',
                     'PAGE 1\nPAGE 2\nPAGE 3\nPAGE 1']:
            for tolerant in (False, True):
                with self.subTest(text=text, tolerant=tolerant), self.assertRaises(ValueError):
                    transcript_to_pages(text, tolerant=tolerant)

    def test_all_exported_targets_are_lossless_and_split_is_disjoint(self):
        folder = Path(__file__).resolve().parents[1] / 'notebooks-ModelFormatter/datasets'
        ids = []
        for name, count in [('train', 72), ('validation', 8)]:
            rows = [json.loads(line) for line in (folder / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), count)
            ids.append({row['sequence_id'] for row in rows})
            for row in rows:
                self.assertEqual(transcript_to_pages(row['transcript_target']), row['target'])
                self.assertTrue(all(not Path(p).is_absolute() for p in row['image_paths']))
                sample = dict(row)
                messages = FormatterPromptBuilder('prompt').build_training_messages(sample)
                self.assertEqual(messages[-1]['content'][0]['text'], row['transcript_target'])
                self.assertEqual(len(messages[0]['content']), 4)
        self.assertTrue(ids[0].isdisjoint(ids[1]))


if __name__ == '__main__':
    unittest.main()
