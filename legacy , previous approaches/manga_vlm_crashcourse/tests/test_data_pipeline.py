from __future__ import annotations

import json
import re
import unittest
from types import SimpleNamespace

import torch

from PIL import Image

from Scripts import dataUtils, trainValSplit
from Scripts.trainer import build_training_args
from Scripts.evaluator import evaluate_model, load_prediction_records
from tests.common import (
    IMAGE_ROOT,
    LABELS,
    ROOT,
    model_path,
    read_jsonl,
    task_prompt_from_notebook,
)


CHARACTER_NAME = re.compile(r"character[1-9]\d*$")
SPECIAL_SPEAKERS = {"NARRATION", "UNKNOWN"}


class CanonicalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(LABELS)
        cls.by_id = {row["sequence_id"]: row for row in cls.rows}
        cls.dataset = dataUtils.MangaSequenceDataset(LABELS, IMAGE_ROOT)

    def assert_pages_schema(self, pages, context):
        self.assertIsInstance(pages, list, context)
        self.assertEqual(len(pages), 3, context)
        for page_index, page in enumerate(pages):
            self.assertIsInstance(page, list, f"{context}, page {page_index}")
            for dialogue_index, dialogue in enumerate(page):
                location = f"{context}, page {page_index}, dialogue {dialogue_index}"
                self.assertEqual(set(dialogue), {"speaker", "text"}, location)
                self.assertIsInstance(dialogue["speaker"], str, location)
                self.assertIsInstance(dialogue["text"], str, location)
                self.assertTrue(dialogue["text"].strip(), location)
                speaker = dialogue["speaker"]
                self.assertTrue(
                    speaker in SPECIAL_SPEAKERS or CHARACTER_NAME.fullmatch(speaker),
                    f"Unexpected speaker {speaker!r} at {location}",
                )

    def test_every_label_is_unique_and_has_valid_schema(self):
        self.assertGreater(len(self.rows), 0)
        self.assertEqual(len(self.by_id), len(self.rows))
        for row in self.rows:
            self.assertEqual(set(row), {"sequence_id", "pages"})
            self.assert_pages_schema(row["pages"], row["sequence_id"])

    def test_every_label_has_one_matching_ordered_image_directory(self):
        self.assertEqual(len(self.dataset), len(self.rows))
        self.assertEqual(
            set(self.by_id),
            {path.name for path in IMAGE_ROOT.iterdir() if path.is_dir()},
        )
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            sequence_id = sample["sequence_id"]
            self.assertEqual(sample["target"], self.by_id[sequence_id]["pages"])
            self.assertEqual(
                [path.name for path in sample["image_paths"]],
                ["01.png", "02.png", "03.png"],
            )
            for path in sample["image_paths"]:
                self.assertEqual(path.parent.name, sequence_id)
                with Image.open(path) as image:
                    image.verify()

    def test_plain_train_validation_split_is_disjoint_and_complete(self):
        sequence_ids = list(self.by_id)
        train_ids, validation_ids = trainValSplit.fit_seq(
            sequence_ids, v_size=0.10, random_state=42
        )
        self.assertFalse(set(train_ids) & set(validation_ids))
        self.assertEqual(set(train_ids) | set(validation_ids), set(sequence_ids))
        self.assertEqual(len(train_ids) + len(validation_ids), len(sequence_ids))


class PromptAndCollatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = task_prompt_from_notebook()
        cls.dataset = dataUtils.MangaSequenceDataset(LABELS, IMAGE_ROOT)
        cls.sample = cls.dataset[0]
        cls.prompt_builder = dataUtils.MangaPromptBuilder(train_text=cls.prompt)

    def test_prompt_and_messages_do_not_leak_targets_to_inference(self):
        prompt_upper = self.prompt.upper()
        self.assertIn("JAPANESE READING ORDER", prompt_upper)
        self.assertIn("VALID JSON", prompt_upper)
        self.assertIn("NARRATION", prompt_upper)
        self.assertIn("UNKNOWN", prompt_upper)

        user_message, assistant_message = self.prompt_builder.build_training_messages(
            self.sample
        )
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(
            [item["type"] for item in user_message["content"]],
            ["image", "image", "image", "text"],
        )
        serialized_target = assistant_message["content"][0]["text"]
        self.assertEqual(json.loads(serialized_target), self.sample["target"])
        inference_messages = self.prompt_builder.build_inference_messages(self.sample)
        self.assertNotIn(
            serialized_target, json.dumps(inference_messages, ensure_ascii=False)
        )

    def test_real_processor_builds_correct_training_label_mask(self):
        local_model_path = model_path()
        if not local_model_path.is_dir():
            self.skipTest(f"Local model processor not found: {local_model_path}")
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
        except ImportError as error:
            self.skipTest(f"VLM processor dependencies are unavailable: {error}")

        processor = AutoProcessor.from_pretrained(
            local_model_path,
            local_files_only=True,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
        training_collator = dataUtils.MangaCollator(
            processor, process_vision_info, self.prompt_builder, train_mode=True
        )
        inference_collator = dataUtils.MangaCollator(
            processor,
            process_vision_info,
            self.prompt_builder,
            train_mode=False,
            return_metadata=True,
        )
        training_inputs = training_collator([self.sample])
        labels = training_inputs["labels"][0]
        active_positions = (labels != -100).nonzero(as_tuple=True)[0]
        self.assertGreater(active_positions.numel(), 0)
        self.assertTrue((labels[: int(active_positions[0])] == -100).all().item())
        decoded_labels = processor.decode(
            labels[active_positions].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.assertIn(
            json.dumps(self.sample["target"], ensure_ascii=False), decoded_labels
        )
        inference_inputs, metadata = inference_collator([self.sample])
        self.assertNotIn("labels", inference_inputs)
        self.assertEqual(metadata[0]["sequence_id"], self.sample["sequence_id"])
        self.assertEqual(metadata[0]["target"], self.sample["target"])
        self.assertEqual(len(metadata[0]["image_paths"]), 3)
        audit = metadata[0]["image_input_diagnostics"]
        self.assertTrue(audit["verified"])
        self.assertEqual(audit["image_count"], 3)
        self.assertEqual(len(audit["images"]), 3)
        self.assertEqual(inference_inputs["image_grid_thw"].shape[0], 3)
        self.assertGreater(audit["total_image_tokens"], 0)

        # Compare each source image processed on its own to the corresponding
        # slice of the batched model pixels. Counts alone cannot prove order
        # or catch a duplicated image replacing one of the three pages.
        images, _ = process_vision_info(self.prompt_builder.build_inference_messages(self.sample))
        offset = 0
        for index, image in enumerate(images):
            individual = processor.image_processor(images=[image], return_tensors="pt")
            torch.testing.assert_close(individual["image_grid_thw"][0], inference_inputs["image_grid_thw"][index])
            patches = individual["pixel_values"].shape[0]
            torch.testing.assert_close(individual["pixel_values"], inference_inputs["pixel_values"][offset:offset + patches])
            offset += patches
        self.assertEqual(offset, inference_inputs["pixel_values"].shape[0])

        # Two sequences must keep all six grids and their own image-token
        # counts, including padding of unequal-length prompts.
        batched_inputs, batched_metadata = inference_collator([self.sample, self.dataset[1]])
        self.assertEqual(batched_inputs["image_grid_thw"].shape[0], 6)
        self.assertEqual([item["image_input_diagnostics"]["image_count"] for item in batched_metadata], [3, 3])


class ImageInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2), image_token_id=99)
        self.sample = {"sequence_id": "three_pages", "image_paths": ["01.png", "02.png", "03.png"]}
        self.inputs = {"image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4], [1, 4, 4]]),
                       "pixel_values": torch.zeros(28, 12),
                       "input_ids": torch.tensor([[99] * 7 + [1]]),
                       "attention_mask": torch.ones(1, 8, dtype=torch.long)}

    def test_all_three_images_are_audited_in_source_order(self):
        audit = dataUtils.verify_image_inputs(self.processor, self.inputs, [self.sample])[0]
        self.assertEqual([item["path"] for item in audit["images"]], self.sample["image_paths"])
        self.assertEqual([item["image_tokens"] for item in audit["images"]], [1, 2, 4])
        self.assertEqual(audit["pixel_patches"], 28)

    def test_dropped_grids_pixels_or_prompt_image_tokens_fail_before_generation(self):
        cases = {
            "missing image pixels": {key: value for key, value in self.inputs.items() if key != "pixel_values"},
            "only two images": {**self.inputs, "image_grid_thw": self.inputs["image_grid_thw"][:2]},
            "missing pixel patches": {**self.inputs, "pixel_values": self.inputs["pixel_values"][:12]},
            "missing image tokens": {**self.inputs, "input_ids": torch.tensor([[99] * 3 + [1] * 5])},
        }
        for reason, inputs in cases.items():
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                dataUtils.verify_image_inputs(self.processor, inputs, [self.sample])

    def test_vision_preprocessing_cannot_silently_drop_a_page(self):
        processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")
        collator = dataUtils.MangaCollator(processor, lambda messages: ([object(), object()], None),
                                          dataUtils.MangaPromptBuilder(dataUtils.DEFAULT_TASK_PROMPT), train_mode=False)
        with self.assertRaisesRegex(ValueError, "must return 3 images, got 2"):
            collator([self.sample])

    def test_generation_receives_image_tensors_and_saves_the_audit(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.generation_config = SimpleNamespace(eos_token_id=2)
                self.received = None

            def generate(self, **kwargs):
                self.received = kwargs
                return torch.cat([kwargs["input_ids"], torch.tensor([[2]])], dim=1)

        processor = SimpleNamespace(image_processor=self.processor.image_processor,
                                    image_token_id=99,
                                    tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0),
                                    batch_decode=lambda *args, **kwargs: ["[[], [], []]"])
        sample = {**self.sample, "target": [[], [], []]}
        model = RecordingModel()
        collator = lambda batch: (self.inputs, [sample])
        directory = ROOT / "tests" / "_image_input_test_outputs"
        directory.mkdir(exist_ok=True)
        try:
            evaluate_model(model, processor, [sample], collator, output_dir=directory)
            self.assertIs(model.received["pixel_values"], self.inputs["pixel_values"])
            self.assertEqual(model.received["image_grid_thw"].shape[0], 3)
            saved = load_prediction_records(f"{directory}/predictions.jsonl")[0]
            self.assertEqual(saved["image_input_diagnostics"]["image_count"], 3)
            self.assertEqual(saved["image_paths"], self.sample["image_paths"])
        finally:
            for name in ("predictions.jsonl", "metrics.json", "per_sequence_results.jsonl"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()
        model.received = None
        self.inputs = {**self.inputs, "image_grid_thw": self.inputs["image_grid_thw"][:2]}
        with self.assertRaisesRegex(ValueError, "Expected 3 image grid rows"):
            evaluate_model(model, processor, [sample], collator)
        self.assertIsNone(model.received)


class TrainingConfigurationTests(unittest.TestCase):
    def test_training_arguments_enable_validation_clipping_and_safe_loading(self):
        arguments = build_training_args(
            output_dir="test-output-not-created",
            batch_size=1,
            eval_batch_size=1,
            gradient_accumulation_steps=10,
            eval_strategy="epoch",
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            warmup_steps=7,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
        )
        evaluation_strategy = getattr(
            arguments,
            "eval_strategy",
            getattr(arguments, "evaluation_strategy", None),
        )
        self.assertEqual(str(evaluation_strategy).lower().split(".")[-1], "epoch")
        self.assertEqual(arguments.max_grad_norm, 1.0)
        self.assertEqual(arguments.gradient_accumulation_steps, 10)
        self.assertEqual(arguments.dataloader_num_workers, 0)
        self.assertTrue(arguments.prediction_loss_only)
        self.assertFalse(arguments.remove_unused_columns)
        self.assertEqual(arguments.warmup_steps, 7)
        self.assertEqual(str(arguments.lr_scheduler_type).lower().split(".")[-1], "cosine")
        self.assertTrue(arguments.load_best_model_at_end)
        self.assertEqual(arguments.metric_for_best_model, "eval_loss")

if __name__ == "__main__":
    unittest.main()
