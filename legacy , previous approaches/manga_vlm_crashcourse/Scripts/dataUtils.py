from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
IMAGES_PER_SEQUENCE = 3

DEFAULT_TASK_PROMPT = """
You are transcribing dialogue and narration from a sequence of exactly three manga page images. Treat them as one continuous story in the order provided. Use all three images as context to identify recurring speakers, but place each transcription under the image where its text appears.

Reading and transcription:
- Read each page in Japanese reading order: panels from right to left and top to bottom, and text within each panel from right to left and top to bottom.
- Transcribe every dialogue bubble and narration block once, using one object per bubble or block. Keep separate bubbles as separate objects, even when the same character speaks consecutively.
- Preserve the language, spelling, capitalization, and punctuation shown in the image. Do not translate, paraphrase, summarize, or invent missing text.

Speaker identity:
- Assign character labels by first appearance in the sequence's reading order: "character1", "character2", and so on. Keep the same person's label across all three images; never restart numbering on a new page.
- Use visual appearance, bubble tails, and surrounding story context to associate dialogue with its speaker. Do not create a new character label just because the person's pose, expression, clothing, or camera angle changes.
- Label caption boxes and non-spoken narrative text as "NARRATION".
- Use "UNKNOWN" only for spoken text whose speaker cannot be identified. Do not use character names or other speaker labels.

Output structure:
Return an array of arrays of JSON objects: [[{"speaker": "...", "text": "..."}], [...], [...]].
- The outer array represents this sequence and must contain exactly three inner arrays, one per input image in the supplied order.
- The first inner array contains image 1's dialogue and narration, the second contains image 2's, and the third contains image 3's.
- Each inner array contains objects in that page's reading order. Each object must have exactly two keys, "speaker" and "text", both with string values.
- Use [] for a page with no dialogue or narration, retaining its position in the outer array.
- Return the nested array itself. Do not wrap it in a "pages" or "sequence_id" object, flatten the pages, or add page numbers, bounding boxes, or other keys.

Example of the output format (illustrative text only; do not copy it):
[
  [
    {"speaker": "character1", "text": "WHERE ARE YOU GOING?"},
    {"speaker": "character2", "text": "TO PRACTICE."}
  ],
  [],
  [
    {"speaker": "NARRATION", "text": "LATER THAT DAY..."},
    {"speaker": "character1", "text": "YOU'RE BACK!"},
    {"speaker": "UNKNOWN", "text": "WATCH OUT!"}
  ]
]

Before responding, check that there are exactly three page arrays, text is in reading order, recurring speakers have consistent labels, and every object has only "speaker" and "text". Output valid JSON only, with double-quoted keys and strings, no trailing commas, no Markdown fences, and no explanation.
""".strip()


def load_labels(path):
    """Load one canonical target per sequence from ``labels.jsonl``."""
    label_path = Path(path)
    records = {}
    with label_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"sequence_id", "pages"}:
                raise ValueError(
                    f"{label_path}:{line_number} must contain only "
                    "'sequence_id' and 'pages'"
                )
            sequence_id = row["sequence_id"]
            if not isinstance(sequence_id, str) or not sequence_id:
                raise ValueError(
                    f"{label_path}:{line_number} has an invalid sequence_id"
                )
            if sequence_id in records:
                raise ValueError(f"Duplicate sequence_id: {sequence_id}")
            pages = row["pages"]
            if not isinstance(pages, list):
                raise ValueError(f"{sequence_id} pages must be a list")
            records[sequence_id] = pages
    if not records:
        raise ValueError(f"No label rows found in {label_path}")
    return records


# Compatibility alias for older canonical-data notebooks.
json_load = load_labels


class MangaSequenceDataset(Dataset):
    """Canonical manga sequences backed by labels.jsonl and image folders."""

    def __init__(self, label_jsonl_path, image_root, sequence_ids=None):
        self.image_root = Path(image_root)
        self.label_dict = load_labels(label_jsonl_path)
        self.sequence_ids = (
            list(sequence_ids)
            if sequence_ids is not None
            else list(self.label_dict)
        )
        unknown_ids = set(self.sequence_ids) - set(self.label_dict)
        if unknown_ids:
            raise KeyError(f"Unknown sequence IDs: {sorted(unknown_ids)}")

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index]
        sequence_dir = self.image_root / sequence_id
        if not sequence_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found for {sequence_id}: {sequence_dir}"
            )
        image_paths = sorted(
            (
                path
                for path in sequence_dir.iterdir()
                if path.is_file()
                and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
        target = self.label_dict[sequence_id]
        if len(target) != IMAGES_PER_SEQUENCE:
            raise ValueError(f"{sequence_id} must have exactly {IMAGES_PER_SEQUENCE} target pages")
        if len(image_paths) != len(target):
            raise ValueError(
                f"{sequence_id} has {len(image_paths)} images but "
                f"{len(target)} target pages"
            )
        return {
            "sequence_id": sequence_id,
            "image_paths": image_paths,
            "target": target,
        }


class MangaPromptBuilder:
    def __init__(self, train_text):
        if not isinstance(train_text, str) or not train_text.strip():
            raise ValueError("train_text must be a non-empty string")
        self.train_text = train_text

    def build_training_messages(self, sample):
        user_message = self.build_user_message(sample)
        assistant_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(sample["target"], ensure_ascii=False),
                }
            ],
        }
        return [user_message, assistant_message]

    def build_user_message(self, sample):
        content = [
            {"type": "image", "image": str(image_path)}
            for image_path in sample["image_paths"]
        ]
        content.append({"type": "text", "text": self.train_text})
        return {"role": "user", "content": content}

    def build_inference_messages(self, sample):
        return [self.build_user_message(sample)]


class MangaCollator:
    def __init__(
        self,
        processor,
        vision_info,
        prompt_builder,
        train_mode=True,
        return_metadata=False,
        return_prompts=False,
    ):
        if train_mode and return_metadata:
            raise ValueError("return_metadata is only supported in inference mode")
        self.processor = processor
        self.vision_info = vision_info
        self.prompt_builder = prompt_builder
        self.train = train_mode
        self.return_metadata = return_metadata
        self.return_prompts = return_prompts

    def __call__(self, batch):
        prompts = []
        batch_image_inputs = []
        user_prompt_lengths = []

        for sample in batch:
            if self.train:
                user_message, assistant_message = (
                    self.prompt_builder.build_training_messages(sample)
                )
                messages = [user_message, assistant_message]
            else:
                messages = self.prompt_builder.build_inference_messages(sample)
                user_message = messages[0]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not self.train,
            )
            image_inputs, _ = self.vision_info(messages)
            if image_inputs is None or len(image_inputs) != IMAGES_PER_SEQUENCE:
                raise ValueError(f"{sample['sequence_id']}: vision preprocessing must return {IMAGES_PER_SEQUENCE} images, got {0 if image_inputs is None else len(image_inputs)}")
            if len(sample["image_paths"]) != IMAGES_PER_SEQUENCE:
                raise ValueError(f"{sample['sequence_id']}: expected {IMAGES_PER_SEQUENCE} source images")
            image_token = getattr(self.processor, "image_token", None)
            if image_token is not None and prompt.count(image_token) != IMAGES_PER_SEQUENCE:
                raise ValueError(f"{sample['sequence_id']}: chat template must contain {IMAGES_PER_SEQUENCE} image placeholders")

            if self.train:
                user_prompt = self.processor.apply_chat_template(
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                user_inputs = self.processor(
                    text=[user_prompt],
                    images=image_inputs,
                    return_tensors="pt",
                )
                user_prompt_lengths.append(user_inputs["input_ids"].shape[1])

            prompts.append(prompt)
            batch_image_inputs.append(image_inputs)

        inputs = self.processor(
            text=prompts,
            images=batch_image_inputs,
            padding=True,
            return_tensors="pt",
        )
        image_audits = verify_image_inputs(self.processor, inputs, batch)

        if self.train:
            labels = inputs["input_ids"].clone()
            for row, user_length in enumerate(user_prompt_lengths):
                labels[row, :user_length] = -100
            labels[inputs["attention_mask"] == 0] = -100
            inputs["labels"] = labels

        if self.return_metadata:
            metadata = [
                {
                    "sequence_id": sample["sequence_id"],
                    "image_paths": [str(path) for path in sample["image_paths"]],
                    "target": sample["target"],
                    "image_input_diagnostics": audit,
                }
                for sample, audit in zip(batch, image_audits)
            ]
            return inputs, metadata

        if self.return_prompts:
            return inputs, prompts

        return inputs


def verify_image_inputs(processor, inputs, batch):
    """Verify Qwen's image grids, pixel patches, and image tokens agree.

    Called before training/generation; returns one serializable audit per
    sequence, with grid rows in the same order as its source image paths.
    """
    grids = inputs.get("image_grid_thw")
    pixels = inputs.get("pixel_values")
    if grids is None or pixels is None:
        raise ValueError("Model inputs are missing image_grid_thw or pixel_values")
    expected_images = sum(len(sample["image_paths"]) for sample in batch)
    if grids.ndim != 2 or grids.shape != (expected_images, 3):
        raise ValueError(f"Expected {expected_images} image grid rows, got {tuple(grids.shape)}")
    if not (grids > 0).all().item():
        raise ValueError("Image grids must contain positive dimensions")
    patch_counts = grids.prod(dim=1)
    expected_patches = int(patch_counts.sum().item())
    if pixels.ndim != 2 or pixels.shape[0] != expected_patches:
        raise ValueError(f"Expected {expected_patches} pixel patches, got {tuple(pixels.shape)}")
    merge_length = processor.image_processor.merge_size ** 2
    if (patch_counts % merge_length != 0).any().item():
        raise ValueError("Image grids are incompatible with the processor merge size")
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    if image_token_id is None:
        raise ValueError("Processor has no image token ID")
    audits = []
    offset = 0
    for row, sample in enumerate(batch):
        count = len(sample["image_paths"])
        if count != IMAGES_PER_SEQUENCE:
            raise ValueError(f"{sample['sequence_id']}: expected {IMAGES_PER_SEQUENCE} images, got {count}")
        sample_grids = grids[offset:offset + count]
        image_tokens = (patch_counts[offset:offset + count] // merge_length).tolist()
        actual_tokens = int(((inputs["input_ids"][row] == image_token_id)
                             & inputs["attention_mask"][row].bool()).sum().item())
        if actual_tokens != sum(image_tokens):
            raise ValueError(f"{sample['sequence_id']}: expected {sum(image_tokens)} image tokens, got {actual_tokens}")
        audits.append({
            "verified": True,
            "image_count": count,
            "images": [{"path": str(path), "grid_thw": grid, "image_tokens": int(tokens)}
                       for path, grid, tokens in zip(sample["image_paths"], sample_grids.tolist(), image_tokens)],
            "total_image_tokens": actual_tokens,
            "pixel_patches": int(patch_counts[offset:offset + count].sum().item()),
        })
        offset += count
    return audits
