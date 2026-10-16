from __future__ import annotations

import logging
import math
import os
import time
import unittest

from Scripts import dataUtils
from Scripts.trainer import build_lora_model
from tests.common import (
    IMAGE_ROOT,
    LABELS,
    model_path,
    task_prompt_from_notebook,
)


RUN_OVERFIT = os.environ.get("RUN_VLM_OVERFIT_TEST") == "1"
LOGGER = logging.getLogger(__name__)
if RUN_OVERFIT:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.setLevel(logging.INFO)


@unittest.skipUnless(
    RUN_OVERFIT,
    "Set RUN_VLM_OVERFIT_TEST=1 to run the real one-sample GPU overfit test",
)
class OneSampleOverfitTest(unittest.TestCase):
    def test_one_real_sample_can_be_overfit(self):
        started_at = time.perf_counter()
        LOGGER.info("Starting one-sample VLM overfit smoke test")

        import torch
        from peft import LoraConfig
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the quantized VLM overfit test")
        local_model_path = model_path()
        if not local_model_path.is_dir():
            self.skipTest(f"Local model not found: {local_model_path}")

        torch.manual_seed(42)
        LOGGER.info("Loading processor from %s", local_model_path)
        processor = AutoProcessor.from_pretrained(
            local_model_path,
            local_files_only=True,
            # Smaller visual token budget keeps this diagnostic reasonably fast.
            min_pixels=64 * 28 * 28,
            max_pixels=128 * 28 * 28,
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        LOGGER.info("Loading 4-bit quantized base model")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0},
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = build_lora_model(base_model, lora_config)
        LOGGER.info("LoRA model initialized")

        LOGGER.info("Loading training sample from %s", LABELS)
        dataset = dataUtils.MangaSequenceDataset(
            label_jsonl_path=LABELS,
            image_root=IMAGE_ROOT,
            sequence_ids=None,
        )
        prompt_builder = dataUtils.MangaPromptBuilder(
            train_text=task_prompt_from_notebook()
        )
        collator = dataUtils.MangaCollator(
            processor=processor,
            vision_info=process_vision_info,
            prompt_builder=prompt_builder,
            train_mode=True,
        )
        batch = collator([dataset[0]])
        device = next(model.parameters()).device
        batch = {
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in batch.items()
        }
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        self.assertTrue(trainable_parameters)
        trainable_parameter_count = sum(
            parameter.numel() for parameter in trainable_parameters
        )
        LOGGER.info(
            "Prepared batch on %s with %d trainable parameters",
            device,
            trainable_parameter_count,
        )
        optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3)

        model.eval()
        with torch.inference_mode():
            initial_loss = float(model(**batch).loss)
        self.assertTrue(math.isfinite(initial_loss))
        LOGGER.info("Initial loss: %.6f", initial_loss)

        losses = []
        model.train()
        for step in range(1, 13):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            step_loss = float(loss.detach())
            losses.append(step_loss)
            LOGGER.info("Training step %d/12, loss: %.6f", step, step_loss)

        model.eval()
        with torch.inference_mode():
            final_loss = float(model(**batch).loss)
        self.assertTrue(math.isfinite(final_loss))
        loss_reduction = (
            1.0 - (final_loss / initial_loss) if initial_loss else float("nan")
        )
        LOGGER.info(
            "Finished overfit smoke test: loss %.6f -> %.6f (%.2f%% reduction) "
            "in %.2fs",
            initial_loss,
            final_loss,
            loss_reduction * 100,
            time.perf_counter() - started_at,
        )
        self.assertLess(
            final_loss,
            initial_loss * 0.90,
            f"Expected at least 10% loss reduction; losses={losses!r}",
        )


if __name__ == "__main__":
    unittest.main()
