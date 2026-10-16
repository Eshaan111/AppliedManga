from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "TaskData" / "development" / "labels.jsonl"
IMAGE_ROOT = ROOT / "TaskData" / "development" / "images"
DEFAULT_MODEL_PATH = Path(r"C:\models\HF\Qwen2.5-VL-3B-Instruct")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def task_prompt_from_notebook() -> str:
    """Return the shared prompt used by baseline and training notebooks."""
    from Scripts.dataUtils import DEFAULT_TASK_PROMPT

    return DEFAULT_TASK_PROMPT


def model_path() -> Path:
    return Path(os.environ.get("MANGA_MODEL_PATH", DEFAULT_MODEL_PATH))
