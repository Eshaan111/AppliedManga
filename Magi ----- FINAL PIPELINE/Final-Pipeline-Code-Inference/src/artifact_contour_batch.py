"""Schedule unchanged per-sequence inference stages across the selected batch."""
import ast
import contextlib
import gc
import json
import re
import sys
import traceback
from pathlib import Path

WORKER = Path(__file__).with_name("artifact_contour_sequence.py")
# Every sequence follows this same order. All original Magi detections precede crop patching.
PHASES = [
    ("magi_raw", 12, {"magi_raw"}, ()),
    ("embeddings", 12, {"embeddings"}, ("magi",)),
    ("paddle", 16, {"paddle_ocr"}, ("paddle",)),
    ("artifacts", 20, {"ocr_selection", "artifact_labels"}, ("gemma", "gemma_tokenizer")),
    ("entities_and_bubbles", 32, {"character_cleanup", "entities", "text_entities", "contour_bubbles"}, ("yolo",)),
    ("contour_and_routing", 42, {"artifact_filter", "contour_roles", "contour_speakers"}, ()),
    ("transcript_cleanup", 44, {"gemma_transcript"}, ("gemma", "gemma_tokenizer")),
]


def load_cells(worker):
    # Reuse the plain-Python snapshot; no notebook execution or duplicated prediction rules.
    sections = re.split(r"^# --- Notebook inference cell (\d+) ---\s*$", worker.read_text(encoding="utf-8"), flags=re.MULTILINE)
    cells = {int(sections[i]): compile(sections[i + 1], str(worker), "exec") for i in range(1, len(sections), 2)}
    raw_source = sections[sections.index("12") + 1]
    tree = ast.parse(raw_source)
    original = next(node for node in tree.body if isinstance(node, ast.Try))
    # The first two statements compute/validate original Magi; patching belongs to the next phase.
    original.body = original.body[:2]
    raw_only = compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(worker), "exec")
    return cells, raw_only


def write_status(path, status):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def release_models(models, names):
    for name in names:
        models.pop(name, None)
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    paddle = sys.modules.get("paddle")
    if "paddle" in names and paddle is not None:
        paddle.device.cuda.empty_cache()


def run_stages(configs, status_path, worker=WORKER):
    cells, raw_only = load_cells(worker)
    if not configs:
        raise ValueError("No sequence configurations")
    resource_keys = ["magi_root", "magi_model", "gemma_model", "bubble_model", "character_checkpoint", "paddle_detector", "paddle_recognizer"]
    expected = {key: configs[0].get(key) for key in resource_keys}
    if any({key: config.get(key) for key in resource_keys} != expected for config in configs):
        raise ValueError("Shared batch stages require the same model resources for every sequence")
    models = {}
    status = dict(status="running", phase=None, sequence_id=None, completed_phases=[])
    try:
        for phase, last_cell, allowed, release in PHASES:
            status.update(phase=phase, completed_in_phase=[])
            write_status(status_path, status)
            print(f"Stage: {phase}", flush=True)
            for index, config in enumerate(configs, 1):
                sid = config["sequence_id"]
                status["sequence_id"] = sid
                write_status(status_path, status)
                output = Path(config["sequence_output_dir"])
                output.mkdir(parents=True, exist_ok=True)
                print(f"  [{index}/{len(configs)}] {sid}", flush=True)
                # A fresh namespace/image set resets all sequence entities and local state.
                namespace = dict(__name__="batch_sequence", __file__=str(worker),
                                 BATCH_SETTINGS=config, SHARED_MODELS=models, ALLOWED_STAGES=allowed,
                                 CACHE_STATUS_STREAM=sys.stdout)
                try:
                    with (output / "inference.log").open("a", encoding="utf-8") as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            print(f"\n=== {phase} ===")
                            try:
                                for number, code in sorted(cells.items()):
                                    if number > last_cell:
                                        break
                                    exec(raw_only if phase == "magi_raw" and number == 12 else code, namespace)
                            except Exception:
                                traceback.print_exc()
                                raise
                finally:
                    namespace.clear()
                    del namespace
                status["completed_in_phase"].append(sid)
                write_status(status_path, status)
            status["completed_phases"].append(phase)
            # Keep original Magi alive across raw and embedding phases, then release it.
            release_models(models, release)
        status.update(status="complete", sequence_id=None)
        write_status(status_path, status)
    except Exception as error:
        status.update(status="failed", error=str(error))
        write_status(status_path, status)
        raise
    finally:
        release_models(models, tuple(models))


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    run_stages(job["sequences"], Path(job["status_path"]))
