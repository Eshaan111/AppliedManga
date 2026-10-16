from pathlib import Path
import json
import shutil

folder = Path(r'C:\Users\ESHAAN\BYTE\Applied-Task\Magi\PaddleOCRvL')
path = folder / 'test_page_ocr.ipynb'
backup = folder / 'test_page_ocr.before_speed_fix.ipynb'
if not backup.exists():
    shutil.copy2(path, backup)
doc = json.loads(path.read_text(encoding='utf-8'))
for cell in doc['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    if 'RUN_FULL_PAGE = True' in source:
        source = source.replace('RUN_FULL_PAGE = True', 'RUN_FULL_PAGE = False  # start with three small text crops')
        source = source.replace('MAX_NEW_TOKENS_PAGE = 2048', 'MAX_NEW_TOKENS_PAGE = 512')
        source = source.replace('MAX_NEW_TOKENS_CROP = 256', '''MAX_NEW_TOKENS_CROP = 128
MAX_CROPS = 3  # set None after the smoke test to process all cached boxes
MAX_PIXELS = 401408  # processor resolution cap; increase later if small text suffers
MAX_TIME_SECONDS = 60  # soft generation budget; cannot interrupt a slow first forward pass
STREAM_TEXT = True''')
    if 'attn_implementation=\'eager\'' in source:
        source = source.replace("AutoModelForCausalLM, AutoProcessor", "AutoModelForCausalLM, AutoProcessor")
        source = source.replace("attn_implementation='eager'", "attn_implementation='sdpa'")
        source += '''
processor.image_processor.max_pixels = MAX_PIXELS
print('Processor max_pixels:', processor.image_processor.max_pixels)
if DEVICE == 'cuda':
    free, total = torch.cuda.mem_get_info()
    print(f'GPU free after loading: {free / 2**30:.2f} / {total / 2**30:.2f} GiB')
'''
    if 'def recognize(' in source:
        source = source.replace('@torch.inference_mode()', 'from transformers import TextStreamer\n\n@torch.inference_mode()', 1)
        source = source.replace("    inputs = processor.apply_chat_template", "    print(f'Preparing OCR input: {image.size}', flush=True)\n    inputs = processor.apply_chat_template", 1)
        source = source.replace('    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)', '''    print(f"Generating: {inputs['input_ids'].shape[1]} input tokens, at most {max_new_tokens} new tokens", flush=True)
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True) if STREAM_TEXT else None
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            use_cache=True, max_time=MAX_TIME_SECONDS, streamer=streamer)''')
        source = source.replace("'hit_token_limit': generated.shape[1] >= max_new_tokens}", "'hit_token_limit': generated.shape[1] >= max_new_tokens,\n            'hit_time_budget': elapsed >= MAX_TIME_SECONDS}")
        source = source.replace("'device': DEVICE, 'dtype': str(DTYPE)}", "'device': DEVICE, 'dtype': str(DTYPE), 'max_pixels': MAX_PIXELS,\n        'max_time_seconds': MAX_TIME_SECONDS, 'attention': 'sdpa'}")
    if "if boxes:\n    overlay" in source:
        source = source.replace("if boxes:\n    overlay", "if MAX_CROPS is not None:\n    boxes, old_ocr = boxes[:MAX_CROPS], old_ocr[:MAX_CROPS]\n\nif boxes:\n    overlay")
    if 'for index, (bbox, previous)' in source:
        source = source.replace('    crop = page.crop(bounds)', "    crop = page.crop(bounds)\n    print(f'Crop {index+1}/{len(boxes)}', flush=True)")
        source = source.replace("    print('Seconds:', round(result['seconds'], 2), '| token limit reached:', result['hit_token_limit'])", "    print('Seconds:', round(result['seconds'], 2), '| token limit reached:', result['hit_token_limit'], '| time budget reached:', result.get('hit_time_budget', False))")
    compile(source, f'{path.name}:{cell["id"]}', 'exec')
    cell['source'] = source.splitlines(True)
    cell['execution_count'] = None
    cell['outputs'] = []
doc['cells'].insert(1, {'cell_type': 'markdown', 'id': 'speed-settings-note', 'metadata': {}, 'source': [
    'Start with three cached text crops on the 8 GB GPU. Full-page OCR is disabled by default. SDPA attention and a processor pixel cap reduce memory use. Interrupt any old run, restart the kernel, and reload this notebook before rerunning. Set MAX_CROPS=None for all boxes after the first test. The 60-second generation limit is soft; the first image forward pass can exceed it.\n']})
path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding='utf-8')
readme = folder / 'README.md'
text = readme.read_text(encoding='utf-8')
text += '''

## Settings for the 8 GB RTX 4060

The original full-page run produced a CUDA out-of-memory error. Updated defaults skip full-page
generation and test only three cached text crops. Attention uses SDPA (supported by the installed
model code), max_pixels=401408, crop generation is capped at 128 tokens, and max_time=60 seconds
is a soft decoding budget. It cannot interrupt an expensive first forward pass. Text streams while
decoding and the notebook prints each crop and input token count before generation.

Interrupt an existing run, restart the kernel, and reload the updated notebook before running it.
Set MAX_CROPS=None for every box after the three-crop smoke test. Increase MAX_PIXELS or token
budgets if outputs lose small characters or are truncated. Existing other GPU workloads may still
cause memory pressure. The original notebook is saved as test_page_ocr.before_speed_fix.ipynb.
The updated settings were syntax/schema checked; no new model instance was loaded while your
GPU was busy, so an actual inference speed improvement has not yet been measured.
'''
readme.write_text(text, encoding='utf-8')
print('Updated:', path)
print('Backup:', backup)
