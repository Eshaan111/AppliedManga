# Manual identity labels

Start the local tool in PowerShell:

```powershell
& C:\Users\ESHAAN\BYTE\Applied-Task\.venvAppliedTask\Scripts\python.exe C:\Users\ESHAAN\BYTE\Applied-Task\1pageTest\manual_labels\server.py
```

Open http://127.0.0.1:8765 in your browser. This server uses only Python's standard library and does not run notebooks or models.

Enter one GT identity per line, such as `Char1 - A1, A4, A7`. Char1 is an alias for character1. Appearance A1 is the existing C1 detection, A2 is C2, etc. Face and body boxes belonging to one character detection share one appearance ID.

Optional false detections go in the rejected field. Click **Save response** (or Ctrl+S). Incomplete responses can be saved; unassigned appearances are listed. Assign an appearance to only one identity or rejection list.

Responses are saved under `manual_labels/responses/`, keyed by sequence and page. Only your response and the IDs/fingerprints needed to associate it with the page are saved. Image overlays and bubble GT labels are read fresh from the existing stage results and are not cached by this tool. The existing VL input image is untouched.

A bubble with no accepted GT match shows **unmatched**; it is not automatically assigned to UNKNOWN. Saved identity groups need to be combined with reviewed bubble speaker labels to create the final training example.

## Bubble speaker corrections

Each bubble row now has a GT speaker dropdown. Choose a character for an unmatched or incorrect bubble, or UNKNOWN for a genuinely unidentifiable visible speaker. The Automatic option removes your override. Save response stores these corrections beside your appearance groups. B8 has been corrected to character3 per your instruction.

After saving, run `6_lora_training/00_make_assistant_targets.ipynb` manually to build the user prompt, joint assistant target and VL input image. The notebook was created without execution.
