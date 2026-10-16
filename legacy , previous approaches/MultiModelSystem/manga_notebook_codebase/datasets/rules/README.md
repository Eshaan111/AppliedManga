# Local text exclusion hints

`text_exclusion_lexicon.json` stores the inclusion/exclusion policy and editable
whole-region hints. Observed entries came from unmatched validation groups;
additional illustrative sound-effect seeds are explicitly marked separately.
All entries require visual confirmation. This file is not a word blacklist.

`../annotations/ocr_exclusion_review.jsonl` preserves all 153 unmatched groups
with page/group IDs, OCR fragment IDs, boxes, text, advisory hints, and initially
null `keep` and `exclusion_category` fields. Review the original page before
setting `keep` to true or false and recording a reason. Unmatched groups can be
valid dialogue, narration, OCR errors, or split regions, not just excluded text.
Do not treat null labels or lexicon matches as gold filtering annotations.

Use `src/text_exclusions.py::exclusion_hints(text)` to retrieve hints for future
review or a context-aware classifier. No OCR candidate or inference output is
removed automatically. Whole-region matching ignores case, whitespace, and
terminal punctuation; it does not strip matching words from sentences or use
fuzzy matches. Numbers are only flagged as possible page/object text. Spoken
vocalisations and punctuation-only speech/thought balloons must be preserved.

To build a new inventory from the workspace root:

```powershell
python manga_notebook_codebase/scripts/build_exclusion_review.py --output manga_notebook_codebase/outputs/diagnostics/exclusion_review_new.jsonl
```

The script refuses to overwrite an existing file to protect manual decisions.
The inventory is derived from validation diagnostics: using it to tune filtering
is validation-driven development, not independent evidence of generalization.
Evaluate the final filter on fresh data. Unknown SFX spellings and categories
such as document text and intro labels still require image context; absence of
a lexicon hint does not imply that text should be kept.
