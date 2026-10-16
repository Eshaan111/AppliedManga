BASELINE_PROMPT = """
Read these three consecutive English manga pages.

Return exactly one JSON object:

{
  "pages": [
    [{"speaker":"character1","text":"..."}],
    [],
    []
  ]
}

Rules:
- Keep story text in reading order.
- Include dialogue, thoughts, narration, clear vocalisations, and punctuation-only balloons.
- Exclude sound effects, signs, credits, page numbers, ads, notes, and text inside documents/interfaces.
- Use NARRATION only for narrator text.
- Use UNKNOWN only when the speaker cannot be determined from the sequence.
- Keep the same character label across all three pages.
- Character names do not need to be real names.
- Do not include explanations or markdown.
"""
