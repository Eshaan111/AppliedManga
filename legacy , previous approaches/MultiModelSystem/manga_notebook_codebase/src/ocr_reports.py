"""Local OCR transcript previews and lookup; no OCR or model execution."""
import html
import json
from pathlib import Path

from dataset_utils import image_paths, save_json
from ocr_joining import draw_ocr_groups


def write_page_report(root, row, sample_index, report, raw_candidates, output):
    from PIL import Image

    output = Path(output)
    page = report["page_index"]
    prefix = f"sample_{sample_index}_page_{page}"
    with Image.open(image_paths(root, row)[page]) as image:
        raw, grouped = draw_ocr_groups(image, raw_candidates, report["groups"])
    raw.save(output / f"{prefix}_raw.png")
    grouped.save(output / f"{prefix}_groups.png")
    transcript = "\n".join(g["text"] for g in report["groups"])
    result = dict(**report, raw_ocr=raw_candidates, transcript=transcript,
                  sample_index=sample_index, preview_file=f"{prefix}.html")
    save_json(result, output / f"{prefix}.json")
    (output / f"{prefix}.txt").write_text(transcript + "\n", encoding="utf-8")
    esc = lambda value: html.escape(str(value))
    table = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
        g["id"], ", ".join(g["candidate_ids"]), g["text"],
        g["target_index"] if g["target_index"] is not None else "",
        g["target_text"] or "", round(g["similarity"], 3) if g["similarity"] is not None else ""))
        + "</tr>" for g in report["groups"])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>{esc(row['sequence_id'])} / page {page}</title>
<style>body{{font:15px system-ui;margin:24px}}.images{{display:flex;gap:20px}}figure{{margin:0;width:48%}}img{{width:100%}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}table{{border-collapse:collapse;width:100%}}pre{{white-space:pre-wrap}}@media(max-width:800px){{.images{{display:block}}figure{{width:100%}}}}</style>
<a href="index.html">All pages</a><h1>{esc(row['sequence_id'])} / page {page}</h1>
<p>{esc(report.get('processing_note', 'Geometry-based joined groups. Order is spatial; panel reading order and bubble boundaries are not verified. References are comparison only.'))}</p>
<p><a href="{prefix}.txt">Transcript TXT</a> · <a href="{prefix}.json">All candidates and results JSON</a></p>
<h2>Joined transcript</h2><pre>{esc(transcript)}</pre>
<div class="images"><figure><figcaption>Raw OCR IDs</figcaption><img src="{prefix}_raw.png" alt="Raw OCR candidate boxes"></figure><figure><figcaption>Joined regions</figcaption><img src="{prefix}_groups.png" alt="Grouped OCR boxes"></figure></div>
<h2>Joined text vs reference transcription</h2><table><tr><th>Group</th><th>OCR IDs</th><th>Joined text</th><th>Target index</th><th>Reference text</th><th>Similarity</th></tr>{table}</table>
<h2>Excluded sound effects</h2><pre>{esc(json.dumps(report['excluded_sound_effects'], indent=2, ensure_ascii=False))}</pre>
<h2>All filtering exclusions</h2><pre>{esc(json.dumps(report.get('filter_exclusions', []), indent=2, ensure_ascii=False))}</pre>
<h2>Unmatched targets</h2><pre>{esc(json.dumps(report['unmatched_targets'], indent=2, ensure_ascii=False))}</pre>
<h2>All raw candidates</h2><pre>{esc(json.dumps(raw_candidates, indent=2, ensure_ascii=False))}</pre></html>'''
    (output / f"{prefix}.html").write_text(document, encoding="utf-8")
    return dict(sequence_id=row["sequence_id"], sample_index=sample_index,
                page_index=page, preview=f"{prefix}.html", result=f"{prefix}.json",
                transcript=f"{prefix}.txt", raw_count=len(raw_candidates),
                group_count=len(report["groups"]))


def write_report_index(entries, output):
    output = Path(output)
    save_json(entries, output / "pages.json")
    links = "\n".join(f'<li><a href="{e["preview"]}">Sample {e["sample_index"]}, '
                      f'page {e["page_index"]}: {html.escape(e["sequence_id"])}</a> '
                      f'— {e["raw_count"]} fragments → {e["group_count"]} groups</li>'
                      for e in entries)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Joined OCR transcripts</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:32px auto}li{margin:10px}</style>'
        f'<h1>Joined OCR transcripts</h1><p>{len(entries)} cached pages. '
        'Page indices start at 0. Transcripts use spatial group order.</p><ul>' + links + '</ul>',
        encoding="utf-8")


def get_ocr_result(root, sample=4, page=0, *, sequence_id=None, output=None):
    """Load a saved page result, including transcript, raw OCR and preview path.

    sample/page are zero-based. sequence_id overrides sample when supplied.
    """
    output = Path(output) if output is not None else Path(root) / "outputs/diagnostics/ocr_joining_validation"
    entries = json.loads((output / "pages.json").read_text(encoding="utf-8"))
    entry = next((e for e in entries if e["page_index"] == page and
                  (e["sequence_id"] == sequence_id if sequence_id is not None else e["sample_index"] == sample)), None)
    if entry is None:
        raise KeyError(f"No saved OCR result for sequence={sequence_id}, sample={sample}, page={page}")
    result = json.loads((output / entry["result"]).read_text(encoding="utf-8"))
    result["preview_path"] = str((output / entry["preview"]).resolve())
    return result
