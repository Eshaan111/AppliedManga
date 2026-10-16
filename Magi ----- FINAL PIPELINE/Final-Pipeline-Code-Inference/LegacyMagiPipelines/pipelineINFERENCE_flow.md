# Magi inference pipeline

Source notebook: [pipelineINFERENCE.ipynb](pipelineINFERENCE.ipynb)

Character overlap removal is included in Section 6, **Remove duplicate character appearances**. An inner character box is removed only when all three conditions hold:

- It is fully contained inside a retained outer box.
- Intersection over union (IoU) is at least **0.90**.
- Character embedding cosine similarity is at least **0.95**.

Text-to-character links are redirected to the retained character, and character indices are remapped. Text boxes, OCR order, and text-to-tail links stay unchanged. Removal decisions are recorded in `character_cleanup.json`.

```mermaid
flowchart TD
    A["Load sequence pages"] --> B["Original Magi inference"]
    B --> C["Text, character and tail boxes<br/>Clusters, associations, OCR and essential flags"]

    C --> D["Character crop embeddings<br/>Fine-tuned final two ViT blocks"]
    C --> E["Paddle OCR on original text boxes"]
    E --> F["Select Paddle or Magi OCR<br/>Validity checks; prefer Paddle if both pass"]
    C --> F

    D --> G["Remove duplicate character boxes<br/>Containment + IoU ≥ 0.90 + cosine ≥ 0.95"]
    G --> H["Redirect character links<br/>Remap indices and preserve text order"]

    H --> I["Match character clusters across pages<br/>Cosine threshold 0.82"]
    I --> J["Sequence entity IDs: E1, E2, ..."]

    J --> K["Map text → character → entity"]
    F --> K

    K --> L["Dropdown page viewer<br/>OCR sources and annotation toggles"]
    K --> M["Keep Magi essential text<br/>Preserve original text-box order"]
    M --> N["Export dialogue_pages.json<br/>speaker: entity ID or null; text: selected OCR"]
```

The notebook uses no ground-truth labels and currently does not run Qwen narration classification. Inference caches and exports are stored under `cache_inference/test/<sequence_id>/`.

The diagram is stored as an editable fenced `mermaid` block. A Markdown viewer with Mermaid support renders it as a flowchart; other viewers may display its source code.
