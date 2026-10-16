# Manga transcript extraction

My goal was to turn three ordered manga pages into a transcript with the text in reading order, a speaker for each line, and narration marked separately.

I tried quite a few things before settling on this pipeline. It still makes mistakes, but I can usually trace what happened. Here is how I got here, what worked, and what I would try next.

I combine **Gemma artifact filtering, contour-based speech/narration detection, joined-bubble grouping and character assignment**.

## What I tried

Before Magi, I tried direct Qwen transcripts, separate YOLO bubbles, Paddle OCR, face/body detection, InternVL association and DINO identity features. I also tested artifact checks, crop variations, repeated seeds and Optuna settings.

### Models I trained or tried training

- **Qwen transcript LoRA:** I tried end-to-end training, but even the baseline struggled with page formatting and repeated text.
- **InternVL weighted-token adapters:** I trained decision-weighted and full-JSON versions, but the generated IDs and quick test outputs were still unreliable.
- **Qwen association LoRA:** I hit GPU memory limits and do not have a confirmed completed comparison.
- **Decoder/projector/vision variants:** I set these up separately, but did not finish every run.
- **Character projection MLP:** I trained projections on cached features, but could not find a clear improvement.
- **Residual character projection:** I got almost the same results, with some metrics up and others down, so I moved on.
- **One-block Magi crop tuning:** I saw validation gains but weaker evaluation results.
- **Two-block Magi crop tuning:** I kept this checkpoint, although the improvement was small.
- **Frozen DINO narration head:** I caught more narration, but also labelled too much speech as narration.
- **Frozen Magi narration MLP:** I got too many false narration labels.
- **Multimodal narration Transformer:** I combined visuals, DeBERTa, geometry and context, but still saw inconsistent decisions when looking through pages.
- **Padded-DINO narration:** I added surrounding visuals; the untuned repeated-seed result did not improve.
- **Bubble-crop and DeBERTa tuning:** I added more context, but preferred the contour results when checking pages myself.
- **Qwen role LoRA:** I got too few usable outputs; I do not have confirmed completed runs for the balanced/visual variants.
- **Manga109 association head:** I set up training, but do not have a verified trained head.
- **QuoteCaster fine-tuning:** I saw a little improvement, but it still struggled to separate speakers.
- **Contour/context MLP:** I missed too many narration examples.
- **Contour CatBoost:** I caught more narration, but still had too many false positives.

I kept the two-block crop checkpoint. The [experiment guide](FINETUNING-EXPERMENTS/README.md) has more detail, including which runs were unfinished. EntityClustering was for inspecting the pipeline, not another model I trained.

While looking for other character models, I later found [Re:Cognize](https://github.com/eternal-f1ame/Re-Cognize), which evaluates MagiV2 alongside other backbones. Its [paper, Section 4 and Tables 12–14](https://arxiv.org/html/2609.34032v1) reports limited gains from encoder-side adaptation. On Re:Verse, released MagiV2 scored 84.2 mAP / 91.0 Rank-1; fine-tuning scored 84.7 / 91.7, and adding memory scored 84.8 / 90.1. After seeing that, I decided I should probably stop spending so much time looking for gains from more fine-tuning and focus on grouping and speaker assignment. Fine-tuning did help a little in their tests, and their baseline trains a BNNeck with the backbone frozen, unlike my crop-block tuning. I still kept my two-block checkpoint.

## What slowed me down

**Local training time was my main bottleneck.** Training, feature extraction and validation took time, and the local 8 GB GPU limited image sizes and model configurations. Some attempts ran out of memory.

I used frozen encoders, small classifiers, LoRA and cached features to keep experiments manageable. I tried repeated seeds and tuning where possible, but could not fully compare every idea.

**Getting enough labelled examples was another challenge.** There were 80 development sequences, with relatively few narration examples. Some validation sets contained only 12 or 13 narration examples, making the metrics sensitive to a handful of mistakes.

The supplied images and transcripts were useful. Box-level training also needed text-to-transcript matches, visible-speaker links and appearance identities. I built these labels using OCR matching, existing model links and manual checks. That took extra work, and I could not confidently label every example.

That limited how much I could try, although it does not explain away the mistakes still in the pipeline.

## The pipeline I chose

```mermaid
flowchart TD
    A["Three ordered pages"] --> B["Magi detections, OCR and original links"]
    B --> C["Paddle OCR per text box"]
    C --> D["Gemma artifact decision and punctuation rules"]
    D --> E["Delete artifacts; retain Text and Review"]
    A --> Y["YOLO bubble candidates"]
    E --> F["Bubble-mask gate for empty-Paddle punctuation fallbacks"]
    Y --> F
    F --> G["Surviving original text IDs"]
    G --> H["Individual contours, faint retries and overlap groups"]
    Y --> H
    H --> I["Concatenate members in Magi reading order"]
    I --> J["White or solid region shape: SPOKEN or NARRATED"]
    J --> T["Linked Magi tail forces SPOKEN"]
    B -->|"Text-to-tail links"| T
    B --> V["Fine-tuned character crop embeddings"]
    V --> X["Duplicate cleanup and cross-page entities"]
    X --> R["Route through original member character links"]
    B -->|"Text-to-character links"| R
    T --> R
    R --> K["Raw transcript and audits"]
    K --> L["Optional final Gemma OCR correction"]
    L --> M["Final transcript JSONL"]
```

[run_sequence_inference.py](Final-Pipeline-Code-Inference/run_sequence_inference.py) selects test or development from TaskData/sequences.json. Reference labels are not inference input.

## How the pieces came together

Once I had Magi detections and Paddle OCR, I could recover text, but not yet a reliable transcript. Watermarks, links, sound effects and drawing details also became OCR. Before deciding who said something, I needed to decide whether it belonged in the transcript at all.

I compared dictionary checks, statistical gibberish detection and OCR agreement. Names, reactions and noisy lettering made these unreliable. That led me to local **Gemma 3 4B IT**, given one box's Paddle text and examples of Text versus Artifact. It sees neither the image nor neighboring dialogue at this stage. Punctuation-only Paddle text is kept directly; empty-Paddle boxes can use Magi punctuation only when the bubble-region check supports it. Ordinary Magi words are not a fallback. Unexpected Gemma answers stay for review.

This gave me a cleaner set of text boxes, but exposed another problem: **one connected speech bubble could become several detections and separate transcript lines**. I started inspecting contours to understand which fragments belonged together. Individual enclosing boundaries were useful, but faint lines and connected bubbles often prevented a complete outline.

So I stopped requiring a perfect outline just to group the text. Overlapping original YOLO rectangles become grouping candidates when at least one individual boundary remains unresolved; verified neighbors can join. Padding alone cannot create an overlap. I match those candidates to surviving Magi/Paddle texts, preserve their IDs and concatenate them in Magi reading order. Unmatched text stays separate, and every surviving box must appear exactly once. A closed merged outline is not required. [contour_grouping.py](Final-Pipeline-Code-Inference/contour_grouping.py) records the members and reasons.

**While working on joined bubbles, I noticed something else:** many narration lines had no surrounding bubble, or sat inside a square or rectangular caption. The contours I was already examining for grouping could also provide evidence about speech versus narration.

I had already tried learning that distinction. My [multimodal narration model](FINETUNING-EXPERMENTS/NarrationImbalanceTraining/README.md) combined Magi text-object visual features, **DeBERTa semantics from the previous two same-page OCR boxes plus the target**, normalized geometry and tail features. I fused these into 256 dimensions, added learned sequence-position and page embeddings, and used a two-layer Transformer to classify all boxes in a three-page sequence together. I used whole-sequence narration oversampling, class-balanced loss, repeated seeds and Optuna. Later variants added padded DINO crops and bubble-crop visuals.

Even with all that information, I was not happy with some of the predictions when I looked through the pages. The contour results looked better to me, and I could see why a decision was made. I did not run a comparison that proves contours beat every model. I also did not want to just call everything SPOKEN: that might score reasonably on common cases, but it would miss the point of detecting narration.

I expanded YOLO crops by about **20% per side** and measured the surrounding white or solid-color region, excluding text rectangles from color measurements. Meaningful white-region curves or supported expressive sharp shapes support SPOKEN; supported straight sides and perpendicular corners support NARRATED. Strong curves can override a box-like interpretation. Weak or missing evidence currently falls back to NARRATED, with review flags where applicable.

This required more care than simply finding a closed contour. Crop edges are not drawn boundaries, faint cases can receive a retry, and clipped or rectangular speech bubbles remain difficult. A Magi text-to-tail link forces SPOKEN, but does not identify the person speaking. [contour_narrator2.py](Final-Pipeline-Code-Inference/contour_narrator2.py) implements these checks, including region inspection after grouping.

With text retained, fragments joined and roles assigned, I could return to **who was speaking**. Magi supplies the original text-character links. My retained checkpoint changes the final two character crop-encoder blocks, not the detector or association head; those features support cross-page identity matching. Narrated groups become NARRATION. Spoken groups use a consistent linked entity, while missing or conflicting links become UNKNOWN. IDs remain sequence-local. Joining text does not prove a shared speaker: 16 of 186 merged groups aligned to multiple reference speakers and need image review.

Finally, joined entries sometimes exposed OCR noise that was harder to notice in isolated fragments. I added a separate Gemma call to correct each final entry without speaker IDs. I tried different prompts and input/output examples: the artifact step covers dialogue to keep, artifacts to remove, SFX, non-SFX reactions, names, numbers and punctuation; the cleanup step covers OCR noise to remove and valid text to preserve. I keep raw and cleaned transcripts, original OCR and deletion reasons because either step can still make mistakes, and later contours cannot restore a deleted box.

## Current results and what they show

All 80 development sequences completed. These are development results, not independent final-test results; development data also supported earlier training and tuning.

| Metric | Result |
| --- | ---: |
| Text order score | 0.902258 |
| Matched token coverage | 0.923951 |
| Speaker accuracy on matched text | 0.638956 |
| Joint F1 | 0.600417 |
| Balanced joint F1 | 0.390426 |

**The main issue I noticed was character clustering.** Text could be linked to the right character crop, but different appearances of the same person sometimes ended up under different entity IDs. So even when the original text-to-character association was right, the final transcript treated that person as multiple speakers. This is why I would focus next on clustering before replacing the association method.

**This directly hurts balanced joint F1:** the scorer can match only one predicted entity to each reference speaker. Splitting one person into several entities makes some correctly linked text count as wrongly attributed, and the extra entities also lower the balanced score. Simply using a different character number is fine; using inconsistent IDs for the same person is the problem. This explains how the clustering failures I observed can compound, although the formula alone does not measure how much of the total error they caused.

[METRICS.md](METRICS.md) explains the calculations. The [score report](score.json) and [predictions](submission_outputs/inference_development/predictions.jsonl) remain available.

The pipeline is better at recovering the text than figuring out who said it. Balanced joint F1 makes that clearer: getting one character's long conversation right does not make up for missing other characters or inventing extra speaker groups.

The [error analysis](submission_outputs/inference_development/analysis/speaker_error_summary.json) found 170 surplus labels across 79 speaker-scored sequences, and 29 of 272 known reference identities with zero identity F1. About 73% of wrong matched-token attribution weight involved spoken identity, and 25% involved speech labelled narration. Narration appeared in 63 predicted sequences versus 14 reference sequences.

The clustering problems I noticed are therefore my first target for improvement. I would still check the original speaker links separately, since association mistakes can also happen. The final Gemma cleanup slightly improved both joint metrics.

## What I would try next

I already have useful inputs: text context, character crops, positions, bubbles, contours and Magi links. Before adding another big model, I would try to get more out of those. The two problems I keep noticing are different people ending up in one identity group, and text being linked to the wrong bubble or character. Different angles, poses and crops make matching harder, but I still need to check whether the feature vectors or the grouping rule caused each mistake.

**I would start by checking a small set of failures.** I would mark which crops are the same person, which texts belong to each bubble, and who speaks each line. A good speaker link still gets the wrong final ID if I have merged two people. Keeping those two problems separate should make it easier to see what actually needs fixing.

**For character grouping, I would be more careful about merging.** Instead of trusting one similar pair, I would compare against several reliable views of each character and check how close the next-best match is. If one new crop connects two groups that otherwise look different, I would flag it. Uncertain crops could stay separate until there is better evidence. That could also split one person into too many groups, so I would check both mistakes. If I train again, I would focus on different views of the same person and similar-looking different people.

**For narration, I would add contours to the existing multimodal Transformer.** It already takes visual features, DeBERTa context, geometry and position/page embeddings. I would add curves, straight sides, white/solid fractions, text coverage, crop-edge contact, leakage and retry flags. I would also tell it when a measurement is missing. Tail-linked text would keep the SPOKEN rule. I would test this against the unchanged model with the same split and seeds. The earlier contour MLP and CatBoost tests only had 12 narration examples, so I would not treat those as the final answer.

**For text assignment, I would try a small model that scores possible matches.** First, bubble-to-text matching could use how much text lies inside each region and whether another bubble fits better. Then text-to-character matching could use Magi scores where available, positions, panels, crop features and tail direction when a reliable tip can be measured. A tail tells me something is spoken, not who said it. Nearby dialogue could help too, but I would not assume consecutive lines have the same speaker. Joined text would keep its original links so conflicting speakers stay visible.

I would start with cached features and small models to keep this manageable. I would keep the deletion and OCR-edit logs, and check any promising change on data I did not use to choose it.

## Code and resources

The [batch guide](Final-Pipeline-Code-Inference/README_BATCH_INFERENCE.md) covers running, caches and shared model loading. The [notebook](Final-Pipeline-Code-Inference/pipelineArtifactContourFullInference.ipynb) lets me look through the results.

I use pretrained [Magi V2](https://huggingface.co/ragavsachdeva/magiv2), [Gemma](https://huggingface.co/google/gemma-3-4b-it), [YOLO bubbles](https://huggingface.co/PSImera/manga_bubbles_detect) and Paddle [detection](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det)/[recognition](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec), plus my two-block character checkpoint. Weights remain separate.

My checkpoint is under FINETUNING-EXPERMENTS/fineTuningMagiCharSimHeads2 and requires the matching Magi base. I run score.py separately.

I renamed folders after the completed run, so the helper-folder settings and old absolute paths still need updating before another run. Workflow tests passed using stubs and saved data; I have not yet measured the GPU speedup or checked numerical agreement on a real GPU run. These README edits do not change the inference code or saved results.

## Model links

These are the models behind the final pipeline and the other experiments. A link here does not mean I kept it or completed training with it.

| Model | Link | Where I used it |
| --- | --- | --- |
| Magi V2 | [Hugging Face](https://huggingface.co/ragavsachdeva/magiv2) · [Code](https://github.com/ragavsachdeva/magi) | Final detections, OCR, links and character features; earlier narration experiments |
| Gemma 3 4B IT | [Hugging Face](https://huggingface.co/google/gemma-3-4b-it) | Final artifact filtering and OCR cleanup |
| YOLO manga bubbles | [Hugging Face](https://huggingface.co/PSImera/manga_bubbles_detect) | Final bubble candidates and earlier bubble tests |
| PP-OCRv5 server detector | [Hugging Face](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det) | Final Paddle text detection |
| English PP-OCRv5 mobile recognizer | [Hugging Face](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec) | Final Paddle text recognition |
| PP-OCRv6 medium detector | [Hugging Face](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det) | Earlier whole-page OCR |
| PP-OCRv6 medium recognizer | [Hugging Face](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec) | Earlier whole-page OCR |
| PaddleOCR-VL for Manga | [Hugging Face](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga) | Alternative manga OCR tests; based on [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) |
| EasyOCR | [Code and checkpoints](https://github.com/JaidedAI/EasyOCR) | Earlier OCR-hint branch; completed execution not confirmed |
| DeBERTa-v3-base | [Hugging Face](https://huggingface.co/microsoft/deberta-v3-base) | Text context for learned narration models |
| DINOv2-small | [Hugging Face](https://huggingface.co/facebook/dinov2-small) | Character similarity, narration baseline and padded-crop features |
| InternVL3-2B | [Hugging Face](https://huggingface.co/OpenGVLab/InternVL3-2B) | Earlier association and LoRA experiments |
| Qwen2.5-VL-3B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Transcript, attribution and role experiments |
| Qwen2.5-1.5B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Earlier text formatter |
| DASS / comicapp character detector | [Hugging Face](https://huggingface.co/comicbox/comicapp-char-detector) | Earlier face/body detection and pairing |
| Anime face detector, YOLOv3 | [Hugging Face](https://huggingface.co/hysts/anime-face-detector-yolov3) · [Code](https://github.com/hysts/anime-face-detector) | Earlier character-crop tests |
| Anime face detector, Faster R-CNN | [Hugging Face](https://huggingface.co/hysts/anime-face-detector-faster-rcnn) | Alternative referenced in the earlier detector setup; a separate completed comparison is not confirmed |
| Anime facial landmarks, HRNetV2 | [Hugging Face](https://huggingface.co/hysts/anime-face-detector-hrnetv2) | Landmark component of the anime-face setup |
| YOLO26n manga panel detector | [Hugging Face](https://huggingface.co/leoxs22/manga-panel-detector-yolo26n) | Earlier panel detection |
| English ByT5 OCR correction | [Hugging Face](https://huggingface.co/yelpfeast/byt5-base-english-ocr-correction) | Standalone correction test; not kept |
| QuoteCaster encoder | [Hugging Face](https://huggingface.co/aNameNobodyChose/quote-caster-encoder) | Text-only speaker tests and fine-tuning |
| Gibberish-detector English model | [Code and model instructions](https://github.com/domanchi/gibberish-detector) | Statistical artifact checks |
| MangaLMM | [Hugging Face](https://huggingface.co/hal-utokyo/MangaLMM) · [Code](https://github.com/manga109/MangaLMM) | Downloaded; no dedicated completed experiment verified |

The [panel-order estimator](https://github.com/manga109/panel-order-estimator) was another earlier component; its local folder did not contain a separate learned checkpoint. I also looked into [Re:Cognize](https://github.com/eternal-f1ame/Re-Cognize) as research, not as a model used in the final pipeline.

### Models I trained locally

These do not have published Hugging Face checkpoints. The folders contain the code and describe the matching base models; generated weights may be excluded from Git.

- Character features: [projection MLP](FINETUNING-EXPERMENTS/charSimilarityMLP/), [residual projection](FINETUNING-EXPERMENTS/charSimResidual/), [one-block Magi tuning](FINETUNING-EXPERMENTS/fineTuningMagiCharSimHeads/), and [two-block Magi tuning, kept in the pipeline](FINETUNING-EXPERMENTS/fineTuningMagiCharSimHeads2/).
- Narration models: [DINO head](FINETUNING-EXPERMENTS/isNarratorText/), [Magi head](FINETUNING-EXPERMENTS/isNarratorTextMagi/), [multimodal Transformer](FINETUNING-EXPERMENTS/NarrationImbalanceTraining/), [padded-DINO version](FINETUNING-EXPERMENTS/NarrationImbalanceTrainingPaddedBox/), and [bubble-crop version](FINETUNING-EXPERMENTS/NarrationModelwBubbles/).
- Other learned tests: [QuoteCaster fine-tuning](FINETUNING-EXPERMENTS/QuoteCaster_Finetuning/), [contour MLP](FINETUNING-EXPERMENTS/ContourNarrationMLP/), and [contour CatBoost](FINETUNING-EXPERMENTS/ContourNarrationCatBoost/).
- Adapter experiments: Qwen transcript/association and InternVL adapters are in the pre-Magi archive; the [Qwen role experiment](FINETUNING-EXPERMENTS/TextRoleVL/) is in this project. They need their matching base model above. Some runs were unfinished.
- [Manga109 association head](FINETUNING-EXPERMENTS/Manga109Association/): training code prepared, but no trained checkpoint verified.
