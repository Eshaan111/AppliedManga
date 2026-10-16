# Fine-tuning experiments: what I tried and what I kept

I used this folder to try improving character matching and speech-versus-narration detection. There are **15 experiment folders**, including a diagnostic pipeline and training setups I did not finish. Here is how the ideas developed, what I learned, and where to find the code.

The [main README](../README.md) tells the full project story. The [final notebook](../Final-Pipeline-Code-Inference/pipelineArtifactContourFullInference.ipynb) and [pipeline guide](../Final-Pipeline-Code-Inference/README_ARTIFACT_CONTOUR_FULL_INFERENCE.md) show what I actually kept. The [metrics guide](../METRICS.md) explains the transcript scores.

## What made it into the final pipeline

I kept the **two-block Magi character-crop checkpoint** from fineTuningMagiCharSimHeads2. Magi still produces the original detections and text-character links before I patch the crop encoder. I use the adapted character features to match appearances across pages. I did not replace the detector or train the active association head.

For text, I kept Paddle OCR, Gemma artifact filtering, contour grouping and contour-based speech/narration decisions. A linked Magi tail forces SPOKEN. Final Gemma text cleanup is optional. The learned narration models, Qwen adapters, QuoteCaster and contour classifiers below are not loaded by the final pipeline.

## The experiments at a glance

These results come from different sets of accepted examples, so I would not read this as one leaderboard. Some entries describe completed comparisons; others describe code I prepared or runs I could not confirm finished.

| Folder | Approach / question | What learns | Kept in the pipeline? | Outcome / lesson |
|---|---|---|---|---|
| [charSimilarityMLP](charSimilarityMLP/) | Project cached character embeddings | 768 → 256 → 128 MLP | No | Cached-feature projection cannot repair detections/associations. Saved checkpoint verified; a numerical gain was not recovered in this review. |
| [charSimResidual](charSimResidual/) | Improve identity features with a residual projection | Residual embedding projection | No | Saved all-pair BA 0.7426 -> 0.7424; clustering-pair BA 0.7142 -> 0.7162. Nearly flat, metric-dependent changes. |
| [fineTuningMagiCharSimHeads](fineTuningMagiCharSimHeads/) | Adapt actual character crops | Final crop-ViT transformer block | No | Validation cross-pair BA improved, but saved evaluation pair BA 0.7576 -> 0.7506 and clustering-pair BA 0.7083 -> 0.7042. No clear transferred gain. |
| [fineTuningMagiCharSimHeads2](fineTuningMagiCharSimHeads2/) | Adapt two crop blocks on the same data | Final two crop-ViT blocks | **Yes** | Retained checkpoint; clustering-pair BA 0.7083 -> 0.7086. Improved different-person recall but reduced same-person recall; benefit is mixed and modest. |
| [isNarratorText](isNarratorText/) | Padded DINO crop narration baseline | Linear head; DINO frozen | No | Narration P/R/F1 0.4848/0.8421/0.6154, support 19. High overall accuracy hid false-positive narration. |
| [isNarratorTextMagi](isNarratorTextMagi/) | Frozen Magi text-object narration baseline | Binary MLP; Magi frozen | No | Narration P/R/F1 0.3000/0.6923/0.4186, support 13; 21 FP versus 9 TP. Frozen text-query visuals alone were weak on this cohort. |
| [NarrationImbalanceTraining](NarrationImbalanceTraining/README.md) | Multimodal context with sparse narration labels | Fusion/sequence Transformer/classifier | No | Baseline five-seed narration F1 mean 0.5754; joint-selected mean 0.7287. Useful tuning but only 13 narration validation examples and validation-based selection. |
| [NarrationImbalanceTrainingPaddedBox](NarrationImbalanceTrainingPaddedBox/README.md) | Add padded DINO crop context | Crop projection and sequence classifier | No | Untuned five-seed narration F1 mean 0.5568; joint-selected 0.6648. Extra DINO context did not automatically beat baseline; joint study did not exceed its architecture winner. |
| [NarrationModelwBubbles](NarrationModelwBubbles/README.md) | Replace covered-text visuals with YOLO bubble-crop features | Classifier; optional DeBERTa tuning | No; alternative learned pipeline | I was not happy with the learned outputs and switched to contour-based roles. Bubble context and semantic tuning were explored; no controlled numerical superiority asserted here. |
| [Manga109Association](Manga109Association/README.md) | Learn explicit visible-speaker associations | Magi text-character matching head | Optional, disabled | Implementation only; no saved trained association-head checkpoint verified. Not a failed benchmark; original Magi head remains active. |
| [TextRoleVL](TextRoleVL/) | Qwen roles from annotated page images/OCR | Zero-shot baseline; language/optional merger LoRA | No | Saved A epoch-1 valid sequences 3/16, scored 34/268 regions, zero narration support. Lower loss did not solve coverage/format failures. B/C execution unconfirmed. |
| [EntityClustering](EntityClustering/README.md) | Isolate entities and inspect tail/character vectors | Existing crop checkpoint; diagnostic pipeline | No; shared concepts | Old viewer omitted unassociated texts rather than deleting them. Added complete logging. Tail-center angle is a proxy, not actual tail-tip direction. |
| [QuoteCaster_Finetuning](QuoteCaster_Finetuning/README.md) | Infer anonymous speaker similarity from dialogue context | Final four BERT blocks with within-sequence triplets | No | Ranking accuracy 0.5499 -> 0.5709, AUC 0.5217 -> 0.5709. Modest learning but weak separation; not reliable speaker identity or visual grounding. |
| [ContourNarrationMLP](ContourNarrationMLP/README.md) | Learn roles from numeric contour history and semantic context | MLP, only no-tail targets | No | No-tail narration P/R/F1 0.8333/0.4167/0.5556 on 12 narration examples. Missed 7/12; I moved on from this model. |
| [ContourNarrationCatBoost](ContourNarrationCatBoost/README.md) | Boosted trees over contour metrics; balancing/text ablations | CatBoost, only no-tail targets | No | Baseline P/R/F1 0.6667/0.8333/0.7407. Removing balancing components/PCA text did not beat baseline; I still had too many false positives. |

## First I tried improving the character features

I started with [charSimilarityMLP](charSimilarityMLP/). Instead of changing Magi, I trained a 768 → 256 → 128 projection on cached character features. I built accepted same/different-person examples from confident OCR-to-reference matches and clear Magi links. Predicted clusters were not treated as correct training labels. That let me try a small model cheaply, but I did not find a clear improvement.

Next, [charSimResidual](charSimResidual/) kept the original feature space and learned a residual correction. The results barely moved: some pair metrics improved slightly and others dropped. So I tried changing the crop encoder itself.

[fineTuningMagiCharSimHeads](fineTuningMagiCharSimHeads/) trains the last crop-ViT block using actual image crops and cosine triplet loss. Despite the folder name, this changes the crop encoder, not the text-character matching head. Validation improved, but the saved evaluation results did not follow it.

[fineTuningMagiCharSimHeads2](fineTuningMagiCharSimHeads2/) trains the final two blocks. I also changed the learning rate to 5e-6, so this was not a clean test of depth alone. I kept the epoch-20 checkpoint with entity threshold 0.82, although the gains were small and mixed. Earlier blocks, the final LayerNorm and the original detection/association components stay frozen.

For these four folders, the usual order is createTrainData.ipynb, the training notebook, then the test notebook. Their configs and saved outputs have the details needed to reproduce each run.

Later, while looking for other character models, I found [Re:Cognize](https://github.com/eternal-f1ame/Re-Cognize). Its [paper, Section 4 and Tables 12–14](https://arxiv.org/html/2609.34032v1) also found small gains from adapting Magi. On Re:Verse, MagiV2 went from 84.2 mAP / 91.0 Rank-1 to 84.7 / 91.7 after fine-tuning; adding memory gave 84.8 / 90.1. Their baseline trains a BNNeck around a frozen backbone, unlike my crop-block tuning. Still, seeing those limited gains made me decide I should probably spend less time trying more fine-tuning and look harder at grouping and speaker assignment.

## Then I tried learning speech versus narration

I first tried a simple visual classifier in [isNarratorText](isNarratorText/): a padded crop, frozen DINO features and a linear head. It caught a lot of narration but also called too much speech narration. [isNarratorTextMagi](isNarratorTextMagi/) instead used frozen Magi text-object features with a small MLP. Its narration precision was also low. Neither experiment fine-tuned the visual backbone.

That led to [NarrationImbalanceTraining](NarrationImbalanceTraining/README.md). I gave the model more than an isolated crop:

- Magi text-object visual features.
- DeBERTa-v3-base semantics from the previous two same-page OCR boxes and the target.
- Normalized text-box geometry and tail features.
- Learned sequence-position and page embeddings.

I projected and fused the inputs to 256 dimensions, then used a two-layer, four-head Transformer and a two-logit classifier. One sample is the whole three-page sequence. Attention runs across all real boxes at once; this is not autoregressive generation. Unlabelled boxes stay as context, with label -100. Padding is masked separately. I did not use is_essential_text.

Because narration was scarce, I oversampled entire narration-containing sequences and used effective-number class weights from the training split only. Repeated draws do not create new independent examples. Shared pages stay on the same side of the split.

The notebooks run **00–06** for preparation, features, training and evaluation. **07** repeats five seeds with the same split, **08** tunes optimization and sampling, **09** tunes heads/feed-forward width, and **10** searches narrow ranges around the stored winners. BestParamTuning exports the selected checkpoint and settings. A best checkpoint score is not the same thing as the average across seeds.

## More visual context was the next idea

In [NarrationImbalanceTrainingPaddedBox](NarrationImbalanceTrainingPaddedBox/README.md), I added frozen DINOv2-small features from text crops padded by 20% per side. Its 384 features project to 128 before joining the other inputs. Magi still supplies its page-conditioned text-object features. The extra input changes the fusion shape, so the original classifier weights cannot simply be reused. I kept the original split and created separate runs and tuning studies.

[NarrationModelwBubbles](NarrationModelwBubbles/README.md) instead used YOLO bubble crops with pooled frozen Magi DETR encoder features. Texts inside the same bubble share its visual vector; uncovered texts keep their original query features. There is no DINO branch here. Stage 05 uses frozen semantic features; stage 07 optionally tunes DeBERTa and the classifier together. That paired encoder must stay with its matching classifier checkpoint.

Even with visual context, text semantics and positions, I was not happy with some page-level predictions. While working on joined bubbles, I found contours useful for grouping, then noticed they could also help distinguish rounded speech regions from rectangular captions. I preferred those results when checking pages myself. I did not establish that contours beat every learned model on a controlled final test. Bubble features alone also do not concatenate text; grouping is a separate step.

## I also tried other ways to learn the decisions

[TextRoleVL](TextRoleVL/) uses **Qwen2.5-VL-3B-Instruct**, not Qwen3. It receives three annotated images, text IDs, OCR and tails, then generates role JSON. I prepared language LoRA, balanced sampling and an optional visual-merger LoRA variant. The saved normal-sampling run had poor usable coverage; I do not have confirmed completed comparisons for every variant. Lower training loss was not enough if the output missed the expected boxes.

[QuoteCaster_Finetuning](QuoteCaster_Finetuning/README.md) fine-tunes the final four BERT blocks with within-sequence triplets. Ranking accuracy improved from 0.5499 to 0.5709, but that did not give reliable speaker separation. Text similarity also does not tell me which visible character spoke a line.

After developing the contour checks, I tried [ContourNarrationMLP](ContourNarrationMLP/README.md) and [ContourNarrationCatBoost](ContourNarrationCatBoost/README.md). I only sent no-tail targets to these models; tail-linked text kept the SPOKEN rule. The MLP missed too much narration. CatBoost improved recall, but false positives remained a problem. There were only 12 narration examples in that validation set, so a few decisions made a large difference.

## Association and clustering are different problems

[Manga109Association](Manga109Association/README.md) prepares training for the Magi text-character matching head using explicit Manga109Dialog links and book-level splits. Ambiguous or off-page links are excluded. I did not verify a trained head checkpoint, and the final pipeline keeps ASSOCIATION_HEAD_PATH=None.

[EntityClustering](EntityClustering/README.md) is for inspecting the existing character pipeline, not another trained model. It also checks disagreement between text-to-tail and text-to-character directions. A tail-center vector is only a proxy; it is not a measured tail-tip direction.

**The main issue I noticed in the final outputs was identity clustering.** A line could be linked to the right character crop, but another view of the same person could receive a different entity ID. The scorer can match only one predicted entity to each reference person, so this hurts attribution and adds an extra-identity penalty to balanced joint F1. That is why clustering is my first target now. The scoring formula confirms the effect; it does not by itself prove how much of the total error came from clustering.

## What I would try next

I would first review a small set of split and merged identities, then test matching against several reliable character views instead of trusting one similar pair. I would check both mistaken merges and unnecessary splits.

For narration, I would add contour measurements and missing-evidence flags to the existing visual + DeBERTa + position model. For attribution, I would try a small candidate-link scorer using bubble coverage, panel/layout information, character features and reliable tail direction where available. Original member links would survive concatenation so conflicting speakers stay visible.

I would keep the split and seeds fixed when comparing a change, and check promising settings on data I did not use to choose them. Local training time and the 8 GB GPU were my main practical limits. Some narration sets contained only 12 or 13 examples, so I would also spend time improving the labels before another large training run.

## Finding the older work and model links

The [main README's model list](../README.md#model-links) links the pretrained models, including ones I did not keep. My checkpoints stay in their experiment folders and need the matching base model and preprocessing.

The [pre-Magi guide](../../legacy%20,%20previous%20approaches/1pageTest/PRE_MAGI_APPROACHES.md) and [full experiment inventory](../../legacy%20,%20previous%20approaches/1pageTest/EXPERIMENT_INVENTORY.md) cover the earlier detection, OCR and Qwen/InternVL attempts. Older contour and learned-pipeline notebooks are in [LegacyMagiPipelines](../Final-Pipeline-Code-Inference/LegacyMagiPipelines/).

I use godfather.venv for the Magi/Paddle work. Training notebooks can start or resume training when run. Saved outputs may be ignored by Git; the per-experiment guides describe the files and settings. A prepared notebook, an unfinished run and a measured poor result are different things, so I have kept those distinctions here.
