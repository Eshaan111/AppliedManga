# ContourNarrationCatBoost

Run TrainContourNarrationCatBoost.ipynb in godfather.venv. You run training. CatBoost is installed into the selected kernel by the training cell if absent. Existing MLP contour caches are reused read-only; changed/missing caches are generated locally by the included preparation cells. Source labels and 64/16 sequence split match NarrationModelwBubbles.

Inputs: 35 current numeric contour/geometry/group features and availability flags, same for previous two same-page texts, plus predecessor presence flags (212 columns). Missing values become NaN. No DeBERTa, image embeddings, tail flags, heuristic role/reason or GT context labels enter CatBoost.

Only known no-tail targets train the model. Tail-linked units bypass CatBoost and force SPOKEN, but can provide context. Unknown/unmatched/mixed merged labels are ignored for loss/metrics. The cohort uses the existing matched OCR cache, not Gemma filtering.

Whole narration-containing training sequences are duplicated with expected multiplier 2.5 (every sequence once; narration sequences two or three copies). Class-balanced effective-number weights use original no-tail training counts. Validation is not duplicated; best iteration uses unweighted validation macro F1. Small CPU baseline: depth 4, learning rate 0.03, 1000 maximum trees, early stopping 100 rounds. No Optuna or new split.

Outputs under runs/baseline_seed42: best.cbm, config.json, split.json, sampling_audit.json, evaluation.json, val_predictions.csv/json, feature_importance.csv, catboost_logs/. Reports no-tail precision/recall/F1, macro F1, balanced accuracy, true-class probability summaries, heuristic comparison and overall tail-routing performance. Optional explain_text() uses CatBoost SHAP values for one no-tail target. Feature importance explains model behavior, not causality.

Native model must be used with saved feature ordering, missing-value conversion and matching contour helper/settings. This notebook does not alter production inference or the MLP notebook/checkpoint. CatBoost references: https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing and https://catboost.ai/docs/en/concepts/loss-functions-multiclassification.


## Precision/recall comparison and viewer
The notebook now reruns the numeric double-balanced baseline in runs/comparison_seed42/baseline_double and trains oversampling_only / class_weights_only. Optional USE_TEXT_VARIANTS=True adds the same two variants with frozen local DeBERTa context and 16 train-only PCA components; compatible MLP semantic caches are reused. Original runs/baseline_seed42 stays unchanged. Numeric-only features are unchanged, and all runs retain the exact split/no-tail targets. Raw TP/FP/FN are compared alongside narration P/R/F1 and macro F1. No threshold changes are imposed. All results use validation, not a separate test set.

Final widget selects model run, sequence, page and error filter. Validation/training sequences are labelled. Wrong known-GT roles are red, narration yellow, spoken blue; original member boxes are dashed. A table shows OCR, GT, probability and contour reason. One page at a time. The viewer never re-trains models. It requires running the training/comparison cells first.
