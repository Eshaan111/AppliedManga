param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$project = Join-Path $workspace 'manga_notebook_codebase'
# Explicit obsolete implementation/artifact paths. OCR, joining, data, splits,
# OCR model cache, and the working OCR environment are intentionally preserved.
$relativeTargets = @(
 'guide(4).md',
 '.venvDino',
 'manga_notebook_codebase/guide(4).md',
 'manga_notebook_codebase/MODELS.md',
 'manga_notebook_codebase/OCR_RECOVERY_TEST.md',
 'manga_notebook_codebase/POTENTIAL_IDEAS.md',
 'manga_notebook_codebase/requirements-gpu-lock.txt',
 'manga_notebook_codebase/checkpoints',
 'manga_notebook_codebase/src/components.py',
 'manga_notebook_codebase/src/experiments.py',
 'manga_notebook_codebase/src/gpu_training.py',
 'manga_notebook_codebase/src/identity_utils.py',
 'manga_notebook_codebase/src/qwen_utils.py',
 'manga_notebook_codebase/src/training.py',
 'manga_notebook_codebase/scripts/build_notebooks.py',
 'manga_notebook_codebase/scripts/overfit_smoke.py',
 'manga_notebook_codebase/scripts/make_submission.py',
 'manga_notebook_codebase/notebooks/01_qwen_baseline.ipynb',
 'manga_notebook_codebase/notebooks/03_train_qwen_lora.ipynb',
 'manga_notebook_codebase/notebooks/04_ocr_experiment.ipynb',
 'manga_notebook_codebase/notebooks/05_identity_embeddings.ipynb',
 'manga_notebook_codebase/notebooks/06_full_pipeline.ipynb',
 'manga_notebook_codebase/notebooks/07_evaluate.ipynb',
 'manga_notebook_codebase/notebooks/08_component_tests.ipynb',
 'manga_notebook_codebase/outputs/experiments',
 'manga_notebook_codebase/outputs/smoke',
 'manga_notebook_codebase/outputs/tmpns3wrd7w',
 'manga_notebook_codebase/src/__pycache__',
 'manga_notebook_codebase/scripts/__pycache__',
 'manga_notebook_codebase/tests/__pycache__'
)
$prefix = $workspace.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
foreach ($relative in $relativeTargets) {
 $target = [IO.Path]::GetFullPath((Join-Path $workspace $relative))
 if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Unsafe deletion target: $target"
 }
 if (Test-Path -LiteralPath $target) {
  Write-Output "Remove: $target"
  if ($Apply) { Remove-Item -LiteralPath $target -Recurse -Force }
 }
}
if (-not $Apply) { Write-Output 'Dry run only. Use -Apply to remove the listed paths.' }
