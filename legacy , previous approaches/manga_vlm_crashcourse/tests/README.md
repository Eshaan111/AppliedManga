# Pipeline tests

Run the dataset, prompt, label-mask, and processor checks:

```powershell
& 'C:\Users\ESHAAN\BYTE\Applied-Task\.venvAppliedTask\Scripts\python.exe' -m unittest discover -s tests -v
```

Run only the canonical label/image pipeline verification:

```powershell
& 'C:\Users\ESHAAN\BYTE\Applied-Task\.venvAppliedTask\Scripts\python.exe' -m unittest tests.test_data_pipeline.CanonicalDatasetTests -v
```

The real 3B-model overfit test is skipped by default. Run it explicitly on a
CUDA machine after closing notebooks or other processes that use GPU memory:

```powershell
$env:RUN_VLM_OVERFIT_TEST = '1'
& 'C:\Users\ESHAAN\BYTE\Applied-Task\.venvAppliedTask\Scripts\python.exe' -m unittest tests.test_overfit_smoke -v
```

Set `MANGA_MODEL_PATH` if the local Qwen model is stored somewhere other than
`C:\models\HF\Qwen2.5-VL-3B-Instruct`.
