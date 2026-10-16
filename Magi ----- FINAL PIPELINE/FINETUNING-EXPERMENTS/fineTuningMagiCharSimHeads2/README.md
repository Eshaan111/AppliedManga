# Two-block Magi character checkpoint

`magi_crop_last_2_blocks.pt` contains my fine-tuned final two character crop-encoder blocks (epoch 20). It requires the original [Magi V2](https://huggingface.co/ragavsachdeva/magiv2); it is not a standalone detector or a trained speaker-association head.

The [final pipeline](../../Final-Pipeline-Code-Inference/src/artifact_contour_sequence.py) loads this file through the `character_checkpoint` model setting. Its existing loader checks the base-model hashes and patches only the saved parameters. The matching entity threshold is **0.82**.

For an already loaded matching Magi model, the patch itself is:

```python
from pathlib import Path
import torch

checkpoint = torch.load(Path("magi_crop_last_2_blocks.pt"),
                        map_location="cpu", weights_only=True)
parameters = dict(model.named_parameters())
with torch.no_grad():
    for name, value in checkpoint["state_dict"].items():
        assert parameters[name].shape == value.shape
        parameters[name].copy_(value.to(parameters[name].device))
model.eval()
```

In the full pipeline, run original Magi detections and associations **before** applying this patch, then extract character crop embeddings. Keep the pipeline's normalization and grouping settings. Training images, caches and `training_data.pt` are not needed to use the checkpoint.
