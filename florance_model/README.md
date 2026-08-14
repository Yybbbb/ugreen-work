# ugipc_1231_15words/epoch_3 Inference Handoff

This package documents how to run inference with:

```text
/data0/work/LuoWeiXing/Florence2-ft/model_checkpoints/ugipc_1231_15words/epoch_3
```

The archive includes the checkpoint weights and processor files under:

```text
checkpoint/
```

## Contents

```text
README.md          This handoff document
infer_ugipc.py     Minimal local inference script
requirements.txt   Python package names
MANIFEST.md        File list and source checkpoint
checkpoint/        Full model checkpoint and processor files
```

## Environment

The known working Python environment on the source machine is:

```bash
/data0/work/LuoWeiXing/miniconda3/envs/Florence-2/bin/python
```

Required packages:

```text
torch
transformers
Pillow
safetensors
```

The package also includes `requirements.txt` with these package names.

The model must be loaded with:

```python
trust_remote_code=True
local_files_only=True
use_fast=False   # processor/tokenizer side
```

## Checkpoint Files

The checkpoint directory contains:

```text
model.safetensors
config.json
generation_config.json
processing_florence2.py
modeling_florence2.py
configuration_florence2.py
preprocessor_config.json
processor_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
added_tokens.json
vocab.json
merges.txt
```

Do not copy only `model.safetensors`. The custom Florence-2 code and processor files in the same directory are required.

## Supported Tasks From processing_florence2.py

The checkpoint processor defines these relevant task tokens:

```text
<CAPTION>
<DETAILED_CAPTION>
<MORE_DETAILED_CAPTION>
<REGION_TO_DESCRIPTION>
```

The processor expands them internally:

```text
<CAPTION>                -> What does the image describe?
<DETAILED_CAPTION>       -> Describe in detail what is shown in the image.
<MORE_DETAILED_CAPTION>  -> Describe with a paragraph what is shown in the image.
<REGION_TO_DESCRIPTION>  -> What does the region {loc_tokens} describe?
```

## Full Image Caption

Use a caption task token only. Do not attach a bbox to full-image caption.

Recommended commands:

```bash
python infer_ugipc.py \
  --image /path/to/image.jpg \
  --task caption
```

For a longer caption:

```bash
python infer_ugipc.py \
  --image /path/to/image.jpg \
  --task detailed_caption \
  --max-new-tokens 100
```

Allowed full-image caption prompts:

```text
<CAPTION>
<DETAILED_CAPTION>
<MORE_DETAILED_CAPTION>
```

Important:

- Do not use `<REGION_TO_DESCRIPTION>` for full-image caption.
- Do not create a fake full-image bbox like `<loc_0><loc_0><loc_999><loc_999>` for caption.
- Choose one caption task for production and log the exact prompt used.
- If `<CAPTION>` is too short or returns an unhelpful generic answer, try `<DETAILED_CAPTION>` before changing model weights.

## Region Description

Use `<REGION_TO_DESCRIPTION>` plus exactly four Florence loc tokens:

```text
<REGION_TO_DESCRIPTION><loc_x1><loc_y1><loc_x2><loc_y2>
```

Example:

```bash
python infer_ugipc.py \
  --image /path/to/image.jpg \
  --task region_description \
  --bbox 100,120,260,420 \
  --max-new-tokens 100
```

The script accepts pixel bboxes and converts them to Florence 0-999 loc tokens.

### Optional Region Name

`processing_florence2.py` has special handling for:

```text
<REGION_TO_DESCRIPTION>person<loc_x1><loc_y1><loc_x2><loc_y2>
```

This expands to a person-specific prompt:

```text
Describe person in the region <loc_x1><loc_y1><loc_x2><loc_y2>, including appearance, actions, and interactions.
```

Use it only when the region is known to contain a person:

```bash
python infer_ugipc.py \
  --image /path/to/image.jpg \
  --task region_description \
  --bbox 100,120,260,420 \
  --name person
```

For general region description, leave `--name` empty.

## Pixel Bbox To Florence Loc Conversion

The script uses Florence-style 1000-bin floor quantization:

```python
loc_x = floor(pixel_x / (image_width / 1000))
loc_y = floor(pixel_y / (image_height / 1000))
```

Values are clamped to `[0, 999]`.

Important:

- Input bbox format is pixel `x1,y1,x2,y2`.
- `x2 > x1` and `y2 > y1`.
- The bbox must be inside image bounds.
- Do not mix normalized `[0, 1]`, COCO `x,y,w,h`, and Florence loc coordinates.
- If you already have Florence loc tokens, do not quantize them again.

## Generation Settings

The script defaults to:

```text
max_new_tokens = 80
num_beams = 1
do_sample = False
no_repeat_ngram_size = 0
```

Notes:

- Keep `do_sample=False` for deterministic inference.
- Keep `no_repeat_ngram_size=0`; repeat blocking can corrupt structured or repetitive outputs.
- Increase `max_new_tokens` for detailed captions or long region descriptions.
- Use `num_beams=1` first. If output quality is poor, try `--num-beams 3` and compare latency.

## Output Format

`infer_ugipc.py` prints JSON:

```json
{
  "checkpoint": ".../ugipc_1231_15words/epoch_3",
  "image": {"path": "/path/to/image.jpg", "width": 1920, "height": 1080},
  "task": "region_description",
  "prompt": "<REGION_TO_DESCRIPTION><loc_52><loc_111><loc_135><loc_388>",
  "bbox_pixel": [100.0, 120.0, 260.0, 420.0],
  "bbox_loc_0_999": [52, 111, 135, 388],
  "raw": "...",
  "text": "..."
}
```

For integration, log at least:

```text
checkpoint
task
prompt
image size
bbox_pixel
bbox_loc_0_999
raw output
cleaned output
```

## Common Mistakes

1. Sending a bbox to full-image caption.
   - Wrong: `<CAPTION><loc_0><loc_0><loc_999><loc_999>`
   - Right: `<CAPTION>`

2. Using the wrong task token for region description.
   - Wrong for this handoff: `<PersonRegion> ...`
   - Right: `<REGION_TO_DESCRIPTION><loc_x1><loc_y1><loc_x2><loc_y2>`

3. Using pixel coordinates directly inside `<loc_*>`.
   - `<loc_*>` tokens must be 0-999 Florence loc coordinates, not raw image pixels.

4. Re-tokenizing or editing `processing_florence2.py`.
   - Use the processor shipped inside the checkpoint.

5. Loading without `trust_remote_code=True`.
   - The model uses custom Florence-2 code in the checkpoint directory.

## Minimal Python API Example

```python
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

checkpoint = "/data0/work/LuoWeiXing/Florence2-ft/model_checkpoints/ugipc_1231_15words/epoch_3"
image = Image.open("/path/to/image.jpg").convert("RGB")
prompt = "<DETAILED_CAPTION>"

processor = AutoProcessor.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False,
)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True,
).to("cuda:0")
model.eval()

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=100,
        num_beams=1,
        do_sample=False,
        no_repeat_ngram_size=0,
    )

print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Recommended Handoff Checklist

- Confirm the recipient has the full checkpoint directory.
- Confirm GPU memory is sufficient for Florence-2.
- Run `python infer_ugipc.py --help`.
- Run one full-image caption example.
- Run one region description example with a known bbox.
- Compare logged `bbox_loc_0_999` against expected region location.
