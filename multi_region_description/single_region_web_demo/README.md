# Single-region person description demo

This page accepts one uploaded image and multiple manually drawn boxes. The browser
sends the boxes sequentially to `/api/predict-one`; every request builds exactly one
`<REGION_TO_DESCRIPTION><loc_x1><loc_y1><loc_x2><loc_y2>` prompt and executes one
`model.generate` call. It never uses `<REGIONS_TO_DESCRIPTIONS>`.

```bash
./run_demo.sh
```

Defaults:

- physical GPU 1
- BF16
- port 8512
- `qwen_single_region_dhash10_grouped_gpu1_ep1_bs24_lr4e6/final`
