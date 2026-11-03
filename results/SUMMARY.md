# NutriSnap results summary

Final (app) model: `gpu_ablation_effnet_b0`

- Test top-1 **83.5%**, top-5 **96.4%**, macro-F1 0.833 on 25,250 images / 101 classes; 4.14 M params; 22.0 ms/img on CPU


## ablation_gpu

| Backbone | Top-1 | Top-5 | Macro-F1 | Params (M) | CPU latency (ms/img) | Train time (s/epoch) |
|---|---|---|---|---|---|---|
| efficientnet_b0 | 83.5% | 96.4% | 0.833 | 4.1 | 23 | 293 |
| convnext_tiny | 89.2% | 98.1% | 0.892 | 27.9 | 55 | 509 |
