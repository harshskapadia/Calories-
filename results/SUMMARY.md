# NutriSnap results summary

Final (app) model: `multi_convnext_tiny`

- Test top-1 **88.8%**, top-5 **97.7%**, macro-F1 0.893 on 26,836 images / 139 classes; 27.93 M params; 242.8 ms/img on CPU


## ablation_gpu

| Backbone | Top-1 | Top-5 | Macro-F1 | Params (M) | CPU latency (ms/img) | Train time (s/epoch) |
|---|---|---|---|---|---|---|
| efficientnet_b0 | 83.5% | 96.4% | 0.833 | 4.1 | 23 | 293 |
| convnext_tiny | 89.2% | 98.1% | 0.892 | 27.9 | 55 | 509 |
