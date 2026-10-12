## Clip classification — accuracy % (ROC-AUC) on each test set

| Method | Hockey Fights | Movies Fights | RWF-2000 (CCTV) | Real-Life Violence | Mean |
|---|---:|---:|---:|---:|---:|
| VGG16+LSTM on Hockey (original repo approach) | 95.0 <sub>(0.985)</sub> | 46.3 <sub>(0.371)</sub> | 66.5 <sub>(0.717)</sub> | 63.2 <sub>(0.693)</sub> | **67.8** |
| CLIP ViT-L/14 zero-shot (threshold set on validation clips) | 95.0 <sub>(0.999)</sub> | 51.2 <sub>(0.888)</sub> | 81.5 <sub>(0.914)</sub> | 95.2 <sub>(0.988)</sub> | **80.7** |
| VideoMAE-B on Hockey only | 100.0 <sub>(1.000)</sub> | 12.2 <sub>(0.043)</sub> | 76.2 <sub>(0.825)</sub> | 76.8 <sub>(0.897)</sub> | **66.3** |
| VGG16+LSTM on all four datasets | 95.5 <sub>(0.990)</sub> | 100.0 <sub>(1.000)</sub> | 69.2 <sub>(0.784)</sub> | 93.5 <sub>(0.990)</sub> | **89.6** |
| VideoMAE-S on all four (lite, 22M) | 96.5 <sub>(0.992)</sub> | 100.0 <sub>(1.000)</sub> | 89.0 <sub>(0.957)</sub> | 97.8 <sub>(0.997)</sub> | **95.8** |
| **VideoMAE-B on all four (ours, 86M)** | 99.0 <sub>(0.995)</sub> | 100.0 <sub>(1.000)</sub> | 95.0 <sub>(0.992)</sub> | 99.0 <sub>(0.999)</sub> | **98.2** |


## Leave-one-dataset-out — accuracy on a dataset never seen in training

| Held-out dataset | VGG16+LSTM | VideoMAE-B | VideoMAE-B trained *with* it (reference) |
|---|---:|---:|---:|
| Hockey Fights | 67.0 <sub>(0.861)</sub> | **80.5** <sub>(0.960)</sub> | 99.0 <sub>(0.995)</sub> |
| Movies Fights | 39.0 <sub>(0.176)</sub> | **51.2** <sub>(1.000)</sub> | 100.0 <sub>(1.000)</sub> |
| RWF-2000 (CCTV) | 56.5 <sub>(0.661)</sub> | **90.8** <sub>(0.970)</sub> | 95.0 <sub>(0.992)</sub> |
| Real-Life Violence | 71.8 <sub>(0.808)</sub> | **91.8** <sub>(0.964)</sub> | 99.0 <sub>(0.999)</sub> |


## Long-video streaming benchmark

| Model | Footage | Frame AUC | Alarm precision | Alarm recall | False alarms / hour | Live alarm delay |
|---|---|---:|---:|---:|---:|---:|
| VideoMAE-B, all four (ours) | CCTV (RWF-2000) | 0.957 | 85.9% | 96.6% | 30.7 | 1.4 s |
| VideoMAE-B, all four (ours) | Web video (RLVS) | 0.982 | 94.4% | 99.1% | 11.6 | 1.6 s |
| VideoMAE-S, all four (lite) | CCTV (RWF-2000) | 0.924 | 73.7% | 94.1% | 64.6 | 1.6 s |
| VideoMAE-S, all four (lite) | Web video (RLVS) | 0.972 | 87.5% | 98.3% | 28.2 | 2.0 s |
| VideoMAE-B, Hockey only | CCTV (RWF-2000) | 0.769 | 57.7% | 83.1% | 106.5 | 0.8 s |
| VideoMAE-B, Hockey only | Web video (RLVS) | 0.887 | 53.3% | 88.9% | 127.8 | 0.6 s |
| VideoMAE-B, all four, no CCTV aug | CCTV (RWF-2000) | 0.952 | 81.4% | 94.9% | 42.0 | 1.5 s |
| VideoMAE-B, all four, no CCTV aug | Web video (RLVS) | 0.982 | 92.9% | 99.1% | 14.9 | 1.6 s |


## Ablation: CCTV-style augmentation — accuracy %

| CCTV augmentation | Hockey Fights | Movies Fights | RWF-2000 (CCTV) | Real-Life Violence |
|---|---:|---:|---:|---:|
| off | 99.0 | 100.0 | 94.5 | 97.5 |
| on (default) | 99.0 | 100.0 | 95.0 | 99.0 |
