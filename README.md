<div align="center">

# Violence Detection

### Finds violence in real-world video (CCTV, phone, broadcast) and says **when** and **where**, with one alarm per incident.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-VideoMAE%20ViT--B-6f42c1)
![Data](https://img.shields.io/badge/trained%20on-4%20datasets%20%C2%B7%205%2C201%20clips-0a7d5a)
![Tests](https://img.shields.io/badge/tests-9%20passing-2ea44f)
![License](https://img.shields.io/badge/license-MIT-blue)

<img src="assets/demo.gif" width="96%" alt="A four-camera control-room wall of held-out test footage. As fights start on CAM 01, CAM 03 and CAM 04, each tile turns red with a VIOLENCE badge and a heatmap on the people fighting, while calm cameras stay green. An incident log on the right fills in as alarms start and clear, and probability traces for all four cameras run along the bottom.">

<sub>Four cameras of held-out test footage (CCTV and street) that the model never saw in training. Every score, alarm and heatmap is real model output.<br>Red tile = incident · heat = where the model is looking (Grad-CAM) · right = incident log · bottom = violence probability per camera.</sub>

[Results](#results) · [Quick start](#quick-start) · [How it works](#how-it-works) · [v1 → v2](#what-changed-from-v1) · [Reproduce](#reproduce)

</div>

---

## Highlights

- **98.2% mean accuracy across four held-out test sets**: Hockey 99.0 · Movies 100 · RWF-2000 CCTV 95.0 · Real-Life Violence 99.0. The v1 approach, re-run on the same splits, averages 67.8%.
- **90.8% on CCTV it has never seen.** Trained without a single CCTV clip (leave-one-dataset-out), it still scores 90.8% on RWF-2000; the v1 approach scores 56.5%.
- **Built for streams.** On 2-minute CCTV videos it catches **97% of incidents**, raising the alarm **1.4 s** after a fight starts, with 3.5× fewer false alarms than the same model trained on Hockey alone.
- **Shows where.** Grad-CAM heatmaps mark the people involved during every alarm.
- **Runs anywhere.** About 20 camera streams in real time on one NVIDIA L4. The lite model runs on a laptop CPU. CLI for files, webcams and RTSP, webhook alerts, evidence clips, a web app and a Python API.

## Results

Every number below comes from test splits that no model saw during training or model selection; validation used a separate 10% of the training data. Full tables: [`results/report.md`](results/report.md). Raw metrics: [`results/summary.json`](results/summary.json).

### 1. One model for every kind of footage

<p align="center"><img src="results/figures/accuracy_by_dataset.png" width="92%" alt="Grouped bar chart of test accuracy on four datasets for four methods. The v1 approach and a Hockey-only VideoMAE drop sharply outside Hockey; VideoMAE trained on all four scores 99, 100, 95 and 99 percent."></p>

| Method | Hockey | Movies | RWF-2000 (CCTV) | Real-Life Violence | Mean |
|---|---:|---:|---:|---:|---:|
| VGG16+LSTM on Hockey (**v1 approach**) | 95.0 | 46.3 | 66.5 | 63.2 | 67.8 |
| CLIP ViT-L/14 zero-shot (threshold set on validation clips) | 95.0 | 51.2 | 81.5 | 95.2 | 80.7 |
| VideoMAE-B on Hockey only | 100.0 | 12.2 | 76.2 | 76.8 | 66.3 |
| VGG16+LSTM on all four datasets | 95.5 | 100.0 | 69.2 | 93.5 | 89.6 |
| VideoMAE-S on all four (lite, 22M params) | 96.5 | 100.0 | 89.0 | 97.8 | 95.8 |
| **VideoMAE-B on all four (this repo, 86M params)** | **99.0** | **100.0** | **95.0** | **99.0** | **98.2** |

<sub>Accuracy % at threshold 0.5 (CLIP: threshold chosen on validation clips). ROC-AUC for every cell is in the full report; the main model's AUCs are 0.995 / 1.000 / 0.992 / 0.999. Test sets: 200 / 41 / 400 / 400 clips.</sub>

A model trained on one dataset learns that footage, not violence. The Hockey-only VideoMAE is perfect on Hockey and gets 12% on Movies. Training on all four fixes it: the combined model matches or beats every single-dataset specialist, even on that specialist's own test set (RWF-2000: 95.0 vs 94.5).

### 2. Footage it has never seen

<p align="center"><img src="results/figures/cross_dataset.png" width="100%" alt="Two heatmaps of cross-dataset accuracy. The VGG16+LSTM grid is mostly pale (46 to 74 percent) away from the diagonal; the VideoMAE grid is darker throughout and reaches 95 to 100 percent when trained on all four."></p>

**Leave-one-dataset-out:** train on three datasets, test on the fourth, which was never seen.

| Never-seen test set | VGG16+LSTM | **VideoMAE-B** | VideoMAE-B trained *with* it |
|---|---:|---:|---:|
| RWF-2000 (CCTV) | 56.5 | **90.8** | 95.0 |
| Real-Life Violence | 71.8 | **91.8** | 99.0 |
| Hockey Fights | 67.0 | **80.5** | 99.0 |
| Movies Fights | 39.0 | **51.2** <sub>(AUC 1.000)</sub> | 100.0 |

Without a single CCTV clip in training, the model still gets 90.8% on real surveillance footage. On unseen Movies it *ranks* every clip correctly (AUC 1.000), but its scores sit on a different scale there, so the default threshold is wrong. On a new kind of camera, recalibrate the threshold on a few labelled clips.

### 3. As a live detector on long videos

Clip benchmarks only ask whether a 2–5 s clip is violent. A deployed system watches hours of footage and has to say *when*, without crying wolf. [`scripts/make_longbench.py`](scripts/make_longbench.py) stitches held-out test clips into 50 two-minute videos (25 CCTV, 25 street), about a quarter of them violent, and the detector streams through them exactly as it would a live feed.

<p align="center"><img src="assets/demo_timeline.png" width="92%" alt="Timeline of violence probability across a two-minute CCTV montage. Grey bands mark real fights; the probability rises above the threshold and red alarm bars appear over each grey band."></p>

| Model | Footage | Frame AUC | Alarm precision | Incidents caught | False alarms / hour | Alarm delay (live) |
|---|---|---:|---:|---:|---:|---:|
| **VideoMAE-B, all four** | CCTV | **0.957** | **86%** | **97%** | **30.7** | 1.4 s |
| **VideoMAE-B, all four** | Street | **0.982** | **94%** | **99%** | **11.6** | 1.6 s |
| VideoMAE-S, all four (lite) | CCTV | 0.924 | 74% | 94% | 64.6 | 1.6 s |
| VideoMAE-B, Hockey only | CCTV | 0.769 | 58% | 83% | 106.5 | 0.8 s |

<p align="center"><img src="results/figures/operating_curves.png" width="100%" alt="Operating curves: incidents detected versus false alarms per hour. The all-four VideoMAE-B curve sits highest and furthest left on both CCTV and street footage; the Hockey-only model sits far right and lower."></p>

**Choosing a threshold.** The default 0.6 favours catching everything. Raising it to **0.9** keeps 95% of CCTV incidents at **19 false alarms per hour** (street footage: 98% at 8 per hour). Because the benchmark is stitched from separate clips, every scene cut is a chance for a spurious alarm; continuous footage from a fixed camera should do better than these numbers.

### 4. Where it looks

<p align="center"><img src="assets/demo_single.gif" width="70%" alt="Close-up of held-out CCTV and street footage: the status bar turns red at VIOLENCE DETECTED and a heatmap highlights the people fighting."></p>

Grad-CAM on the last transformer block's 1,568 space-time tokens. The heat follows the people involved, so a reviewer can check an alarm at a glance.

### 5. Speed

| Hardware | VideoMAE-B (86M) | VideoMAE-S (22M) |
|---|---|---|
| NVIDIA L4, fp16 | 54 windows/s → about **21 live streams** at a 0.4 s stride | 108 windows/s → about **43 streams** |
| Laptop CPU (Intel i5-12500H), fp32 | 2.4 s per window | 1.07 s per window |
| Laptop CPU, `--int8` | 1.63 s per window | **0.80 s per window**, real time at a 1 s stride |

### 6. Ablation: CCTV-style augmentation

Training with simulated surveillance conditions (greyscale night footage, low resolution, blur, sensor noise) mostly pays off where it should, on CCTV. Clip accuracy on RWF-2000 rises from 94.5 to 95.0 and on RLVS from 97.5 to 99.0. On the CCTV long videos, false alarms fall from 42.0 to **30.7 per hour (−27%)**, while incidents caught rise from 95% to 97%.

## Quick start

```bash
git clone https://github.com/harshakalluri1403/Violence-Detection.git
cd Violence-Detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_weights.py                   # 172 MB base + 44 MB lite model
```

**Analyse a video** → incidents in JSON, an annotated MP4 and a timeline plot in `out/`:

```bash
python detect.py examples/hockey_montage.mp4
python detect.py my_clip.mp4 --explain       # + Grad-CAM heatmaps during incidents
python detect.py my_clip.mp4 --clips         # + one evidence clip per incident
```

```text
examples/hockey_montage.mp4: VIOLENCE | 1 event(s) | max p=1.00 | 14.8s video in 42.0s
       3.6s ->     9.6s   peak 1.00
```

<sub>Real output on a laptop CPU. The fight clips in that montage run from 4.9 s to 9.8 s.</sub>

**Watch a live camera** (webcam index or RTSP/HTTP URL), with optional alerts to Slack, Discord, Teams or n8n:

```bash
python detect.py --live 0
python detect.py --live rtsp://user:pass@192.168.1.20/stream1 --headless --log alarms.jsonl
python detect.py --live 0 --webhook https://hooks.slack.com/services/XXX
```

**Web app**: `python app.py` → http://127.0.0.1:7860. Upload a video and get the annotated copy, an interactive timeline and an incidents table.

**Python API**

```python
from vdet.stream import ViolenceDetector

det = ViolenceDetector("checkpoints/vdet_base.pt")     # uses the GPU when there is one
timeline = det.analyze("clip.mp4")
for e in timeline.events(on=0.6, off=0.4):
    print(f"{e.start:.1f}s - {e.end:.1f}s   peak {e.peak:.2f}")
```

No GPU? The lite model with int8 runs faster than real time on a laptop CPU (the 14.8 s sample in 9.5 s on an Intel i5-12500H):

```bash
python detect.py clip.mp4 --checkpoint checkpoints/vdet_small.pt --int8 --stride 1.0
```

## How it works

```mermaid
flowchart LR
    S[Video file<br/>webcam / RTSP] --> B[Sample 5 fps<br/>16-frame ring buffer]
    B --> M[VideoMAE-B<br/>scores the last 3.2 s<br/>every 0.4 s]
    M --> H[Smooth + hysteresis<br/>open at 0.6, close below 0.4]
    H --> I((Incident<br/>start · end · peak))
    I --> J[JSON report]
    I --> V[Annotated MP4<br/>+ Grad-CAM]
    I --> C[Evidence clips]
    I --> W[Webhook /<br/>alarm log]
```

1. **Model.** [VideoMAE](https://arxiv.org/abs/2203.12602) ViT-B/16, pretrained on Kinetics-400 (which already contains punching, wrestling and boxing), with its 400-class head replaced by one violence logit and every layer fine-tuned. 16 frames become 1,568 space-time tokens, and attention runs across all of them, so motion is modelled inside the backbone rather than bolted on afterwards.
2. **Data.** 5,201 clips from four public datasets: sports broadcast (Hockey Fights), films (Movies Fights), real CCTV (RWF-2000) and street/phone video (Real-Life Violence Situations). Each dataset is sampled ∝ √size so the small ones still count.
3. **Training.** AdamW with layer-wise learning-rate decay (0.75), cosine schedule, bf16, 12 epochs on one NVIDIA L4. Augmentation adds a **CCTV branch** (greyscale night footage, low resolution, blur, sensor noise) and random temporal spans, so the model is robust to camera quality and clip speed.
4. **Streaming.** Every 0.4 s the last 3.2 s are scored; scores are smoothed and turned into incidents by hysteresis thresholds, which gives one alarm per fight instead of a flickering label.
5. **Explainability.** Grad-CAM on the last transformer block's tubelet tokens gives an 8 × 14 × 14 space-time heatmap, overlaid during alarms.

## What changed from v1

v1 of this repo was the classic Kaggle notebook: frozen VGG16 frame features into an LSTM, trained on Hockey Fights. Re-reading it closely turned up real problems:

| | v1 (notebook, kept in [`legacy/`](legacy/)) | v2 (this repo) |
|---|---|---|
| Frames | First 20 *consecutive* frames, so only 0.8 s of each 1.6 s clip | 16 frames spread over the whole clip |
| Pre-processing | 0–1 RGB fed to Keras VGG16, which expects mean-subtracted BGR 0–255 | Each backbone gets the normalisation it was trained with |
| Reported accuracy | "~96% validation", which was actually *training* accuracy; the notebook's own test set gives **87.5%** | Fixed held-out test sets, used once |
| Data | Hockey only | 4 datasets, plus cross-dataset and leave-one-dataset-out tests |
| Model | VGG16 (2014) + LSTM | VideoMAE video transformer, fine-tuned end to end |
| Output | One label per clip | Timeline, incidents, alarms, heatmaps, live streams |

The v1 architecture is still here as a baseline (`scripts/baselines.py vgg_lstm`), with its bugs fixed and trained on exactly the same splits, so the comparison above is fair to it.

## Reproduce

Everything runs from scripts. On one spot NVIDIA L4 the full experiment plan takes about 4.5 GPU-hours, roughly US$1.50–2.

```bash
bash scripts/setup_vm.sh                      # packages + the 4 Kaggle datasets (needs ~/.kaggle/kaggle.json)
bash scripts/run_all.sh                       # frames, 10 VideoMAE runs, baselines, long-video benchmark
python scripts/make_report.py                 # results/report.md + figures
python scripts/make_assets.py                 # demo GIF and annotated videos
python -m pytest -q                           # unit tests (CPU, no downloads)
```

`train.py` resumes from its last epoch and `run_all.sh` skips finished work, so a preempted spot VM just runs the same command again. `scripts/vm_startup.sh` and `scripts/watchdog.sh` automate that, and `scripts/finisher.sh` powers the VM off when everything is done. A single experiment:

```bash
python train.py --name my_run --train-sets rwf,rlvs --epochs 10     # --arch small, --no-cctv-aug, --fold k
```

## Project structure

```text
train.py              fine-tune VideoMAE on any mix of datasets (resumable)
detect.py             files, webcams, RTSP: incidents, annotated video, Grad-CAM, clips, webhooks
app.py                Gradio web app
vdet/
  data.py             dataset discovery, splits, frame cache, augmentation (incl. CCTV branch)
  models.py           VideoMAE wrapper, fp16 checkpoints, layer-wise LR decay
  stream.py           sliding-window scorer, smoothing, hysteresis incidents, ViolenceDetector API
  explain.py          spatio-temporal Grad-CAM
  metrics.py          clip metrics + alarm metrics (false alarms/hour, delay)
  viz.py              annotated video, timeline plots, evidence clips, GIFs
scripts/              setup, data prep, run_all, baselines, long-video benchmark, report, assets, VM helpers
tests/                unit tests
results/              report.md, summary.json, figures
legacy/               the original v1 notebook
```

## Limitations and responsible use

- Alarms are prompts for a person to review, not grounds for automatic action. Tune the threshold on the actual camera using the operating curves above.
- Contact sports, dancing and play-fighting can look like fights. Night, top-down and fisheye cameras are under-represented in public data; the CCTV augmentation only simulates them.
- The long-video benchmark is stitched from clips, so its false-alarm rate is a guide, not a guarantee for a given camera.
- Video surveillance is regulated (for example the DPDP Act 2023 in India and the GDPR in the EU). The datasets are research-only; check their terms before commercial use.

## Datasets and credits

[Hockey Fights](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes) (Bermejo Nievas et al., 2011) · [Movies Fights](https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset) · [RWF-2000](https://www.kaggle.com/datasets/vulamnguyen/rwf2000) (Cheng et al., 2021) · [Real-Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) (Soliman et al., 2019) · [VideoMAE](https://github.com/MCG-NJU/VideoMAE) (Tong et al., NeurIPS 2022) via 🤗 Transformers · CLIP (Radford et al., 2021) for the zero-shot baseline.

## License

[MIT](LICENSE) © Pavan Sri Harsha Kalluri
