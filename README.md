<div align="center">

# Violence Detection — Hockey Fights

### Spotting fights in video with transfer learning: VGG16 frames → LSTM sequence model

![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/model-VGG16%20%2B%20LSTM-6f42c1)
![Accuracy](https://img.shields.io/badge/val%20accuracy-~96%25-2ea44f)
![Status](https://img.shields.io/badge/status-learning%20project-lightgrey)

[How it works](#how-it-works) ·
[Results](#results) ·
[Dataset](#dataset) ·
[Run it](#run-it)

</div>

---

A video is just a sequence of frames, and a fight is a pattern *across* those
frames — not something any single frame can prove. This project pairs a
pre-trained image network with a recurrent one to classify short hockey clips
as **fight** or **no fight**: VGG16 turns each frame into a feature vector, and
an LSTM reads the sequence of vectors to make the call.

<p align="center">
<img src="download%20(2).png" width="70%" alt="Sample detection output on a hockey clip">
</p>

## How it works

The pipeline is classic **transfer learning + sequence modelling**, built in the
[`violence-detection.ipynb`](violence-detection.ipynb) notebook:

```
clip ──▶ sample 20 frames ──▶ VGG16 (frozen) ──▶ 20 × 4096 transfer values
                                                        │
                                            cached to disk (slow step, done once)
                                                        │
                                                        ▼
                                      LSTM ──▶ Dense ──▶ softmax ──▶ {fight, no-fight}
```

1. **Sample.** 20 evenly spaced frames are pulled from each clip and resized to
   224×224 — the input size VGG16 expects.
2. **Extract features.** Each frame passes through VGG16 (ImageNet weights),
   producing a 4096-dim *transfer value*. These are cached to disk so the
   expensive CNN pass runs only once.
3. **Model the sequence.** The 20×4096 sequence feeds an LSTM followed by dense
   layers and a 2-way softmax. If violence appears anywhere in the sequence,
   the clip is labelled a fight.

## Results

The LSTM reaches roughly **96% validation accuracy** on a held-out 20% of the
clips. Accuracy/loss curves and sample predictions are at the end of the
notebook.

<p align="center">
<img src="download%20(1).png" width="45%" alt="Training curve">
<img src="download.png" width="45%" alt="Evaluation plot">
</p>

> Accuracy is reported on this dataset's own split. This is a learning project,
> not a benchmarked production detector.

## Dataset

The **Hockey Fight** dataset — 1,000 short clips (`fiXXX` = fight,
`noXXX` = no fight) — lives in [`data/`](data/). The clips are small `.avi`
files, so the repo is self-contained and the notebook runs end to end after
cloning.

## Run it

```bash
git clone https://github.com/harshakalluri1403/Violence-Detection.git
cd Violence-Detection

# a virtual environment is recommended
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install tensorflow opencv-python numpy matplotlib h5py

jupyter notebook violence-detection.ipynb
```

Then run the cells top to bottom. The first full pass caches VGG16 transfer
values to disk; later runs reuse the cache and are much faster.

## Tech stack

TensorFlow / Keras · VGG16 (transfer learning) · LSTM · OpenCV · NumPy ·
Matplotlib · Jupyter
