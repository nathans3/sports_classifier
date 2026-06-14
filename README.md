# 🏀⚽ Sport Classifier

A fastai image classifier that distinguishes **basketball players** 🏀 from **soccer players** ⚽.
Built on the Chapter 2 Bear Classifier workflow from *Deep Learning for Coders with fastai and PyTorch*.

## Quick Start

```bash
source venv/bin/activate
python3 src/download_data.py   # ~400 images
python3 src/train.py           # fine-tune ResNet18
python3 src/evaluate.py        # validation grid + plots
python3 src/predict.py data/basketball/img_0001_abc123.jpg
```

## Commands

| Command | What it does |
|---------|-------------|
| `python3 src/download_data.py` | Download ~200 images per sport via DuckDuckGo |
| `python3 src/train.py` | Train ResNet18 with transfer learning |
| `python3 src/evaluate.py` | Generate eval_grid.png, confusion_matrix.png, top_losses.png |
| `python3 src/predict.py <image>` | Classify a single image |
