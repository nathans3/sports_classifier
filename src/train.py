"""
Sport Classifier — Model Training Script
==========================================
Fine-tunes ResNet18 to classify basketball vs soccer players.
Same pipeline as the fastai Chapter 2 Bear Classifier.

Usage::
    cd <project_root>
    python3 src/train.py
"""
import sys, os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock,
    get_image_files, parent_label, RandomSplitter,
    RandomResizedCrop, aug_transforms,
    vision_learner, resnet18, accuracy,
    ClassificationInterpretation,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_FILE   = "sport_classifier.pkl"

BATCH_SIZE = 32
IMAGE_SIZE = 224
VALID_PCT  = 0.2
SEED       = 42
EPOCHS     = 4


def build_dataloaders(data_dir: Path, bs: int = BATCH_SIZE):
    """Build DataLoaders with the DataBlock API.
    parent_label maps folder name -> label:
      data/basketball/ -> "basketball"
      data/soccer/     -> "soccer"
    """
    print("Building DataLoaders…")
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=RandomSplitter(valid_pct=VALID_PCT, seed=SEED),
        item_tfms=RandomResizedCrop(IMAGE_SIZE, min_scale=0.5),
        batch_tfms=aug_transforms(),
    ).dataloaders(data_dir, bs=bs)
    print(f"  Classes          : {dls.vocab}")
    print(f"  Training images  : {len(dls.train_ds)}")
    print(f"  Validation images: {len(dls.valid_ds)}")
    return dls


def train_model(dls):
    """Fine-tune ResNet18 (pretrained on ImageNet) with transfer learning.
    fine_tune(4): epoch 0 = head only, epochs 1-4 = all layers.
    """
    print("\nCreating learner (ResNet18 backbone, pretrained on ImageNet)…")
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    print(f"Fine-tuning for {EPOCHS} epochs…\n")
    learn.fine_tune(EPOCHS)
    return learn


def evaluate_model(learn):
    print("\n" + "═" * 48)
    print("  📊  Model Evaluation")
    print("═" * 48)
    val_loss, val_acc = learn.validate()
    print(f"\n  Validation Loss     : {val_loss:.4f}")
    print(f"  Validation Accuracy : {val_acc:.4f}  ({val_acc*100:.1f} %)\n")
    interp = ClassificationInterpretation.from_learner(learn)
    MODELS_DIR.mkdir(exist_ok=True)
    interp.plot_confusion_matrix(figsize=(5, 4))
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved → models/confusion_matrix.png")
    interp.plot_top_losses(9, nrows=3, figsize=(12, 10))
    plt.savefig(MODELS_DIR / "top_losses.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved → models/top_losses.png")


def export_model(learn) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / MODEL_FILE
    learn.export(path)
    print(f"\n  Model exported → {path}  ({path.stat().st_size/1_048_576:.1f} MB)")
    return path


def main():
    print("🏀⚽ Sport Classifier — Model Training")
    print("═" * 48)
    n_bball  = len(get_image_files(DATA_DIR/"basketball")) if (DATA_DIR/"basketball").exists() else 0
    n_soccer = len(get_image_files(DATA_DIR/"soccer"))     if (DATA_DIR/"soccer").exists()     else 0
    print(f"  Basketball images : {n_bball}")
    print(f"  Soccer images     : {n_soccer}")
    if n_bball == 0 or n_soccer == 0:
        print("\n✗  No images found. Run: python3 src/download_data.py")
        sys.exit(1)
    dls   = build_dataloaders(DATA_DIR)
    learn = train_model(dls)
    evaluate_model(learn)
    path  = export_model(learn)
    print("\n✅  Training complete!")
    print(f"   Model → {path}")
    print("   Next: python3 src/predict.py <image_path>")


if __name__ == "__main__":
    main()
