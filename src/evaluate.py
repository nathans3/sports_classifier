"""
Sport Classifier — Evaluation Script
======================================
Evaluates the trained model on the validation set (seed=42, 20%).
Saves eval_grid.png, confusion_matrix.png, top_losses.png.

Usage::
    cd <project_root>
    python3 src/evaluate.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fastai.vision.all import (
    load_learner, get_image_files,
    DataBlock, ImageBlock, CategoryBlock,
    parent_label, RandomSplitter,
    RandomResizedCrop, aug_transforms,
    ClassificationInterpretation, PILImage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_PATH   = MODELS_DIR / "sport_classifier.pkl"


def load_model():
    if not MODEL_PATH.exists():
        print(f"✗  Model not found: {MODEL_PATH}\n   Run: python3 src/train.py")
        sys.exit(1)
    print(f"Loading model from {MODEL_PATH}…")
    return load_learner(MODEL_PATH)


def rebuild_dataloaders():
    print("Rebuilding validation set (seed=42, valid_pct=0.2)…")
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms(),
    ).dataloaders(DATA_DIR, bs=32)


def print_metrics(learn):
    val_loss, val_acc = learn.validate()
    print("\n" + "═" * 44)
    print("  📊  Validation Results")
    print("═" * 44)
    print(f"  Loss     : {val_loss:.4f}")
    print(f"  Accuracy : {val_acc:.4f}  ({val_acc*100:.1f} %)")
    print("═" * 44)


def save_confusion_matrix(interp):
    interp.plot_confusion_matrix(figsize=(5, 4))
    plt.tight_layout()
    out = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out.relative_to(PROJECT_ROOT)}")


def save_top_losses(interp, n=9):
    interp.plot_top_losses(n, nrows=3, figsize=(13, 11))
    out = MODELS_DIR / "top_losses.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out.relative_to(PROJECT_ROOT)}")


def save_validation_grid(learn, max_images=48):
    """Colour-coded grid: green border = correct, red border = wrong."""
    val_ds = learn.dls.valid_ds
    n      = min(max_images, len(val_ds))
    vocab  = learn.dls.vocab
    cols   = 8
    rows   = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes_flat = axes.flat

    correct = wrong = 0
    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue
        _, actual_idx  = val_ds[i]
        actual_label   = vocab[actual_idx]
        img            = PILImage.create(val_ds.items[i])
        pred, pred_idx, probs = learn.predict(img)
        conf       = float(probs[pred_idx]) * 100
        is_correct = (pred == actual_label)
        colour     = "#27ae60" if is_correct else "#e74c3c"
        if is_correct: correct += 1
        else:          wrong   += 1

        ax.imshow(img.resize((200, 150)))
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(4); spine.set_visible(True)

        caption = f"{pred}\n{conf:.0f}%"
        if not is_correct:
            caption += f"\n(actual: {actual_label})"
        ax.set_title(caption, fontsize=7, color=colour, fontweight="bold", pad=2)

    total = correct + wrong
    acc   = correct / total * 100 if total else 0
    legend = [
        mpatches.Patch(color="#27ae60", label=f"Correct ({correct})"),
        mpatches.Patch(color="#e74c3c", label=f"Wrong   ({wrong})"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2,
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Validation Set Predictions  —  {correct}/{total} correct  ({acc:.1f}%)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.4)
    out = MODELS_DIR / "eval_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out.relative_to(PROJECT_ROOT)}")
    return correct, wrong


def print_per_image_log(learn):
    val_ds = learn.dls.valid_ds
    vocab  = learn.dls.vocab
    W = 52
    print()
    print(f"{'No.':<4} {'File':<{W}} {'Actual':<14} {'Predicted':<14} {'Conf%':>6}  Result")
    print("─" * 98)
    correct = wrong = 0
    for i in range(len(val_ds)):
        fp         = val_ds.items[i]
        _, act_idx = val_ds[i]
        actual     = vocab[act_idx]
        img        = PILImage.create(fp)
        pred, pred_idx, probs = learn.predict(img)
        conf = float(probs[pred_idx]) * 100
        ok   = "✓" if pred == actual else "✗"
        if pred == actual: correct += 1
        else:              wrong   += 1
        fname = fp.parent.name + "/" + fp.name
        print(f"{i:<4} {fname:<{W}} {actual:<14} {str(pred):<14} {conf:>5.1f}%  {ok}")
    total = correct + wrong
    print("─" * 98)
    print(f"  Correct: {correct}/{total}  ({correct/total*100:.1f}%)")


def main():
    print("🏀⚽ Sport Classifier — Evaluation")
    print("═" * 44)
    learn     = load_model()
    dls       = rebuild_dataloaders()
    learn.dls = dls
    print_metrics(learn)
    print("\nBuilding ClassificationInterpretation…")
    interp = ClassificationInterpretation.from_learner(learn)
    print("Saving plots…")
    save_confusion_matrix(interp)
    save_top_losses(interp)
    print("Generating validation grid…")
    correct, wrong = save_validation_grid(learn)
    answer = input("\nPrint per-image prediction log? [y/N] ").strip().lower()
    if answer == "y":
        print_per_image_log(learn)
    total = correct + wrong
    print(f"\n✅  Done!  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print("   Open: open models/eval_grid.png models/confusion_matrix.png models/top_losses.png")


if __name__ == "__main__":
    main()
