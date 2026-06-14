"""
Sport Classifier — Prediction Script
======================================
Classify an image as a BASKETBALL or SOCCER player.

Usage::
    python3 src/predict.py data/basketball/img_0001_abc12345.jpg
    python3 src/predict.py ~/Downloads/athlete.jpg
"""
import sys, os, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
from fastai.vision.all import load_learner, PILImage

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "sport_classifier.pkl"


def load_model(model_path: Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\n  Run: python3 src/train.py")
    print(f"Loading model from {model_path}…")
    return load_learner(model_path)


def predict_image(learn, image_path: Path) -> tuple[str, float, dict[str, float]]:
    image_path = Path(image_path)
    if not image_path.exists():
        hints = [f"Image not found: {image_path}"]
        for cat in ("basketball", "soccer"):
            folder = Path("data") / cat
            if folder.exists():
                ex = sorted(f for f in folder.iterdir() if not f.name.startswith("."))[:3]
                if ex:
                    hints.append(f"  Available in data/{cat}/: " + ", ".join(f.name for f in ex) + " …")
        raise FileNotFoundError("\n".join(hints))
    img = PILImage.create(image_path)
    label, idx, probs = learn.predict(img)
    confidence = float(probs[idx]) * 100
    all_probs  = {cls: float(probs[i]) * 100 for i, cls in enumerate(learn.dls.vocab)}
    return str(label), confidence, all_probs


def format_results(image_path, label, confidence, all_probs) -> str:
    icon = "🏀" if label == "basketball" else "⚽"
    lines = ["",
        f"  📸 Image      : {Path(image_path).name}",
        f"  {icon} Prediction : {label}",
        f"  📊 Confidence : {confidence:.1f}%",
        "", "  Class probabilities:",
    ]
    for cls, prob in sorted(all_probs.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        lines.append(f"    {cls:<14}: {prob:5.1f}%  {bar}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Predict BASKETBALL or SOCCER player.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n  python3 src/predict.py data/basketball/img_0001_abc.jpg")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    try:
        learn = load_model(Path(args.model))
    except FileNotFoundError as e:
        print(f"\n✗  {e}"); sys.exit(1)
    try:
        label, conf, all_probs = predict_image(learn, Path(args.image_path))
    except FileNotFoundError as e:
        print(f"\n✗  {e}"); sys.exit(1)
    except Exception as e:
        print(f"\n✗  Prediction failed: {e}"); sys.exit(1)

    print(format_results(args.image_path, label, conf, all_probs))
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.1f}%")


if __name__ == "__main__":
    main()
