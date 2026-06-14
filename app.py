import gradio as gr
from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

learn = load_learner(Path("models/sport_classifier.pkl"))
labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {str(labels[i]): float(probs[i]) for i in range(len(labels))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="🏀⚽ Sport Classifier",
    description="Upload a photo and the model will classify it as **basketball** or **soccer**.",
    examples=[],
).launch()
