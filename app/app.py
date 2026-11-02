"""NutriSnap Gradio demo: photo -> dish -> portion -> calories & macros.

Run locally:  python app/app.py   (http://localhost:7860)
On Hugging Face Spaces this file is the entry point (see README "Deploy").
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nutrition import PORTION_MULTIPLIERS, load_db  # noqa: E402
from src.portion import estimate_grams_from_reference  # noqa: E402
from src.predict import Predictor  # noqa: E402

CKPT = Path(os.environ.get("NUTRISNAP_CKPT", ROOT / "app" / "model" / "nutrisnap.pt"))
EXAMPLES_DIR = ROOT / "app" / "examples"

DB = load_db()
PREDICTOR = Predictor(CKPT) if CKPT.exists() else None
CHOICES = [(DB.get(c).display_name, c) for c in DB.classes]
MACRO_COLORS = {"Protein": "#2a9d8f", "Carbs": "#e9c46a", "Fat": "#e76f51"}

DISCLAIMER = (
    "⚠️ **Estimates only.** Calories and macros are approximations based on a typical recipe "
    "(USDA FoodData Central) and an estimated portion. Real values vary with ingredients, oil, "
    "sauces and portion size. Not medical or dietary advice."
)

CSS = """
.gradio-container {max-width: 980px !important; margin: auto;}
#kcal-card {text-align:center; border-radius:16px; padding:14px 8px;
            background: linear-gradient(135deg, #264653 0%, #2a9d8f 100%); color: white;}
#kcal-card .big {font-size: 2.6rem; font-weight: 800; line-height: 1.1;}
#kcal-card .macros {display:flex; justify-content:space-around; margin-top:8px; font-size:1rem;}
#kcal-card .macros b {display:block; font-size:1.3rem;}
@media (max-width: 640px) { #kcal-card .big {font-size: 2.1rem;} }
"""


# ----------------------------------------------------------------------------- helpers
def _card(name: str, grams: float, kcal: float, p: float, c: float, f: float, note: str = "") -> str:
    return f"""
<div id="kcal-card">
  <div>{name} · {grams:.0f} g</div>
  <div class="big">{kcal:.0f} kcal</div>
  <div class="macros">
    <div><b>{p:.0f} g</b>protein</div><div><b>{c:.0f} g</b>carbs</div><div><b>{f:.0f} g</b>fat</div>
  </div>
  {f'<div style="margin-top:6px;font-size:.85rem;opacity:.9">{note}</div>' if note else ''}
</div>"""


def _nutrition_outputs(label: str, grams: float):
    est = DB.estimate(label, float(grams))
    info = DB.get(label)
    note = "" if info.match == "exact" else f"nutrition source: {info.match} match ({info.source})"
    mk = est.macro_kcal()
    chart = pd.DataFrame(
        {"macro": ["Protein", "Carbs", "Fat"],
         "kcal": [round(mk["protein"]), round(mk["carbs"]), round(mk["fat"])],
         "grams": [est.protein_g, est.carbs_g, est.fat_g]}
    )
    return _card(est.display_name, est.grams, est.kcal, est.protein_g, est.carbs_g, est.fat_g, note), chart


def _preset_grams(label: str, preset: str) -> float:
    return round(DB.get(label).serving_g * PORTION_MULTIPLIERS[preset])


# ---------------------------------------------------------------------------- callbacks
def analyze(image: Image.Image | None, preset: str):
    if image is None:
        raise gr.Error("Please upload or take a photo first.")
    if PREDICTOR is None:
        raise gr.Error(f"Model checkpoint not found at {CKPT}. Train one or set NUTRISNAP_CKPT.")
    img = PREDICTOR.load_image(image)
    preds = PREDICTOR.topk(img, k=3)
    top = preds[0].label
    cam = PREDICTOR.explain(img, top)
    grams = _preset_grams(top, preset)
    card, chart = _nutrition_outputs(top, grams)
    conf = {p.display_name: p.confidence for p in preds}
    warn = ("🤔 Low confidence: please check the dish name below and correct it if needed."
            if preds[0].confidence < 0.4 else "")
    return (conf, cam, gr.update(value=top), gr.update(value=grams), card, chart, warn, img)


def on_label_change(label: str, preset: str, image: Image.Image | None):
    """User corrected the dish: re-scale the portion to that dish and re-explain the photo."""
    if not label:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    grams = _preset_grams(label, preset)
    card, chart = _nutrition_outputs(label, grams)
    cam = PREDICTOR.explain(image, label) if (image is not None and PREDICTOR) else gr.skip()
    return gr.update(value=grams), card, chart, cam


def on_preset_change(label: str, preset: str):
    if not label:
        return gr.skip(), gr.skip(), gr.skip()
    grams = _preset_grams(label, preset)
    card, chart = _nutrition_outputs(label, grams)
    return gr.update(value=grams), card, chart


def on_grams_change(label: str, grams: float):
    if not label:
        return gr.skip(), gr.skip()
    return _nutrition_outputs(label, grams)


def on_reference(image: Image.Image | None, label: str, ref: str):
    if image is None or not label:
        raise gr.Error("Analyze a photo first.")
    res = estimate_grams_from_reference(image, label, ref)
    if res is None:
        raise gr.Error("Couldn't find the reference object. Place it flat next to the plate, fully visible.")
    card, chart = _nutrition_outputs(label, res.grams)
    msg = (f"Reference found → food area ≈ {res.food_area_cm2:.0f} cm² → ≈ {res.grams:.0f} g "
           f"(rough estimate; assumes typical food thickness).")
    return gr.update(value=round(res.grams)), card, chart, res.debug_image, msg


# --------------------------------------------------------------------------------- UI
with gr.Blocks(title="NutriSnap: food photo to calories") as demo:
    gr.Markdown("# 🍽️ NutriSnap\nSnap a meal → dish recognition → portion → calories & macros.")
    gr.Markdown(DISCLAIMER)
    img_state = gr.State(None)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            image = gr.Image(label="Meal photo", type="pil", sources=["upload", "webcam", "clipboard"],
                             height=320, webcam_options=gr.WebcamOptions(mirror=False))
            preset = gr.Radio(list(PORTION_MULTIPLIERS), value="medium", label="Portion size")
            btn = gr.Button("Analyze", variant="primary", size="lg")
            examples = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
            if examples:
                gr.Examples([[str(p)] for p in examples], inputs=[image], label="Try an example")
        with gr.Column(scale=1, min_width=300):
            warn = gr.Markdown()
            top3 = gr.Label(num_top_classes=3, label="Top-3 predictions")
            label = gr.Dropdown(CHOICES, label="Dish (correct it if wrong)", filterable=True)
            grams = gr.Slider(0, 1200, value=0, step=5, label="Portion (grams)")
            card = gr.HTML()
            chart = gr.BarPlot(x="macro", y="kcal", color="macro", color_map=MACRO_COLORS,
                               title="Energy by macronutrient (kcal)", height=220, show_label=False)

    with gr.Accordion("🔍 What did the model look at? (Grad-CAM)", open=True):
        cam = gr.Image(label="Grad-CAM overlay", height=300, interactive=False)
        gr.Markdown("Warm colours show the regions that most influenced the predicted dish.")

    with gr.Accordion("📏 Estimate portion from a reference object (experimental)", open=False):
        gr.Markdown("Put a **credit card** or a **US quarter** flat next to the plate, shot from above. "
                    "The food area is measured relative to the reference and converted to grams.")
        ref = gr.Radio(["credit card", "coin (US quarter)"], value="credit card", label="Reference object")
        ref_btn = gr.Button("Estimate grams from reference")
        ref_msg = gr.Markdown()
        ref_dbg = gr.Image(label="Detected regions", height=260, interactive=False)

    btn.click(analyze, [image, preset], [top3, cam, label, grams, card, chart, warn, img_state])
    label.input(on_label_change, [label, preset, img_state], [grams, card, chart, cam])
    preset.input(on_preset_change, [label, preset], [grams, card, chart])
    grams.release(on_grams_change, [label, grams], [card, chart])
    ref_btn.click(on_reference, [img_state, label, ref], [grams, card, chart, ref_dbg, ref_msg])

    gr.Markdown("Model: EfficientNet-B0 fine-tuned on Food-101 · Nutrition: USDA FoodData Central (FNDDS) · "
                "[Source code](https://github.com/harshskapadia/Calories-)")

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                theme=gr.themes.Soft(primary_hue="teal"), css=CSS)
