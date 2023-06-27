import os
import warnings

import gradio as gr
import lightning as L
import numpy as np
from lightning.app.components.serve import ServeGradio
from PIL import Image

from lang_sam import LangSAM
from lang_sam import SAM_MODELS
from lang_sam.utils import draw_image
from lang_sam.utils import load_image

warnings.filterwarnings("ignore")


def build_model(sam_type="vit_h"):
    model = LangSAM(sam_type)

    return model


model = build_model(sam_type="vit_h")


def predict(sam_type, box_threshold, text_threshold, image_path, text_prompt):
    print("Predicting... ", sam_type, box_threshold, text_threshold, image_path, text_prompt)
    if sam_type != model.sam_type:
        model.build_sam(sam_type)
    image_pil = load_image(image_path)
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    return image


with gr.Blocks() as demo:
    inputs = [
        gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM model", value="vit_h"),
        gr.Slider(0, 1, value=0.3, label="Box threshold"),
        gr.Slider(0, 1, value=0.25, label="Text threshold"),
        gr.Image(type="filepath", label='Image'),
        gr.Textbox(lines=1, label="Text Prompt"),
    ]
    outputs = [gr.outputs.Image(type="pil", label="Output Image")]
    btn = gr.Button(label="Run")
    btn.click(fn=predict, inputs=inputs, outputs=outputs)

    examples = [
        [
            'vit_h',
            0.36,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
            "kiwi",
        ],
        [
            'vit_h',
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "car.jpeg"),
            "wheel",
        ],
        [
            'vit_h',
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "food.jpg"),
            "food",
        ],
    ]

demo.launch(enable_queue=False, share=False, debug=True)

# app = L.LightningApp(LitGradio())
