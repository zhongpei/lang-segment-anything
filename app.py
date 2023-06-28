import os
import warnings
import gradio as gr
import numpy as np
from PIL import Image
import time
from lang_sam import LangSAM
from lang_sam import SAM_MODELS
from lang_sam.utils import draw_image
from lang_sam.utils import load_image
import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline
import random
import string

warnings.filterwarnings("ignore")


def init_sam(sam_type="vit_h"):
    model = LangSAM(sam_type)

    return model


def sd_init(sd_ckpt="stabilityai/stable-diffusion-2-inpainting"):
    # Build Stable Diffusion Inpainting

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_ckpt, torch_dtype=torch.float16, resume_download=True,
    )
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to('cuda')
    return pipe


sd_pipe = None
sam_model = init_sam(sam_type="vit_h")


def multi_mask2one_mask(masks):
    masks_length, _, h, w = masks.shape
    for i, mask in enumerate(masks):
        mask_image = mask.cpu().numpy().reshape(h, w, 1)
        whole_mask = mask_image if i == 0 else whole_mask + mask_image
    whole_mask = np.where(whole_mask == False, 0, 255)
    return whole_mask


def gen_whole_mask(masks, iterations=2):
    ori_mask = multi_mask2one_mask(masks=masks)

    # Dilate the mask region to promote the following erasing quality
    mask_img = ori_mask[:, :, 0].astype('uint8')
    kernel = np.ones((5, 5), np.int8)
    whole_mask = cv2.dilate(
        mask_img, kernel, iterations=iterations
    )
    return whole_mask


def numpy2PIL(numpy_image):
    out = Image.fromarray(numpy_image.astype(np.uint8))
    return out


def sd_inpainting(pipe, ori_input, whole_mask, prompt: str):
    # Data preparation
    sd_img = Image.open(ori_input).convert("RGB")
    w, h = sd_img.size
    sd_mask_img = numpy2PIL(
        numpy_image=whole_mask
    ).convert("RGB")
    # sd_mask_img = sd_mask_img.resize((w_resize, h_resize))
    # sd_mask_img.save(os.path.join(self.args.outdir, f'whole_mask.png'))
    # sd_img = sd_img.resize((w_resize, h_resize))

    if w % 64 != 0 or h % 64 != 0:
        w_resize, h_resize = w // 64 * 64, h // 64 * 64
        sd_img = sd_img.resize((w_resize, h_resize))
        sd_mask_img = sd_mask_img.resize((w_resize, h_resize))
    else:
        w_resize, h_resize = w, h

    print(f"{w} {h} Resize to {w_resize}x{h_resize}")

    # Stable Diffusion for Erasing
    if prompt is None:
        prompt = 'No text, clean background'
    image = pipe(
        prompt=prompt,
        image=sd_img,
        mask_image=sd_mask_img,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=h_resize,
        width=w_resize,
    ).images[0]

    # Save image
    if w != w_resize or h != h_resize:
        image = image.resize((w, h))
    return image


def sd_erase(ori_input, prompt: str, sam_prompt: str, box_threshold, text_threshold):
    image_pil = load_image(ori_input)
    all_boxes = []
    for p in sam_prompt.split("|"):
        boxes, _, _ = sam_model.predict_dino(image_pil, p, box_threshold, text_threshold)
        print(f"Boxes: {boxes}")
        all_boxes.append(boxes)

    boxes = torch.cat(all_boxes, dim=0)
    print(f"all Boxes: {boxes}")
    masks = sam_model.predict_sam(image_pil, boxes=boxes)

    whole_mask = gen_whole_mask(masks)

    global sd_pipe
    if sd_pipe is None:
        sd_pipe = sd_init()

    image = sd_inpainting(sd_pipe, ori_input, whole_mask, prompt=prompt)
    return image


def segment_sam(sam_type, box_threshold, text_threshold, image_path, text_prompt):
    print("Predicting... ", sam_type, box_threshold, text_threshold, image_path, text_prompt)
    if sam_type != sam_model.sam_type:
        sam_model.build_sam(sam_type)
    image_pil = load_image(image_path)

    boxes, logits, phrases = sam_model.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
    masks = torch.tensor([])
    if len(boxes) > 0:
        masks = sam_model.predict_sam(image_pil, boxes)
        draw_masks = masks.squeeze(1)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, draw_masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    return image, boxes


def crop_sam_image(box_threshold: float, text_threshold: float, image_path, sam_prompt: str, ):
    output_dir = "output"
    fn_prefix = f"{sam_prompt.replace(' ', '_')}_{random.randint(0, 1000)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_pil = load_image(image_path)
    all_boxes = []
    for p in sam_prompt.split("|"):
        boxes, _, _ = sam_model.predict_dino(image_pil, p, box_threshold, text_threshold)
        print(f"Boxes: {boxes}")
        all_boxes.append(boxes)
        print("Predicting... ", sam_type, box_threshold, text_threshold, image_path, p)
    boxes = torch.cat(all_boxes, dim=0)

    output_images = []
    input_image = cv2.imread(image_path)
    org_height, org_width, _ = input_image.shape
    img_boxes = boxes.to(torch.int64).tolist()
    for i in range(len(img_boxes)):
        x, y, xm, ym = img_boxes[i]

        print(f"Box {i}: {x}  {xm} {y} {ym} ")
        cropped_image = input_image[y:ym, x:xm]

        filename = os.path.join(output_dir, f"{fn_prefix}-{i}.jpg")

        cv2.imwrite(filename, cropped_image)
        output_images.append(filename)
    return output_images


with gr.Blocks() as demo:
    with gr.Row():
        sam_type = gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM model", value="vit_h")
        sam_box_threshold = gr.Slider(0, 1, value=0.3, label="Box threshold")
        sam_text_threshold = gr.Slider(0, 1, value=0.25, label="Text threshold")
    with gr.Row():
        input_file = gr.Image(type="filepath", label='Image')

    with gr.Row():
        sam_prompt = gr.Textbox(lines=1, label="Text segment Prompt", value="text")
        sd_prompt = gr.Textbox(lines=1, label="Text erase Prompt", value="No text, clean background")
    with gr.Row():
        sam_btn = gr.Button(label="Sam Segment", value="sam")
        crop_btn = gr.Button(label="Crop", value="crop")
        erase_btn = gr.Button(label="Erase", value="erase")

    with gr.Row():
        mark_image = gr.outputs.Image(type="pil", label="Output Image")
        output_image = gr.outputs.Image(type="pil", label="Output Image")
        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(object_fit="contain", height="auto")

    boxes = gr.State(label="Masks")

    crop_btn.click(
        fn=crop_sam_image,
        inputs=[sam_box_threshold, sam_text_threshold, input_file, sam_prompt],
        outputs=gallery
    )

    sam_btn.click(
        fn=segment_sam,
        inputs=[sam_type, sam_box_threshold, sam_text_threshold, input_file, sam_prompt],
        outputs=[mark_image, boxes]
    )

    erase_btn.click(
        fn=sd_erase,
        inputs=[input_file, sd_prompt, sam_prompt, sam_box_threshold, sam_text_threshold],
        outputs=[output_image]
    )

demo.launch(enable_queue=False, share=False, debug=True)
