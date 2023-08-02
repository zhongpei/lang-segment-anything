import json
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
from diffusers import StableDiffusionXLInpaintPipeline
import random
import glob
import tqdm
import matplotlib.pyplot as plt
import io

warnings.filterwarnings("ignore")

sd_pipe = None
current_sd_model = None
sam_model = None

SD_MODELS = {
    "Stable Diffusion 2 Inpainting": "stabilityai/stable-diffusion-2-inpainting",
    "1.5": "runwayml/stable-diffusion-v1-5",
    "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
}


def sd_init(sd_ckpt_name="Stable Diffusion 2 Inpainting"):
    global sd_pipe
    global current_sd_model
    global SD_MODELS

    # Build Stable Diffusion Inpainting
    sd_ckpt = SD_MODELS.get(sd_ckpt_name, "stabilityai/stable-diffusion-2-inpainting")

    if sd_pipe is not None:
        if current_sd_model != sd_ckpt:
            del sd_pipe
        else:
            return sd_pipe
    current_sd_model = sd_ckpt
    if sd_ckpt_name == "SDXL":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            sd_ckpt, torch_dtype=torch.float16, variant="fp16", resume_download=True, safety_checker=None
        )
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_ckpt, torch_dtype=torch.float16, resume_download=True, safety_checker=None
        )
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to('cuda')
    return pipe


def get_sam_model(sam_type="vit_h"):
    global sam_model
    if sam_model is None:
        sam_model = LangSAM(sam_type)
    return sam_model


def remove_border(input_dir: str, output_dir: str, image_path: str):
    input_fns = get_images_from_dir(input_dir, image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_fns = []
    for image_file in tqdm.tqdm(input_fns, total=len(input_fns), desc="remove border"):
        output_fn = remove_border_one(image_file, output_dir)
        output_fns.append(output_fn)

    return output_fns


def remove_border_one(image_file: str, output_dir: str):
    image = cv2.imread(image_file)  # 读取图片

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = gray_image.astype(np.uint8)

    # 使用自适应阈值计算
    threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 进行边缘检测
    # edges = cv2.Canny(threshold_image, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    cropped_image = image[y:y + h, x:x + w]
    output_fn = os.path.join(output_dir, os.path.basename(image_file))

    cv2.imwrite(output_fn, cropped_image)
    return output_fn


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


def sd_inpainting(
        pipe,
        ori_input,
        whole_mask,
        prompt: str,
        negative_prompt: str = None,
        steps=50,
        cfg=7.5,
        denoise=1
):
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
        negative_prompt=negative_prompt,
        image=sd_img,
        mask_image=sd_mask_img,
        num_inference_steps=steps,
        guidance_scale=cfg,
        height=h_resize,
        width=w_resize,
        strength=denoise,
    ).images[0]

    # Save image
    if w != w_resize or h != h_resize:
        image = image.resize((w, h))
    return image


def sd_erase(
        input_dir,
        output_dir,
        ori_input,
        prompt: str,
        sam_prompt: str,
        box_threshold,
        text_threshold,
        box_json,
        use_sam_box,
        negative_prompt: str = None,
        sd_model_name: str = None,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fns = get_images_from_dir(input_dir, ori_input)
    output_fns = []
    for image_file in tqdm.tqdm(fns, total=len(fns), desc="sd erase"):
        image = sd_erase_one(
            image_file,
            prompt,
            sam_prompt,
            box_threshold,
            text_threshold,
            box_json,
            use_sam_box,
            negative_prompt=negative_prompt,
            sd_model_name=sd_model_name
        )
        output_fn = os.path.join(output_dir, os.path.basename(image_file))
        image.save(output_fn)
        output_fns.append(output_fn)
    return output_fns


def sd_erase_one(
        ori_input,
        prompt: str,
        sam_prompt: str,
        box_threshold,
        text_threshold,
        box_json,
        use_sam_box=False,
        negative_prompt=None,
        sd_model_name=None,
):
    image_pil = load_image(ori_input)
    all_boxes = []
    if box_json is not None and use_sam_box:
        all_boxes.append(torch.Tensor(box_json))
    for p in sam_prompt.split("|"):
        boxes, _, _ = get_sam_model().predict_dino(image_pil, p, box_threshold, text_threshold)
        print(f"Boxes: {boxes}")
        all_boxes.append(boxes)

    print(f"all Boxes: {all_boxes}")
    boxes = torch.cat(all_boxes, dim=0)

    masks = get_sam_model().predict_sam(image_pil, boxes=boxes)

    whole_mask = gen_whole_mask(masks)

    sd_pipe = sd_init(sd_model_name)

    image = sd_inpainting(sd_pipe, ori_input, whole_mask, prompt=prompt, negative_prompt=negative_prompt)
    return image


def segment_sam(sam_type, box_threshold, text_threshold, image_path, text_prompt):
    print("Predicting... ", sam_type, box_threshold, text_threshold, image_path, text_prompt)
    if sam_type != get_sam_model().sam_type:
        sam_model.build_sam(sam_type)
    image_pil = load_image(image_path)

    boxes, logits, phrases = get_sam_model().predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
    draw_masks = torch.tensor([])

    if len(boxes) > 0:
        masks = get_sam_model().predict_sam(image_pil, boxes)
        draw_masks = masks.squeeze(1)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, draw_masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    print(f"{boxes.to(torch.int64).tolist()}")

    return image, boxes.to(torch.int64).tolist()


def segment_sam(sam_type, box_threshold, text_threshold, image_path, text_prompt):
    print("Predicting... ", sam_type, box_threshold, text_threshold, image_path, text_prompt)
    if sam_type != get_sam_model().sam_type:
        sam_model.build_sam(sam_type)
    image_pil = load_image(image_path)

    boxes, logits, phrases = get_sam_model().predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
    draw_masks = torch.tensor([])

    if len(boxes) > 0:
        masks = get_sam_model().predict_sam(image_pil, boxes)
        draw_masks = masks.squeeze(1)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, draw_masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    print(f"{boxes.to(torch.int64).tolist()}")

    return image, boxes.to(torch.int64).tolist()


def draw_boxes(image_path, box_x, box_y, box_w, box_h, box_json):
    image_pil = load_image(image_path)
    image_array = np.asarray(image_pil)
    draw_masks = torch.tensor([])
    raw_boxes = [[box_x, box_y, box_x + box_w, box_y + box_h]]
    if box_json is not None:
        raw_boxes.extend(box_json)

    boxes = torch.tensor(
        raw_boxes
    )
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(["box", ] * len(raw_boxes), (1.0,) * len(raw_boxes))]
    image = draw_image(image_array, draw_masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    return image


def sd_erase_box(ori_input, prompt, box_x, box_y, box_w, box_h, box_json):
    image_pil = load_image(ori_input)
    raw_boxes = [[box_x, box_y, box_x + box_w, box_y + box_h]]
    if box_json is not None:
        raw_boxes.extend(box_json)
    boxes = torch.tensor(raw_boxes)
    print(f"all Boxes: {boxes}")
    masks = get_sam_model().predict_sam(image_pil, boxes=boxes)
    whole_mask = gen_whole_mask(masks)

    sd_pipe = sd_init()

    image = sd_inpainting(sd_pipe, ori_input, whole_mask, prompt=prompt)
    return image


def get_images_from_dir(input_dir: str, input_file: str) -> list[str]:
    input_fns = []
    if input_dir is not None and os.path.exists(input_dir) and os.path.isdir(input_dir):
        input_fns.extend(glob.glob(os.path.join(input_dir, "*.jpg")))
        input_fns.extend(glob.glob(os.path.join(input_dir, "*.png")))
        input_fns.extend(glob.glob(os.path.join(input_dir, "*.jpeg")))

    if input_file is not None and os.path.exists(input_file) and os.path.isfile(input_file):
        input_fns.append(input_file)

    print(f"Input files count: {len(input_fns)}")
    return input_fns


def crop_sam_image(
        input_dir,
        output_dir: str,
        box_threshold: float,
        text_threshold: float,
        image_path,
        sam_prompt: str,
):
    output_images = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_fns = get_images_from_dir(input_dir, image_path)

    print(f"Input files: {input_fns}")
    for fn in input_fns:
        images = crop_sam_image_one(
            output_dir=output_dir,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            image_path=fn,
            sam_prompt=sam_prompt,
        )
        output_images.extend(images)

    return output_images


def update_image(image_path):
    image = cv2.imread(image_path)
    org_height, org_width, _ = image.shape

    plt.imshow(image)
    # 将图像转换为PIL格式
    buf = io.BytesIO()  # 创建内存存储区
    plt.savefig(buf, format='png')  # 将图像保存到内存中
    buf.seek(0)  # 将文件指针移到开头

    pil_image = Image.open(buf)  # 从内存中读取图像数据
    image = Image.fromarray(np.uint8(pil_image)).convert("RGB")

    return image


def crop_sam_image_one(

        output_dir: str,
        box_threshold: float,
        text_threshold: float,
        image_path,
        sam_prompt: str,
):
    fn_prefix = f"{sam_prompt.replace(' ', '_')}_{random.randint(0, 1000)}"
    image_pil = load_image(image_path)
    all_boxes = []
    for p in sam_prompt.split("|"):
        boxes, _, _ = get_sam_model().predict_dino(image_pil, p, box_threshold, text_threshold)
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


import json

with gr.Blocks() as demo:
    with gr.Row():
        sam_type = gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM model", value="vit_h")
        sam_box_threshold = gr.Slider(0, 1, value=0.3, label="Box threshold")
        sam_text_threshold = gr.Slider(0, 1, value=0.25, label="Text threshold")
    with gr.Row():
        sd_model = gr.Dropdown(choices=list(SD_MODELS.keys()), label="SD model", value="Stable Diffusion 2 Inpainting")
        sd_cfg = gr.Slider(0, 20, value=7.5, label="SD cfg", step=0.1)
        sd_denoise = gr.Slider(0, 1, value=0.75, label="Denoise", step=0.01)
    with gr.Row():
        with gr.Column(scale=4):
            input_file = gr.Image(type="filepath", label='Image')
        with gr.Column(scale=1):
            input_dir = gr.Textbox(lines=1, label="Input dir", value="")
            output_dir = gr.Textbox(lines=1, label="Output dir", value="output")
            remove_border_btn = gr.Button(label="Remove Border", value="remove_border")

    with gr.Row():
        with gr.Column(scale=2):
            box_x = gr.Slider(0, 2000, value=0, label="Box X", step=1)
            box_y = gr.Slider(0, 2000, value=0, label="Box Y", step=1)
            box_w = gr.Slider(0, 4000, value=0, label="Box W", step=1)
            box_h = gr.Slider(0, 4000, value=0, label="Box H", step=1)
            use_sam_box = gr.Checkbox(label="Use SAM Box", value=True)
            with gr.Row():
                draw_box_btn = gr.Button(label="Draw Box", value="draw_box")
                erase_box_btn = gr.Button(label="Erase Box", value="erase_box")

        with gr.Column(scale=3):
            with gr.Row():
                box_text = gr.Textbox(lines=6, label="Box Text", value="")
                box_json = gr.Json(label="Box JSON", value=None)
            with gr.Row():
                text2json_btn = gr.Button(label="Text to JSON", value="text2json")
                json2text_btn = gr.Button(label="JSON to Text", value="json2text")

    with gr.Row():
        sam_prompt = gr.Textbox(lines=1, label="Text segment Prompt", value="text")
        with gr.Row():
            sd_prompt = gr.Textbox(lines=1, label="Text erase Prompt", value="No text, clean background")
            sd_negative_prompt = gr.Textbox(lines=1, label="Negative Prompt", value="text,watermark")

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

    text2json_btn.click(
        fn=json.loads,
        inputs=[box_text],
        outputs=[box_json],
    )
    json2text_btn.click(
        fn=json.dumps,
        inputs=[box_json],
        outputs=[box_text],
    )
    input_file.change(
        fn=update_image,
        inputs=[input_file],
        outputs=mark_image,
    )
    draw_box_btn.click(
        fn=draw_boxes,
        inputs=[input_file, box_x, box_y, box_w, box_h, box_json],
        outputs=mark_image,
    )
    erase_box_btn.click(
        fn=sd_erase_box,
        inputs=[input_file, sd_prompt, box_x, box_y, box_w, box_h, box_json],
        outputs=output_image,
    )
    crop_btn.click(
        fn=crop_sam_image,
        inputs=[input_dir, output_dir, sam_box_threshold, sam_text_threshold, input_file, sam_prompt],
        outputs=gallery
    )

    remove_border_btn.click(
        fn=remove_border,
        inputs=[input_dir, output_dir, input_file],
        outputs=gallery
    )

    sam_btn.click(
        fn=segment_sam,
        inputs=[sam_type, sam_box_threshold, sam_text_threshold, input_file, sam_prompt],
        outputs=[mark_image, box_json]
    )

    erase_btn.click(
        fn=sd_erase,
        inputs=[
            input_dir,
            output_dir,
            input_file,
            sd_prompt,
            sam_prompt,
            sam_box_threshold,
            sam_text_threshold,
            box_json,
            use_sam_box,
            sd_negative_prompt,
            sd_model,
        ],
        outputs=gallery
    )

demo.launch(enable_queue=False, share=False, debug=True)
