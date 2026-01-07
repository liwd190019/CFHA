#!/usr/bin/env python3
"""
Single-crop object removal using AttentiveEraser (Stable Diffusion XL custom pipeline).

- For each image/mask pair:
    - Find the top-left anchor bounding box covering all white pixels in the mask.
    - Add a margin for context.
    - Crop that region from both image and mask.
    - Resize to 1024x1024, run AE, then resize back and paste in.
    - Save the updated image and mask.
"""

import os, sys, argparse
from typing import Optional, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
from diffusers import DDIMScheduler, DiffusionPipeline

# ----------------------------
# Constants
# ----------------------------
AE_SIZE = 1024
BLUR_KERNEL_1024 = 77

# ----------------------------
# Pipeline Initialization
# ----------------------------
def init_ae_pipeline(device: torch.device, dtype: torch.dtype):
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        custom_pipeline="pipeline_stable_diffusion_xl_attentive_eraser",
        scheduler=scheduler,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=dtype,
    ).to(device)
    return pipe

# ----------------------------
# Clamp and crop utils
# ----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))

def prep_crop_for_ae_1024(
    img_crop_pil: Image.Image,
    mask_crop_pil: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
):
    image = to_tensor(img_crop_pil).unsqueeze(0).float() * 2 - 1
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (AE_SIZE, AE_SIZE))
    image = image.to(dtype=dtype, device=device)

    mask = to_tensor(mask_crop_pil.convert('L')).unsqueeze(0).float()
    mask = F.interpolate(mask, (AE_SIZE, AE_SIZE))
    mask = gaussian_blur(mask, kernel_size=(BLUR_KERNEL_1024, BLUR_KERNEL_1024))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(dtype=dtype, device=device)
    return image, mask

@torch.no_grad()
def call_ae_on_crop_upsized(
    pipeline: DiffusionPipeline,
    device: torch.device,
    dtype: torch.dtype,
    img_crop_pil: Image.Image,
    mask_crop_pil: Image.Image,
    generator: torch.Generator,
) -> Image.Image:
    orig_size = img_crop_pil.size
    image_t, mask_t = prep_crop_for_ae_1024(img_crop_pil, mask_crop_pil, device, dtype)

    out = pipeline(
        prompt="",
        image=image_t,
        mask_image=mask_t,
        height=AE_SIZE,
        width=AE_SIZE,
        AAS=True,
        strength=0.8,
        rm_guidance_scale=9,
        ss_steps=9,
        ss_scale=0.3,
        AAS_start_step=0,
        AAS_start_layer=34,
        AAS_end_layer=70,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=1,
    )
    out_1024 = out.images[0]
    return out_1024.resize(orig_size, Image.LANCZOS)

# ----------------------------
# Main per-image logic
# ----------------------------
def process_single_crop(
    pipeline: DiffusionPipeline,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    img_path: str,
    mask_path: str,
    out_image_path: str,
    out_mask_path: str,
    margin: int,
):
    image = np.array(Image.open(img_path).convert("RGB"))
    work_mask = np.array(Image.open(mask_path).convert("L"))
    work_mask = np.where(work_mask > 0, 255, 0).astype(np.uint8)

    H, W = work_mask.shape
    m = (work_mask > 0)
    if not m.any():
        print(f"[SKIP] No masked area in {os.path.basename(img_path)}")
        Image.fromarray(image).save(out_image_path)
        Image.fromarray(work_mask).save(out_mask_path)
        return

    ys, xs = np.where(m)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x0 = clamp(x_min - margin, 0, W - 1)
    y0 = clamp(y_min - margin, 0, H - 1)
    x1 = clamp(x_max + margin, x0 + 1, W)
    y1 = clamp(y_max + margin, y0 + 1, H)

    img_crop_pil = Image.fromarray(image[y0:y1, x0:x1])
    mask_crop_pil = Image.fromarray(work_mask[y0:y1, x0:x1])

    out_crop_pil = call_ae_on_crop_upsized(pipeline, device, dtype, img_crop_pil, mask_crop_pil, generator)
    image[y0:y1, x0:x1] = np.array(out_crop_pil)
    work_mask[y0:y1, x0:x1][np.array(mask_crop_pil) > 0] = 0

    Image.fromarray(image).save(out_image_path)
    Image.fromarray(work_mask).save(out_mask_path)

# ----------------------------
# Driver
# ----------------------------
def valid_image_name(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png"))

def main():
    ap = argparse.ArgumentParser(description="Single-crop Attentive Eraser object removal.")
    ap.add_argument("--input_image_dir", required=True)
    ap.add_argument("--input_mask_dir", required=True)
    ap.add_argument("--out_image_dir", required=True)
    ap.add_argument("--out_mask_dir", required=True)
    ap.add_argument("--mask_suffix", default="_mask.png")
    ap.add_argument("--margin", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out_image_dir, exist_ok=True)
    os.makedirs(args.out_mask_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipeline = init_ae_pipeline(device, dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    img_files = sorted([f for f in os.listdir(args.input_image_dir) if valid_image_name(f)])

    for img_name in img_files:
        base, _ = os.path.splitext(img_name)
        img_path = os.path.join(args.input_image_dir, img_name)
        mask_path = os.path.join(args.input_mask_dir, base + args.mask_suffix)

        out_image_path = os.path.join(args.out_image_dir, img_name)
        out_mask_path  = os.path.join(args.out_mask_dir, base + args.mask_suffix)

        if not os.path.exists(mask_path):
            print(f"[WARN] Missing mask for {img_name}")
            continue

        if os.path.exists(out_image_path) and os.path.exists(out_mask_path):
            print(f"[SKIP] Already processed: {img_name}")
            continue

        process_single_crop(
            pipeline, device, dtype, generator,
            img_path, mask_path,
            out_image_path, out_mask_path,
            margin=args.margin
        )

    print("[DONE] All images processed.")

if __name__ == "__main__":
    main()
