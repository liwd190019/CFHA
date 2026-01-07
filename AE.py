#!/usr/bin/env python3
"""
Batch AttentiveEraser with anchored crop planning and 1024×1024 AE inputs.

Pairing rule:
  image: <base>.jpg (or .jpeg/.png) in --input_image_dir
  mask : <base><mask_suffix> in --input_mask_dir, default mask_suffix="_mask.png"

- White in the mask = regions to remove.
- Crop size on canvas: 512 or 1024 (no resize on the canvas).
- AE always runs at 1024×1024; output is resized back to the crop size and pasted.
- Saves the edited image to --out_image_dir and the updated working mask to --out_mask_dir.
"""

import argparse, os, sys
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
from diffusers import DDIMScheduler, DiffusionPipeline

from typing import Optional, List, Tuple


# ----------------------------
# AE init
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
# Anchored crop planning (no dilation)
# ----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))

def plan_crops_from_mask_anchored(
    binary_mask: np.ndarray,
    patch_size: int = 1024,
    stride: Optional[int] = None,
    margin: int = 64,
) -> List[Tuple[int, int, int, int]]:
    H, W = binary_mask.shape
    if stride is None or stride <= 0:
        stride = patch_size // 2

    m = (binary_mask > 0)
    if not m.any():
        return []

    ys, xs = np.where(m)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    x0_anchor = clamp(x_min - margin, 0, max(0, W - patch_size))
    y0_anchor = clamp(y_min - margin, 0, max(0, H - patch_size))

    xs_list = list(range(x0_anchor, max(0, W - patch_size) + 1, stride))
    ys_list = list(range(y0_anchor, max(0, H - patch_size) + 1, stride))

    x_right_need = clamp(x_max + margin - patch_size + 1, 0, max(0, W - patch_size))
    y_bot_need   = clamp(y_max + margin - patch_size + 1, 0, max(0, H - patch_size))
    if (not xs_list) or (xs_list[-1] < x_right_need): xs_list.append(x_right_need)
    if (not ys_list) or (ys_list[-1] < y_bot_need):   ys_list.append(y_bot_need)

    xs_list = sorted(set(xs_list)); ys_list = sorted(set(ys_list))

    windows = []
    m_u8 = m.astype(np.uint8)
    for y0 in ys_list:
        for x0 in xs_list:
            x1, y1 = x0 + patch_size, y0 + patch_size
            if m_u8[y0:y1, x0:x1].any():
                windows.append((x0, y0, x1, y1))
    windows.sort(key=lambda w: -m_u8[w[1]:w[3], w[0]:w[2]].sum())
    return windows

# ----------------------------
# Preprocess crop -> 1024×1024 tensors for AE
# ----------------------------
AE_SIZE = 1024        # always feed 1024×1024 to AE
BLUR_KERNEL_1024 = 77 # match the official example

def prep_crop_for_ae_1024(
    img_crop_pil: Image.Image,
    mask_crop_pil: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
):
    # Image -> [-1,1], then upsample to 1024×1024
    image = to_tensor(img_crop_pil).unsqueeze(0).float() * 2 - 1
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (AE_SIZE, AE_SIZE))
    image = image.to(dtype=dtype, device=device)

    # Mask -> [0,1], upsample to 1024×1024, blur, threshold
    mask = to_tensor(mask_crop_pil.convert('L')).unsqueeze(0).float()
    mask = F.interpolate(mask, (AE_SIZE, AE_SIZE))
    mask = gaussian_blur(mask, kernel_size=(BLUR_KERNEL_1024, BLUR_KERNEL_1024))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(dtype=dtype, device=device)
    return image, mask

# ----------------------------
# Run AE on a crop, then downscale back to crop size
# ----------------------------
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
    out_1024 = out.images[0]  # PIL 1024×1024
    return out_1024.resize(orig_size, Image.LANCZOS)

# ----------------------------
# Process a single (image, mask) pair
# ----------------------------
def process_pair(
    pipeline: DiffusionPipeline,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    img_path: str,
    mask_path: str,
    out_image_path: str,
    out_mask_path: str,
    patch_size: int,
    stride: Optional[int],
    margin: int,
):
    image = np.array(Image.open(img_path).convert("RGB"))
    work_mask = np.array(Image.open(mask_path).convert("L"))
    work_mask = np.where(work_mask > 0, 255, 0).astype(np.uint8)  # force 0/255

    H, W = work_mask.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Patch {patch_size} does not fit inside {img_path} ({W}x{H}).")

    windows = plan_crops_from_mask_anchored(work_mask, patch_size=patch_size, stride=stride, margin=margin)
    print(f"[INFO] {os.path.basename(img_path)}: {len(windows)} crop(s) planned")

    for (x0, y0, x1, y1) in windows:
        if not (work_mask[y0:y1, x0:x1] > 0).any():
            continue

        img_crop_pil  = Image.fromarray(image[y0:y1, x0:x1])
        mask_crop_pil = Image.fromarray(work_mask[y0:y1, x0:x1])

        out_crop_pil = call_ae_on_crop_upsized(pipeline, device, dtype, img_crop_pil, mask_crop_pil, generator)
        image[y0:y1, x0:x1] = np.array(out_crop_pil)

        # Clear processed mask area
        mc = np.array(mask_crop_pil)
        work_mask[y0:y1, x0:x1][mc > 0] = 0

    Image.fromarray(image).save(out_image_path)
    Image.fromarray(work_mask).save(out_mask_path)

# ----------------------------
# Batch driver
# ----------------------------
def valid_image_name(name: str) -> bool:
    n = name.lower()
    return n.endswith(".jpg") or n.endswith(".jpeg") or n.endswith(".png")

def main():
    ap = argparse.ArgumentParser(description="Batch AttentiveEraser over image/mask directories.")
    ap.add_argument("--input_image_dir", required=True, help="Dir with images (e.g., *.jpg).")
    ap.add_argument("--input_mask_dir",  required=True, help="Dir with masks (base + mask_suffix).")
    ap.add_argument("--out_image_dir",   required=True, help="Where to save edited images.")
    ap.add_argument("--out_mask_dir",    required=True, help="Where to save updated masks.")
    ap.add_argument("--mask_suffix",     default="_mask.png", help="Suffix appended to base for mask names.")
    ap.add_argument("--patch_size",      type=int, default=1024, choices=[512, 1024], help="Crop size on canvas.")
    ap.add_argument("--stride",          type=int, default=-1, help="Stride; -1 => half‑patch.")
    ap.add_argument("--margin",          type=int, default=64, help="Anchor margin for context.")
    ap.add_argument("--seed",            type=int, default=123, help="Random seed.")
    args = ap.parse_args()

    os.makedirs(args.out_image_dir, exist_ok=True)
    os.makedirs(args.out_mask_dir,  exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    stride = None if args.stride is None or args.stride <= 0 else args.stride

    pipeline  = init_ae_pipeline(device, dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Build pairs from image dir
    img_files = sorted([f for f in os.listdir(args.input_image_dir) if valid_image_name(f)])
    if not img_files:
        print(f"[ERROR] No images found in {args.input_image_dir}", file=sys.stderr); sys.exit(1)

    processed = 0
    for img_name in img_files:
        if 'okutama' in img_name:
            continue
        base, _ = os.path.splitext(img_name)
        mask_name = f"{base}{args.mask_suffix}"
        img_path  = os.path.join(args.input_image_dir, img_name)
        mask_path = os.path.join(args.input_mask_dir,  mask_name)

        if not os.path.exists(mask_path):
            print(f"[WARN] Mask missing for {img_name}: expected {mask_path}. Skipping.")
            continue

        out_image_path = os.path.join(args.out_image_dir, img_name)           # keep original ext
        out_mask_path  = os.path.join(args.out_mask_dir,  mask_name)          # keep mask suffix

        process_pair(
            pipeline, device, dtype, generator,
            img_path, mask_path,
            out_image_path, out_mask_path,
            patch_size=args.patch_size,
            stride=stride,
            margin=args.margin,
        )
        processed += 1

    print(f"[DONE] Processed {processed} / {len(img_files)} image(s).")

if __name__ == "__main__":
    main()
