import random
from typing import List
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# ---------- helpers ----------

def _extract_preprocess_info(preprocess):
    """
    Infer CLIP input crop size and normalization stats from OpenCLIP's preprocess.
    Defaults to standard CLIP stats if not found.
    """
    crop_size = 224
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    if hasattr(preprocess, "transforms"):
        for t in preprocess.transforms:
            name = t.__class__.__name__.lower()
            if "centercrop" in name:
                crop_size = t.size if isinstance(t.size, int) else max(t.size)
            if "normalize" in name:
                mean = tuple(t.mean)
                std  = tuple(t.std)
    return int(crop_size), mean, std

def _to_clip_tensor(im: Image.Image, mean, std):
    x = TF.to_tensor(im)
    x = TF.normalize(x, mean, std)
    return x

def _safe_resample(name="lanczos"):
    try:
        if name == "lanczos":
            return Image.Resampling.LANCZOS
        if name == "bicubic":
            return Image.Resampling.BICUBIC
    except AttributeError:
        # Pillow < 9 fallback
        return Image.LANCZOS if name == "lanczos" else Image.BICUBIC

def _apply_anisotropic_scale(im: Image.Image, sx: float, sy: float) -> Image.Image:
    """Scale with independent factors along x/y (up-only)."""
    w, h = im.size
    new_w = max(1, int(round(w * sx)))
    new_h = max(1, int(round(h * sy)))
    return im.resize((new_w, new_h), _safe_resample("lanczos"))

def _rotate_expand(im: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate with expand=True to avoid cropping content; fill corners with mid-gray."""
    return im.rotate(
        angle_deg,
        resample=_safe_resample("bicubic"),
        expand=True,
        fillcolor=(127, 127, 127)
    )

def _letterbox_to_square_uponly(im: Image.Image, target: int) -> Image.Image:
    """
    Pad the image into a target×target canvas without any downscale.
    If either side is larger than target (shouldn't happen with our guards), we keep as-is
    and paste centered (some edges may be truncated if truly oversized).
    """
    w, h = im.size
    canvas = Image.new("RGB", (target, target), color=(127, 127, 127))
    off_x = max(0, (target - w) // 2)
    off_y = max(0, (target - h) // 2)
    # If w/h exceed target, paste will naturally crop at canvas boundaries (rare for tiny inputs).
    canvas.paste(im, (off_x, off_y))
    return canvas

def _max_fit_scalar_after_rotation(w: int, h: int, sx: float, sy: float, angle_rad: float, target: int) -> float:
    """
    Compute the largest scalar k <= 1 such that after applying sx*k, sy*k and rotating by angle,
    the expanded bounding-box fits within 'target' along both axes.
    Derived bounds for rotated rectangle (expand=True):
        W' = |sx*k*w*cosθ| + |sy*k*h*sinθ|
        H' = |sx*k*w*sinθ| + |sy*k*h*cosθ|
    We need W' <= target and H' <= target -> k <= min( target / denomW, target / denomH ).
    """
    import math
    c, s = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
    denomW = sx * w * c + sy * h * s
    denomH = sx * w * s + sy * h * c
    # If denom is 0 (degenerate tiny), allow k=1
    kW = target / denomW if denomW > 0 else 1.0
    kH = target / denomH if denomH > 0 else 1.0
    return max(0.0, min(1.0, kW, kH))

# ---------- encoder (flip + rotate + anisotropic scale; up-only; no crop) ----------

@torch.no_grad()
def enc_img(
    paths: List[str],
    model,
    preprocess,         # just to read CLIP size & stats
    device,
    batch: int = 64,
    tta_views: int = 4,
    flip_p: float = 0.5,
    max_rotate_deg: float = 10.0,    # magnitude sampled in [0, max_rotate_deg]
    rotate_two_sided: bool = True,   # if True, random sign ±; else always >=0
    scale_x_max: float = 2.0,        # anisotropic up-only scaling ranges: sx in [1, scale_x_max]
    scale_y_max: float = 2.0,        # sy in [1, scale_y_max]
):
    """
    For each image, build 'tta_views' augmented views by:
      1) optional horizontal flip,
      2) rotation by a random angle with magnitude U[0, max_rotate_deg] (sign random if rotate_two_sided),
      3) anisotropic up-only scaling (sx ∈ [1,scale_x_max], sy ∈ [1,scale_y_max]),
      4) ensure final size fits into CLIP input via a soft shrink (still >=1) BEFORE transform is applied,
      5) rotate (expand) and letterbox-pad to target square (no cropping, no downscale).
    """
    import math

    input_size, mean, std = _extract_preprocess_info(preprocess)
    feats = []

    for i in range(0, len(paths), batch):
        batch_paths = paths[i:i+batch]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        for im in imgs:
            w0, h0 = im.size
            views = []

            for _ in range(max(1, int(tta_views))):
                # 1) Flip
                im_aug = im.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < flip_p else im

                # 2) Sample rotation
                mag = random.uniform(0.0, max(0.0, float(max_rotate_deg)))
                if rotate_two_sided and mag > 0:
                    mag = mag if random.random() < 0.5 else -mag
                angle_deg = mag
                angle_rad = math.radians(abs(mag))

                # 3) Sample anisotropic up-only scales
                sx = random.uniform(1.0, max(1.0, float(scale_x_max)))
                sy = random.uniform(1.0, max(1.0, float(scale_y_max)))

                # 4) Guard: ensure that after scaling+rotation, we will not exceed input_size
                # Compute max shrink factor k <= 1 to fit into target; keep >= 0.0
                k = _max_fit_scalar_after_rotation(w0, h0, sx, sy, angle_rad, input_size)
                # We want up-only: ensure sx,sy remain >=1 after shrinking toward 1.
                # Shrink multiplicatively toward 1 by mixing: sx' = 1 + (sx-1)*k, same for sy.
                sx_fit = 1.0 + (sx - 1.0) * k
                sy_fit = 1.0 + (sy - 1.0) * k

                # Apply anisotropic scale
                im_scaled = _apply_anisotropic_scale(im_aug, sx_fit, sy_fit)

                # Apply rotation (expand=True)
                im_rot = _rotate_expand(im_scaled, angle_deg)

                # 5) Letterbox-pad to exact CLIP input size (no downscale)
                im_final = _letterbox_to_square_uponly(im_rot, input_size)

                # Encode
                x = _to_clip_tensor(im_final, mean, std).unsqueeze(0).to(device)
                f = model.encode_image(x)  # [1,D]
                views.append(f[0])

            f_img = torch.stack(views, 0)               # [V,D]
            f_img = F.normalize(f_img, dim=-1).mean(0)  # average TTA views
            feats.append(f_img)

    F_stack = torch.stack(feats, 0).to(device)          # [N,D]
    return F.normalize(F_stack, dim=-1)
