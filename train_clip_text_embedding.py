#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn a CLIP text-side reference embedding t_* from positive images (no negatives).
Saves a .pt file containing t_star, (optional) t_anchor, and model config.

Example:
  python train_text_ref.py \
      --pos_dir /path/to/positives \
      --out ref_person.pt \
      --anchor_text "a photo of a person" \
      --model ViT-L-14 \
      --pretrained laion2b_s32b_b82k \
      --steps 400 --lr 5e-2 --lam_anchor 0.2 --tta_flip
"""

import os, argparse, json, time
from glob import glob
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np
import open_clip

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def list_images(path):
    if os.path.isdir(path):
        files = []
        for root, _, names in os.walk(path):
            for n in names:
                if os.path.splitext(n.lower())[1] in EXTS:
                    files.append(os.path.join(root, n))
        return sorted(files)
    else:
        # treat as a text file with one image path per line
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()]

def set_seed(s=123):
    torch.manual_seed(s); np.random.seed(s)

def load_model(name, pretrained, device):
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model.eval().to(device)
    return model, preprocess

@torch.no_grad()
def enc_img(paths, model, preprocess, device, batch=64, tta_flip=False):
    """Return L2-normalized image embeddings stacked."""
    feats = []
    for i in range(0, len(paths), batch):
        batch_paths = paths[i:i+batch]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        x = torch.stack([preprocess(im) for im in imgs]).to(device)
        f = model.encode_image(x)
        if tta_flip:
            imgs_f = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs]
            x_f = torch.stack([preprocess(im) for im in imgs_f]).to(device)
            f = 0.5 * (f + model.encode_image(x_f))
        feats.append(F.normalize(f, dim=-1))
    return torch.cat(feats, 0)  # [N, D]

@torch.no_grad()
def enc_txt(text, model, device):
    tok = open_clip.get_tokenizer("ViT-L-14")
    t = model.encode_text(tok([text]).to(device))
    return F.normalize(t[0], dim=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dir", required=True,
                    help="Directory of positive images OR a text file listing image paths.")
    ap.add_argument("--out", required=True, help="Output .pt file to save the learned reference.")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--anchor_text", default="a photo of a person")

    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--lam_anchor", type=float, default=0.2,
                    help="Strength to keep t_* near anchor_text.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta_flip", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--closed_form", action="store_true",
                    help="If set, skip SGD and compute t_* = normalize(c + lam*anchor).")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect positives
    pos_paths = list_images(args.pos_dir)
    if len(pos_paths) == 0:
        raise SystemExit(f"No images found under {args.pos_dir}")

    # Load CLIP
    model, preprocess = load_model(args.model, args.pretrained, device)

    # Compute positive center (mean of L2-normalized embeddings)
    pos_feats = enc_img(pos_paths, model, preprocess, device, batch=args.batch, tta_flip=args.tta_flip)
    pos_center = F.normalize(pos_feats.mean(0), dim=0)  # [D]

    # Anchor text vector
    t_anchor = enc_txt(args.anchor_text, model, device)  # [D]

    # -------- Learn t_* --------
    if args.closed_form:
        # Fast closed-form solution (good approximation of the SGD objective on the sphere):
        # maximize v·pos_center + lam * v·t_anchor  s.t. ||v||=1  -> v ∝ pos_center + lam * t_anchor
        t_star = F.normalize(pos_center + args.lam_anchor * t_anchor, dim=0)
    else:
        v = t_anchor.clone().detach().requires_grad_(True)  # start from anchor
        opt = torch.optim.Adam([v], lr=args.lr)
        for step in range(args.steps):
            v_n = F.normalize(v, dim=0)
            loss = -(pos_center @ v_n) + args.lam_anchor * (1.0 - (v_n @ t_anchor))
            opt.zero_grad()
            loss.backward()
            opt.step()
            logging.info(f"[step {step+1:4d}/{args.steps}] loss = {loss.item():.6f}")
        t_star = F.normalize(v.detach(), dim=0)

    # Diagnostics
    with torch.no_grad():
        pos_sim = float((pos_feats @ t_star).mean().item())
        anchor_sim = float((pos_feats @ t_anchor).mean().item())

    payload = {
        "t_star": t_star.cpu(),              # [D] tensor
        "t_anchor": t_anchor.cpu(),          # [D] tensor
        "model": args.model,
        "pretrained": args.pretrained,
        "anchor_text": args.anchor_text,
        "tta_flip": bool(args.tta_flip),
        "train_pos_count": len(pos_paths),
        "pos_mean_sim_to_t_star": pos_sim,
        "pos_mean_sim_to_anchor": anchor_sim,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "closed_form" if args.closed_form else "sgd",
        "lam_anchor": args.lam_anchor,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
    }
    torch.save(payload, args.out)
    print(f"Saved reference to: {args.out}")
    print(f"Mean cos(pos, t_*): {pos_sim:.4f} | Mean cos(pos, anchor): {anchor_sim:.4f}")

if __name__ == "__main__":
    main()
