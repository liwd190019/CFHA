#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score a single image against a learned reference (t_*) saved by train_text_ref.py.

Example:
  python score_image.py \
      --ref ref_person.pt \
      --image /path/to/test.jpg \
      --json
"""
import argparse
from PIL import Image
import json
import torch
import torch.nn.functional as F
import open_clip

def load_model(name, pretrained, device):
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model.eval().to(device)
    return model, preprocess

@torch.no_grad()
def enc_one_image(path, model, preprocess, device, tta_flip=False):
    im = Image.open(path).convert("RGB")
    x = preprocess(im).unsqueeze(0).to(device)
    f = model.encode_image(x)
    if tta_flip:
        imf = im.transpose(Image.FLIP_LEFT_RIGHT)
        xf = preprocess(imf).unsqueeze(0).to(device)
        f = 0.5 * (f + model.encode_image(xf))
    return F.normalize(f[0], dim=-1)  # [D]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help=".pt file produced by train_text_ref.py")
    ap.add_argument("--image", required=True, help="Path to image to score")
    ap.add_argument("--json", action="store_true", help="Output as a JSON line")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load reference
    ref = torch.load(args.ref, map_location="cpu")
    t_star = ref["t_star"].to(device)
    t_anchor = ref.get("t_anchor", None)
    if t_anchor is not None:
        t_anchor = t_anchor.to(device)

    model, preprocess = load_model(ref["model"], ref["pretrained"], device)

    # Image embedding
    feat = enc_one_image(args.image, model, preprocess, device, tta_flip=bool(ref.get("tta_flip", False)))

    # 1) Cosine similarity
    cos_sim = float((feat @ t_star).item())  # [-1, 1]

    # 2) Optional relative probability vs anchor (2-way softmax)
    rel_prob = None
    if t_anchor is not None:
        scale = float(model.logit_scale.exp().item())
        logits = scale * torch.stack([feat @ t_star, feat @ t_anchor], dim=0)
        rel_prob = float(torch.softmax(logits, dim=0)[0].item())  # P(concept)

    out = {
        "image": args.image,
        "cosine_similarity": round(cos_sim, 6),
        "relative_probability": (round(rel_prob, 6) if rel_prob is not None else None),
        "model": ref["model"],
        "pretrained": ref["pretrained"],
        "anchor_text": ref.get("anchor_text", None),
    }

    if args.json:
        print(json.dumps(out))
    else:
        print(f"cosine_similarity: {out['cosine_similarity']}")
        if rel_prob is not None:
            print(f"relative_probability (vs anchor): {out['relative_probability']}")

if __name__ == "__main__":
    main()
