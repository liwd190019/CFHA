# import os
# import sys
# import glob
# import argparse
# import torch
# from torchvision import transforms
# import torchvision.transforms.functional as F
# import numpy as np
# from PIL import Image

# from unify_model2 import UnifiedModel
# from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
# from ram.models.ram_lora import ram
# from ram import inference_ram as inference

# # Define transforms for input images
# tensor_transforms = transforms.Compose([
#     transforms.ToTensor(),
# ])

# ram_transforms = transforms.Compose([
#     transforms.Resize((384, 384)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def get_validation_prompt(args, image, ram_model, device='cuda'):
#     """
#     Generate a validation prompt using RAM model and user-provided prompt.
    
#     Args:
#         args: Command-line arguments.
#         image (PIL.Image): Input content image.
#         ram_model: RAM model for caption generation.
#         device (str): Device to run on.
    
#     Returns:
#         tuple: (validation prompt, preprocessed low-quality image tensor).
#     """
#     lq = tensor_transforms(image).unsqueeze(0).to(device)
#     lq_ram = ram_transforms(lq).to(dtype=args.weight_dtype)
#     captions = inference(lq_ram, ram_model)
#     validation_prompt = f"{captions[0]}, {args.prompt},"
#     return validation_prompt, lq

# def load_image(path, size, upscale, device='cuda'):
#     """
#     Load and preprocess an image, ensuring size is a multiple of 8.
    
#     Args:
#         path (str): Path to the image file.
#         size (int): Target processing size (pre-upscale).
#         upscale (int): Upscale factor.
#         device (str): Device to place the tensor.
    
#     Returns:
#         tuple: (preprocessed tensor, PIL image, resize flag, original width, original height).
#     """
#     img = Image.open(path).convert('RGB')
#     ori_width, ori_height = img.size
#     resize_flag = False
#     if ori_width < size // upscale or ori_height < size // upscale:
#         scale = (size // upscale) / min(ori_width, ori_height)
#         img = img.resize((int(scale * ori_width), int(scale * ori_height)), Image.LANCZOS)
#         resize_flag = True
#     img = img.resize((img.size[0] * upscale, img.size[1] * upscale), Image.LANCZOS)
#     new_width = img.width - img.width % 8
#     new_height = img.height - img.height % 8
#     img = img.resize((new_width, new_height), Image.LANCZOS)
#     tensor = tensor_transforms(img).unsqueeze(0).to(device) * 2 - 1  # Normalize to [-1, 1]
#     return tensor, img, resize_flag, ori_width, ori_height


# def save_image(tensor, path, resize_flag, ori_width, ori_height, upscale, align_method, source_img):
#     """
#     Save a tensor as an image with optional color correction and resizing.
    
#     Args:
#         tensor (torch.Tensor): Output image tensor of shape (1, 3, H, W) in [0, 1].
#         path (str): Output file path.
#         resize_flag (bool): Whether the input was resized.
#         ori_width (int): Original width of input image.
#         ori_height (int): Original height of input image.
#         upscale (int): Upscale factor.
#         align_method (str): Color alignment method ('adain', 'wavelet', 'nofix').
#         source_img (PIL.Image): Source image for color correction.
#     """
#     output_pil = transforms.ToPILImage()(tensor.squeeze(0).cpu())
#     if align_method == 'adain':
#         output_pil = adain_color_fix(target=output_pil, source=source_img)
#     elif align_method == 'wavelet':
#         output_pil = wavelet_color_fix(target=output_pil, source=source_img)
#     if resize_flag:
#         output_pil = output_pil.resize((int(upscale * ori_width), int(upscale * ori_height)), Image.LANCZOS)
#     output_pil.save(path)

# def main():
#     parser = argparse.ArgumentParser(description="Run UnifiedModel for super-resolution and style transfer")
#     parser.add_argument('--content_image', '-i', type=str, required=True, help='Path to content image or directory')
#     parser.add_argument('--style_image', '-s', type=str, required=True, help='Path to style image or directory')
#     parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', help='Directory to save output images')
#     parser.add_argument('--prompt', type=str, default='', help='User prompt to append to RAM captions')
#     parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/stable-diffusion-2-1-base', help='Pretrained model name or path')
#     parser.add_argument('--osediff_path', type=str, default=None, help='Path to LoRA checkpoint (optional)')
#     parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM pretrained model')
#     parser.add_argument('--ram_ft_path', type=str, default=None, help='Path to RAM finetuned model')
#     parser.add_argument('--save_prompts', type=bool, default=True, help='Save generated prompts to txt files')
#     parser.add_argument('--process_size', type=int, default=512, help='Processing size (pre-upscale)')
#     parser.add_argument('--upscale', type=int, default=4, help='Upscale factor')
#     parser.add_argument('--align_method', type=str, choices=['wavelet', 'adain', 'nofix'], default='adain', help='Color alignment method')
#     parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'fp32'], default='fp16', help='Precision for computation')
#     parser.add_argument('--vae_decoder_tiled_size', type=int, default=224)
#     parser.add_argument('--vae_encoder_tiled_size', type=int, default=1024)
#     parser.add_argument('--latent_tiled_size', type=int, default=96)
#     parser.add_argument('--latent_tiled_overlap', type=int, default=32)
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')

#     args = parser.parse_args()

#     # Set random seed
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)

#     # Initialize model
#     model_args = argparse.Namespace(
#         pretrained_model_name_or_path=args.pretrained_model_name_or_path,
#         osediff_lora_path=args.osediff_path,
#         vae_encoder_tiled_size=args.vae_encoder_tiled_size,
#         vae_decoder_tiled_size=args.vae_decoder_tiled_size,
#         latent_tiled_size=args.latent_tiled_size,
#         latent_tiled_overlap=args.latent_tiled_overlap,
#         mixed_precision=args.mixed_precision,
#         filter_percentile=0.5,
#         enable_filtering=True
#     )
#     model = UnifiedModel(model_args).to('cuda')

#     # Initialize RAM model
#     ram_model = ram(
#         pretrained=args.ram_path,
#         pretrained_condition=args.ram_ft_path,
#         image_size=384,
#         vit='swin_l'
#     )
#     ram_model.eval().to('cuda')
#     args.weight_dtype = torch.float16 if args.mixed_precision == 'fp16' else torch.float32
#     ram_model = ram_model.to(dtype=args.weight_dtype)

#     # Get input images
#     if os.path.isdir(args.content_image):
#         content_image_names = sorted(glob.glob(f'{args.content_image}/*.jpeg'))
#     else:
#         content_image_names = [args.content_image]
    
#     if os.path.isdir(args.style_image):
#         style_image_names = sorted(glob.glob(f'{args.style_image}/*.jpg'))
#         if len(style_image_names) != len(content_image_names):
#             raise ValueError("Number of style images must match number of content images")
#     else:
#         style_image_names = [args.style_image] * len(content_image_names)

#     # Create output and prompt directories
#     os.makedirs(args.output_dir, exist_ok=True)
#     if args.save_prompts:
#         txt_path = os.path.join(args.output_dir, 'txt')
#         os.makedirs(txt_path, exist_ok=True)
    
#     print(f'[INFO] Processing {len(content_image_names)} image pairs.')

#     # one_content_image = Image.open(content_image_names[0]).convert('RGB')


#     # Process each image pair
#     for content_name, style_name in zip(content_image_names, style_image_names):
#         bname = os.path.basename(content_name)

#         # Load and preprocess content and style images
#         content_tensor, content_pil, resize_flag, ori_width, ori_height = load_image(
#             content_name, args.process_size, args.upscale, device='cuda'
#         )

#         style_tensor, _, _, _, _ = load_image(
#             style_name, args.process_size, args.upscale, device='cuda'
#         )

#         # Get the target spatial size from the content tensor
#         content_size = content_tensor.shape[2:]

#         # Resize the style tensor to match the content tensor if their dimensions differ
#         if style_tensor.shape[2:] != content_size:
#             print(f"[INFO] Resizing style tensor from {style_tensor.shape[2:]} to {content_size} to match content.")
#             style_tensor = F.interpolate(style_tensor, size=content_size, mode='bicubic', align_corners=False)


#         # Generate prompt
#         validation_prompt, _ = get_validation_prompt(args, content_pil, ram_model)
#         if args.save_prompts:
#             txt_save_path = os.path.join(txt_path, f"{bname.split('.')[0]}.txt")
#             with open(txt_save_path, 'w', encoding='utf-8') as f:
#                 f.write(validation_prompt)
#         print(f"[INFO] Processing {bname}, tag: {validation_prompt}")

#         # Run inference
#         with torch.no_grad():
#             output_image = model(content_tensor, style_tensor, [validation_prompt])

#         # Save output with color correction
#         output_path = os.path.join(args.output_dir, bname)
#         save_image(
#             output_image, output_path, resize_flag, ori_width, ori_height,
#             args.upscale, args.align_method, content_pil
#         )
#         print(f"[INFO] Saved output to {output_path}")

# if __name__ == "__main__":
#     main()

import os
import sys
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from unify_model4 import UnifiedModel
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from ram.models.ram_lora import ram
from ram import inference_ram as inference

# Define transforms for input images
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_validation_prompt(args, image, ram_model, device='cuda'):
    """
    Generate a validation prompt using RAM model and user-provided prompt.
    
    Args:
        args: Command-line arguments.
        image (PIL.Image): Input content image.
        ram_model: RAM model for caption generation.
        device (str): Device to run on.
    
    Returns:
        tuple: (validation prompt, preprocessed low-quality image tensor).
    """
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=args.weight_dtype)
    captions = inference(lq_ram, ram_model)
    validation_prompt = f"{captions[0]}, {args.prompt},"
    return validation_prompt, lq

def load_image(path, size, upscale, device='cuda', target_size=None):
    """
    Load and preprocess an image, ensuring size is a multiple of 8.
    
    Args:
        path (str): Path to the image file.
        size (int): Target processing size (pre-upscale).
        upscale (int): Upscale factor.
        device (str): Device to place the tensor.
        target_size (tuple): Optional (width, height) to resize image.
    
    Returns:
        tuple: (preprocessed tensor, PIL image, resize flag, original width, original height).
    """
    img = Image.open(path).convert('RGB')
    ori_width, ori_height = img.size
    resize_flag = False
    if ori_width < size // upscale or ori_height < size // upscale:
        scale = (size // upscale) / min(ori_width, ori_height)
        img = img.resize((int(scale * ori_width), int(scale * ori_height)), Image.LANCZOS)
        resize_flag = True
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    else:
        img = img.resize((img.size[0] * upscale, img.size[1] * upscale), Image.LANCZOS)
    new_width = img.width - img.width % 8
    new_height = img.height - img.height % 8
    img = img.resize((new_width, new_height), Image.LANCZOS)
    tensor = tensor_transforms(img).unsqueeze(0).to(device) * 2 - 1  # Normalize to [-1, 1]
    tensor = tensor.to(dtype=args.weight_dtype)
    return tensor, img, resize_flag, ori_width, ori_height

def save_image(tensor, path, resize_flag, ori_width, ori_height, upscale, align_method, source_img):
    """
    Save a tensor as an image with optional color correction and resizing.
    
    Args:
        tensor (torch.Tensor): Output image tensor of shape (1, 3, H, W) in [0, 1].
        path (str): Output file path.
        resize_flag (bool): Whether the input was resized.
        ori_width (int): Original width of input image.
        ori_height (int): Original height of input image.
        upscale (int): Upscale factor.
        align_method (str): Color alignment method ('adain', 'wavelet', 'nofix').
        source_img (PIL.Image): Source image for color correction.
    """
    output_pil = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    if align_method == 'adain':
        output_pil = adain_color_fix(target=output_pil, source=source_img)
    elif align_method == 'wavelet':
        output_pil = wavelet_color_fix(target=output_pil, source=source_img)
    if resize_flag:
        output_pil = output_pil.resize((int(upscale * ori_width), int(upscale * ori_height)), Image.LANCZOS)
    output_pil.save(path)

def main():
    parser = argparse.ArgumentParser(description="Run UnifiedModel for super-resolution and style transfer")
    parser.add_argument('--content_image', '-i', type=str, required=True, help='Path to content image or directory')
    parser.add_argument('--style_image', '-s', type=str, required=True, help='Path to single style image')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', help='Directory to save output images')
    parser.add_argument('--prompt', type=str, default='', help='User prompt to append to RAM captions')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/stable-diffusion-2-1-base', help='Pretrained model name or path')
    parser.add_argument('--osediff_path', type=str, default=None, help='Path to LoRA checkpoint (optional)')
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM pretrained model')
    parser.add_argument('--ram_ft_path', type=str, default=None, help='Path to RAM finetuned model')
    parser.add_argument('--save_prompts', type=bool, default=True, help='Save generated prompts to txt files')
    parser.add_argument('--process_size', type=int, default=512, help='Processing size (pre-upscale)')
    parser.add_argument('--upscale', type=int, default=4, help='Upscale factor')
    parser.add_argument('--align_method', type=str, choices=['wavelet', 'adain', 'nofix'], default='adain', help='Color alignment method')
    parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'fp32'], default='fp16', help='Precision for computation')
    parser.add_argument('--vae_decoder_tiled_size', type=int, default=224)
    parser.add_argument('--vae_encoder_tiled_size', type=int, default=1024)
    parser.add_argument('--latent_tiled_size', type=int, default=96)
    parser.add_argument('--latent_tiled_overlap', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    global args  # For load_image access to weight_dtype
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize model
    model_args = argparse.Namespace(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        osediff_lora_path=args.osediff_path,
        vae_encoder_tiled_size=args.vae_encoder_tiled_size,
        vae_decoder_tiled_size=args.vae_decoder_tiled_size,
        latent_tiled_size=args.latent_tiled_size,
        latent_tiled_overlap=args.latent_tiled_overlap,
        mixed_precision=args.mixed_precision,
        filter_percentile=0.5,
        enable_filtering=True
    )
    model = UnifiedModel(model_args).to('cuda')

    # Initialize RAM model
    ram_model = ram(
        pretrained=args.ram_path,
        pretrained_condition=args.ram_ft_path,
        image_size=384,
        vit='swin_l'
    )
    ram_model.eval().to('cuda')
    args.weight_dtype = torch.float16 if args.mixed_precision == 'fp16' else torch.float32
    ram_model = ram_model.to(dtype=args.weight_dtype)

    # Get content images
    if os.path.isdir(args.content_image):
        content_image_names = sorted(glob.glob(f'{args.content_image}/*.jpeg'))
    else:
        content_image_names = [args.content_image]
    
    # Load first content image to get target resolution
    first_content_tensor, first_content_img, _, _, _ = load_image(
        content_image_names[0], args.process_size, args.upscale, device='cuda')
    target_size = first_content_img.size
    print(f"[INFO] All content images will use resolution: {target_size}")
    
    # Load and resize single style image to match content resolution
    style_tensor, style_img, _, _, _ = load_image(
        args.style_image, args.process_size, args.upscale, device='cuda', target_size=target_size)
    print(f"Style tensor shape: {style_tensor.shape}")
    
    # Create output and prompt directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)
    
    print(f'[INFO] Processing {len(content_image_names)} content images with one style image.')

    # Process each content image
    for content_name in content_image_names:
        bname = os.path.basename(content_name)

        # Load and preprocess content image
        content_tensor, content_pil, resize_flag, ori_width, ori_height = load_image(
            content_name, args.process_size, args.upscale, device='cuda', target_size=target_size)
        
        print(f"[INFO] Processing {bname}")
        print(f"Content tensor shape: {content_tensor.shape}")

        # Generate prompt
        validation_prompt, _ = get_validation_prompt(args, content_pil, ram_model)
        if args.save_prompts:
            txt_save_path = os.path.join(txt_path, f"{bname.split('.')[0]}.txt")
            with open(txt_save_path, 'w', encoding='utf-8') as f:
                f.write(validation_prompt)
        print(f"[INFO] Tag: {validation_prompt}")

        # Run inference
        with torch.no_grad():
            output_image = model(content_tensor, style_tensor, [validation_prompt])

        # Save output with color correction
        output_path = os.path.join(args.output_dir, bname)
        save_image(
            output_image, output_path, resize_flag, ori_width, ori_height,
            args.upscale, args.align_method, content_pil
        )
        print(f"[INFO] Saved output to {output_path}")

if __name__ == "__main__":
    main()
