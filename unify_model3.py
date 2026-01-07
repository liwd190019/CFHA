import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttentionProcessor
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig
from einops import rearrange
from my_utils.my_adain import adain
from utils.model_utils import get_stable_diffusion_model
from my_utils.vaehook import VAEHook
from my_utils.wavelet_color_fix import adain_color_fix

from torchvision.transforms import ToPILImage, ToTensor
from my_utils.wavelet_color_fix import adain_color_fix

OUT_INDEX, CONTENT_INDEX, STYLE_INDEX = 0, 1, 2

class UnifiedModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Validate and set default args
        defaults = {
            'pretrained_model_name_or_path': 'stabilityai/stable-diffusion-2-1-base',
            'osediff_lora_path': None,
            'vae_encoder_tiled_size': 256,
            'vae_decoder_tiled_size': 256,
            'latent_tiled_size': 256,
            'latent_tiled_overlap': 32,
            'mixed_precision': 'fp16',
            'filter_percentile': 0.5,
            'enable_filtering': True
        }
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
                print(f"[INFO] Set default {key} = {value}")

        # 1. Load Base Diffusion Components (from OSEDiff)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

        # 2. Set OSEDiff's Unchangeable Parameters
        self.timesteps = torch.tensor([999], device=self.device).long()
        self.noise_scheduler.set_timesteps(1, device=self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        # 3. Load OSEDiff's LoRA Checkpoint
        if args.osediff_lora_path:
            lora_checkpoint = torch.load(args.osediff_lora_path, map_location="cpu")
            self.load_lora_ckpt(lora_checkpoint)
            print("[INFO] Successfully loaded OSEDiff LoRA weights.")

        # 4. Initialize Tiling Mechanisms (from OSEDiff)
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size
        )

        # 5. Register CACTIF's Custom Attention Control
        self.register_attention_control()
        print("[INFO] CACTIF Attention Processor has been registered.")

        # 6. Set Precision and Move to Device
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        
        if self.weight_dtype == torch.float16:
            print("[INFO] Applying VAE quant_conv dtype fix.")
            quant_conv_original_forward = self.vae.quant_conv.forward

            def patched_quant_conv_forward(h):
                return quant_conv_original_forward(h.to(self.weight_dtype))

            self.vae.quant_conv.forward = patched_quant_conv_forward

    @torch.no_grad()
    def forward(self, content_image, style_image, prompt):
        """
        Performs super-resolution with global style transfer.

        Args:
            content_image (torch.Tensor): Low-quality input image, shape (batch_size, 3, H, W).
            style_image (torch.Tensor): Style reference image, shape (batch_size, 3, H, W).
            prompt (list[str]): Text prompts for conditioning, length batch_size or 1.

        Returns:
            torch.Tensor: Super-resolved and styled image, shape (batch_size, 3, H, W).
        """
        batch_size = content_image.shape[0]
        if style_image.shape[0] != batch_size:
            print(f"[INFO] Broadcasting style image from batch size {style_image.shape[0]} to {batch_size}")
            style_image = style_image.repeat(batch_size, 1, 1, 1)

        # Handle single prompt or per-image prompts
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        elif len(prompt) != batch_size:
            raise ValueError("Prompt list must match batch size or be a single string")

        prompt_embeds = self._encode_prompt(prompt)

        content_image = content_image.to(self.weight_dtype)
        style_image = style_image.to(self.weight_dtype)

        latent_content = self.vae.encode(content_image).latent_dist.sample() * self.vae.config.scaling_factor
        latent_style = self.vae.encode(style_image).latent_dist.sample() * self.vae.config.scaling_factor

        print(f"latent_content shape: {latent_content.shape}")
        print(f"latent_style shape: {latent_style.shape}")

        # Fallback: Resize latent_style if dimensions don't match
        if latent_style.shape[2:] != latent_content.shape[2:]:
            print(f"[WARNING] Resizing latent_style from {latent_style.shape[2:]} to {latent_content.shape[2:]}")
            latent_style = F.interpolate(latent_style, size=latent_content.shape[2:], mode='bilinear', align_corners=False)

        # styled_latent = adain(latent_content, latent_style)
        styled_latent = latent_content

        print(f"styled_latent shape: {styled_latent.shape}")

        # Verify shapes before concatenation
        if styled_latent.shape[2:] != latent_content.shape[2:] or styled_latent.shape[2:] != latent_style.shape[2:]:
            raise ValueError(f"Shape mismatch: styled_latent {styled_latent.shape}, "
                            f"latent_content {latent_content.shape}, latent_style {latent_style.shape}")

        unet_input_latents = torch.cat([styled_latent, latent_content, latent_style], dim=0)
        unet_prompt_embeds = prompt_embeds.repeat(3, 1, 1)

        noise_pred = self._tiled_unet_pass(unet_input_latents, unet_prompt_embeds)

        denoised_latent = self.noise_scheduler.step(noise_pred, self.timesteps, styled_latent).prev_sample

        output_image = self.vae.decode(denoised_latent / self.vae.config.scaling_factor, return_dict=False)[0]
        output_image = (output_image / 2 + 0.5).clamp(0, 1)

        to_pil = ToPILImage()
        to_tensor = ToTensor()

        style_rgb = style_image
        if style_rgb.min() < 0:
            style_rgb = (style_rgb + 1) * 0.5
        style_pil = to_pil(style_rgb[0].cpu())

        out_tensors = []
        for i in range(output_image.shape[0]):
            base_pil = to_pil(output_image[i].cpu())
            fixed_pil = adain_color_fix(base_pil, style_pil)
            out_t = to_tensor(fixed_pil).to(self.device)
            out_tensors.append(out_t)
        
        final = torch.stack(out_tensors, dim=0)
        return final.clamp(0,1)

    def _encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        self.vae.encoder.forward = VAEHook(
            self.vae.encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder,
            fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu
        )
        self.vae.decoder.forward = VAEHook(
            self.vae.decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder,
            fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu
        )

    def load_lora_ckpt(self, lora_checkpoint):
        lora_conf_encoder = LoraConfig(r=lora_checkpoint["rank_unet"], init_lora_weights="gaussian", target_modules=lora_checkpoint["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=lora_checkpoint["rank_unet"], init_lora_weights="gaussian", target_modules=lora_checkpoint["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=lora_checkpoint["rank_unet"], init_lora_weights="gaussian", target_modules=lora_checkpoint["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n and n in lora_checkpoint["state_dict_unet"]:
                p.data.copy_(lora_checkpoint["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        
        # Load VAE LoRA
        vae_lora_conf_encoder = LoraConfig(
            r=lora_checkpoint["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=lora_checkpoint["vae_lora_encoder_modules"]
        )
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n and n in lora_checkpoint["state_dict_vae"]:
                p.data.copy_(lora_checkpoint["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

    def register_attention_control(self):
        model_self = self
        class AttentionProcessor:
            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet

            def attention_filtering(self, attn_map, V):
                prc_values = model_self.args.filter_percentile
                num_pixels = attn_map.shape[2]
                batch_size, num_heads, _, _ = attn_map.shape

                # Get tile dimensions from V shape
                head_dim = V.shape[-1]
                # pixels_per_head = V.shape[1] // num_heads
                pixels_per_head = V.shape[2]

                if pixels_per_head < 16:
                    print(f"[INFO] Skipping attention filtering for small feature map: pixels_per_head={pixels_per_head}")
                    return attn_map, V
                
                print(f"[INFO] attention filtering not skipped: pixels_per_head={pixels_per_head}")

                # attn_h = int(model_self.args.latent_tiled_size * (model_self.args.latent_tiled_size / (model_self.args.latent_tiled_size - model_self.args.latent_tiled_overlap)))
                # attn_w = pixels_per_head // attn_h
                # if attn_h * attn_w != pixels_per_head:
                #     raise ValueError(f"Cannot determine attention map dimensions: {pixels_per_head} pixels, tried {attn_h}x{attn_w}")

                side_length = int(pixels_per_head ** 0.5)
                if side_length * side_length != pixels_per_head:
                    # This might happen for non-square feature maps, handle as needed
                    print(f"[WARNING] Non-square feature map detected ({pixels_per_head} pixels). Falling back to original logic.")
                    attn_h = int(model_self.args.latent_tiled_size * (model_self.args.latent_tiled_size / (model_self.args.latent_tiled_size - model_self.args.latent_tiled_overlap)))
                    attn_w = pixels_per_head // attn_h
                else:
                    attn_h = attn_w = side_length

                print(f"[DEBUG] attention_filtering: num_pixels={num_pixels}, batch_size={batch_size}, num_heads={num_heads}, "
                      f"attn_h={attn_h}, attn_w={attn_w}, head_dim={head_dim}, V.shape={V.shape}")

                max_map = attn_map[OUT_INDEX].abs().sum(dim=0)
                max_idx = torch.argmax(max_map, dim=-1)

                V_rearranged = V.view(batch_size, num_heads, attn_h, attn_w, head_dim)
                cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
                cos_v_mean = []

                for i in range(attn_h):
                    for j in range(attn_w):
                        v_content_pixel = V_rearranged[CONTENT_INDEX, :, i, j, :]
                        style_pixel_idx = max_idx[i * attn_w + j].item()
                        style_i, style_j = style_pixel_idx // attn_w, style_pixel_idx % attn_w
                        v_style_pixel = V_rearranged[STYLE_INDEX, :, style_i, style_j, :]
                        cos_v_idx = cos(v_content_pixel.unsqueeze(0), v_style_pixel.unsqueeze(0))
                        cos_v_mean.append(torch.sum(torch.abs(cos_v_idx)))
                
                tensor_mean = torch.tensor(cos_v_mean, device=V.device)
                threshold_value = torch.quantile(tensor_mean, prc_values)
                is_weak = tensor_mean < threshold_value
                
                if torch.any(is_weak):
                    attn_map_rearranged = rearrange(attn_map, 'b c (h w) d -> b c h w d', h=attn_h, w=attn_w)
                    rows, cols = torch.nonzero(is_weak.view(attn_h, attn_w), as_tuple=True)
                    attn_map_rearranged[OUT_INDEX, :, rows, cols, :] = attn_map_rearranged[CONTENT_INDEX, :, rows, cols, :]
                    V_rearranged[OUT_INDEX, :, rows, cols, :] = V_rearranged[CONTENT_INDEX, :, rows, cols, :]
                    attn_map = rearrange(attn_map_rearranged, 'b c h w d -> b c (h w) d', h=attn_h, w=attn_w)
                    V = V_rearranged.view(batch_size, num_heads * attn_h * attn_w, head_dim)
                
                return attn_map, V

            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
                residual = hidden_states
                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = hidden_states.shape
                
                is_cross = encoder_hidden_states is not None
                encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
                
                query = attn.to_q(hidden_states)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                
                should_mix = not is_cross and "up" in self.place_in_unet
                
                if should_mix:
                    key[OUT_INDEX] = key[STYLE_INDEX]
                    value[OUT_INDEX] = value[STYLE_INDEX]

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = torch.baddbmm(
                    torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                    query,
                    key.transpose(-1, -2),
                    beta=0,
                    alpha=attn.scale,
                )
                attention_probs = attention_probs.softmax(dim=-1).to(query.dtype)

                if should_mix and model_self.args.enable_filtering:
                    num_heads = attn.heads
                    head_dim = query.shape[-1]
                    
                    attn_map_for_filter = attention_probs.view(batch_size, num_heads, sequence_length, sequence_length)
                    value_for_filter = value.view(batch_size, num_heads, sequence_length, head_dim)

                    attn_map_for_filter, value_for_filter = self.attention_filtering(attn_map_for_filter, value_for_filter)
                    
                    attention_probs = attn_map_for_filter.view(batch_size * num_heads, sequence_length, sequence_length)
                    value = value_for_filter.view(batch_size * num_heads, sequence_length, head_dim)

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)

                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor
                return hidden_states

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'Attention':
                net_.set_processor(AttentionProcessor(place_in_unet=place_in_unet))
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        for name, net in self.unet.named_children():
            if "down" in name or "up" in name or "mid" in name:
                register_recr(net, 0, name)

    def _gaussian_weights(self, tile_width, tile_height, n_batches=1):
        latent_width = tile_width
        latent_height = tile_height
        var = 0.01
        midpoint_x = (latent_width - 1) / 2
        x_probs = [np.exp(-(x - midpoint_x) * (x - midpoint_x) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var) for x in range(latent_width)]
        midpoint_y = (latent_height - 1) / 2
        y_probs = [np.exp(-(y - midpoint_y) * (y - midpoint_y) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var) for y in range(latent_height)]
        weights = np.outer(y_probs, x_probs)

        num_channels = self.unet.config.in_channels
        return torch.tile(torch.tensor(weights, device=self.device), (n_batches, num_channels, 1, 1))

    def _tiled_unet_pass(self, latents, prompt_embeds):
        batch_size, channels, h, w = latents.shape
        tile_size = self.args.latent_tiled_size
        tile_overlap = self.args.latent_tiled_overlap

        if tile_size <= 0 or tile_overlap < 0:
            raise ValueError("tile_size must be positive and tile_overlap non-negative")
        if tile_size > min(h, w):
            tile_size = min(h, w)
            print(f"[INFO] Adjusted tile_size to {tile_size} to fit latent dimensions")
        
        if h * w <= tile_size * tile_size:
            print("[INFO] Input is small, no latent tiling needed.")
            model_output = self.unet(latents, self.timesteps, encoder_hidden_states=prompt_embeds).sample
            return model_output[OUT_INDEX].unsqueeze(0)

        print(f"[INFO] Input size {h}x{w} requires latent tiling.")
        
        grid_rows = 0
        cur_y = 0
        while cur_y < h:
            cur_y = max(grid_rows * (tile_size - tile_overlap), 0) + tile_size
            grid_rows += 1

        grid_cols = 0
        cur_x = 0
        while cur_x < w:
            cur_x = max(grid_cols * (tile_size - tile_overlap), 0) + tile_size
            grid_cols += 1
        
        noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                ofs_y = max(row * (tile_size - tile_overlap), 0)
                ofs_x = max(col * (tile_size - tile_overlap), 0)
                end_y = min(ofs_y + tile_size, h)
                end_x = min(ofs_x + tile_size, w)
                ofs_y = end_y - tile_size
                ofs_x = end_x - tile_size

                input_tile = latents[:, :, ofs_y:end_y, ofs_x:end_x]
                print(f"[DEBUG] Processing tile: row={row}, col={col}, shape={input_tile.shape}")
                model_out_tile = self.unet(input_tile, self.timesteps, encoder_hidden_states=prompt_embeds).sample
                noise_preds.append(model_out_tile)
        
        tile_weights = self._gaussian_weights(tile_size, tile_size)
        stitched_noise_pred = torch.zeros((1, channels, h, w), device=self.device, dtype=self.weight_dtype)
        contributors = torch.zeros((1, channels, h, w), device=self.device, dtype=self.weight_dtype)
        
        for i, pred_tile in enumerate(noise_preds):
            row = i // grid_cols
            col = i % grid_cols
            ofs_y = max(row * (tile_size - tile_overlap), 0)
            ofs_x = max(col * (tile_size - tile_overlap), 0)
            end_y = min(ofs_y + tile_size, h)
            end_x = min(ofs_x + tile_size, w)
            ofs_y = end_y - tile_size
            ofs_x = end_x - tile_size

            styled_pred_tile = pred_tile[OUT_INDEX].unsqueeze(0)
            stitched_noise_pred[:, :, ofs_y:end_y, ofs_x:end_x] += styled_pred_tile * tile_weights
            contributors[:, :, ofs_y:end_y, ofs_x:end_x] += tile_weights

        stitched_noise_pred /= contributors
        
        return stitched_noise_pred
