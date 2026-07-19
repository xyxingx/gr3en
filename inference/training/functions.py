# pylint: disable=all
# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append('..')
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from datetime import timedelta
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch.distributed as dist


# from diffusers.utils.export_utils import export_to_video
# from diffusers.video_processor import VideoProcessor


def prepare_batch(batch, model, device, use_autocast=True):

    args = model.args
    dtype = model.vae.dtype
    transformer_config = model.transformer.module.config if hasattr(model.transformer, "module") else model.transformer.config
    assert not (args.interpolate_traj and args.first_image_condition), "interpolate_traj and first_image_condition cannot be True at the same time"

    videos = batch['videos'].to(device, dtype=dtype)
    poses = batch['poses'].to(device, dtype=dtype)
    prompts = batch['prompts']
    img_index = batch['index']

    # ====videos, prompts, rotary embeds====
    video_latents = encode_video(videos, model.vae, device)
    # prompt_embeds = compute_prompt_embeddings(
    #                 model.tokenizer,
    #                 model.text_encoder,
    #                 prompts,
    #                 transformer_config.max_text_seq_length,
    #                 device,
    #                 model.vae.dtype,
    #                 requires_grad=False,
    #             )
    
    # shape: 1, 226, 4096
    prompt_embeds = model.saved_prompt_embeds.to(device, dtype=dtype)
    prompt_embeds = prompt_embeds.repeat(video_latents.shape[0], 1, 1)

    # image_rotary_emb = (
    #     prepare_rotary_positional_embeddings(
    #         height=videos.shape[-2],
    #         width=videos.shape[-1],
    #         num_frames=video_latents.shape[1],
    #         vae_scale_factor_spatial=transformer_config.vae_scale_factor_spatial,
    #         patch_size=transformer_config.patch_size,
    #         attention_head_dim=transformer_config.attention_head_dim,
    #         device=vae.device,
    #     )
    #     if transformer_config.use_rotary_positional_embeddings # False for cogvideox 2B
    #     else None
    # )    
    
    image_rotary_emb = None

    # ====additional conditioning====
    first_image_latent = encode_video(videos[:,0:1], model.vae, device) if args.first_image_condition else None
    
    if args.interpolate_traj:
        start_end_latents = []
        start_latent = encode_video(videos[:,0:1], model.vae, device) 
        last_latent = encode_video(videos[:,-1:], model.vae, device)
        start_end_latents.append(start_latent)
        start_end_latents.append(last_latent)
    else:
        start_end_latents = None

    
    if args.use_vggt_latents:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                vggt_inputs = batch['vggt_inputs'].to(device, dtype=dtype)
                aggregated_tokens_list, patch_start_idx = model.vggt_encoder(vggt_inputs, return_intermediates=True)

            # sanity check vggt encoder is used correctly
            # predictions = vggt_encoder(vggt_inputs, return_intermediates=False)
            # from vggt.visualize_predictions import predictions_to_glb
            # scene = predictions_to_glb(predictions)
            # scene.export(f'scene_{img_index[0]}.glb')
            # import ipdb; ipdb.set_trace()


        shape_placeholder = torch.zeros(vggt_inputs.shape[0], vggt_inputs.shape[1], 3, 350, 518)
        # vggt latents shape: B, N, 256, 200, 296
        vggt_latents = model.latent_dpt(aggregated_tokens_list, images=shape_placeholder, patch_start_idx=patch_start_idx).contiguous().to(dtype=dtype)
        # vggt_latents = torch.randn((1, 17, 256, 200, 296)).to(device, dtype=dtype).contiguous()

    else:
        vggt_latents = None


    return dict(
        videos=videos,
        poses=poses,
        video_latents=video_latents,
        prompts=prompts,
        prompt_embeds=prompt_embeds,
        image_rotary_emb=image_rotary_emb,
        first_image_latent=first_image_latent,
        start_end_latents=start_end_latents,
        vggt_latents=vggt_latents,
    )


def prepare_conditioning(noisy_model_input, batch, timesteps, model):
    
    controlnet, condition_preprocess = model.controlnet, model.condition_preprocess
    args = model.args
    
    
    if controlnet is not None:
        assert args.interpolate_traj is False, "still need to implement interpolate_conditions with controlnet"
        assert condition_preprocess is None, "condition_preprocess cannot be used with controlnet"
        controlnet_states = controlnet(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=batch['prompt_embeds'],
            image_rotary_emb=batch['image_rotary_emb'],
            controlnet_states=batch['poses'],
            start_end_latents=batch['start_end_latents'],
            first_image_latent=batch['first_image_latent'],
            vggt_latents=batch['vggt_latents'],
            return_dict=False,
        )[0]
        if isinstance(controlnet_states, (tuple, list)):
            controlnet_states = [x.to(dtype=noisy_model_input.dtype) for x in controlnet_states]
        else:
            controlnet_states = controlnet_states.to(dtype=noisy_model_input.dtype)


        noisy_model_input = noisy_model_input

    else:
        controlnet_states = None


    if condition_preprocess is not None:
        if args.use_vggt_latents:
            print("vggt latents not implemented for condition_preprocess yet")
            import ipdb; ipdb.set_trace()
        noisy_model_input = condition_preprocess(noisy_model_input, batch['poses'], 
            batch['first_image_latent'], batch['vggt_latents'], batch['start_end_latents'])


    return noisy_model_input, controlnet_states


def diffusion_forward(batch, model, overwrite_timesteps=None):
    
    args = model.args
    scheduler = model.scheduler
    
    
    video_latents = batch['video_latents']
    noise = torch.randn_like(video_latents)
    batch_size, num_frames, num_channels, height, width = video_latents.shape

    # sample timesteps
    if args.enable_time_sampling:
        if torch.rand(1).item() < args.percentage_of_truncated_timesteps:
        
            if args.time_sampling_type == "truncated_normal":
                time_sampling_dict = {
                    'mean': args.time_sampling_mean,
                    'std': args.time_sampling_std,
                    'a': 1 - args.controlnet_guidance_end, # lower bound of the sampled timesteps 
                    'b': 1 - args.controlnet_guidance_start,
                }
                timesteps = torch.nn.init.trunc_normal_(
                    torch.empty(batch_size, device=video_latents.device), **time_sampling_dict
                    ) * scheduler.config.num_train_timesteps
            elif args.time_sampling_type == "truncated_uniform":
                timesteps = torch.randint(
                    int((1- args.controlnet_guidance_end) * scheduler.config.num_train_timesteps),
                    int((1 - args.controlnet_guidance_start) * scheduler.config.num_train_timesteps),
                    (batch_size,), device=video_latents.device
                )
        else:
            timesteps = torch.randint(
                0, 750, (batch_size,), device=video_latents.device
            )
    else:    
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (batch_size,), device=video_latents.device
        )

    
    # timesteps = torch.randint(
    #         999, 1000, (batch_size,), device=model_input.device
    #     )

    
    if overwrite_timesteps is not None:
        timesteps = overwrite_timesteps
        
    timesteps = timesteps.long()

    noisy_model_input = scheduler.add_noise(video_latents, noise, timesteps)

    return noisy_model_input, timesteps


def get_loss(target, model_output, noisy_model_input, timesteps, model):
    model_pred = model.scheduler.get_velocity(model_output, noisy_model_input, timesteps)
    alphas_cumprod = model.scheduler.alphas_cumprod[timesteps]
    weights = 1 / (1 - alphas_cumprod)
    while len(weights.shape) < len(model_pred.shape):
        weights = weights.unsqueeze(-1)
        
    weights = weights.to(model_pred.dtype)

    loss = torch.mean((weights * (model_pred - target) ** 2).reshape(model_pred.shape[0], -1), dim=1)
    loss = loss.mean()


    return loss, model_pred


# def vis_training(batch, model_pred, vae, args, step, timesteps):
#     # if accelerator.is_main_process: # and step == 0:
#     with torch.no_grad():
#         # output_path_file = os.path.join('training_vis', f"traj10_out_{epoch}_{timesteps.item()}.mp4")
#         # output_path_file_reference = os.path.join('training_vis', f"traj10_reference_{epoch}_{timesteps.item()}.mp4")
#         tmp_outdir = args.output_dir.split('/')[-1]
#         output_path_file_out_reference = os.path.join('training_vis', f"test_{step}.mp4")

#         video_processor = VideoProcessor(vae_scale_factor=8)
#         video = decode_latents(vae, model_pred[0:1]) # 1, 3, N, H, W
#         video = video_processor.postprocess_video(video=video, output_type='pil')[0] # N, H, W, 3
#         reference_frames = batch["videos"].permute(0, 2, 1, 3, 4)[0:1] # 1, 3, N, H, W
#         reference_frames = video_processor.postprocess_video(video=reference_frames, output_type='pil')[0] # N, H, W, 3
#         # export_to_video(video, output_path_file, fps=8)
#         # export_to_video(reference_frames, output_path_file_reference, fps=8)
#         out_reference_frames = [
#             stack_images_horizontally(frame_reference, frame_out)
#             for frame_out, frame_reference in zip(video, reference_frames)
#             ]
#         export_to_video(out_reference_frames, output_path_file_out_reference, fps=8)

#     print("timestep / step: ", timesteps.item(), "/", step)

@torch.inference_mode
def validation_loop(val_loader, model, device, world_size):
    # model.transformer.eval(); model.controlnet.eval(); model.latent_dpt.eval()
    
    num_val_batches = 0
    fixed_timesteps = [999, 900, 700]
    total_val_losses = {timestep: 0.0 for timestep in fixed_timesteps}
    
    for val_step, batch in enumerate(val_loader):
        batch = prepare_batch(batch, model, device)
        num_val_batches += 1
        
        for timestep in fixed_timesteps:
            timesteps = torch.ones(1, device=device) * timestep
            noisy_model_input, timesteps = diffusion_forward(batch, model, overwrite_timesteps=timesteps)
            noisy_model_input, controlnet_states = prepare_conditioning(noisy_model_input, batch, timesteps, model)
            
            model_output = model.transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=batch['prompt_embeds'],
                timestep=timesteps,
                image_rotary_emb=batch['image_rotary_emb'],
                controlnet_states=controlnet_states,
                controlnet_weights=model.args.controlnet_weights,
                return_dict=False,
                interpolate_traj=(model.args.interpolate_traj and model.args.use_condition_embedding),
            )[0]
            
            loss, model_pred = get_loss(batch['video_latents'], model_output, noisy_model_input, timesteps, model)   
            total_val_losses[timestep] += loss
        
    local_val_losses = {timestep: total_val_losses[timestep] / num_val_batches for timestep in fixed_timesteps}
    avg_val_losses = {}
    
    for key, val in local_val_losses.items():
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        avg_val_losses[key] = val / world_size
            
    # model.transformer.train(); model.controlnet.train(); model.latent_dpt.train()
    torch.cuda.empty_cache() 
    
    return avg_val_losses
        
        
        
        
        
        
        
        

#====functions one level below====


def decode_latents(vae, latents):
    latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    latents = 1 / vae.config.scaling_factor * latents

    frames = vae.decode(latents.to(vae.dtype)).sample
    return frames

def encode_video(video, vae, device):
    video = video.to(device, dtype=vae.dtype)
    video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
    return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)







#====helper functions====

# def stack_images_horizontally(image1: Image.Image, image2: Image.Image):
#     from PIL import Image
#     # Ensure both images have the same height
#     height = max(image1.height, image2.height)
#     width = image1.width + image2.width
    
#     # Create a new blank image with the combined width and the maximum height
#     new_image = Image.new('RGB', (width, height))
    
#     # Paste the images into the new image
#     new_image.paste(image1, (0, 0))
#     new_image.paste(image2, (image1.width, 0))
    
#     return new_image

def stack_images_horizontally(*images: Image.Image, align_to_top: bool = False) -> Image.Image:
    """
    Stitches a sequence of PIL Images together horizontally.

    Args:
        *images: A variable number of PIL Image objects.
        align_to_top: If True, all images are aligned to the top edge. 
                      If False (default), they are centered vertically.

    Returns:
        A new PIL Image object containing all images stitched together.
    """
    if not images:
        raise ValueError("At least one image must be provided.")

    # Convert all images to RGB mode to ensure they can be pasted onto the new image
    images = [img.convert('RGB') for img in images]

    # Calculate the total width and the maximum height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the combined width and the maximum height
    new_image = Image.new('RGB', (total_width, max_height))

    current_width = 0
    for img in images:
        if align_to_top:
            # Align to the top
            paste_position = (current_width, 0)
        else:
            # Center vertically
            paste_position = (current_width, (max_height - img.height) // 2)
            
        new_image.paste(img, paste_position)
        current_width += img.width

    return new_image

def plot_validation_losses(val_loss_history, output_dir, global_step):
    """
    Plot validation loss curves and save the plot.
    
    Args:
        val_loss_history: dict with timestep -> list of (step, loss) tuples
        output_dir: directory to save the plot
        global_step: current global step
    """
    plt.figure(figsize=(10, 6))
    
    for timestep, losses in val_loss_history.items():
        if losses:  # Only plot if we have data
            steps, loss_values = zip(*losses)
            plt.plot(steps, loss_values, label=f'Timestep {timestep}', marker='o', markersize=3)
    
    plt.xlabel('Global Step')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss Curves (Step {global_step})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'validation_loss_curves.png')
    with open(plot_path, 'wb') as f:
        plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close()
    
    # logging.info(f"Validation loss plot saved to {plot_path}")


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw"]
    if args.optimizer not in supported_optimizers:
        logging.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.optimizer.lower() == "adamw":
        optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    return optimizer

