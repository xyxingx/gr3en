# pylint: disable=all
# This file has been modified by Google DeepMind.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
import mediapy as media

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict, get_model_state_dict, 
        set_optimizer_state_dict, get_optimizer_state_dict,
        StateDictOptions
    )

import torch.distributed.checkpoint as dcp
# from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from wan.distributed.fsdp import AppState


from wan.distributed.fsdp import shard_model, wrap_with_fsdp2
from wan.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from wan.distributed.util import get_world_size
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)   
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from training import flow_match_scheduler #import FlowMatchScheduler
from training import functions
from training import controlnet

from diffusers.utils.export_utils import export_to_video
from diffusers.video_processor import VideoProcessor

class WanI2V(nn.Module):

    def __init__(
        self,
        config,
        model_configs=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        training=False,
        **kwargs
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        
        super().__init__()
        self.args = model_configs
        self.training = training
        if self.training:
            self.low_model_or_high_model = self.args.low_model_or_high_model

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        checkpoint_dir = self.args.ckpt_dir

        
        # shard_fn = partial(shard_model, device_id=device_id) # FSDP 1
        shard_fn = wrap_with_fsdp2

        # self.text_encoder = T5EncoderModel(
        #     text_len=config.text_len,
        #     dtype=config.t5_dtype,
        #     device=torch.device('cpu'),
        #     checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
        #     tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        #     shard_fn=shard_fn if t5_fsdp else None,
        # )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device, dtype=torch.bfloat16)
        

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        if not training:
            # load both models during inference
            self.low_noise_model = WanModel.from_pretrained(
                checkpoint_dir, subfolder=config.low_noise_checkpoint)
            self.low_noise_model = self._configure_model(
                model=self.low_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype)
            
            # assume low noise model frozen; but this will change if low noise model is loaded
            self.low_noise_loaded=False

            self.high_noise_model = WanModel.from_pretrained(
                checkpoint_dir, subfolder=config.high_noise_checkpoint)
            self.high_noise_model = self._configure_model(
                model=self.high_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype)
            self.high_noise_loaded=False
            
            
        else:
            if self.low_model_or_high_model == 'low':
                subfolder = config.low_noise_checkpoint
            elif self.low_model_or_high_model == 'high':
                subfolder = config.high_noise_checkpoint
            self.dit = WanModel.from_pretrained(
                checkpoint_dir, subfolder=subfolder)
            self.dit = self._configure_model(
                model=self.dit,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype)
            

        # to process poses  
        self.controlnet = controlnet.ControlNet().to(self.device, dtype=torch.bfloat16)
        self.latent_dpt = None

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

        if self.training:
            # https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/wan_video_new.py#L37
            self.scheduler = flow_match_scheduler.FlowMatchScheduler(
                shift=5, sigma_min=0.0, extra_one_step=True
            )
            self.scheduler.set_timesteps(1000, training=True)

        self.text_embeds = torch.load(os.path.join(checkpoint_dir, 'prompt_embed.pt')).to(self.device, dtype=torch.bfloat16)
        
        
    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """

        if not self.training:
            model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def set_requires_grad_and_lr(self, learning_rate):
        
        params_to_optimize = []
        self.dit.requires_grad_(True)
        t_params = {"params": self.dit.parameters(), "lr": 0.1*learning_rate}
        params_to_optimize.append(t_params)
        self.dit.train()

        self.controlnet.requires_grad_(True)
        p_params = {"params": self.controlnet.parameters(), "lr": learning_rate}
        params_to_optimize.append(p_params)
        self.controlnet.train()

        return params_to_optimize
    
    def load_checkpoint(self, checkpoint_path, optimizer, lr_scheduler):
        full_pt = torch.load(checkpoint_path, mmap=True, weights_only=True, map_location="cpu")
        
        options = StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
        )

        set_model_state_dict(
            model=self.dit,
            model_state_dict=full_pt['dit_state_dict'],
            options=options
        )
        
        if self.controlnet is not None:
            set_model_state_dict(
                model=self.controlnet,
                model_state_dict=full_pt['controlnet_state_dict'],
                options=options
            )
        
        if self.latent_dpt is not None:
            set_model_state_dict(
                model=self.latent_dpt,
                model_state_dict=full_pt['latent_dpt_state_dict'],
                options=options
            )
        
        set_optimizer_state_dict(
            model=self,
            optimizers=optimizer,
            optim_state_dict=full_pt['optimizer_state_dict'],
            options=options
        )
        
        lr_scheduler.load_state_dict(full_pt['lr_scheduler_state_dict'])
        global_step = full_pt['global_step']
        epoch = full_pt['epoch']
        
        return global_step, epoch
        
        
    
    def save_checkpoint(self, checkpoint_path, optimizer, lr_scheduler, global_step, epoch, is_main):
        
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        
        dit_state_dict = get_model_state_dict(
            model=self.dit,
            options=options
        )
                
        if self.controlnet is not None:
            controlnet_state_dict = get_model_state_dict(
                model=self.controlnet,
                options=options
            )
            
        if self.latent_dpt is not None:
            latent_dpt_state_dict = get_model_state_dict(
                model=self.latent_dpt,
                options=options
            )
        
        optimizer_state_dict = get_optimizer_state_dict(
            model=self,
            optimizers=optimizer,
            options=options
        )
        
        
        if is_main:
            full_pt = {
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'optimizer_state_dict':optimizer_state_dict,
                'dit_state_dict': dit_state_dict,
            }
            if self.controlnet is not None:
                full_pt['controlnet_state_dict'] = controlnet_state_dict
            if self.latent_dpt is not None:
                full_pt['latent_dpt_state_dict'] = latent_dpt_state_dict
        
            torch.save(full_pt, checkpoint_path)
            logging.info(f"checkpoint saved to {checkpoint_path}")
            
    def save_sharded_checkpoint(self, checkpoint_path, optimizer, lr_scheduler, global_step, epoch, is_main):
        '''
        https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
        '''
        
        sharded_dir = os.path.join(checkpoint_path, f'sharded-{global_step}')
        os.makedirs(sharded_dir, exist_ok=True)
        
        state_dict = { "app": AppState(self, optimizer) }
        dcp.save(state_dict, checkpoint_id=sharded_dir)
            
        if is_main:
            local_state_dict = {
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch
            }
            
            if self.controlnet is not None:
                local_state_dict["controlnet_state_dict"] = self.controlnet.state_dict()
            if self.latent_dpt is not None:
                local_state_dict["latent_dpt_state_dict"] = self.latent_dpt.state_dict()
    
            torch.save(local_state_dict, os.path.join(checkpoint_path, f"local_state_dict_{global_step}.pt"))
            logging.info(f"checkpoint saved to {sharded_dir}")


    def diffusion_forward(self, batch, overwrite_timesteps=None):
        
        if self.low_model_or_high_model == 'low':
            max_timestep_boundary = 0.9; min_timestep_boundary=0
        if self.low_model_or_high_model == 'high':
            max_timestep_boundary = 1; min_timestep_boundary=0.9
        max_timestep_boundary = int(max_timestep_boundary*self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(min_timestep_boundary*self.scheduler.num_train_timesteps)

        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))

        if overwrite_timesteps is not None:
            timestep_id = overwrite_timesteps

        # TODO: from https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/wan_video_new.py#L79
        # but this doesn't make sense; timesteps is from [1000,999,...,4];
        # but timestep_id is sampled between [0,900] for 'low' or [900,999] for 'high',
        # so 'low' would actually mean 'high' noise
        # timestep = self.scheduler.timesteps[timestep_id].to(dtype=torch.bfloat16, device=self.device)
        
        # so instead, treat timestep_id as the actual timestep, and the following functions
        # e.g. add_noise() would be correct
        timestep = timestep_id
        
        video_latents = batch['video_latents']
        noise = torch.randn_like(video_latents)
        # takes the sampled timestep from randint and returns the actual timestep
        # i.e. the closest item in scheduler.timesteps
        noisy_model_input, timestep = self.scheduler.add_noise(video_latents, noise, timestep)

        # from https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/wan_video_new.py#L82
        # and https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/schedulers/flow_match.py#L98
        # this matches the velocity prediction
        target = self.scheduler.training_target(video_latents, noise)

        return noisy_model_input, target, timestep
    

    def prepare_batch(self, batch, device, dtype=torch.bfloat16):
        videos = batch['videos'].to(device, dtype=dtype)
        poses = batch['poses'].to(device, dtype=dtype)

        # preprocess / validate for vae 
        # check spatial dimensions:
        b,f,c,h,w = videos.shape
        assert (h,w)==(480,832) or (h,w)==(720,1280)
        pose_h, pose_w = poses.shape[-2:]
        assert (pose_h, pose_w)==(60,104) or (pose_h, pose_w)==(90,160) 

        # input to vae (and its corresponding output) should be B C F H W
        video_latents = self.vae.encode_batch(videos.transpose(1,2))

        # first image embed
        # vae input needs to be padded to same frame num count
        first_image = videos[:,0:1]
        vae_input = torch.cat([first_image, first_image.new_zeros((b,f-1,c,h,w))], dim=1)
        first_image_latent = self.vae.encode_batch(vae_input.transpose(1,2))
        # then the latent needs to be padded to have channel dim 20 (from vae output of 16)
        msk = torch.ones(1, f, pose_h, pose_w, device=self.device, dtype=first_image_latent.dtype) # pose_hw = latent_hw
        msk[:, 1:] = 0
        msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), 
                         msk[:, 1:]],
                        dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, pose_h, pose_w)
        msk = msk.transpose(1, 2)[0]
        msk = msk[None].repeat(b,1,1,1,1) # add batch dimension
        first_image_latent = torch.cat([msk, first_image_latent], dim=1) # B,20,f,h,w

        # text embed
        text_embeds = self.text_embeds[None].repeat(b,1,1) # b,L,D

        # preprocess poses 
        poses = self.controlnet.preprocess_poses(poses)
        pb,pc,pf,ph,pw = poses.shape
        assert (pb,video_latents.shape[1],pf,ph,pw)==video_latents.shape

        # TODO: vggt latents
        vggt_latents = None
        
        return dict(
            videos=videos,
            poses=poses,
            video_latents=video_latents,
            text_embeds=text_embeds,
            first_image_latent=first_image_latent,
            vggt_latents=vggt_latents,
        )
        
    
    
    def run_one_step(self, batch, noisy_model_input, timestep):
        model_out = self.dit(
            noisy_model_input,
            timestep,
            batch['text_embeds'],
            None, # batch['max_seq_len'],
            batch['first_image_latent'],
            self.training,
            poses=batch['poses'],
            pose_proj=self.controlnet.pose_proj if self.controlnet is not None else None
        )

        return model_out

    @torch.inference_mode
    def validation_loop(self, val_loader, device, world_size, global_step, outdir):
        # model.transformer.eval(); model.controlnet.eval(); model.latent_dpt.eval()
        
        num_val_batches = 0
        if self.low_model_or_high_model == 'high':
            fixed_timesteps = [999, 900]
        elif self.low_model_or_high_model == 'low':
            fixed_timesteps = [899,300]
        total_val_losses = {timestep: 0.0 for timestep in fixed_timesteps}
        
        for val_step, batch in enumerate(val_loader):
            batch = self.prepare_batch(batch, device)
            num_val_batches += 1
            
            for defined_timestep in fixed_timesteps:
                noisy_model_input, target, timesteps = self.diffusion_forward(batch, overwrite_timesteps=defined_timestep)
                # noisy_model_input, controlnet_states = prepare_conditioning(noisy_model_input, batch, timesteps, model)
                
                model_output = self.run_one_step(batch, noisy_model_input, timesteps)
                
                loss = torch.nn.functional.mse_loss(model_output.float(), target.float())
                loss = loss * self.scheduler.training_weight(timesteps)
                total_val_losses[defined_timestep] += loss

                # save mp4s
                self.vis_training(model_out=model_output, noisy_model_input=noisy_model_input, timesteps=defined_timestep,
                             step=global_step, outdir=outdir, ref=batch['videos'])

        local_val_losses = {timestep: total_val_losses[timestep] / num_val_batches for timestep in fixed_timesteps}
        avg_val_losses = {}
        
        for key, val in local_val_losses.items():
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            avg_val_losses[key] = val / world_size
                
        # model.transformer.train(); model.controlnet.train(); model.latent_dpt.train()
        torch.cuda.empty_cache() 
        
        return avg_val_losses
    
    @torch.inference_mode
    def vis_training(self, ref, model_out, noisy_model_input, step, timesteps, outdir):
        # if accelerator.is_main_process: # and step == 0:
        # output_path_file = os.path.join('training_vis', f"traj10_out_{epoch}_{timesteps.item()}.mp4")
        # output_path_file_reference = os.path.join('training_vis', f"traj10_reference_{epoch}_{timesteps.item()}.mp4")
        output_path_file_out_reference = os.path.join(outdir, f"{step}_{dist.get_rank()}_{timesteps}.mp4")
        
        pred_x0 = self.scheduler.velocity_to_x0(noisy_model_input, model_out, timesteps)

        video_processor = VideoProcessor(vae_scale_factor=8)
        torch.cuda.empty_cache() 
        video = self.vae.decode_batch(pred_x0).cpu()
        video = video_processor.postprocess_video(video=video, output_type='pil')[0] # N, H, W, 3
        reference_frames = ref.permute(0, 2, 1, 3, 4)[0:1].cpu() # 1, 3, N, H, W
        reference_frames = video_processor.postprocess_video(video=reference_frames, output_type='pil')[0] # N, H, W, 3
        # export_to_video(video, output_path_file, fps=8)
        # export_to_video(reference_frames, output_path_file_reference, fps=8)
        out_reference_frames = [
            functions.stack_images_horizontally(frame_reference, frame_out)
            for frame_out, frame_reference in zip(video, reference_frames)
            ]
        
        # export_to_video(out_reference_frames, output_path_file_out_reference, fps=10)
        media.write_video(output_path_file_out_reference, np.array(out_reference_frames), fps=10)

        del video, reference_frames
        torch.cuda.empty_cache() 

        # print("timestep / step: ", timesteps, "/", step)



    def generate(self,
                 input_prompt,
                 img,
                 data_batch=None,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        
        F = frame_num
        
        if data_batch is None:
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
            h, w = img.shape[1:]
            aspect_ratio = h / w
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
                self.patch_size[1] * self.patch_size[1])
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
                self.patch_size[2] * self.patch_size[2])
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]
            poses = None
        else:
            img = data_batch['videos'][0].to(self.device, torch.bfloat16) # CHW
            h,w = 480,832
            lat_h,lat_w = 60, 104
            poses = data_batch['poses'][None].to(self.device, torch.bfloat16) # 1 F 7 h w 
            assert F==poses.shape[1]

            if getattr(self, 'controlnet_low_noise', None) is not None:
                poses_low_noise = self.controlnet_low_noise.preprocess_poses(poses) 
            poses = self.controlnet.preprocess_poses(poses) # 1 c f h w


        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        # if not self.t5_cpu:
        #     self.text_encoder.model.to(self.device)
        #     context = self.text_encoder([input_prompt], self.device)
        #     context_null = self.text_encoder([n_prompt], self.device)
        #     if offload_model:
        #         self.text_encoder.model.cpu()
        # else:
        #     context = self.text_encoder([input_prompt], torch.device('cpu'))
        #     context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        #     context = [t.to(self.device) for t in context]
        #     context_null = [t.to(self.device) for t in context_null]


        context = torch.load('prompt_embed.pt').to(self.device, dtype=torch.bfloat16)
        context_null = torch.load('null_prompt_embed.pt').to(self.device, dtype=torch.bfloat16)
        context = [context]
        context_null = [context_null]

        
        # input to vae shape: C, F, H, W (3, 81, 512, 768)
        # output y is a list with len 1, shape [16, 21, 64, 96] (cfhw)
        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)
            ], dim=1).to(self.device)
        ])[0]

        # msk shape: 4,21,64,96; fhw dims match; 4 is needed for embedding later (not sure why)
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y]
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y]
            }

            if offload_model:
                torch.cuda.empty_cache()


            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(t, boundary, offload_model)
                # model = self.low_noise_model.to(self.device)

                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                if poses is not None:
                    # always provide poses for high noise (t>boundary)
                    # also provide if low noise model was loaded (i.e. trained with the condition)
                    if t.item() >= boundary:
                        arg_c['poses']=arg_null['poses']=poses
                        arg_c['pose_proj']=arg_null['pose_proj']=self.controlnet.pose_proj
                    elif getattr(self, 'controlnet_low_noise', None) is not None:
                        arg_c['poses']=arg_null['poses']=poses_low_noise
                        arg_c['pose_proj']=arg_null['pose_proj']=self.controlnet_low_noise.pose_proj


                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
