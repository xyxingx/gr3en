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
from contextlib import contextmanager
from functools import partial
import gc
import logging
import math
import os
import random
import sys
import types
from diffusers.utils.export_utils import export_to_video
from diffusers.video_processor import VideoProcessor
import mediapy as media
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
from training import controlnet  # import ControlNet
from training import flow_match_scheduler  # import FlowMatchScheduler
from training import functions
from wan.distributed.fsdp import AppState
from wan.distributed.fsdp import shard_model, wrap_with_fsdp2
from wan.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from wan.distributed.util import get_world_size
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import Wan2_2_VAE
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import best_output_size, masks_like, masks_like_batch


@contextmanager
def without_submodules(module: torch.nn.Module, names):
  """Temporarily remove one or more direct child submodules from `module`.

  `names` can be a string or an iterable of strings. Missing names are ignored.
  """
  if isinstance(names, str):
    names = [names]

  removed = []  # (name, submodule_or_None)
  for name in names:
    sub = module._modules.pop(name, None)  # unregister if present
    removed.append((name, sub))

  try:
    yield
  finally:
    # restore in reverse order
    for name, sub in reversed(removed):
      if sub is not None:
        module.add_module(name, sub)


class WanTI2V(nn.Module):

  def __init__(
      self,
      config,
      checkpoint_dir=None,
      model_configs=None,
      device_id=0,
      rank=0,
      t5_fsdp=False,
      dit_fsdp=False,
      use_sp=False,
      init_on_cpu=True,
      convert_model_dtype=False,
      training=False,
      enable_cfg=True,
      **kwargs,
  ):
    r"""Initializes the Wan text-to-video generation model components.

    Args:
        config (EasyDict): Object containing model parameters initialized from
          config.py
        checkpoint_dir (`str`): Path to directory containing model checkpoints
        device_id (`int`,  *optional*, defaults to 0): Id of target GPU device
        rank (`int`,  *optional*, defaults to 0): Process rank for distributed
          training
        t5_fsdp (`bool`, *optional*, defaults to False): Enable FSDP sharding
          for T5 model
        dit_fsdp (`bool`, *optional*, defaults to False): Enable FSDP sharding
          for DiT model
        use_sp (`bool`, *optional*, defaults to False): Enable distribution
          strategy of sequence parallel.
        t5_cpu (`bool`, *optional*, defaults to False): Whether to place T5
          model on CPU. Only works without t5_fsdp.
        init_on_cpu (`bool`, *optional*, defaults to True): Enable initializing
          Transformer Model on CPU. Only works without FSDP or USP.
        convert_model_dtype (`bool`, *optional*, defaults to False): Convert DiT
          model parameters dtype to 'config.param_dtype'. Only works without
          FSDP.
    """
    super().__init__()
    self.device = torch.device(f'cuda:{device_id}')
    self.config = config
    self.rank = rank
    self.t5_cpu = None
    self.init_on_cpu = init_on_cpu
    self.enable_cfg = enable_cfg
    self.num_train_timesteps = config.num_train_timesteps
    self.param_dtype = config.param_dtype

    if t5_fsdp or dit_fsdp or use_sp:
      self.init_on_cpu = False

    checkpoint_dir = model_configs.ckpt_dir

    # FSDP 1
    # shard_fn = partial(shard_model, device_id=device_id)
    shard_fn = wrap_with_fsdp2

    self.vae_stride = config.vae_stride
    self.patch_size = config.patch_size
    self.vae = Wan2_2_VAE(
        vae_pth=os.path.join(checkpoint_dir, 'Wan2.2_VAE.pth'),
        device=self.device,
        dtype=torch.bfloat16,
    )

    logging.info(f'Creating WanModel from {checkpoint_dir}')
    self.dit = WanModel.from_pretrained(checkpoint_dir)
    # magic REGR add-on
    self.dit.add_REGR_modules(ae_scale=model_configs.ae_scale)
    self.dit = self._configure_model(
        model=self.dit,
        use_sp=use_sp,
        dit_fsdp=dit_fsdp,
        shard_fn=shard_fn,
        convert_model_dtype=convert_model_dtype,
    )

    # simple scalar time_mask to replace latent_dpt.time_mask
    self.time_mask = 10.0

    self.controlnet = controlnet.ControlNet(
        use_5b_model=True, use_vggt_latents=False
    ).to(self.device, dtype=torch.bfloat16)

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

    self.text_embeds = torch.load(
        os.path.join(checkpoint_dir, 'prompt_embed.pt')
    ).to(self.device, dtype=torch.bfloat16)
    self.null_text_embeds = torch.load(
        os.path.join(checkpoint_dir, 'null_prompt_embed.pt')
    ).to(self.device, dtype=torch.bfloat16)

  def _configure_model(
      self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype
  ):
    """Configures a model object.

    This includes setting evaluation modes, applying distributed parallel
    strategy, and handling device placement.

    Args:
        model (torch.nn.Module): The model instance to configure.
        use_sp (`bool`): Enable distribution strategy of sequence parallel.
        dit_fsdp (`bool`): Enable FSDP sharding for DiT model.
        shard_fn (callable): The function to apply FSDP sharding.
        convert_model_dtype (`bool`): Convert DiT model parameters dtype to
          'config.param_dtype'. Only works without FSDP.

    Returns:
        torch.nn.Module:
            The configured model.
    """
    if not self.training:
      model.eval().requires_grad_(False)

    if use_sp:
      for block in model.blocks:
        block.self_attn.forward = types.MethodType(
            sp_attn_forward, block.self_attn
        )
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

  def set_requires_grad_and_lr(self, learning_rate):
    params_to_optimize = []

    self.dit.requires_grad_(True)
    t_params = {'params': self.dit.parameters(), 'lr': 0.1 * learning_rate}
    params_to_optimize.append(t_params)
    self.dit.train()

    self.controlnet.requires_grad_(True)
    p_params = {'params': self.controlnet.parameters(), 'lr': learning_rate}
    params_to_optimize.append(p_params)
    self.controlnet.train()

    return params_to_optimize

  def load_checkpoint(
      self, checkpoint_path, optimizer=None, lr_scheduler=None, is_main=False
  ):
    full_pt = torch.load(
        checkpoint_path, mmap=True, weights_only=True, map_location='cpu'
    )

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
    )

    set_model_state_dict(
        model=self.dit,
        model_state_dict=full_pt['dit_state_dict'],
        options=options,
    )

    if self.controlnet is not None:
      set_model_state_dict(
          model=self.controlnet,
          model_state_dict=full_pt['controlnet_state_dict'],
          options=options,
      )

    if optimizer is not None:
      set_optimizer_state_dict(
          model=self,
          optimizers=optimizer,
          optim_state_dict=full_pt['optimizer_state_dict'],
          options=options,
      )

    if lr_scheduler is not None:
      lr_scheduler.load_state_dict(full_pt['lr_scheduler_state_dict'])
    global_step = full_pt['global_step']
    epoch = full_pt['epoch']

    return global_step, epoch

  def load_sharded_checkpoint(
      self, checkpoint_path, global_step, optimizer=None, lr_scheduler=None
  ):
    """https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html"""

    sharded_dir = os.path.join(checkpoint_path, f'sharded-{global_step}')

    model_to_load = self
    state_dict = {'app': AppState(model_to_load)}
    with without_submodules(self, ['controlnet']):
      dcp.load(state_dict, checkpoint_id=sharded_dir)

    local_state_dict = torch.load(
        os.path.join(checkpoint_path, f'local_state_dict_{global_step}.pt'),
        map_location='cpu',
        weights_only=False,
    )
    self.dit = self.dit.to(dtype=torch.bfloat16)

    if lr_scheduler is not None:
      lr_scheduler.load_state_dict(local_state_dict['lr_scheduler_state_dict'])

    global_step = local_state_dict['global_step']

    dist.barrier()

    return global_step

  def save_checkpoint(
      self,
      checkpoint_path,
      optimizer,
      lr_scheduler,
      global_step,
      epoch,
      is_main,
  ):
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )

    dit_state_dict = get_model_state_dict(model=self.dit, options=options)

    if self.controlnet is not None:
      controlnet_state_dict = get_model_state_dict(
          model=self.controlnet, options=options
      )

    optimizer_state_dict = get_optimizer_state_dict(
        model=self, optimizers=optimizer, options=options
    )

    if is_main:
      full_pt = {
          'lr_scheduler_state_dict': lr_scheduler.state_dict(),
          'global_step': global_step,
          'epoch': epoch,
          'optimizer_state_dict': optimizer_state_dict,
          'dit_state_dict': dit_state_dict,
      }
      if self.controlnet is not None:
        full_pt['controlnet_state_dict'] = controlnet_state_dict

      torch.save(full_pt, checkpoint_path)
      logging.info(f'checkpoint saved to {checkpoint_path}')

    dist.barrier()

  def save_sharded_checkpoint(
      self,
      checkpoint_path,
      optimizer,
      lr_scheduler,
      global_step,
      epoch,
      is_main,
  ):
    """https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html"""

    sharded_dir = os.path.join(checkpoint_path, f'sharded-{global_step}')
    os.makedirs(sharded_dir, exist_ok=True)

    state_dict = {'app': AppState(self, optimizer)}
    dcp.save(state_dict, checkpoint_id=sharded_dir)

    if is_main:
      local_state_dict = {
          'lr_scheduler_state_dict': lr_scheduler.state_dict(),
          'global_step': global_step,
          'epoch': epoch,
      }

      if self.controlnet is not None:
        local_state_dict['controlnet_state_dict'] = self.controlnet.state_dict()

      torch.save(
          local_state_dict,
          os.path.join(checkpoint_path, f'local_state_dict_{global_step}.pt'),
      )
      logging.info(f'checkpoint saved to {sharded_dir}')

    dist.barrier()

  # Original implementation from:
  # https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/wan_video_new.py#L79
  def diffusion_forward(self, batch, overwrite_timesteps=None):
    # basic logic here: input video latent (original video) + noise
    # and then predict the target video latent (relit video)
    # The supervision signal is the target video latent (relit video)

    video_latents = batch['video_latents']
    input_video_latents = batch['input_video_latents']
    noise = torch.randn_like(video_latents)

    max_timestep_boundary = int(
        batch.get('max_timestep_boundary', 1)
        * self.scheduler.num_train_timesteps
    )
    min_timestep_boundary = int(
        batch.get('min_timestep_boundary', 0)
        * self.scheduler.num_train_timesteps
    )
    noise_sample_prob = random.random()
    if noise_sample_prob > 0.15:
      timestep_id = torch.randint(
          min_timestep_boundary, int(max_timestep_boundary * 0.4), (1,)
      )
    else:
      timestep_id = torch.randint(
          int(max_timestep_boundary * 0.4), max_timestep_boundary, (1,)
      )
    print(f'timestep_id: {timestep_id}')
    # for eval part
    if overwrite_timesteps is not None:
      timestep_id = overwrite_timesteps
      if isinstance(timestep_id, int):
        timestep_id = torch.tensor([timestep_id])

    timestep = self.scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=self.device
    )

    noisy_model_input = self.scheduler.add_noise(
        batch['video_latents'], noise, timestep
    )
    target = self.scheduler.training_target(batch['video_latents'], noise)

    return noisy_model_input, target, timestep

  def prepare_batch(self, batch, device, dtype=torch.bfloat16):
    videos = batch['videos'].to(device, dtype=dtype)
    input_video = batch['videos_input'].to(device, dtype=dtype)
    masks = batch['masks'].to(device, dtype=dtype)
    ambient_embeds = batch['abt_embed'].to(device, dtype=dtype)
    ae_embeds = batch['ae_embed'].to(device, dtype=dtype)

    # preprocess / validate for vae
    # check spatial dimensions:
    b, f, c, h, w = videos.shape
    video_latents = self.vae.encode_batch(videos.transpose(1, 2))
    input_video_latents = self.vae.encode_batch(input_video.transpose(1, 2))
    masks_latnents = self.vae.encode_batch(masks.transpose(1, 2))

    if self.enable_cfg:
      content_mask = torch.zeros_like(input_video_latents)

    cfg_p = random.random()
    if cfg_p < 0.12 and self.enable_cfg:
      input_video_latents = torch.cat(
          [input_video_latents, content_mask], dim=2
      )
    else:
      # add the mask as an extra condition
      input_video_latents = torch.cat(
          [input_video_latents, masks_latnents], dim=2
      )

    first_image = input_video[:, 0:1]
    vae_input = first_image
    first_image_latent = self.vae.encode_batch(vae_input.transpose(1, 2))
    text_embeds = self.text_embeds[None].repeat(b, 1, 1)  # b, L, D
    if (cfg_p < 0.06 or cfg_p > 0.94) and self.enable_cfg:
      ambient_embeds = torch.zeros_like(ambient_embeds)
    ambient_embeds = ambient_embeds[None].repeat(b, 3, 1)
    print(f'ambient_embeds: {ambient_embeds.shape}')
    ae_embeds = ae_embeds[None].repeat(b, 3, 1)

    vggt_latents = None
    return dict(
        videos=videos,
        input_video=input_video,
        input_video_latents=input_video_latents,
        masks_latnents=masks_latnents,
        video_latents=video_latents,
        text_embeds=text_embeds,
        first_image_latent=first_image_latent,
        vggt_latents=vggt_latents,
        ambient_embeds=ambient_embeds,
        ae_embeds=ae_embeds,
    )

  def run_one_step(self, batch, noisy_model_input, timestep):
    # in the 5B model they do this operation
    # which effectively replaces the first "frame" of video_latents with first_image_latent,
    # while leaving all subsequent frames unchanged
    mask, _ = masks_like_batch(noisy_model_input, zero=True)
    first_image_latent = batch['first_image_latent']
    input_video_latents = batch['input_video_latents']
    masks_latnents = batch['masks_latnents']
    noisy_model_input = noisy_model_input

    b = first_image_latent.shape[0]
    timestep_reshaped = timestep.view(b, 1, 1, 1)
    temp_ts_batched = timestep_reshaped
    timestep = temp_ts_batched.view(b, -1)
    ambient_embeds = batch['ambient_embeds']
    ae_embeds = batch['ae_embeds']

    regr_kwargs = dict(
        time_mask=self.time_mask,
        poses=None,
        pose_proj=None,
        ambient_embeds=ambient_embeds,
        ae_embeds=ae_embeds,
    )

    model_out = self.dit(
        x=noisy_model_input,
        t=timestep,
        context=batch['text_embeds'],
        seq_len=None,
        y=input_video_latents,
        is_training=self.training,
        **regr_kwargs,
    )

    return model_out

  @torch.inference_mode
  def validation_loop(
      self, val_loader, device, world_size, global_step, outdir
  ):
    num_val_batches = 0
    fixed_timesteps = [999, 900, 100, 0]
    total_val_losses = {timestep: 0.0 for timestep in fixed_timesteps}

    for val_step, batch in enumerate(val_loader):
      batch = self.prepare_batch(batch, device)
      num_val_batches += 1

      for defined_timestep in fixed_timesteps:
        noisy_model_input, target, timesteps = self.diffusion_forward(
            batch, overwrite_timesteps=defined_timestep
        )

        model_output = self.run_one_step(batch, noisy_model_input, timesteps)

        loss = torch.nn.functional.mse_loss(
            model_output.float(), target.float()
        )
        total_val_losses[defined_timestep] += loss

        # save mp4s
        self.vis_training(
            model_out=model_output,
            noisy_model_input=noisy_model_input,
            timesteps=defined_timestep,
            step=global_step,
            outdir=outdir,
            ref=batch['videos'],
            inputs=batch['input_video'],
        )

      if num_val_batches > 2:
        break

    local_val_losses = {
        timestep: total_val_losses[timestep] / num_val_batches
        for timestep in fixed_timesteps
    }
    avg_val_losses = {}

    for key, val in local_val_losses.items():
      dist.all_reduce(val, op=dist.ReduceOp.SUM)
      avg_val_losses[key] = val / world_size

    torch.cuda.empty_cache()

    return avg_val_losses

  @torch.inference_mode
  def vis_training(
      self, ref, model_out, noisy_model_input, step, timesteps, outdir, inputs
  ):
    output_path_file_out_reference = os.path.join(
        outdir, f'{step}_{dist.get_rank()}_{timesteps}.mp4'
    )

    pred_x0 = self.scheduler.velocity_to_x0(
        noisy_model_input, model_out, timesteps
    )

    video_processor = VideoProcessor(vae_scale_factor=16)
    torch.cuda.empty_cache()
    video = self.vae.decode_batch(pred_x0).cpu()
    video = video_processor.postprocess_video(video=video, output_type='pil')[
        0
    ]  # N, H, W, 3
    input_frames = inputs.permute(0, 2, 1, 3, 4)[0:1]  # 1, 3, N, H, W
    input_frames = video_processor.postprocess_video(
        video=input_frames, output_type='pil'
    )[
        0
    ]  # N, H, W, 3
    reference_frames = ref.permute(0, 2, 1, 3, 4)[0:1]  # 1, 3, N, H, W
    reference_frames = video_processor.postprocess_video(
        video=reference_frames, output_type='pil'
    )[
        0
    ]  # N, H, W, 3

    out_reference_frames = [
        functions.stack_images_horizontally(
            frame_input, frame_reference, frame_out
        )
        for frame_input, frame_out, frame_reference in zip(
            input_frames, video, reference_frames
        )
    ]
    media.write_video(
        output_path_file_out_reference, np.array(out_reference_frames), fps=10
    )

    del video, reference_frames
    torch.cuda.empty_cache()

  def generate(
      self,
      input_prompt,
      img=None,
      data_batch=None,
      size=(1280, 704),
      max_area=704 * 1280,
      frame_num=81,
      shift=5.0,
      sample_solver='unipc',
      sampling_steps=50,
      guide_scale=5.0,
      n_prompt='',
      seed=-1,
      offload_model=True,
  ):
    r"""Generates video frames from text prompt using diffusion process.

    Args:
        input_prompt (`str`): Text prompt for content generation
        img (PIL.Image.Image): Input image tensor. Shape: [3, H, W]
        size (`tuple[int]`, *optional*, defaults to (1280,704)): Controls video
          resolution, (width,height).
        max_area (`int`, *optional*, defaults to 704*1280): Maximum pixel area
          for latent space calculation. Controls video resolution scaling
        frame_num (`int`, *optional*, defaults to 81): How many frames to sample
          from a video. The number should be 4n+1
        shift (`float`, *optional*, defaults to 5.0): Noise schedule shift
          parameter. Affects temporal dynamics
        sample_solver (`str`, *optional*, defaults to 'unipc'): Solver used to
          sample the video.
        sampling_steps (`int`, *optional*, defaults to 50): Number of diffusion
          sampling steps. Higher values improve quality but slow generation
        guide_scale (`float`, *optional*, defaults 5.0): Classifier-free
          guidance scale. Controls prompt adherence vs. creativity.
        n_prompt (`str`, *optional*, defaults to ""): Negative prompt for
          content exclusion. If not given, use `config.sample_neg_prompt`
        seed (`int`, *optional*, defaults to -1): Random seed for noise
          generation. If -1, use random seed.
        offload_model (`bool`, *optional*, defaults to True): If True, offloads
          models to CPU during generation to save VRAM

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from size)
            - W: Frame width from size)
    """
    # i2v
    if img is not None or data_batch is not None:
      return self.v2v(
          data_batch=data_batch,
          input_prompt=input_prompt,
          image=img,
          max_area=max_area,
          shift=shift,
          sample_solver=sample_solver,
          sampling_steps=sampling_steps,
          guide_scale=guide_scale,
          n_prompt=n_prompt,
          seed=seed,
          offload_model=offload_model,
      )

  def v2v(
      self,
      input_prompt,
      data_batch,
      image=None,  # list[PIL.Image] length F, same size
      strength=1,  # EDIT STRENGTH: 0..1
      sample_solver='unipc',
      sampling_steps=100,
      guide_scale=5.0,
      n_prompt='',
      seed=-1,
      offload_model=True,
      shift=5.0,
      max_area=704 * 1280,
  ):
    # 1) Preprocess all frames: resize/crop to (oh, ow) exactly like you do for img.
    frames = []
    img = data_batch['videos'].to(self.device, torch.bfloat16).transpose(0, 1)
    condition = (
        data_batch['videos_input']
        .to(self.device, torch.bfloat16)
        .transpose(0, 1)
    )
    mask = data_batch['masks'].to(self.device, torch.bfloat16).transpose(0, 1)

    F = len(img)

    # 3) Encode entire video into latents
    z = self.vae.encode([img])  # shaped like your model expects, same as before
    y_c = self.vae.encode([condition])
    if (mask - condition).mean() != 0:
      y_mask = self.vae.encode([mask])
      y_cu = torch.cat([y_c[0], torch.zeros_like(y_mask[0])], dim=1)
      y_c = torch.cat([y_c[0], y_mask[0]], dim=1)
      y_cu = [y_cu]
      y_c = [y_c]

    print('***' * 10)
    print('v2v:z shape here!!!', z[0].shape)  # [48,5,44,80]

    # 4) Prepare noise with same latent shape
    if seed < 0:
      seed = random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    print('***' * 10)
    print('img_shape!!', img.shape)
    oh, ow = img.shape[-2], img.shape[-1]
    noise = torch.randn_like(z[0])
    print('***' * 10)
    print('v2v:noise shape here!!!', noise.shape)  # [48,5,44,80]

    # 5) Initialize latent by strength (no masks; whole clip is editable)
    latent = noise
    print('***' * 10)
    print('v2v:latent shape here!!!', latent.shape)  # [48,5,44,80]

    # 6) Text encodings (same as your code)
    if n_prompt == '':
      n_prompt = self.sample_neg_prompt
    context = [self.text_embeds]
    context_null = [self.null_text_embeds]
    ambient_embed = data_batch['abt_embed']
    if ambient_embed is not None:
      ambient_embed = ambient_embed.to(self.device, torch.bfloat16)
      ambient_embed = ambient_embed[None].repeat(1, 3, 1)

    ae_embed = data_batch['ae_embed']
    if ae_embed is not None:
      ae_embed = ae_embed.to(self.device, torch.bfloat16)
      ae_embed = ae_embed[None].repeat(1, 3, 1)

    # 7) Scheduler (unchanged)
    if sample_solver == 'unipc':
      sample_scheduler = FlowUniPCMultistepScheduler(
          num_train_timesteps=self.num_train_timesteps,
          shift=1,
          use_dynamic_shifting=False,
      )
      sample_scheduler.set_timesteps(
          sampling_steps, device=self.device, shift=shift
      )
      timesteps = sample_scheduler.timesteps
    elif sample_solver == 'dpm++':
      sample_scheduler = FlowDPMSolverMultistepScheduler(
          num_train_timesteps=self.num_train_timesteps,
          shift=1,
          use_dynamic_shifting=False,
      )
      sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
      timesteps, _ = retrieve_timesteps(
          sample_scheduler, device=self.device, sigmas=sampling_sigmas
      )
    else:
      raise NotImplementedError

    # 8) Start index according to strength
    start_idx = int(round((1 - strength) * (len(timesteps) - 1)))
    timesteps = timesteps[start_idx:]

    # 9) seq_len for your transformer (same computation as your code)
    seq_len = (
        ((F - 1) // self.vae_stride[0] + 1)
        * (oh // self.vae_stride[1])
        * (ow // self.vae_stride[2])
        // (self.patch_size[1] * self.patch_size[2])
    )
    seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size
    print('***' * 10)
    print('v2v:seq_len here!!!', seq_len)
    print('v2v:patch_size here!!!', self.patch_size)

    arg_c = {
        'context': [context[0]],
        'seq_len': seq_len,
        'time_mask': self.time_mask,
        'ambient_embeds': ambient_embed,
        'ae_embeds': ae_embed,
    }
    arg_unc = {
        'context': context_null,
        'seq_len': seq_len,
        'time_mask': self.time_mask,
        'ambient_embeds': torch.zeros_like(ambient_embed),
        'ae_embeds': ae_embed,
    }

    # 10) Denoising loop (NO mask-based overwrite)
    if offload_model or self.init_on_cpu:
      self.dit.to(self.device)
      torch.cuda.empty_cache()
    guide_scale = 1.0
    with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
      for t in timesteps:
        latent_in = [latent.to(self.device)]
        timestep = torch.stack([t]).to(self.device)
        temp_ts = timestep.repeat(seq_len)
        timestep_vec = temp_ts.unsqueeze(0)

        eps_c = self.dit(latent_in, t=timestep_vec, y=y_c, **arg_c)[0]
        # eps_u = self.dit(latent_in, t=timestep_vec, y=y_cu, **arg_unc)[0]
        # eps = eps_u + guide_scale * (eps_c - eps_u)
        eps = eps_c

        new_lat = sample_scheduler.step(
            eps.unsqueeze(0),
            t,
            latent.unsqueeze(0),
            return_dict=False,
            generator=seed_g,
        )[0]
        latent = new_lat.squeeze(0)

    if offload_model:
      self.model.cpu()
      torch.cuda.synchronize()
      torch.cuda.empty_cache()

    # 11) Decode the full clip
    videos = self.vae.decode([latent])  # -> (3, F, H, W)
    return videos[0] if self.rank == 0 else None

  def v2v_ae_only(
      self,
      input_prompt,
      data_batch,
      image=None,  # list[PIL.Image] length F, same size
      strength=1,  # EDIT STRENGTH: 0..1
      sample_solver='unipc',
      sampling_steps=50,
      guide_scale=5.0,
      n_prompt='',
      seed=-1,
      offload_model=True,
      shift=5.0,
      max_area=704 * 1280,
  ):
    img = data_batch['videos'].to(self.device, torch.bfloat16).transpose(0, 1)
    condition = (
        data_batch['videos_input']
        .to(self.device, torch.bfloat16)
        .transpose(0, 1)
    )
    F = len(img)

    z = self.vae.encode([img])
    videos = self.vae.decode(z)
    return videos[0] if self.rank == 0 else None
