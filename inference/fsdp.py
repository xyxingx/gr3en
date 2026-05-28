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

r"""FSDP launcher for WAN + GR3EN_dataset evaluation / generation."""

from collections.abc import Sequence
from datetime import timedelta
import json
import os
import time
from types import SimpleNamespace

from absl import flags
from absl import logging
from diffusers.optimization import get_scheduler
import torch
from torch import optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# imports from project code
from training.relit_dataset import GR3EN_dataset
from wan import textimage2video
from wan.configs.init import WAN_CONFIGS
from wan.utils.utils import save_video
import yaml

# ========================= Flags =========================

_MODEL_CONFIGS_STRING = flags.DEFINE_string(
    "model_configs_string",
    default=None,
    help="config dict wrapped in a string",
)

_ENABLE_FLASH = flags.DEFINE_bool(
    "enable_flash",
    default=True,
    help="Whether to enable flash attention.",
)

_WORKDIR = flags.DEFINE_string("workdir", None, "Output directory.")

_MIN_FSDP_PARAMS = flags.DEFINE_integer(
    "min_fsdp_params",
    20000,
    "Minimum number of parameters for FSDP (for logging only).",
)

_FSDP_VERSION = flags.DEFINE_integer(
    "fsdp_version",
    default=1,
    lower_bound=1,
    upper_bound=2,
    help="FSDP version (for logging only).",
)

_SEED = flags.DEFINE_integer("seed", 1337, "Seed")

_TB_SUMMARY_LOGGING_DIR = flags.DEFINE_string(
    "tb_summary_logging_dir",
    default=os.environ.get("TB_SUMMARY_LOGGING_DIR", None),
    help="TensorBoard summary logging directory.",
)


# ========================= Helpers =========================


def get_writer(global_rank):
  """Get the writer for TensorBoard."""
  if global_rank == 0 and _TB_SUMMARY_LOGGING_DIR.value:
    return SummaryWriter(_TB_SUMMARY_LOGGING_DIR.value)
  return None


def main_logging(info_str, is_main):
  if is_main:
    logging.info(info_str)


# ========================= Main training / eval fn =========================


def fsdp_train_and_test(
    local_rank: int,
    nproc_per_node: int,
    global_rank: int,
    node_rank: int,
    world_size: int,
    workdir: str | None,
    min_fsdp_params: int,
    fsdp_version: int,
    model_configs,
    enable_flash: bool,
    mp_context=None,  # kept for interface completeness, logged below
):

  # Enable flash attention if requested
  torch.backends.cuda.enable_flash_sdp(enable_flash)

  is_cuda = torch.cuda.device_count() > 0
  assert is_cuda, "CUDA is required"

  device = torch.device(f"cuda:{local_rank}")
  torch.cuda.set_device(device)

  dist.init_process_group(
      backend="NCCL",
      rank=global_rank,
      world_size=world_size,
      timeout=timedelta(minutes=30),
      device_id=device,
  )
  is_main = global_rank == 0

  if is_main and workdir is not None:
    os.makedirs(workdir, exist_ok=True)
    logging.info("workdir: %s", workdir)

  main_logging(f"main device: {device}", is_main)
  main_logging(f"enable flash attention: {enable_flash}", is_main)
  logging.info(
      "FSDP details: local_rank=%s, nproc_per_node=%s, node_rank=%s, "
      "world_size=%s, min_fsdp_params=%s, fsdp_version=%s, mp_context=%s",
      local_rank,
      nproc_per_node,
      node_rank,
      world_size,
      min_fsdp_params,
      fsdp_version,
      type(mp_context).__name__ if mp_context is not None else None,
  )

  # ----------------- Configs & base dataset -----------------

  task = model_configs.task
  cfg = WAN_CONFIGS[task]

  # A base GR3EN_dataset to define effective "train size" for lr scheduler / logging.
  # (We don't actually train here, but optimizer + scheduler are passed into
  #  load_sharded_checkpoint.)
  base_dataset = GR3EN_dataset(
      split="test",
      sample_n_frames=model_configs.max_num_frames,
      use_5b_model=(task == "ti2v-5B"),
      test_root=model_configs.test_root,
      image_size=model_configs.image_size,
      input_name=getattr(model_configs, "input_name", "frames"),
      mask_name=getattr(model_configs, "mask_name", "mask"),
      ae_scale=getattr(model_configs, "ae_scale", None),
      frame_step=getattr(model_configs, "frame_step", 5),
  )

  base_sampler = DistributedSampler(
      base_dataset,
      rank=global_rank,
      num_replicas=world_size,
      shuffle=False,
  )

  base_dataloader = DataLoader(
      base_dataset,
      batch_size=1,
      sampler=base_sampler,
      num_workers=0,
      pin_memory=True,
      persistent_workers=False,
  )

  # ----------------- Model init -----------------

  init_start_event = torch.cuda.Event(enable_timing=True)
  init_end_event = torch.cuda.Event(enable_timing=True)

  model_args = dict(
      device_id=local_rank,
      rank=global_rank,
      dit_fsdp=True,
      use_sp=False,
      convert_model_dtype=True,
      config=cfg,
      training=False,
      model_configs=model_configs,
  )

  if task == "i2v-A14B":
    from wan import image2video  # lazy import

    model = image2video.WanI2V(**model_args)
    val_loss_history = {999: [], 900: [], 899: [], 300: []}
  elif task == "ti2v-5B":
    model = textimage2video.WanTI2V(**model_args)
    val_loss_history = {999: [], 900: [], 500: [], 300: []}
  else:
    raise ValueError(f"Unknown task: {task}")

  train_psnr_history = {"psnr": []}
  eval_psnr_history = {"psnr": []}

  main_logging(f"model: {model}", is_main)

  # Only initialize optimizer/scheduler if we are actually training
  # (Saves massive VRAM for 5B model inference)
  optimizer = None
  lr_scheduler = None
  if model_args["training"]:
    params_to_optimize = model.set_requires_grad_and_lr(
        model_configs.learning_rate
    )
    optimizer = optim.AdamW(params_to_optimize)
    lr_scheduler = get_scheduler(
        model_configs.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=model_configs.lr_warmup_steps,
        num_training_steps=len(base_dataloader) * 1000,
        num_cycles=model_configs.lr_num_cycles,
    )

  # ----------------- Checkpoint resume (if any) -----------------

  if model_configs.resume_from_checkpoint is not None:
    global_step = model.load_sharded_checkpoint(
        model_configs.resume_from_checkpoint,
        model_configs.resume_step,
        optimizer,
        lr_scheduler,
    )
    main_logging(f"loaded checkpoint from {global_step}", is_main)
    init_step = global_step
  else:
    global_step = init_step = 0
    current_epoch = 0
    assert global_step == 0

  if init_start_event:
    init_start_event.record()

  # ----------------- Logging / writer -----------------

  writer = get_writer(global_rank)
  start_time = time.time()

  progress_bar = tqdm(
      range(0, len(base_dataloader)),
      initial=init_step,
      desc="Steps",
      disable=not is_main,
  )

  # ================== GENERATION / EVAL WITH GR3EN_dataset ==================

  torch.cuda.empty_cache()
  generation_dir = (
      os.path.join(workdir, "gen") if workdir is not None else "gen"
  )
  if is_main:
    os.makedirs(generation_dir, exist_ok=True)

  with torch.inference_mode():

    # ---- Gather GR3EN config from model_configs (with sane defaults) ----
    gen_start_idx = getattr(model_configs, "start_idx", 0)
    print(f"gen_start_idx: {gen_start_idx}")
    gen_mask_intensity = getattr(
        model_configs,
        "mask_intensity",
        {"110": 1.0},  # default: one light spec on
    )
    gen_light_color = getattr(
        model_configs,
        "light_color",
        {"110": [1.0, 1.0, 1.0]},  # default: white
    )
    gen_ambient_scale = getattr(model_configs, "ambient_scale", 0.5)
    gen_test_root = getattr(
        model_configs,
        "test_root",
        "./data/kitchen/",
    )
    gen_input_name = getattr(model_configs, "input_name", "frames")
    gen_mask_name = getattr(model_configs, "mask_name", "mask")

    # ---- New GR3EN_dataset for relighting control ----
    gen_dataset = GR3EN_dataset(
        split="test",
        sample_n_frames=model_configs.max_num_frames,
        start_idx=gen_start_idx,  # fixed starting frame if provided
        mask_intensity=gen_mask_intensity,  # dict: spec -> intensity
        light_color=gen_light_color,  # dict: spec -> [R,G,B]
        ambient_scale=gen_ambient_scale,
        use_5b_model=(task == "ti2v-5B"),
        test_root=gen_test_root,
        image_size=model_configs.image_size,
        input_name=gen_input_name,
        mask_name=gen_mask_name,
        vary_intensity=False,
        vary_color=False,
        ae_scale=model_configs.ae_scale,
        zipnerf=model_configs.zipnerf,
        frame_step=model_configs.frame_step,
    )

    for gen_step in range(len(gen_dataset)):
      batch = gen_dataset[gen_step]

      video = model.generate(
          input_prompt=None,
          img=None,
          offload_model=False,
          data_batch=batch,
          frame_num=model_configs.max_num_frames,
      )

      if is_main:
        if video is None:
          logging.warning(
              "gen: model.generate returned None for step %d", gen_step
          )
        else:
          start_idx = batch["start_idx"]

          # Save the generated video
          save_file_output = os.path.join(
              generation_dir, f"{gen_step}_{start_idx}_output.mp4"
          )
          save_video(
              tensor=video[None],
              save_file=save_file_output,
              fps=10,
              nrow=1,
              normalize=True,
              value_range=(-1, 1),
          )

          # Save the input video
          input_video = batch["videos_input"]  # (F, C, H, W)
          input_video_tensor = input_video.permute(1, 0, 2, 3).unsqueeze(0)
          save_file_input = os.path.join(
              generation_dir, f"{gen_step}_{start_idx}_input.mp4"
          )
          save_video(
              tensor=input_video_tensor,
              save_file=save_file_input,
              fps=10,
              nrow=1,
              normalize=True,
              value_range=(-1, 1),
          )

          # Save the mask (control) video
          mask_video_save = batch["mask_values_vis"]  # (F, C, H, W)
          mask_video_tensor = mask_video_save.permute(1, 0, 2, 3).unsqueeze(0)
          save_file_mask = os.path.join(
              generation_dir, f"{gen_step}_{start_idx}_control_input.mp4"
          )
          save_video(
              tensor=mask_video_tensor,
              save_file=save_file_mask,
              fps=10,
              nrow=1,
              normalize=True,
              value_range=(-1, 1),
          )
          del video

      if gen_step >= 10:  # safety cap
        break

  dist.barrier()
  progress_bar.update(1)
  global_step += 1

  # ================== Wrap up ==================

  if global_rank == 0 and writer is not None:
    writer.add_scalar("training_time_sec", time.time() - start_time)
    writer.close()

  if init_end_event:
    init_end_event.record()

  if is_cuda:
    print(
        f"[rank={global_rank}]",
        "Peak CUDA memory usage:",
        torch.cuda.max_memory_allocated(device),
    )
  if is_main:
    if init_start_event and init_end_event:
      print(
          "CUDA event elapsed time:"
          f" {init_start_event.elapsed_time(init_end_event) / 1000}sec"
      )
    print(f"{model}")

  dist.destroy_process_group()


# ========================= main() =========================


def main(argv: Sequence[str]):
  del argv
  torch.manual_seed(_SEED.value)
  torch.cuda.manual_seed_all(_SEED.value)

  local_rank = int(os.environ["LOCAL_RANK"])
  num_worker_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
  global_rank = int(os.environ["RANK"])
  node_rank = int(os.environ["GROUP_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  model_config_dict = yaml.safe_load(_MODEL_CONFIGS_STRING.value)
  model_configs = json.loads(
      json.dumps(model_config_dict),
      object_hook=lambda d: SimpleNamespace(**d),
  )

  fsdp_train_and_test(
      local_rank=local_rank,
      nproc_per_node=num_worker_per_node,
      global_rank=global_rank,
      node_rank=node_rank,
      world_size=world_size,
      workdir=_WORKDIR.value,
      min_fsdp_params=_MIN_FSDP_PARAMS.value,
      fsdp_version=_FSDP_VERSION.value,
      model_configs=model_configs,
      enable_flash=_ENABLE_FLASH.value,
  )


if __name__ == "__main__":
  from absl import app

  app.run(main)
