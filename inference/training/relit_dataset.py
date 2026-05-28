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

import math
import os
from os.path import join

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# ---------------------- Palette utilities ----------------------


def _parse_color_spec(spec, hval=0.5):
  """'110' → [1,1,0], '1h1' → [1,0.5,1], 'hh1' → [0.5,0.5,1]."""
  table = {'1': 1.0, '0': 0.0, 'h': float(hval)}
  if len(spec) != 3 or any(c not in table for c in spec):
    raise ValueError(f"Bad color spec '{spec}'. Expected 3 chars of 0/1/h.")
  return np.array(
      [table[spec[0]], table[spec[1]], table[spec[2]]],
      dtype=np.float32,
  )


def _to_plain_dict(obj):
  """Convert config-like objects (dict or SimpleNamespace) to a plain dict."""
  if obj is None:
    return {}
  if isinstance(obj, dict):
    return obj
  # Handles SimpleNamespace and similar "object with __dict__"
  if hasattr(obj, '__dict__'):
    return dict(vars(obj))
  raise TypeError(f'Expected dict or dict-like object, got {type(obj)}')


def build_palette(spec_list, hval=0.5):
  """Return dict: spec -> RGB float array in [0,1]."""
  return {spec: _parse_color_spec(spec, hval=hval) for spec in spec_list}


# ---------------------- Core masking ----------------------


def color_region_masks_palette(
    img_seq,
    *,
    specs=(
        '110',
        '101',
        '011',
        '111',
        '11h',
        '1h1',
        'h11',
        'hh1',
        'h1h',
        '1hh',
    ),
    hval=0.5,
    intensity_thr=0.30,
    tol_per_channel=0.18,
    require_saturation=False,
    sat_thr=0.15,
):
  """img_seq: (T,H,W,3) float in [0,1]

  Returns: dict { '110': (T,H,W) bool, ..., '1hh': (T,H,W) bool }
  Matching rule: bright gate AND max(|ch - target_ch|) <= tol_per_channel
  """
  assert (
      img_seq.ndim == 4 and img_seq.shape[-1] == 3
  ), 'img_seq must be (T,H,W,3)'
  palette = build_palette(specs, hval=hval)  # np arrays in [0,1]

  # 1) Brightness gate
  bright = img_seq.max(axis=-1) >= float(intensity_thr)

  # 2) (Optional) saturation gate
  if require_saturation:
    mx = img_seq.max(axis=-1)
    mn = img_seq.min(axis=-1)
    sat_val = (mx - mn) / (mx + 1e-6)
    gate = bright & (sat_val >= float(sat_thr))
  else:
    gate = bright

  # 3) Per-spec matching
  masks = {}
  for spec, target in palette.items():
    diff = np.abs(
        img_seq - target[None, None, None, :]
    )  # broadcast to (T,H,W,3)
    match = (diff <= float(tol_per_channel)).all(
        axis=-1
    )  # per-pixel & per-channel
    masks[spec] = gate & match

  return masks


def to_3ch(mask_bool):
  """(T,H,W) -> (T,H,W,3) uint8 (0/1)."""
  return np.repeat(mask_bool[..., None], 3, axis=-1).astype(np.uint8)


# ---------------------- Fourier embed (for abt_embed) ----------------------


def fourier_embed(x, n_freqs=12):
  """Embeds x with Fourier features.

  If x is scalar, returns (2*n_freqs,) tensor. If x is (d,), returns (d,
  2*n_freqs) tensor. If x is (..., d), returns (..., d, 2*n_freqs) tensor.
  """
  if not torch.is_tensor(x):
    x = torch.tensor(x)
  x_in = x
  if x.ndim == 0:
    x = x[None]  # treat scalar as (1,) tensor

  freqs = 2.0 ** torch.arange(n_freqs) * math.pi
  x_proj = x.unsqueeze(-1) * freqs  # (..., d, L)

  ff = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (..., d, 2L)

  if x_in.ndim == 0:
    return ff.squeeze(0)  # if input was scalar, return (2L,)
  return ff


# ---------------------- Image helpers ----------------------

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _is_image(fn):
  """Checks if a filename corresponds to a known image extension."""
  _, ext = os.path.splitext(fn.lower())
  return ext in IMG_EXTS


def _load_and_process_image(
    image_path,
    crop_type=None,
    resolution=None,
    resize_factor=1.0,
    eye_full=False,
    **kwargs,
):
  """Loads and preprocesses an image.

  Args:
      image_path: Path to the image file.
      crop_type: Type of cropping ('center', 'random', or None).
      resolution: Target resolution (height, width) for resizing/cropping.
      resize_factor: Factor to resize image before cropping.
      eye_full: If True, resize to specific dimensions (1280, 704).
      **kwargs: Additional unused arguments.

  Returns:
      Preprocessed image as a numpy array in [0, 1].
  """
  # Load RGB image in [0,1]
  image = np.array(Image.open(image_path).convert('RGB')) / 255.0

  h, w = image.shape[:2]

  if crop_type is not None and resolution is not None:
    target_h, target_w = resolution  # (H, W)

    new_h, new_w = int(h * resize_factor), int(w * resize_factor)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image = np.clip(image, 0, 1)
    h, w = new_h, new_w

    if crop_type == 'center':
      # Resize to cover target, then center crop
      scale = max(target_h / h, target_w / w)
      new_h = int(h * scale)
      new_w = int(w * scale)
      image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
      image = np.clip(image, 0, 1)

      start_y = (new_h - target_h) // 2
      start_x = (new_w - target_w) // 2
      image = image[start_y : start_y + target_h, start_x : start_x + target_w]

    elif crop_type == 'random':
      # Ensure it's at least target size
      scale = max(target_h / h, target_w / w)
      if scale > 1:
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        image = np.clip(image, 0, 1)
        h, w = new_h, new_w

      max_y = h - target_h
      max_x = w - target_w
      y = np.random.randint(0, max(1, max_y + 1))
      x = np.random.randint(0, max(1, max_x + 1))
      image = image[y : y + target_h, x : x + target_w]

    if eye_full:
      image = cv2.resize(image, (1280, 704), interpolation=cv2.INTER_CUBIC)
      image = np.clip(image, 0, 1)

  elif resolution is not None:
    target_h, target_w = resolution
    image = cv2.resize(
        image, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )
    image = np.clip(image, 0, 1)

  return image


# ---------------------- Dataset ----------------------


class GR3EN_dataset(Dataset):

  def __init__(
      self,
      split='train',
      stride=(1, 1),
      sample_n_frames=81,
      start_idx=None,
      # dicts that control which palette color → which intensity / light color
      mask_intensity=None,  # e.g. {"110": 1.0, "100": 0.0, ...}
      light_color=None,  # e.g. {"110": [1,1,0.8], "100": [1,0.8,0.7], ...}
      ambient_scale=0.5,  # 0.5 → embedded to 0 after mapping to [-1,1]
      image_size=(480, 832),
      use_5b_model=False,
      test_root='./data/test',
      input_name=None,
      mask_name=None,
      # random-combo mode: list of specs to randomly combine, e.g. ["110","010","001"]
      random_light_specs=None,
      # NEW: deterministic variation controls (dict mode only)
      vary_intensity=True,
      intensity_min=0.5,  # target minimal scale (e.g. 0.5)
      variation_steps=10,  # e.g. 6: 1.0→0.5 in steps of 0.1 when base_int=1
      vary_color=False,
      color_target=(0.2, 0.4, 1.0),  # blue-ish target to drag towards
      ae_scale=None,
      zipnerf=False,
      frame_step=5,
      **kwargs,
  ):
    """Initializes the GR3EN dataset.

    Args:
        split: Dataset split ('train', 'val', 'test').
        stride: Tuple of (minimum_sample_stride, sample_stride).
        sample_n_frames: Number of frames to sample in a sequence.
        start_idx: Fixed start index for sampling frames; if None, random start.
        mask_intensity: Dictionary mapping color spec to light intensity.
        light_color: Dictionary mapping color spec to light color [R, G, B].
        ambient_scale: Ambient light scale factor.
        image_size: Target image size (height, width).
        use_5b_model: Whether to use settings for the 5B parameter model.
        test_root: Root directory for test data.
        input_name: Name of the input image directory.
        mask_name: Name of the mask image directory.
        random_light_specs: List of specs for random light combination mode.
        vary_intensity: If True, vary light intensity based on index.
        intensity_min: Minimum intensity for variation.
        variation_steps: Number of steps for intensity/color variation.
        vary_color: If True, vary light color based on index.
        color_target: Target color [R, G, B] for color variation.
        ae_scale: Auto-exposure scale factor.
        zipnerf: Whether to use ZipNeRF specific settings.
        frame_step: Step between frames when sampling a sequence.
        **kwargs: Additional unused arguments.
    """
    minimum_sample_stride, sample_stride = stride

    self.sample_stride = sample_stride
    self.minimum_sample_stride = minimum_sample_stride
    self.sample_n_frames = sample_n_frames
    self.frame_step = frame_step
    if zipnerf:
      self.frame_step = 7
    self.use_5b_model = use_5b_model

    # Fixed start index for the sequence; if None → random each __getitem__
    self.fixed_start_idx = start_idx

    # Light control dicts (static config mode)
    self.mask_intensity = _to_plain_dict(mask_intensity)  # spec -> float
    self.light_color = _to_plain_dict(light_color)  # spec -> [R,G,B]

    self.ambient_scale = ambient_scale

    # Random-combo mode: list/tuple of specs
    self.random_light_specs = (
        tuple(random_light_specs) if random_light_specs else tuple()
    )

    # Deterministic variation params (for dict mode)
    self.vary_intensity = bool(vary_intensity)
    self.intensity_min = float(intensity_min)
    self.variation_steps = max(int(variation_steps), 1)
    self.vary_color = bool(vary_color)
    self.color_target = np.array(color_target, dtype=np.float32)
    self.ae_scale = ae_scale

    # Default specs used if user doesn't specify any
    default_specs = (
        '110',
        '101',
        '011',
        '111',
        '11h',
        '1h1',
        'h11',
        'hh1',
        'h1h',
        '1hh',
        '001',
        '010',
        '100',  # pure B, G, R
    )
    self.default_specs = default_specs
    # Palette specs = union of keys from static dicts and random specs
    combined_keys = (
        set(self.mask_intensity.keys())
        | set(self.light_color.keys())
        | set(self.random_light_specs)
    )
    if not combined_keys:
      self.palette_specs = default_specs
    else:
      self.palette_specs = tuple(sorted(combined_keys))

    # Resolution / pose grid
    # Only use hardcoded defaults if the user didn't provide a specific size
    # (Assuming (480, 832) is the generic default we want to override if 5B)
    # Only use hardcoded defaults if the user didn't provide a specific size
    # (Assuming (480, 832) is the generic default we want to override if 5B)
    if image_size == (480, 832) or image_size == [480, 832]:
      if self.use_5b_model:
        image_size = (608, 1120)  # h, w
      else:
        image_size = (480, 832)

    # Calculate pose_hw based on VAE stride (16 for 5B, 8/unspecified for others)
    stride_s = 16 if self.use_5b_model else 8
    self.pose_hw = (image_size[0] // stride_s, image_size[1] // stride_s)

    self.resolution_configs = {
        'min_sequence_length': sample_n_frames,
        'image_size': image_size,
    }

    # ---------- Paths ----------
    self.split = split
    self.root = test_root
    self.zipnerf = zipnerf

    # Input / mask / target dirs
    if input_name is None:
      self.input_dir = join(self.root, 'input')
    else:
      self.input_dir = join(self.root, input_name)

    if mask_name is not None:
      self.mask_dir = join(self.root, mask_name)
    else:
      self.mask_dir = join(self.root, 'mask')

    self.target_dir = self.input_dir  # relit target = same folder as input here

    self.overfit = False
    self.pixel_transforms = [
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            inplace=True,
        )
    ]

    # Collect frames from mask folder
    self.dataset = sorted(
        [f for f in os.listdir(self.mask_dir) if _is_image(f)]
    )
    if 0 < len(self.dataset) < self.sample_n_frames:
      last_frame = self.dataset[-1]
      self.dataset.extend(
          [last_frame] * (self.sample_n_frames - len(self.dataset))
      )
      self.frame_step = 1
    self.length = max(
        0,
        len(self.dataset) - (self.sample_n_frames - 1) * self.frame_step,
    )

    print(
        f'[RelitDataset] split={split} | subseqs={self.length} |'
        f' palette_specs={self.palette_specs} |'
        f' random_light_specs={self.random_light_specs} |'
        f' vary_intensity={self.vary_intensity},'
        f' intensity_min={self.intensity_min},'
        f' variation_steps={self.variation_steps},'
        f' vary_color={self.vary_color}, color_target={self.color_target}'
    )

  def __len__(self):
    """Returns the number of possible subsequences in the dataset."""
    return self.length

  def __getitem__(self, idx):
    """Returns a dictionary containing a sequence of data for the given index.

    Args:
        idx: Index of the sample to retrieve.

    Returns:
        A dictionary containing video frames, masks, prompts, and other
        metadata.
    """
    sample_n_frames = self.resolution_configs['min_sequence_length']
    sample_size = self.resolution_configs['image_size']
    k = self.frame_step
    n = sample_n_frames

    # Number of valid start positions for a contiguous window
    n_frames = len(self.dataset)
    n_seqs = n_frames - (self.sample_n_frames - 1) * self.frame_step
    max_start = max(0, n_seqs - 1)

    # Decide start_idx: either fixed (from __init__) or random
    if self.fixed_start_idx is not None:
      # Clamp to valid range
      start_idx = max(0, min(self.fixed_start_idx, max_start))
    else:
      if self.overfit or n_seqs <= 0:
        start_idx = 0
      else:
        # random in [0, max_start]
        start_idx = np.random.randint(0, max_start + 1)

    frames = [self.dataset[start_idx + t * k] for t in range(n)]
    input_paths = [join(self.input_dir, f) for f in frames]
    mask_paths = [join(self.mask_dir, f) for f in frames]
    target_paths = [join(self.target_dir, f) for f in frames]

    # ---- Load all images ----
    video_frames_list = []
    mask_frames_list = []
    relit_video_frames_list = []

    for ip, mp, tp in zip(input_paths, mask_paths, target_paths):
      video_frames_list.append(
          _load_and_process_image(
              ip,
              crop_type='center',
              resolution=sample_size,
              eye_full=False,
          )
      )
      mask_frames_list.append(
          _load_and_process_image(
              mp,
              crop_type='center',
              resolution=sample_size,
              eye_full=False,
          )
      )
      relit_video_frames_list.append(
          _load_and_process_image(
              tp,
              crop_type='center',
              resolution=sample_size,
              eye_full=False,
          )
      )

    video_frames = np.array(video_frames_list, dtype=np.float32)  # (T,H,W,3)
    mask_frames = np.array(mask_frames_list, dtype=np.float32)  # (T,H,W,3)
    relit_video_frames = np.array(relit_video_frames_list, dtype=np.float32)

    # Ambient scalar in [-1,1] via Fourier embedding
    abti_tensor = torch.tensor([self.ambient_scale])
    abti_tensor = (abti_tensor * 2) - 1
    abti_tensor_base = fourier_embed(abti_tensor[0], n_freqs=12)
    if self.ae_scale is not None:
      ae_tensor = torch.tensor([self.ae_scale])
      ae_tensor_base = fourier_embed(ae_tensor[0])
      ae_tensor_base = ae_tensor_base.float()
    else:
      ae_tensor_base = None

    # ----------- Build per-spec masks from the palette -----------
    masks = color_region_masks_palette(
        mask_frames,  # (T,H,W,3) in [0,1]
        specs=self.default_specs,
        hval=0.5,
        intensity_thr=0.30,
        tol_per_channel=0.18,
        require_saturation=False,
    )

    # ----------- Compute variation factor from idx -----------
    # frac = 0 at idx=0 → original; frac = 1 at/after last step → fully scaled/shifted.
    if self.variation_steps > 1:
      step_idx = min(idx, self.variation_steps - 1)
      frac = step_idx / float(self.variation_steps - 1)
    else:
      frac = 0.0

    # ----------- Decide intensities & colors for THIS sample -----------

    if self.random_light_specs:
      # Random-combo mode (unchanged): choose 1 or 2 specs from random_light_specs
      cand = [s for s in self.random_light_specs if s in self.palette_specs]
      if len(cand) == 0:
        # fallback: no valid candidates → treat as all-off
        effective_mask_intensity = {spec: 0.5 for spec in self.palette_specs}
        effective_light_color = {}
      else:
        max_lights = min(2, len(cand))
        n_on = np.random.randint(1, max_lights + 1)  # 1 or 2
        on_specs = np.random.choice(cand, size=n_on, replace=False)

        # base: everything off
        effective_mask_intensity = {spec: 0.5 for spec in self.palette_specs}
        effective_light_color = {}

        # turn selected specs on with random colors
        for spec in on_specs:
          effective_mask_intensity[spec] = 1.0
          color = np.random.uniform(0.7, 1.0, size=(3,))
          effective_light_color[spec] = color.tolist()
    else:
      # Dict-controlled mode with **deterministic** variation
      effective_mask_intensity = {}
      effective_light_color = {}

      for spec in self.palette_specs:
        base_int = float(self.mask_intensity.get(spec, 0.5))
        # base_int = 1 / (1 + np.exp(-base_int))
        if base_int == 0.5:
          # light doesn't exist or is off
          effective_mask_intensity[spec] = 0.5
          continue

        # ---- intensity: drag from base_int → intensity_min (e.g. 1 → 0.5) ----
        if self.vary_intensity:
          # linear interpolation between base_int and intensity_min
          target_int = self.intensity_min
          new_int = (1.0 - frac) * base_int + frac * target_int
          new_int = float(np.clip(new_int, 0.0, 1.0))
          # new_int = 1 / (1 + np.exp(-new_int))
        else:
          new_int = base_int

          # new_int = 1 / (1 + np.exp(-new_int))
        effective_mask_intensity[spec] = new_int

        # ---- color: drag from base_color → color_target (e.g. yellow → blue) ----
        base_color = np.array(
            self.light_color.get(spec, [1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        if self.vary_color:
          new_color = (1.0 - frac) * base_color + frac * self.color_target
          new_color = np.clip(new_color, 0.0, 1.0)
        else:
          new_color = base_color
        effective_light_color[spec] = new_color.tolist()

    # ----------- Composite final light mask using effective dicts -----------
    light_mask = np.zeros_like(mask_frames, dtype=np.float32)  # (T,H,W,3)

    for spec in self.default_specs:

      # If the light is not in effective_mask_intensity, it does NOT exist → intensity = 0.5
      intensity = float(effective_mask_intensity.get(spec, 0.5))
      # if intensity == 0.5:
      #     continue  # completely off

      mask_bool = masks[spec][..., None].astype(np.float32)  # (T,H,W,1)

      # Color defaults to white if not provided, but only for existing lights
      color = np.array(
          effective_light_color.get(spec, [1.0, 1.0, 1.0]),
          dtype=np.float32,
      )  # (3,)
      color = color.reshape(1, 1, 1, 3)  # broadcast

      # Add contribution of this light: mask * intensity * color
      light_mask += mask_bool * intensity * color

    # Clamp to [0,1] in case of overlaps
    light_mask = np.clip(light_mask, 0.0, 1.0)

    # Use this as the conditioning mask
    mask_frames = light_mask

    # ----------- Torch & normalization -----------
    pixel_values = (
        torch.from_numpy(video_frames).permute(0, 3, 1, 2).contiguous()
    )
    relit_pixels = (
        torch.from_numpy(relit_video_frames).permute(0, 3, 1, 2).contiguous()
    )
    mask_values = torch.from_numpy(mask_frames).permute(0, 3, 1, 2).contiguous()

    composed_transforms = transforms.Compose(self.pixel_transforms)
    stacked = torch.cat([pixel_values, relit_pixels, mask_values], dim=0)
    transformed = composed_transforms(stacked)
    pixel_values, relit_pixels, mask_values = torch.chunk(transformed, 3, dim=0)
    if self.zipnerf:
      mask_values = torch.full_like(mask_values, -1)
    mask_values_vis = 0.15 * pixel_values + 0.85 * mask_values

    video_captions = (
        'a photorealistic fisheye video of an indoor room, the lamp is turned'
        ' off, camera is rotating around the room to capture a 360 degree view'
        ' of the room'
    )

    data = {
        'videos_input': pixel_values.float(),
        'videos': relit_pixels.float(),
        'prompts': video_captions,
        'masks': mask_values.float(),
        'index': idx,
        'abt_embed': abti_tensor_base.float(),
        'start_idx': start_idx,  # actual used start frame
        'mask_values_vis': mask_values_vis.float(),
        'ae_embed': ae_tensor_base,
    }
    return data
