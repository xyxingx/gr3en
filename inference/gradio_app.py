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
"""Interactive GR3EN relighting UI (Gradio + SAM2), single-GPU, resident model.

Workflow:
  1. upload a video (conformed to the model's 81-frame requirement),
  2. click each light source; SAM2 segments it and propagates the mask
     through all 81 frames,
  3. set each light to No change / Off / On (+ color & intensity for On),
     adjust ambient / auto-exposure / sampling steps,
  4. relight. The GR3EN 5B model lives in this process on ONE GPU and is
     loaded once at startup, so each relight only pays the diffusion time.

Light-value encoding (matches the palette-mask / GR3EN_dataset convention;
the mask is normalized x -> 2x-1 before the model):
    no change / background : pre-norm 0.0        -> -1
    light OFF              : pre-norm 0.5        ->  0  (explicit off level)
    light ON (colored)     : pre-norm i*color<=1 -> up to 1

Run (SLURM):  sbatch run_gradio_slurm.sh            # UI on port 7860
Headless GPU test:  python gradio_app.py --selftest
"""

import argparse
import glob
import os
import re
import time
from types import SimpleNamespace

import cv2
import gradio as gr
import imageio
import numpy as np
import torch

# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------
INFERENCE_DIR = os.path.abspath(os.path.dirname(__file__))
# All assets/weights now live INSIDE the repo (self-contained); GR3EN_ASSETS
# can still point elsewhere if needed.
ASSETS_ROOT = os.environ.get("GR3EN_ASSETS", INFERENCE_DIR)
SESSIONS_ROOT = os.path.join(INFERENCE_DIR, "output", "gradio_sessions")

SAM2_CKPT = os.environ.get(
    "SAM2_CKPT",
    os.path.join(ASSETS_ROOT, "checkpoints", "sam2", "sam2.1_hiera_large.pt"))
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # ships inside the sam2 package

WAN_CKPT_DIR = os.environ.get(
    "WAN_CKPT_DIR", os.path.join(ASSETS_ROOT, "checkpoints", "wan2.2-ti2v-5b"))
# Consolidated GR3EN checkpoint ({dit_state_dict, controlnet_state_dict, ...}).
# Override with GR3EN_WEIGHTS_PT to serve a different training run.
GR3EN_WEIGHTS = os.environ.get(
    "GR3EN_WEIGHTS_PT",
    os.path.join(ASSETS_ROOT, "checkpoints", "gr3en_weights.pt"),
)

N_FRAMES = 81
MAX_LIGHTS = 6
RESOLUTIONS = {
    "352x640 (fast)": (352, 640),
    "608x1120 (high res)": (608, 1120),
}
DEFAULT_RES = "352x640 (fast)"

# Distinct overlay colors per light index (RGB 0-255), for visualization only.
LIGHT_VIS_COLORS = [
    (255, 64, 64), (64, 160, 255), (64, 220, 120),
    (255, 200, 40), (200, 90, 255), (0, 220, 220),
]


# ----------------------------------------------------------------------------
# Resident GR3EN pipeline (single GPU, no FSDP, no torch.distributed)
# ----------------------------------------------------------------------------
class RelightPipeline:
  """Loads WanTI2V + GR3EN weights once and serves relight requests."""

  def __init__(self, device_id=0):
    from training.relit_dataset import fourier_embed
    from wan import textimage2video
    from wan.configs.init import WAN_CONFIGS

    self._fourier_embed = fourier_embed
    self.device = torch.device(f"cuda:{device_id}")

    t0 = time.time()
    # Peek at the checkpoint to detect the REGR variant: newer runs condition
    # on a 24-dim auto-exposure fourier embedding (add_REGR_modules(ae_scale=x)),
    # older runs have a 16-dim ae_embedding that is never fed at inference
    # (model.forward skips ae when ae_embeds is None).
    print(f"[pipeline] peeking checkpoint: {GR3EN_WEIGHTS}", flush=True)
    full = torch.load(GR3EN_WEIGHTS, mmap=True, weights_only=True,
                      map_location="cpu")
    self.ae_in_dim = full["dit_state_dict"]["ae_embedding.0.weight"].shape[1]
    self.use_ae = self.ae_in_dim == 24
    print(f"[pipeline] ae_embedding input dim: {self.ae_in_dim} "
          f"({'ae conditioning ACTIVE' if self.use_ae else 'ae conditioning DISABLED for this checkpoint'})",
          flush=True)

    print("[pipeline] building WanTI2V (base Wan2.2-TI2V-5B + REGR modules)...",
          flush=True)
    cfg = WAN_CONFIGS["ti2v-5B"]
    model_configs = SimpleNamespace(
        ckpt_dir=WAN_CKPT_DIR, ae_scale=(0.99 if self.use_ae else None))
    self.model = textimage2video.WanTI2V(
        config=cfg,
        model_configs=model_configs,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        init_on_cpu=True,
        convert_model_dtype=True,
        training=False,
    )

    print(f"[pipeline] loading GR3EN weights: {GR3EN_WEIGHTS}", flush=True)
    self.model.dit.load_state_dict(full["dit_state_dict"], strict=True)
    # ControlNet is unused in the v2v inference path (poses=None); some
    # training runs checkpoint a larger controlnet variant, so load loosely
    # but report any mismatch.
    cn_result = self.model.controlnet.load_state_dict(
        full["controlnet_state_dict"], strict=False)
    if cn_result.missing_keys or cn_result.unexpected_keys:
      print(f"[pipeline] controlnet load (non-strict): "
            f"{len(cn_result.missing_keys)} missing, "
            f"{len(cn_result.unexpected_keys)} unexpected keys "
            f"(controlnet is unused at inference)", flush=True)
    del full

    print("[pipeline] moving DiT to GPU...", flush=True)
    self.model.dit.to(self.device)
    torch.cuda.empty_cache()
    print(f"[pipeline] ready in {time.time() - t0:.0f}s. "
          f"GPU mem: {torch.cuda.memory_allocated(self.device)/2**30:.1f} GiB",
          flush=True)

  @torch.inference_mode()
  def relight(self, frames, mask_seq, ambient_scale, ae_scale, sampling_steps,
              seed):
    """frames: list of (H,W,3) uint8 RGB; mask_seq: (T,H,W,3) float32 in [0,1]
    (pre-normalization light values). Returns (3,F,H,W) tensor in [-1,1]."""
    vid = torch.from_numpy(
        np.stack(frames).astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)                      # (T,3,H,W) in [0,1]
    vid = vid * 2.0 - 1.0                      # Normalize(0.5, 0.5)
    mask = torch.from_numpy(mask_seq).permute(0, 3, 1, 2) * 2.0 - 1.0

    abt = torch.tensor([float(ambient_scale)])
    abt_embed = self._fourier_embed((abt * 2 - 1)[0]).float()
    if self.use_ae:
      ae_embed = self._fourier_embed(torch.tensor([float(ae_scale)])[0]).float()
    else:
      ae_embed = None  # old checkpoint variant: model.forward skips ae

    batch = {
        "videos": vid.float(),          # v2v encodes this as the source clip
        "videos_input": vid.float(),
        "masks": mask.float(),
        "abt_embed": abt_embed,
        "ae_embed": ae_embed,
        "start_idx": 0,
    }
    video = self.model.generate(
        input_prompt=None,
        img=None,
        data_batch=batch,
        frame_num=N_FRAMES,
        sampling_steps=int(sampling_steps),
        seed=int(seed),
        offload_model=False,
    )
    torch.cuda.empty_cache()
    return video  # (3,F,H,W) in [-1,1]


PIPELINE = None  # set in main()


# ----------------------------------------------------------------------------
# Single-user session state (research demo; one user at a time).
# ----------------------------------------------------------------------------
class Session:
  def __init__(self):
    self.predictor = None          # SAM2 video predictor (loaded lazily)
    self.state = None              # SAM2 inference state
    self.workdir = None            # per-session output dir
    self.sam2_dir = None           # JPEG frames (SAM2 requires JPEG)
    self.frames = []               # list of np.uint8 (H,W,3), len N_FRAMES
    self.img_h, self.img_w = RESOLUTIONS[DEFAULT_RES]
    self.n_source = 0              # frames decoded from the uploaded video
    # per-light click prompts: light_idx -> {frame_idx: [(x,y,label), ...]}
    self.points = {}
    self.n_lights = 0
    self.masks = None              # np.bool_ (n_lights, T, H, W) after propagate
    self.mask_seq = None           # (T,H,W,3) float composited pre-norm mask
    self._live = {}

SESS = Session()


def _load_predictor():
  """Load SAM2 once (on the same GPU as the model; ~2.3 GiB)."""
  if SESS.predictor is not None:
    return
  from sam2.build_sam import build_sam2_video_predictor
  SESS.predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")


# ----------------------------------------------------------------------------
# Frame preparation
# ----------------------------------------------------------------------------
def _center_cover_resize(img, out_h, out_w):
  """Resize-to-cover then center-crop -- matches the dataset's crop_type='center'."""
  h, w = img.shape[:2]
  scale = max(out_h / h, out_w / w)
  nh, nw = int(round(h * scale)), int(round(w * scale))
  img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
  y0 = (nh - out_h) // 2
  x0 = (nw - out_w) // 2
  return img[y0:y0 + out_h, x0:x0 + out_w]


def _decode_video(path):
  cap = cv2.VideoCapture(path)
  frames = []
  while True:
    ok, bgr = cap.read()
    if not ok:
      break
    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
  cap.release()
  return frames


def _conform_to_81(frames, policy, factor):
  n = len(frames)
  if n >= N_FRAMES:
    if policy == "Downsample to 81 (keep every k-th)":
      k = max(1, int(factor))
      frames = frames[::k]
    frames = frames[:N_FRAMES]
  if len(frames) < N_FRAMES:
    frames = frames + [frames[-1]] * (N_FRAMES - len(frames))
  return frames[:N_FRAMES]


def _init_session_from_frames(frames):
  """frames: list of RGB uint8 already at (img_h, img_w). Sets up SAM2 state."""
  SESS.frames = frames
  SESS.workdir = os.path.join(SESSIONS_ROOT, f"sess_{int(time.time())}")
  SESS.sam2_dir = os.path.join(SESS.workdir, "sam2_frames")
  os.makedirs(SESS.sam2_dir, exist_ok=True)
  for i, f in enumerate(frames):
    cv2.imwrite(os.path.join(SESS.sam2_dir, f"{i:05d}.jpg"),
                cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
  _load_predictor()
  SESS.state = SESS.predictor.init_state(video_path=SESS.sam2_dir)
  SESS.predictor.reset_state(SESS.state)
  SESS.points = {}
  SESS.n_lights = 0
  SESS.masks = None
  SESS.mask_seq = None
  SESS._live = {}


# ----------------------------------------------------------------------------
# Gradio callbacks (UI-only imports kept inside functions where possible)
# ----------------------------------------------------------------------------
def on_video_uploaded(video_path):
  import gradio as gr
  if not video_path:
    return (gr.update(value="Upload a video to begin."),
            gr.update(visible=False), gr.update(visible=False))
  frames = _decode_video(video_path)
  n = len(frames)
  SESS.n_source = n
  if n == 0:
    return (gr.update(value="Could not decode any frames from this video."),
            gr.update(visible=False), gr.update(visible=False))
  if n < N_FRAMES:
    msg = (f"Video has **{n}** frames (< {N_FRAMES}). The last frame will be "
           f"repeated to pad the sequence to {N_FRAMES}.")
    return (gr.update(value=msg), gr.update(visible=False),
            gr.update(visible=False))
  if n == N_FRAMES:
    return (gr.update(value=f"Video has exactly {N_FRAMES} frames. Ready."),
            gr.update(visible=False), gr.update(visible=False))
  auto_factor = int(np.ceil(n / N_FRAMES))
  msg = (f"Video has **{n}** frames (> {N_FRAMES}). Choose how to reach "
         f"{N_FRAMES}: trim to the first {N_FRAMES}, or keep every k-th frame "
         f"(downsample). Suggested factor k = {auto_factor}.")
  return (gr.update(value=msg),
          gr.update(visible=True),
          gr.update(visible=True, value=auto_factor))


def prepare_frames(video_path, policy, factor, resolution):
  import gradio as gr
  if not video_path:
    raise gr.Error("Please upload a video first.")
  frames = _decode_video(video_path)
  if not frames:
    raise gr.Error("Could not decode the video.")
  SESS.img_h, SESS.img_w = RESOLUTIONS[resolution]
  frames = _conform_to_81(frames, policy, factor)
  frames = [_center_cover_resize(f, SESS.img_h, SESS.img_w) for f in frames]
  _init_session_from_frames(frames)
  info = (f"Prepared {N_FRAMES} frames at {SESS.img_w}x{SESS.img_h}. "
          f"Now click on the light sources in tab 2.")
  return (frames[0], info, gr.update(maximum=N_FRAMES - 1, value=0),
          _lights_summary())


def _overlay_masks_on(frame_idx):
  base = SESS.frames[frame_idx].astype(np.float32)
  for li in range(SESS.n_lights):
    m = _light_mask_at(li, frame_idx)
    if m is None or not m.any():
      continue
    col = np.array(LIGHT_VIS_COLORS[li % len(LIGHT_VIS_COLORS)], np.float32)
    base[m] = 0.45 * base[m] + 0.55 * col
  return base.clip(0, 255).astype(np.uint8)


def _light_mask_at(light_idx, frame_idx):
  if SESS.masks is not None:
    return SESS.masks[light_idx, frame_idx]
  return SESS._live.get((light_idx, frame_idx))


def add_click(frame_idx, light_id, point_type, evt: gr.SelectData):
  # NOTE: the `evt: gr.SelectData` annotation is REQUIRED — Gradio only
  # injects the click event when the parameter is annotated with the event
  # dataclass; without it evt arrives as None.
  if SESS.state is None:
    raise gr.Error("Prepare the 81 frames first (tab 1).")
  frame_idx = int(frame_idx)
  x, y = int(evt.index[0]), int(evt.index[1])
  li = int(light_id)
  label = 1 if point_type == "Positive (on the light)" else 0

  SESS.points.setdefault(li, {}).setdefault(frame_idx, []).append((x, y, label))
  SESS.n_lights = max(SESS.n_lights, li + 1)
  SESS.masks = None  # invalidate stale propagation

  pts = np.array([[px, py] for px, py, _ in SESS.points[li][frame_idx]],
                 np.float32)
  lbs = np.array([lb for _, _, lb in SESS.points[li][frame_idx]], np.int32)

  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    _, obj_ids, mask_logits = SESS.predictor.add_new_points_or_box(
        inference_state=SESS.state, frame_idx=frame_idx, obj_id=li,
        points=pts, labels=lbs,
    )
  for j, oid in enumerate(obj_ids):
    SESS._live[(int(oid), frame_idx)] = (mask_logits[j, 0] > 0).cpu().numpy()

  return _overlay_masks_on(frame_idx), _lights_summary()


def new_light():
  import gradio as gr
  nxt = SESS.n_lights
  if nxt >= MAX_LIGHTS:
    raise gr.Error(f"This demo supports up to {MAX_LIGHTS} lights.")
  SESS.n_lights = nxt + 1
  return gr.update(value=nxt), _lights_summary()


def _replay_prompts():
  SESS.predictor.reset_state(SESS.state)
  SESS._live = {}
  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for li in sorted(SESS.points):
      for fidx, plist in SESS.points[li].items():
        if not plist:
          continue
        pts = np.array([[px, py] for px, py, _ in plist], np.float32)
        lbs = np.array([lb for _, _, lb in plist], np.int32)
        _, obj_ids, mask_logits = SESS.predictor.add_new_points_or_box(
            inference_state=SESS.state, frame_idx=fidx, obj_id=li,
            points=pts, labels=lbs,
        )
        for j, oid in enumerate(obj_ids):
          SESS._live[(int(oid), fidx)] = (mask_logits[j, 0] > 0).cpu().numpy()


def reset_light(light_id):
  li = int(light_id)
  SESS.points.pop(li, None)
  SESS._live = {k: v for k, v in SESS._live.items() if k[0] != li}
  _replay_prompts()
  SESS.masks = None
  return _overlay_masks_on(0), _lights_summary()


def clear_all():
  if SESS.state is not None:
    SESS.predictor.reset_state(SESS.state)
  SESS.points = {}
  SESS.n_lights = 0
  SESS.masks = None
  SESS._live = {}
  frame = SESS.frames[0] if SESS.frames else None
  return frame, _lights_summary()


def show_frame(frame_idx):
  if not SESS.frames:
    return None
  return _overlay_masks_on(int(frame_idx))


def _lights_summary():
  if SESS.n_lights == 0:
    return "No lights yet. Click on a light source to create Light 0."
  rows = []
  for li in range(SESS.n_lights):
    npt = sum(len(v) for v in SESS.points.get(li, {}).values())
    rows.append(f"- Light {li}: {npt} click(s)")
  return "**Lights:**\n" + "\n".join(rows)


def propagate():
  import gradio as gr
  if SESS.state is None or SESS.n_lights == 0:
    raise gr.Error("Add at least one light before propagating.")
  masks = np.zeros((SESS.n_lights, N_FRAMES, SESS.img_h, SESS.img_w),
                   dtype=bool)
  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for fidx, obj_ids, mask_logits in SESS.predictor.propagate_in_video(
        SESS.state):
      for j, oid in enumerate(obj_ids):
        if 0 <= int(oid) < SESS.n_lights:
          masks[int(oid), fidx] = (mask_logits[j, 0] > 0).cpu().numpy()
  SESS.masks = masks
  cover = [f"Light {li}: {masks[li].any(axis=(1, 2)).sum()} frames"
           for li in range(SESS.n_lights)]
  return (_overlay_masks_on(0), "Propagation done.\n" + "\n".join(cover))


def _hex_to_rgb01(h):
  """Parse a Gradio ColorPicker value ('#rrggbb' or 'rgb(a)(...)') -> RGB in [0,1]."""
  if h is None:
    return np.ones(3, np.float32)
  h = str(h).strip()
  if h.startswith("#"):
    h = h[1:]
    if len(h) == 3:
      h = "".join(c * 2 for c in h)
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
  if h.lower().startswith("rgb"):
    nums = re.findall(r"[-+]?\d*\.?\d+", h)[:3]
    return np.clip(np.array([float(x) for x in nums], np.float32) / 255.0, 0, 1)
  return np.ones(3, np.float32)


def _write_mp4(path, frames_uint8, fps=10):
  writer = imageio.get_writer(path, fps=fps, codec="libx264",
                              macro_block_size=1, pixelformat="yuv420p")
  for f in frames_uint8:
    writer.append_data(f)
  writer.close()
  return path


def _intensity_to_mask(i):
  """Raw light intensity (0-5, the training scale) -> pre-norm mask value.

  Training passed the whole mask through a sigmoid, so brightness must be
  encoded as sigmoid(i). The useful range is [1, 5] (5 = training max);
  anything below 1 is clamped up to 1 — 'off' is a separate state."""
  return 1.0 / (1.0 + float(np.exp(-np.clip(float(i), 1.0, 5.0))))


def build_mask(*light_settings):
  """Compose the pre-normalization light mask from per-light settings.

  light_settings is a flat list: for each of MAX_LIGHTS -> (state, color, intensity)
  """
  import gradio as gr
  if SESS.masks is None:
    raise gr.Error("Propagate the masks through the video first.")
  states = light_settings[0::3]
  colors = light_settings[1::3]
  intens = light_settings[2::3]

  mask_seq = np.zeros((N_FRAMES, SESS.img_h, SESS.img_w, 3), np.float32)
  for li in range(SESS.n_lights):
    st = states[li]
    region = SESS.masks[li]  # (T,H,W)
    if st == "No change":
      continue
    if st == "Off":
      # palette-mask convention: explicit off = 0.5 pre-norm -> 0 normalized
      # (distinct from the -1 background / no-change level).
      val = np.array([0.5, 0.5, 0.5], np.float32)
    else:  # On
      val = _intensity_to_mask(intens[li]) * _hex_to_rgb01(colors[li])
    for t in range(N_FRAMES):
      mask_seq[t][region[t]] = val
  mask_seq = np.clip(mask_seq, 0.0, 1.0)
  SESS.mask_seq = mask_seq

  preview = os.path.join(SESS.workdir, "mask_preview.mp4")
  blend = [
      (0.15 * SESS.frames[t].astype(np.float32) + 0.85 * mask_seq[t] * 255.0)
      .clip(0, 255).astype(np.uint8)
      for t in range(N_FRAMES)
  ]
  _write_mp4(preview, blend)
  return preview, "Mask composited. Ready to relight."


def _relight_once(ae_percentile, ambient_scale, sampling_steps, seed):
  """One relight with the current session mask; returns (out_path, seconds)."""
  import gradio as gr
  if SESS.mask_seq is None:
    raise gr.Error("Build the mask before relighting.")
  if PIPELINE is None:
    raise gr.Error("Model pipeline is not loaded (startup failed?).")

  p = float(np.clip(ae_percentile, 0.5, 0.999999))
  ae_scale = float(-np.log10(p))

  t0 = time.time()
  video = PIPELINE.relight(
      SESS.frames, SESS.mask_seq,
      ambient_scale=ambient_scale, ae_scale=ae_scale,
      sampling_steps=sampling_steps, seed=seed,
  )
  dur = time.time() - t0

  out = ((video.clamp(-1, 1).permute(1, 2, 3, 0).float().cpu().numpy() + 1.0)
         / 2.0 * 255.0).astype(np.uint8)
  out_path = os.path.join(SESS.workdir,
                          f"relit_seed{seed}_{int(time.time())}.mp4")
  _write_mp4(out_path, list(out))
  return out_path, dur


def _ensure_input_mp4():
  in_path = os.path.join(SESS.workdir, "input.mp4")
  if not os.path.exists(in_path):
    _write_mp4(in_path, SESS.frames)
  return in_path


def run_relight(ae_percentile, ambient_scale, sampling_steps, seed):
  import random as _random
  seed = int(seed)
  if seed < 0:
    seed = _random.randint(0, 2**31 - 1)
  out_path, dur = _relight_once(ae_percentile, ambient_scale, sampling_steps,
                                seed)
  in_path = _ensure_input_mp4()
  p = float(np.clip(ae_percentile, 0.5, 0.999999))
  msg = (f"Done in {dur:.0f}s (seed={seed}, steps={int(sampling_steps)}, "
         f"ae p={p:.3f} -> ae_scale={-np.log10(p):.4f}, "
         f"ambient={ambient_scale:.2f}). Saved to {out_path}")
  return out_path, in_path, msg


N_MULTI = 5


def run_relight_multi(ae_percentile, ambient_scale, sampling_steps,
                      progress=None):
  """Run N_MULTI relights with fresh random seeds; returns N_MULTI videos."""
  import gradio as gr
  import random as _random
  if progress is None:
    progress = gr.Progress()

  seeds = [_random.randint(0, 2**31 - 1) for _ in range(N_MULTI)]
  outs, durs = [], []
  for i, seed in enumerate(seeds):
    progress((i, N_MULTI), desc=f"Relighting run {i + 1}/{N_MULTI} "
                                f"(seed {seed})")
    out_path, dur = _relight_once(ae_percentile, ambient_scale,
                                  sampling_steps, seed)
    outs.append(out_path)
    durs.append(dur)
  _ensure_input_mp4()

  updates = [gr.update(value=path, label=f"Run {i + 1} — seed {seed}")
             for i, (path, seed) in enumerate(zip(outs, seeds))]
  msg = (f"{N_MULTI} runs done in {sum(durs):.0f}s total "
         f"(steps={int(sampling_steps)}). Seeds: "
         + ", ".join(str(s) for s in seeds))
  return updates + [msg]


# ----------------------------------------------------------------------------
# Palette-mask mode: relight from a PRE-PAINTED palette mask video (the
# original GR3EN workflow) instead of SAM2 clicks. Lights are auto-detected
# from the mask's palette colors; one control row appears per detected light.
# ----------------------------------------------------------------------------
PALETTE_SPECS = ['110', '101', '011', '111', '11h', '1h1', 'h11', 'hh1',
                 'h1h', '1hh', '001', '010', '100']
SPEC_NAMES = {
    '100': 'red', '010': 'green', '001': 'blue', '110': 'yellow',
    '101': 'magenta', '011': 'cyan', '111': 'white', '11h': 'warm yellow',
    '1h1': 'pink', 'h11': 'spring green', 'hh1': 'lavender',
    'h1h': 'mint', '1hh': 'salmon',
}
EXAMPLE_DIRS = [
    os.path.join(INFERENCE_DIR, "examples_videos"),
    os.path.join(INFERENCE_DIR, "assets", "demo_video"),
]
# pairs whose filenames don't follow the <name>_rgb/_mask convention
EXTRA_PAIRS = {
    "office_view2": (
        os.path.join(INFERENCE_DIR, "assets", "demo_video",
                     "office_view2_rgb.mp4"),
        os.path.join(INFERENCE_DIR, "assets", "demo_video",
                     "office_view_2_mask.mp4"),
    ),
}


def _parse_spec_rgb(spec, hval=0.5):
  table = {'1': 1.0, '0': 0.0, 'h': float(hval)}
  return np.array([table[c] for c in spec], np.float32)


def find_example_pairs():
  """Scan EXAMPLE_DIRS for <name>_rgb.mp4 + <name>_mask(.mp4|_rgb.mp4) pairs."""
  pairs = {}
  for d in EXAMPLE_DIRS:
    if not os.path.isdir(d):
      continue
    for f in sorted(os.listdir(d)):
      if not f.endswith("_rgb.mp4") or "_mask" in f:
        continue
      base = f[:-len("_rgb.mp4")]
      for cand in (f"{base}_mask.mp4", f"{base}_mask_rgb.mp4"):
        if os.path.exists(os.path.join(d, cand)):
          pairs[base] = (os.path.join(d, f), os.path.join(d, cand))
          break
  for k, v in EXTRA_PAIRS.items():
    if os.path.exists(v[0]) and os.path.exists(v[1]):
      pairs.setdefault(k, v)
  return pairs


class PaletteSession:
  def __init__(self):
    self.workdir = None
    self.img_h, self.img_w = RESOLUTIONS[DEFAULT_RES]
    self.frames = []          # list of (H,W,3) uint8 RGB, len N_FRAMES
    self.mask_frames = None   # (T,H,W,3) float32 in [0,1] — raw palette mask
    self.region_masks = {}    # spec -> (T,H,W) bool, only detected specs
    self.detected = []        # list of specs
    self.mask_seq = None      # (T,H,W,3) float32 pre-norm composited mask

PSESS = PaletteSession()


def _center_cover_resize_interp(img, out_h, out_w, interp):
  h, w = img.shape[:2]
  scale = max(out_h / h, out_w / w)
  nh, nw = int(round(h * scale)), int(round(w * scale))
  img = cv2.resize(img, (nw, nh), interpolation=interp)
  y0 = (nh - out_h) // 2
  x0 = (nw - out_w) // 2
  return img[y0:y0 + out_h, x0:x0 + out_w]


def palette_prepare(example_name, video_up, mask_up, resolution):
  """Load + align an input/mask video pair and detect palette lights."""
  dummy_mask = False
  if video_up and mask_up:
    vid_path, mask_path = video_up, mask_up
    src = "uploaded videos"
  elif video_up and not mask_up:
    # No mask provided: use a completely black dummy mask (no explicit light
    # sources; background/-1 everywhere — same encoding as the zipnerf mode).
    vid_path, mask_path = video_up, None
    dummy_mask = True
    src = "uploaded video + BLACK dummy mask"
  else:
    pairs = find_example_pairs()
    if example_name not in pairs:
      raise gr.Error("Upload a video (mask optional), or pick an example scene.")
    vid_path, mask_path = pairs[example_name]
    src = f"example '{example_name}'"

  vid = _decode_video(vid_path)
  if not vid:
    raise gr.Error("Could not decode the input video.")
  if dummy_mask:
    msk = [np.zeros_like(vid[0])]
  else:
    msk = _decode_video(mask_path)
    if not msk:
      raise gr.Error("Could not decode the mask video.")

  PSESS.img_h, PSESS.img_w = RESOLUTIONS[resolution]

  # same resolution: bring the mask to the video's resolution first
  vh, vw = vid[0].shape[:2]
  if msk[0].shape[:2] != (vh, vw):
    msk = [cv2.resize(m, (vw, vh), interpolation=cv2.INTER_NEAREST)
           for m in msk]

  # same length: sample both at matching normalized-time positions
  vi = np.linspace(0, len(vid) - 1, N_FRAMES).round().astype(int)
  mi = np.linspace(0, len(msk) - 1, N_FRAMES).round().astype(int)
  frames = [_center_cover_resize_interp(vid[i], PSESS.img_h, PSESS.img_w,
                                        cv2.INTER_CUBIC) for i in vi]
  masks = [_center_cover_resize_interp(msk[i], PSESS.img_h, PSESS.img_w,
                                       cv2.INTER_NEAREST) for i in mi]

  PSESS.frames = frames
  mask_f = np.stack(masks).astype(np.float32) / 255.0   # (T,H,W,3)
  PSESS.mask_frames = mask_f
  PSESS.mask_seq = None

  # ---- detect palette lights (same rule as GR3EN_dataset) ----
  bright = mask_f.max(axis=-1) >= 0.30
  PSESS.region_masks = {}
  PSESS.detected = []
  coverage = {}
  for spec in PALETTE_SPECS:
    target = _parse_spec_rgb(spec)[None, None, None, :]
    match = (np.abs(mask_f - target) <= 0.18).all(axis=-1) & bright
    peak = int(match.sum(axis=(1, 2)).max())
    if peak >= 300:  # ignore compression-noise specks
      PSESS.region_masks[spec] = match
      PSESS.detected.append(spec)
      coverage[spec] = peak

  PSESS.workdir = os.path.join(SESSIONS_ROOT, f"palette_{int(time.time())}")
  os.makedirs(PSESS.workdir, exist_ok=True)

  # raw-mask preview (blend, same viz convention as the model's control viz)
  preview = os.path.join(PSESS.workdir, "raw_mask_preview.mp4")
  blend = [
      (0.15 * frames[t].astype(np.float32) + 0.85 * mask_f[t] * 255.0)
      .clip(0, 255).astype(np.uint8)
      for t in range(N_FRAMES)
  ]
  _write_mp4(preview, blend)

  n_src = f"video {len(vid)}f / mask {len(msk)}f"
  if not PSESS.detected:
    # No lights (e.g. black dummy mask): auto-build an all-background control
    # mask so the user can go straight to Relight (ambient-only relighting).
    PSESS.mask_seq = np.zeros(
        (N_FRAMES, PSESS.img_h, PSESS.img_w, 3), np.float32)
    if dummy_mask:
      info = (f"Loaded {src} ({len(vid)} frames) at "
              f"{PSESS.img_w}x{PSESS.img_h}. Using a **black dummy mask** — "
              f"no explicit light sources. You can *Relight* directly "
              f"(ambient / auto-exposure still apply).")
    else:
      info = (f"Loaded {src} ({n_src}) at {PSESS.img_w}x{PSESS.img_h} — "
              f"**no palette lights found** in the mask (is it "
              f"palette-painted?). An all-background control mask was built; "
              f"you can still *Relight* (ambient-only).")
  else:
    rows = [f"- **'{s}'** ({SPEC_NAMES.get(s, s)} paint): up to "
            f"{coverage[s]} px/frame" for s in PSESS.detected]
    info = (f"Loaded {src} ({n_src}), conformed to {N_FRAMES} frames at "
            f"{PSESS.img_w}x{PSESS.img_h}.\n\n**Detected "
            f"{len(PSESS.detected)} light(s):**\n" + "\n".join(rows) +
            "\n\nSet each light below, then *Build light mask* and *Relight*.")

  row_updates = []
  for spec in PALETTE_SPECS:
    if spec in PSESS.detected:
      row_updates.append(gr.update(
          visible=True))
    else:
      row_updates.append(gr.update(visible=False))
  label_updates = [
      gr.update(value=f"**Light '{s}'** — {SPEC_NAMES.get(s, s)} paint")
      for s in PALETTE_SPECS
  ]
  return [info, preview, frames[0]] + row_updates + label_updates


def palette_build(*settings):
  """Composite the pre-norm control mask from per-light settings.

  settings: flat (state, color, intensity) per PALETTE_SPECS entry.
  Encoding (matches GR3EN_dataset dict mode): background / 'No change' -> 0
  (normalized -1), 'Off' -> 0.5*white (normalized 0), 'On' -> intensity*color.
  """
  if PSESS.mask_frames is None:
    raise gr.Error("Prepare a video (+ optional mask) first.")
  if not PSESS.detected:
    # black/blank mask: all-background control mask (ambient-only relight)
    PSESS.mask_seq = np.zeros(
        (N_FRAMES, PSESS.img_h, PSESS.img_w, 3), np.float32)
    return None, ("No lights in the mask — built an all-background control "
                  "mask (ambient-only relight).")
  states = settings[0::3]
  colors = settings[1::3]
  intens = settings[2::3]

  mask_seq = np.zeros((N_FRAMES, PSESS.img_h, PSESS.img_w, 3), np.float32)
  applied = []
  for i, spec in enumerate(PALETTE_SPECS):
    if spec not in PSESS.detected:
      continue
    region = PSESS.region_masks[spec][..., None].astype(np.float32)
    st = states[i]
    if st == "No change":
      continue
    if st == "Off":
      val = np.array([0.5, 0.5, 0.5], np.float32)  # explicit off level
      applied.append(f"'{spec}' off")
    else:
      inten = _intensity_to_mask(intens[i])
      val = inten * _hex_to_rgb01(colors[i])
      applied.append(f"'{spec}' on ({inten:.2f})")
    mask_seq = mask_seq * (1 - region) + region * val[None, None, None, :]
  mask_seq = np.clip(mask_seq, 0.0, 1.0)
  PSESS.mask_seq = mask_seq

  preview = os.path.join(PSESS.workdir, "control_mask_preview.mp4")
  blend = [
      (0.15 * PSESS.frames[t].astype(np.float32) + 0.85 * mask_seq[t] * 255.0)
      .clip(0, 255).astype(np.uint8)
      for t in range(N_FRAMES)
  ]
  _write_mp4(preview, blend)
  return preview, "Control mask built: " + ", ".join(applied)


# ----------------------------------------------------------------------------
# Mask authoring: export the SAM2-selected light regions as a PALETTE mask
# (each light painted with an [R,G,B] color whose channels are in {0, 0.5, 1}),
# aligned with the palette relighting protocol.
# ----------------------------------------------------------------------------
# preferred spec order when auto-assigning colors to lights
PREFERRED_SPECS = ['100', '010', '001', '110', '101', '011',
                   '111', '11h', '1h1', 'h11', 'hh1', 'h1h', '1hh']


def export_palette_mask(*spec_choices):
  """Paint each SAM2 light region with its palette color and export the mask.

  Writes lossless PNGs (the exact format GR3EN_dataset consumes) plus an mp4
  preview/download. Channel values are 0 / 0.5 / 1 (stored as 0 / 128 / 255).
  """
  if not SESS.frames:
    raise gr.Error("Prepare frames and click the lights first (tabs 1-2).")
  if SESS.n_lights == 0:
    raise gr.Error("Click at least one light source first (tab 2).")
  if SESS.masks is None:
    propagate()  # auto-propagate through the video

  specs = [str(s) for s in spec_choices[:SESS.n_lights]]
  if len(set(specs)) != len(specs):
    raise gr.Error(f"Each light needs a UNIQUE palette color; got {specs}.")

  mask_seq = np.zeros((N_FRAMES, SESS.img_h, SESS.img_w, 3), np.float32)
  for li in range(SESS.n_lights):
    rgb = _parse_spec_rgb(specs[li])
    region = SESS.masks[li][..., None].astype(np.float32)
    mask_seq = mask_seq * (1 - region) + region * rgb[None, None, None, :]

  out_dir = os.path.join(SESS.workdir, "palette_mask")
  os.makedirs(out_dir, exist_ok=True)
  for t in range(N_FRAMES):
    png = (mask_seq[t] * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{t:05d}.png"),
                cv2.cvtColor(png, cv2.COLOR_RGB2BGR))

  mp4_path = os.path.join(SESS.workdir, "palette_mask.mp4")
  _write_mp4(mp4_path, [(mask_seq[t] * 255.0 + 0.5).astype(np.uint8)
                        for t in range(N_FRAMES)])
  # also export the conformed input frames for a ready-to-use pair
  in_path = os.path.join(SESS.workdir, "input.mp4")
  if not os.path.exists(in_path):
    _write_mp4(in_path, SESS.frames)

  assign = ", ".join(
      f"Light {li} -> '{specs[li]}' ({SPEC_NAMES.get(specs[li], specs[li])})"
      for li in range(SESS.n_lights))
  msg = (f"Exported palette mask: {assign}\n\n"
         f"- PNG frames (lossless, dataset format): `{out_dir}/`\n"
         f"- Mask video: `{mp4_path}`\n"
         f"- Matching input video: `{in_path}`\n\n"
         f"Use this pair in the *Palette-mask relight* tab, with "
         f"`random_relight.py`, or with the yaml/fsdp.py workflow.")
  return mp4_path, mp4_path, msg


def palette_relight(ae_percentile, ambient_scale, sampling_steps, seed):
  import random as _random
  if PSESS.mask_seq is None:
    raise gr.Error("Build the light mask first.")
  if PIPELINE is None:
    raise gr.Error("Model pipeline is not loaded (startup failed?).")
  seed = int(seed)
  if seed < 0:
    seed = _random.randint(0, 2**31 - 1)

  p = float(np.clip(ae_percentile, 0.5, 0.999999))
  ae_scale = float(-np.log10(p))
  t0 = time.time()
  video = PIPELINE.relight(
      PSESS.frames, PSESS.mask_seq,
      ambient_scale=ambient_scale, ae_scale=ae_scale,
      sampling_steps=sampling_steps, seed=seed,
  )
  dur = time.time() - t0

  out = ((video.clamp(-1, 1).permute(1, 2, 3, 0).float().cpu().numpy() + 1.0)
         / 2.0 * 255.0).astype(np.uint8)
  out_path = os.path.join(PSESS.workdir,
                          f"relit_seed{seed}_{int(time.time())}.mp4")
  _write_mp4(out_path, list(out))
  in_path = os.path.join(PSESS.workdir, "input.mp4")
  if not os.path.exists(in_path):
    _write_mp4(in_path, PSESS.frames)

  msg = (f"Done in {dur:.0f}s (seed={seed}, steps={int(sampling_steps)}, "
         f"ae p={p:.3f}, ambient={ambient_scale:.2f}). Saved to {out_path}")
  return out_path, in_path, msg


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def build_full_ui():
  """Combined relighting UI (served on 7861): SAM2 interactive relight +
  palette-mask relight + palette export — the original all-in-one layout."""
  import gradio as gr

  with gr.Blocks(title="GR3EN — Interactive Relighting") as demo:
    gr.Markdown(
        "# GR3EN — Interactive Relighting (SAM2 + palette)\n"
        "Load a video, click each light source, set its **state / color / "
        "intensity**, then relight — or use the *Palette-mask relight* tab "
        "with a pre-painted mask. The 5B model is resident on one GPU."
    )

    with gr.Tab("1. Load video"):
      video_in = gr.Video(label="Input video", sources=["upload"])
      frame_msg = gr.Markdown("Upload a video to begin.")
      long_policy = gr.Radio(
          ["Trim to first 81", "Downsample to 81 (keep every k-th)"],
          value="Downsample to 81 (keep every k-th)",
          label="If longer than 81 frames", visible=False,
      )
      down_factor = gr.Number(label="Downsample factor k", value=1,
                              precision=0, visible=False)
      resolution = gr.Radio(
          list(RESOLUTIONS.keys()), value=DEFAULT_RES,
          label="Working resolution (352x640 is ~3x faster; 608x1120 is the "
                "native training resolution)",
      )
      prep_btn = gr.Button("Prepare 81 frames", variant="primary")
      prep_info = gr.Markdown()

    with gr.Tab("2. Select lights (click)"):
      with gr.Row():
        with gr.Column(scale=3):
          canvas = gr.Image(label="Click on a light source", type="numpy",
                            interactive=True, height=420)
          frame_slider = gr.Slider(0, N_FRAMES - 1, value=0, step=1,
                                   label="Preview frame")
        with gr.Column(scale=1):
          light_id = gr.Number(label="Active light id", value=0, precision=0)
          point_type = gr.Radio(
              ["Positive (on the light)", "Negative (not the light)"],
              value="Positive (on the light)", label="Click type")
          new_light_btn = gr.Button("+ New light")
          reset_light_btn = gr.Button("Reset this light")
          clear_btn = gr.Button("Clear all")
          lights_md = gr.Markdown("No lights yet.")

    with gr.Tab("3. Assign & relight"):
      prop_btn = gr.Button("Propagate masks through video", variant="primary")
      prop_info = gr.Markdown()
      gr.Markdown("### Per-light settings")
      light_rows = []
      for li in range(MAX_LIGHTS):
        with gr.Row(visible=(li == 0)) as row:
          gr.Markdown(f"**Light {li}**")
          st = gr.Dropdown(["On", "Off", "No change"], value="On",
                           label="State", scale=2)
          col = gr.ColorPicker(value="#ffffff", label="Color (On)", scale=1)
          inten = gr.Slider(1.0, 5.0, value=5.0, step=0.05,
                            label="Intensity (1 = dim · 5 = max)", scale=2)
        light_rows.append((row, st, col, inten))
      build_btn = gr.Button("Build mask & preview")
      mask_preview = gr.Video(label="Control-mask preview")
      build_info = gr.Markdown()

      with gr.Row():
        ae_slider = gr.Slider(
            0.90, 0.999, value=0.99, step=0.001,
            label="Auto-exposure percentile p (ae = -log10(p); lower p = brighter)")
        ambient_slider = gr.Slider(
            0.0, 1.0, value=0.5, step=0.05,
            label="Ambient scale (0.5 = neutral)")
      with gr.Row():
        steps_slider = gr.Slider(10, 75, value=50, step=1,
                                 label="Diffusion sampling steps")
        seed_box = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

      with gr.Row():
        relight_btn = gr.Button("Relight", variant="primary")
        relight5_btn = gr.Button(f"Relight {N_MULTI}x (random seeds)")
      with gr.Row():
        in_video = gr.Video(label="Input (conformed)")
        out_video = gr.Video(label="Relit output")
      relight_info = gr.Markdown()
      gr.Markdown(f"### Seed variations ({N_MULTI} runs, random seeds)")
      multi_videos = []
      with gr.Row():
        for i in range(N_MULTI):
          multi_videos.append(gr.Video(label=f"Run {i + 1}", scale=1))
      multi_info = gr.Markdown()

    with gr.Tab("Palette-mask relight"):
      gr.Markdown(
          "### Relight from a pre-painted palette mask\n"
          "Provide an **input video** and a **mask video** whose light sources "
          "are painted with palette colors (author one in the separate "
          "Light-Mask Maker demo). Aligned automatically; each detected "
          "light gets its own control row."
      )
      example_pairs = find_example_pairs()
      with gr.Row():
        example_dd = gr.Dropdown(
            sorted(example_pairs.keys()),
            value=(sorted(example_pairs.keys())[0] if example_pairs else None),
            label="Example scene (used when no uploads given)")
        p_resolution = gr.Radio(list(RESOLUTIONS.keys()), value=DEFAULT_RES,
                                label="Working resolution")
      with gr.Row():
        p_video_in = gr.Video(label="Input video (optional upload)",
                              sources=["upload"])
        p_mask_in = gr.Video(
            label="Palette mask video (optional — leave empty to use a black "
                  "dummy mask: no explicit light sources)",
            sources=["upload"])
      p_prep_btn = gr.Button("Load & detect lights", variant="primary")
      p_info = gr.Markdown()
      with gr.Row():
        p_first = gr.Image(label="First frame", interactive=False)
        p_mask_preview = gr.Video(label="Mask preview (raw palette mask)")

      gr.Markdown("### Per-light settings (auto-detected)")
      p_rows, p_labels, p_settings = [], [], []
      for spec in PALETTE_SPECS:
        with gr.Row(visible=False) as prow:
          plabel = gr.Markdown(f"**Light '{spec}'**")
          pst = gr.Dropdown(["On", "Off", "No change"], value="On",
                            label="State", scale=2)
          pcol = gr.ColorPicker(value="#ffffff", label="Color (On)", scale=1)
          pint = gr.Slider(1.0, 5.0, value=5.0, step=0.05,
                           label="Intensity (1 = dim · 5 = max)", scale=2)
        p_rows.append(prow)
        p_labels.append(plabel)
        p_settings += [pst, pcol, pint]

      p_build_btn = gr.Button("Build light mask & preview")
      p_ctrl_preview = gr.Video(label="Control-mask preview (what the model sees)")
      p_build_info = gr.Markdown()

      with gr.Row():
        p_ae = gr.Slider(0.90, 0.999, value=0.99, step=0.001,
                         label="Auto-exposure percentile p")
        p_ambient = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                              label="Ambient scale")
      with gr.Row():
        p_steps = gr.Slider(10, 75, value=50, step=1,
                            label="Diffusion sampling steps")
        p_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
      p_relight_btn = gr.Button("Relight", variant="primary")
      with gr.Row():
        p_in_video = gr.Video(label="Input (conformed)")
        p_out_video = gr.Video(label="Relit output")
      p_relight_info = gr.Markdown()

    # ---- wiring ----
    video_in.change(on_video_uploaded, [video_in],
                    [frame_msg, long_policy, down_factor])
    prep_btn.click(prepare_frames,
                   [video_in, long_policy, down_factor, resolution],
                   [canvas, prep_info, frame_slider, lights_md])
    canvas.select(add_click, [frame_slider, light_id, point_type],
                  [canvas, lights_md])
    frame_slider.change(show_frame, [frame_slider], [canvas])
    new_light_btn.click(new_light, None, [light_id, lights_md])
    reset_light_btn.click(reset_light, [light_id], [canvas, lights_md])
    clear_btn.click(clear_all, None, [canvas, lights_md])
    prop_btn.click(propagate, None, [canvas, prop_info])

    def _sync_rows():
      import gradio as gr
      return [gr.update(visible=(li < max(1, SESS.n_lights)))
              for li in range(MAX_LIGHTS)]
    prop_btn.click(_sync_rows, None, [r[0] for r in light_rows])

    flat_settings = []
    for _, st, col, inten in light_rows:
      flat_settings += [st, col, inten]
    build_btn.click(build_mask, flat_settings, [mask_preview, build_info])
    relight_btn.click(run_relight,
                      [ae_slider, ambient_slider, steps_slider, seed_box],
                      [out_video, in_video, relight_info])
    relight5_btn.click(run_relight_multi,
                       [ae_slider, ambient_slider, steps_slider],
                       multi_videos + [multi_info])

    p_prep_btn.click(
        palette_prepare,
        [example_dd, p_video_in, p_mask_in, p_resolution],
        [p_info, p_mask_preview, p_first] + p_rows + p_labels)
    p_build_btn.click(palette_build, p_settings,
                      [p_ctrl_preview, p_build_info])
    p_relight_btn.click(palette_relight,
                        [p_ae, p_ambient, p_steps, p_seed],
                        [p_out_video, p_in_video, p_relight_info])

  return demo


def build_mask_ui():
  """Standalone LIGHT-MASK MAKER UI: SAM2 selection + palette export only.

  No relighting here — masks authored in this demo are consumed by the
  separate relighting demo (or random_relight.py / the yaml workflow).
  """
  import gradio as gr

  with gr.Blocks(title="GR3EN — Light-Mask Maker (SAM2)") as demo:
    gr.Markdown(
        "# GR3EN — Light-Mask Maker (SAM2)\n"
        "Author a palette light-source mask: load a video, **click** each "
        "light source (SAM2 segments and tracks it), then export the palette "
        "mask — each light painted with an [R,G,B] color whose channels are "
        "in **{0, 0.5, 1}**, aligned with the relighting protocol. "
        "Relighting itself lives in the separate relighting demo."
    )

    with gr.Tab("1. Load video"):
      video_in = gr.Video(label="Input video", sources=["upload"])
      frame_msg = gr.Markdown("Upload a video to begin.")
      long_policy = gr.Radio(
          ["Trim to first 81", "Downsample to 81 (keep every k-th)"],
          value="Downsample to 81 (keep every k-th)",
          label="If longer than 81 frames", visible=False,
      )
      down_factor = gr.Number(label="Downsample factor k", value=1,
                              precision=0, visible=False)
      resolution = gr.Radio(
          list(RESOLUTIONS.keys()), value=DEFAULT_RES,
          label="Working resolution (352x640 is ~3x faster; 608x1120 is the "
                "native training resolution)",
      )
      prep_btn = gr.Button("Prepare 81 frames", variant="primary")
      prep_info = gr.Markdown()

    with gr.Tab("2. Select lights (click)"):
      with gr.Row():
        with gr.Column(scale=3):
          canvas = gr.Image(label="Click on a light source", type="numpy",
                            interactive=True, height=420)
          frame_slider = gr.Slider(0, N_FRAMES - 1, value=0, step=1,
                                   label="Preview frame")
        with gr.Column(scale=1):
          light_id = gr.Number(label="Active light id", value=0, precision=0)
          point_type = gr.Radio(
              ["Positive (on the light)", "Negative (not the light)"],
              value="Positive (on the light)", label="Click type")
          new_light_btn = gr.Button("+ New light")
          reset_light_btn = gr.Button("Reset this light")
          clear_btn = gr.Button("Clear all")
          lights_md = gr.Markdown("No lights yet.")

    with gr.Tab("3. Export palette mask"):
      gr.Markdown(
          "### Author a palette light-source mask (mask only)\n"
          "Uses the lights you selected with SAM2 (tabs 1-2). Each light is "
          "painted with a palette color whose **R,G,B channels are in "
          "{0, 0.5, 1}**, aligned with the relighting protocol. Masks are "
          "propagated through the whole clip automatically if needed.\n\n"
          "Output: lossless PNG frames (the dataset format) + a mask video + "
          "the matching conformed input video."
      )
      exp_specs = []
      with gr.Row():
        for li in range(MAX_LIGHTS):
          exp_specs.append(gr.Dropdown(
              PREFERRED_SPECS,
              value=PREFERRED_SPECS[li % len(PREFERRED_SPECS)],
              label=f"Light {li} palette color"))
      export_btn = gr.Button("Propagate & export palette mask",
                             variant="primary")
      with gr.Row():
        export_video = gr.Video(label="Exported palette mask")
        export_file = gr.File(label="Download mask video")
      export_info = gr.Markdown()

    # ---- wiring (mask maker) ----
    video_in.change(on_video_uploaded, [video_in],
                    [frame_msg, long_policy, down_factor])
    prep_btn.click(prepare_frames,
                   [video_in, long_policy, down_factor, resolution],
                   [canvas, prep_info, frame_slider, lights_md])
    canvas.select(add_click, [frame_slider, light_id, point_type],
                  [canvas, lights_md])
    frame_slider.change(show_frame, [frame_slider], [canvas])
    new_light_btn.click(new_light, None, [light_id, lights_md])
    reset_light_btn.click(reset_light, [light_id], [canvas, lights_md])
    clear_btn.click(clear_all, None, [canvas, lights_md])
    export_btn.click(export_palette_mask, exp_specs,
                     [export_video, export_file, export_info])

  return demo


def build_relight_ui():
  """RELIGHTING demo UI: palette-mask relight only (masks come from the
  separate Light-Mask Maker demo, pre-painted masks, or uploads)."""
  import gradio as gr

  with gr.Blocks(title="GR3EN — Palette Relighting") as demo:
    gr.Markdown(
        "# GR3EN — Relighting demo (palette mask)\n"
        "Relight a video from a palette light-source mask. To CREATE a mask "
        "by clicking light sources (SAM2), use the separate **Light-Mask "
        "Maker** demo, then bring the exported pair here."
    )

    with gr.Tab("Palette-mask relight"):
      gr.Markdown(
          "### Relight from a pre-painted palette mask\n"
          "Provide an **input video** and a **mask video** whose light sources "
          "are painted with palette colors (red/green/blue/yellow/...). The "
          "two are aligned automatically (resolution + duration). Each "
          "detected light gets its own control row."
      )
      example_pairs = find_example_pairs()
      with gr.Row():
        example_dd = gr.Dropdown(
            sorted(example_pairs.keys()),
            value=(sorted(example_pairs.keys())[0] if example_pairs else None),
            label="Example scene (used when no uploads given)")
        p_resolution = gr.Radio(list(RESOLUTIONS.keys()), value=DEFAULT_RES,
                                label="Working resolution")
      with gr.Row():
        p_video_in = gr.Video(label="Input video (optional upload)",
                              sources=["upload"])
        p_mask_in = gr.Video(
            label="Palette mask video (optional — leave empty to use a black "
                  "dummy mask: no explicit light sources)",
            sources=["upload"])
      p_prep_btn = gr.Button("Load & detect lights", variant="primary")
      p_info = gr.Markdown()
      with gr.Row():
        p_first = gr.Image(label="First frame", interactive=False)
        p_mask_preview = gr.Video(label="Mask preview (raw palette mask)")

      gr.Markdown("### Per-light settings (auto-detected)")
      p_rows, p_labels, p_settings = [], [], []
      for spec in PALETTE_SPECS:
        with gr.Row(visible=False) as prow:
          plabel = gr.Markdown(f"**Light '{spec}'**")
          pst = gr.Dropdown(["On", "Off", "No change"], value="On",
                            label="State", scale=2)
          pcol = gr.ColorPicker(value="#ffffff", label="Color (On)", scale=1)
          pint = gr.Slider(1.0, 5.0, value=5.0, step=0.05,
                           label="Intensity (1 = dim · 5 = max)", scale=2)
        p_rows.append(prow)
        p_labels.append(plabel)
        p_settings += [pst, pcol, pint]

      p_build_btn = gr.Button("Build light mask & preview")
      p_ctrl_preview = gr.Video(label="Control-mask preview (what the model sees)")
      p_build_info = gr.Markdown()

      with gr.Row():
        p_ae = gr.Slider(0.90, 0.999, value=0.99, step=0.001,
                         label="Auto-exposure percentile p")
        p_ambient = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                              label="Ambient scale")
      with gr.Row():
        p_steps = gr.Slider(10, 75, value=50, step=1,
                            label="Diffusion sampling steps")
        p_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
      p_relight_btn = gr.Button("Relight", variant="primary")
      with gr.Row():
        p_in_video = gr.Video(label="Input (conformed)")
        p_out_video = gr.Video(label="Relit output")
      p_relight_info = gr.Markdown()

    # ---- wiring ----
    p_prep_btn.click(
        palette_prepare,
        [example_dd, p_video_in, p_mask_in, p_resolution],
        [p_info, p_mask_preview, p_first] + p_rows + p_labels)
    p_build_btn.click(palette_build, p_settings,
                      [p_ctrl_preview, p_build_info])
    p_relight_btn.click(palette_relight,
                        [p_ae, p_ambient, p_steps, p_seed],
                        [p_out_video, p_in_video, p_relight_info])

  return demo


# ----------------------------------------------------------------------------
# Headless end-to-end selftest (no UI): demo frames -> brightest-pixel click ->
# SAM2 propagate -> red light -> 10-step relight.
# ----------------------------------------------------------------------------
def selftest():
  print("===== gradio_app selftest (SAM2 -> mask -> single-GPU relight) =====",
        flush=True)
  src = sorted(glob.glob(os.path.join(INFERENCE_DIR,
                                      "data/demo_video/frames/*")))
  assert src, f"no demo frames under {INFERENCE_DIR}/data/demo_video/frames"
  SESS.img_h, SESS.img_w = RESOLUTIONS["352x640 (fast)"]
  frames = [
      _center_cover_resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
                           SESS.img_h, SESS.img_w)
      for p in src[:N_FRAMES]
  ]
  frames += [frames[-1]] * (N_FRAMES - len(frames))
  _init_session_from_frames(frames)

  gray = frames[0].mean(2)
  y, x = np.unravel_index(int(gray.argmax()), gray.shape)
  print(f"[selftest] click Light 0 at x={x} y={y}", flush=True)
  SESS.points = {0: {0: [(int(x), int(y), 1)]}}
  SESS.n_lights = 1
  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    _, oids, ml = SESS.predictor.add_new_points_or_box(
        inference_state=SESS.state, frame_idx=0, obj_id=0,
        points=np.array([[x, y]], np.float32), labels=np.array([1], np.int32))
    for j, oid in enumerate(oids):
      SESS._live[(int(oid), 0)] = (ml[j, 0] > 0).cpu().numpy()

  masks = np.zeros((1, N_FRAMES, SESS.img_h, SESS.img_w), dtype=bool)
  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for fidx, obj_ids, mask_logits in SESS.predictor.propagate_in_video(
        SESS.state):
      for j, oid in enumerate(obj_ids):
        if int(oid) == 0:
          masks[0, fidx] = (mask_logits[j, 0] > 0).cpu().numpy()
  SESS.masks = masks
  print(f"[selftest] propagation done: light covers "
        f"{masks[0].any(axis=(1, 2)).sum()} frames", flush=True)

  mask_seq = np.zeros((N_FRAMES, SESS.img_h, SESS.img_w, 3), np.float32)
  red = np.array([1.0, 0.2, 0.2], np.float32)
  for t in range(N_FRAMES):
    mask_seq[t][masks[0, t]] = red
  SESS.mask_seq = mask_seq

  out_path, in_path, msg = run_relight(
      ae_percentile=0.99, ambient_scale=0.5, sampling_steps=10, seed=1337)
  print("[selftest]", msg, flush=True)
  assert os.path.exists(out_path) and os.path.getsize(out_path) > 10000
  print("SELFTEST PASSED ->", out_path, flush=True)


def selftest_palette():
  """Headless test of the palette-mask flow on an example pair."""
  print("===== palette-mask selftest =====", flush=True)
  pairs = find_example_pairs()
  assert pairs, "no example pairs found"
  name = sorted(pairs.keys())[0]
  print(f"[selftest-palette] scene: {name}", flush=True)
  out = palette_prepare(name, None, None, "352x640 (fast)")
  print("[selftest-palette]", out[0].replace("\n", " | ")[:400], flush=True)
  assert PSESS.detected, "no lights detected in example mask"

  settings = []
  for spec in PALETTE_SPECS:
    settings += ["On", "#ffffff", 1.0]
  preview, msg = palette_build(*settings)
  print("[selftest-palette]", msg, flush=True)
  assert os.path.exists(preview)

  out_path, _, msg = palette_relight(0.99, 0.5, 10, 1337)
  print("[selftest-palette]", msg, flush=True)
  assert os.path.exists(out_path) and os.path.getsize(out_path) > 10000
  print("PALETTE SELFTEST PASSED ->", out_path, flush=True)


def main():
  global PIPELINE
  parser = argparse.ArgumentParser()
  parser.add_argument("--selftest", action="store_true",
                      help="headless end-to-end GPU test (SAM2 flow), then exit")
  parser.add_argument("--selftest-palette", action="store_true",
                      help="headless end-to-end GPU test (palette flow), then exit")
  parser.add_argument("--port", type=int, default=7860)
  args = parser.parse_args()

  os.makedirs(SESSIONS_ROOT, exist_ok=True)
  assert torch.cuda.is_available(), "CUDA GPU required"
  print(f"[app] GPU: {torch.cuda.get_device_name(0)}", flush=True)

  PIPELINE = RelightPipeline(device_id=0)

  if args.selftest:
    selftest()
    return
  if args.selftest_palette:
    selftest_palette()
    return

  ui = build_full_ui()
  ui.queue()
  print(f"[app] launching Gradio on 0.0.0.0:{args.port} "
        f"(node: {os.uname().nodename})", flush=True)
  try:
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=False)
  except OSError:
    # port taken (e.g. another demo on the same node) -> let Gradio pick one;
    # the actual port is printed in the "Running on local URL" line.
    print(f"[app] port {args.port} is busy; picking the next free port...",
          flush=True)
    ui.launch(server_name="0.0.0.0", server_port=None, share=False)


if __name__ == "__main__":
  main()
