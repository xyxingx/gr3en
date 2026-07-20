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
"""GR3EN Relighting Studio — streamlined single-page wizard UI (port 7862).

One vertical flow, no tabs. A stepper at the top always shows where you are:

  1 Upload  ->  2 Select lights  ->  3 Configure lights  ->  4 Relight

- Upload: video is analyzed instantly (frames / fps / resolution); if longer
  than 81 frames the user picks "first 81" or downsampling at a recommended
  fps, then the SAM2 click canvas opens automatically.
- Select lights: same click logic as before; one confirm button propagates
  all masks through the clip and advances the flow.
- Configure lights: per-light state / color / intensity; confirming renders
  the control mask automatically.
- Relight: single run or 5 random seeds; denoising steps default to 10 for
  speed.
- Example scenes (palette video+mask pairs) sit at the bottom; picking one
  loads the pair and jumps straight to the light-configuration stage.

Reuses the session/back-end logic from gradio_app.py; this file is UI only.

Run:  PYTHONPATH=. python studio_app.py --port 7862
"""

import argparse
import math
import os
import random
import time

import cv2
import gradio as gr
import numpy as np
import torch

import gradio_app as core

MAX_CFG = 8   # generic light-config rows (covers SAM2's 6 and example specs)

STEPS = ["Upload", "Select lights", "Configure lights", "Relight"]

CSS = """
#stepper-box {position: sticky; top: 0; z-index: 50; background: var(--body-background-fill); padding: 6px 0 2px 0;}
.stepper {display: flex; gap: 10px; justify-content: center; margin: 4px 0;}
.step {display: flex; align-items: center; gap: 8px; padding: 8px 18px; border-radius: 999px;
       background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
       color: var(--body-text-color-subdued); font-weight: 600; font-size: 0.95em;}
.step .num {display: inline-flex; width: 22px; height: 22px; border-radius: 50%;
            align-items: center; justify-content: center; background: var(--border-color-primary);
            color: var(--body-text-color); font-size: 0.85em;}
.step.active {border-color: #ff7b26; color: var(--body-text-color); box-shadow: 0 0 0 2px #ff7b2633;}
.step.active .num {background: #ff7b26; color: white;}
.step.done {border-color: #22a06b; color: var(--body-text-color);}
.step.done .num {background: #22a06b; color: white;}
.hint {text-align: center; font-size: 1.05em; margin: 2px 0 8px 0; color: var(--body-text-color);}
.hint b {color: #ff7b26;}
"""


def stepper_html(cur, hint):
  pills = []
  for i, s in enumerate(STEPS):
    cls = "done" if i < cur else ("active" if i == cur else "")
    mark = "✓" if i < cur else str(i + 1)
    pills.append(f'<div class="step {cls}"><span class="num">{mark}</span>{s}</div>')
  return (f'<div class="stepper">{"".join(pills)}</div>'
          f'<div class="hint"><b>{hint}</b></div>')


# --------------------------- studio session state ---------------------------
STUDIO = {
    "mode": None,       # 'sam2' | 'palette'
    "keys": [],         # sam2: light indices; palette: detected specs
    "video_meta": None, # (n, fps, w, h)
}


def _row_updates_for_keys(labels):
  """Visibility + label updates for the MAX_CFG config rows."""
  row_ups, label_ups = [], []
  for i in range(MAX_CFG):
    if i < len(labels):
      row_ups.append(gr.update(visible=True))
      label_ups.append(gr.update(value=labels[i]))
    else:
      row_ups.append(gr.update(visible=False))
      label_ups.append(gr.update(value=""))
  return row_ups, label_ups


EXTRA_VIDEO_DIR = os.path.join(core.INFERENCE_DIR, "assets", "extra_video")


# ------------------------------- step 1: upload ------------------------------
def on_upload(video_path, resolution):
  if not video_path:
    return (stepper_html(0, "Upload a video to begin"),
            gr.update(value="", visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            None, gr.update(), "No lights yet.")
  cap = cv2.VideoCapture(video_path)
  n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  cap.release()
  STUDIO["video_meta"] = (n, fps, w, h)

  if n <= core.N_FRAMES:
    # nothing to choose — prepare right away and open the SAM2 canvas
    first, info, slider_up, lights = core.prepare_frames(
        video_path, "Trim to first 81", 1, resolution)
    analysis = (f"**{n} frames** @ {fps:.1f} fps, {w}x{h} — short clip, "
                f"padded to {core.N_FRAMES} frames automatically.")
    return (stepper_html(1, "Click each light source"),
            gr.update(value=analysis, visible=True),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=True), first, slider_up, lights)

  k = max(1, math.ceil(n / core.N_FRAMES))
  rec_fps = fps / k
  analysis = (f"**{n} frames** @ {fps:.1f} fps, {w}x{h} — longer than "
              f"{core.N_FRAMES} frames: pick how to sample it.")
  choices = [
      f"Downsample to ~{rec_fps:.1f} fps (recommended — covers the whole clip)",
      "Use the first 81 frames only",
  ]
  return (stepper_html(0, "Pick a frame sampling, then continue"),
          gr.update(value=analysis, visible=True),
          gr.update(choices=choices, value=choices[0], visible=True),
          gr.update(visible=True),
          gr.update(visible=False), None, gr.update(), "No lights yet.")


def on_continue(video_path, choice, resolution):
  if not video_path:
    raise gr.Error("Upload a video first.")
  n, fps, _, _ = STUDIO["video_meta"] or (0, 30.0, 0, 0)
  if choice and choice.startswith("Downsample"):
    policy, k = "Downsample to 81 (keep every k-th)", max(1, math.ceil(n / core.N_FRAMES))
  else:
    policy, k = "Trim to first 81", 1
  first, info, slider_up, lights = core.prepare_frames(
      video_path, policy, k, resolution)
  return (stepper_html(1, "Click each light source"),
          gr.update(visible=True), first, slider_up, lights)


# --------------------------- step 2 -> 3: lights done ------------------------
def confirm_lights():
  if core.SESS.n_lights == 0:
    raise gr.Error("Click at least one light source first.")
  overlay, prop_msg = core.propagate()
  STUDIO["mode"] = "sam2"
  STUDIO["keys"] = list(range(core.SESS.n_lights))
  labels = [f"**Light {i}**" for i in STUDIO["keys"]]
  row_ups, label_ups = _row_updates_for_keys(labels)
  return ([stepper_html(2, "Set state, color and intensity"),
           overlay, gr.update(visible=True),
           f"Masks propagated through all {core.N_FRAMES} frames. {prop_msg.splitlines()[0]}"]
          + row_ups + label_ups)


# ------------------------- step 3 -> 4: render mask --------------------------
def render_mask(*settings):
  # per light: (color, state, intensity) with intensity on the RAW training
  # scale (0-5); core._intensity_to_mask applies the training sigmoid when
  # composing the mask, so values pass through untouched here.
  colors = list(settings[0::3])
  states = list(settings[1::3])
  intens = [float(v) for v in settings[2::3]]

  if STUDIO["mode"] == "palette":
    full = []
    for spec in core.PALETTE_SPECS:
      if spec in STUDIO["keys"]:
        i = STUDIO["keys"].index(spec)
        full += [states[i], colors[i], intens[i]]
      else:
        full += ["No change", "#ffffff", 1.0]
    preview, msg = core.palette_build(*full)
  elif STUDIO["mode"] == "sam2":
    full = []
    for i in range(core.MAX_LIGHTS):
      if i < len(STUDIO["keys"]):
        full += [states[i], colors[i], intens[i]]
      else:
        full += ["No change", "#ffffff", 1.0]
    preview, msg = core.build_mask(*full)
  else:
    raise gr.Error("Select lights (or pick an example) first.")

  return (stepper_html(3, "Ready — hit Relight!"),
          preview, msg, gr.update(visible=True))


# ------------------------------- step 4: relight -----------------------------
def _session():
  if STUDIO["mode"] == "palette":
    return core.PSESS.frames, core.PSESS.mask_seq, core.PSESS.workdir
  return core.SESS.frames, core.SESS.mask_seq, core.SESS.workdir


def _do_relight(ae_percentile, ambient, steps, seed):
  frames, mask_seq, workdir = _session()
  if mask_seq is None:
    raise gr.Error("Render the control mask first.")
  if core.PIPELINE is None:
    raise gr.Error("Model pipeline not loaded (startup failed?).")
  p = float(np.clip(ae_percentile, 0.5, 0.999999))
  t0 = time.time()
  video = core.PIPELINE.relight(
      frames, mask_seq, ambient_scale=ambient,
      ae_scale=float(-np.log10(p)), sampling_steps=steps, seed=seed)
  dur = time.time() - t0
  out = ((video.clamp(-1, 1).permute(1, 2, 3, 0).float().cpu().numpy() + 1.0)
         / 2.0 * 255.0).astype(np.uint8)
  out_path = os.path.join(workdir, f"relit_seed{seed}_{int(time.time())}.mp4")
  core._write_mp4(out_path, list(out))
  in_path = os.path.join(workdir, "input.mp4")
  if not os.path.exists(in_path):
    core._write_mp4(in_path, frames)
  return out_path, in_path, dur


def studio_relight(steps, ae, ambient_ui, seed):
  seed = int(seed)
  if seed < 0:
    seed = random.randint(0, 2**31 - 1)
  # UI shows -1..1 (0 = unchanged); the model uses 0..1 (0.5 = neutral)
  ambient = (float(ambient_ui) + 1.0) / 2.0
  out_path, in_path, dur = _do_relight(ae, ambient, int(steps), seed)
  msg = f"Done in {dur:.0f}s (seed={seed}, steps={int(steps)}). Saved: {out_path}"
  return stepper_html(3, "Done — tweak lights & rerun!"), out_path, in_path, msg


def studio_relight5(steps, ae, ambient_ui, progress=None):
  if progress is None:
    progress = gr.Progress()
  ambient = (float(ambient_ui) + 1.0) / 2.0
  seeds = [random.randint(0, 2**31 - 1) for _ in range(5)]
  outs, total = [], 0.0
  for i, seed in enumerate(seeds):
    progress((i, 5), desc=f"Run {i + 1}/5 (seed {seed})")
    out_path, _, dur = _do_relight(ae, ambient, int(steps), seed)
    outs.append(gr.update(value=out_path, label=f"Seed {seed}"))
    total += dur
  msg = (f"5 runs in {total:.0f}s (steps={int(steps)}). "
         f"Seeds: {', '.join(map(str, seeds))}")
  return [stepper_html(3, "5 variations done — compare!")] + outs + [msg]


# ------------------------------ example scenes -------------------------------
def _example_thumbs():
  pairs = core.find_example_pairs()
  names, thumbs = [], []
  for name in sorted(pairs.keys()):
    vid_path, _ = pairs[name]
    cap = cv2.VideoCapture(vid_path)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
      continue
    names.append(name)
    thumbs.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), name))
  return names, thumbs


def _extra_thumbs():
  """Input-only extra scenes (RE10K / robotics / ego-centric)."""
  names, paths, thumbs = [], [], []
  if os.path.isdir(EXTRA_VIDEO_DIR):
    for f in sorted(os.listdir(EXTRA_VIDEO_DIR)):
      if not f.lower().endswith(".mp4"):
        continue
      p = os.path.join(EXTRA_VIDEO_DIR, f)
      cap = cv2.VideoCapture(p)
      ok, bgr = cap.read()
      cap.release()
      if not ok:
        continue
      name = os.path.splitext(f)[0].replace("_video", "")
      names.append(name)
      paths.append(p)
      thumbs.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), name))
  return names, paths, thumbs


def load_extra_example(resolution, evt: gr.SelectData):
  paths = STUDIO.get("extra_paths", [])
  if evt.index >= len(paths):
    raise gr.Error("Unknown scene.")
  path = paths[evt.index]
  ups = on_upload(path, resolution)
  return [gr.update(value=path)] + list(ups)


def load_example(resolution, evt: gr.SelectData):
  names = STUDIO.get("example_names", [])
  name = names[evt.index] if evt.index < len(names) else None
  if not name:
    raise gr.Error("Unknown example.")
  out = core.palette_prepare(name, None, None, resolution)
  info, mask_preview, first = out[0], out[1], out[2]
  if not core.PSESS.detected:
    raise gr.Error(f"No palette lights found in example '{name}'.")
  STUDIO["mode"] = "palette"
  STUDIO["keys"] = list(core.PSESS.detected)
  labels = [f"**Light '{s}'** — {core.SPEC_NAMES.get(s, s)} paint"
            for s in STUDIO["keys"]]
  row_ups, label_ups = _row_updates_for_keys(labels)
  summary = (f"Example **{name}** loaded: {len(STUDIO['keys'])} light(s) "
             f"detected ({', '.join(STUDIO['keys'])}). Configure them below.")
  return ([stepper_html(2, "Example loaded — set the lights"),
           gr.update(visible=True), summary, first, mask_preview]
          + row_ups + label_ups)


# ---------------------------------- the UI -----------------------------------
def build_studio_ui():
  names, thumbs = _example_thumbs()
  STUDIO["example_names"] = names
  ex_names, ex_paths, ex_thumbs = _extra_thumbs()
  STUDIO["extra_names"] = ex_names
  STUDIO["extra_paths"] = ex_paths

  with gr.Blocks(title="GR3EN Relighting Studio", css=CSS,
                 theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# GR3EN Relighting Studio")
    gr.Markdown(
        "**GR3EN** (SIGGRAPH 2026) is a generative relighting model: give it "
        "a video, tell it which light sources to turn on, off, or recolor, "
        "and it re-renders the whole video under the new lighting. "
        "[Project page](https://gr3en-relight.github.io)\n\n"
        "**How to use**\n"
        "1. **Upload** a video — or click an example scene at the bottom of "
        "the page.\n"
        "2. **Click each light source** in the frame; SAM 2 segments it "
        "automatically. Use *New light* to add the next one.\n"
        "3. **Set each light**: pick a color, choose On / Off / No change, "
        "adjust intensity, then render the control mask.\n"
        "4. **Relight** — one run, or ×5 for five random-seed variations. "
        "A run takes under 30 seconds on an RTX 6000 Ada.\n\n"
        "*Paper scenes come with ready-made light masks and jump straight to "
        "step 3.*\n\n"
        "This is not an official Google product."
    )
    with gr.Column(elem_id="stepper-box"):
      stepper = gr.HTML(stepper_html(0, "Upload a video to begin"))

    # ---- step 1: upload ----
    with gr.Group():
      video_in = gr.Video(label="1 · Upload your video", sources=["upload"],
                          height=260)
      res_radio = gr.Radio(
          list(core.RESOLUTIONS.keys()), value=core.DEFAULT_RES,
          label="Working resolution — lower is recommended for speed")
      analysis_md = gr.Markdown(visible=False)
      with gr.Row():
        sample_radio = gr.Radio([], label="Frame sampling", visible=False,
                                scale=4)
        continue_btn = gr.Button("Continue →", variant="primary",
                                 visible=False, scale=1)

    # ---- step 2: SAM2 light selection ----
    with gr.Column(visible=False) as mask_section:
      gr.Markdown("### 2 · Click each light source")
      with gr.Row():
        with gr.Column(scale=3):
          canvas = gr.Image(label="Click on a light (SAM2 segments it live)",
                            type="numpy", interactive=True, height=400)
          frame_slider = gr.Slider(0, core.N_FRAMES - 1, value=0, step=1,
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
      lights_done_btn = gr.Button("✓ All lights selected — continue",
                                  variant="primary")

    # ---- step 3: light configuration ----
    with gr.Column(visible=False) as config_section:
      gr.Markdown("### 3 · Configure the lights")
      config_summary = gr.Markdown()
      with gr.Row(visible=False) as example_preview_row:
        ex_first = gr.Image(label="First frame", interactive=False)
        ex_mask_preview = gr.Video(label="Palette mask preview")
      cfg_rows, cfg_labels, cfg_settings = [], [], []
      for i in range(MAX_CFG):
        with gr.Row(visible=False) as row:
          lab = gr.Markdown(f"**Light {i}**")
          col = gr.ColorPicker(value="#ffffff", label="Color", scale=1)
          state = gr.Radio(["On", "Off", "No change"], value="On",
                           label="State", scale=2)
          inten = gr.Slider(0.0, 5.0, value=5.0, step=0.05, scale=2,
                            label="Intensity (0 = off · 5 = max)")
        cfg_rows.append(row)
        cfg_labels.append(lab)
        cfg_settings += [col, state, inten]
      render_btn = gr.Button("✓ Lights set — render control mask",
                             variant="primary")
      ctrl_preview = gr.Video(label="Control mask (what the model sees)")
      render_info = gr.Markdown()

    # ---- step 4: relight ----
    with gr.Column(visible=False) as relight_section:
      gr.Markdown("### 4 · Relight")
      with gr.Row():
        steps_slider = gr.Slider(10, 75, value=10, step=1, scale=3,
                                 label="Denoising steps (10 = fast preview, "
                                       "50 = best quality)")
        relight_btn = gr.Button("Relight", variant="primary", scale=1)
        relight5_btn = gr.Button("Relight ×5 (random seeds)", scale=1)
      with gr.Row():
        ambient_slider = gr.Slider(
            -1.0, 1.0, value=0.0, step=0.05,
            label="External lighting scale (-1 = unchanged · 0 = off · "
                  "1 = brighter)")
        ae_slider = gr.Slider(
            0.90, 0.999, value=0.99, step=0.001,
            label="Auto-exposure percentile p — do not adjust unless necessary")
      with gr.Accordion("Advanced (seed)", open=False):
        seed_box = gr.Number(label="Seed (-1 = random)", value=-1,
                             precision=0)
      with gr.Row():
        in_video = gr.Video(label="Input (conformed)")
        out_video = gr.Video(label="Relit output")
      relight_info = gr.Markdown()
      with gr.Row():
        multi_videos = [gr.Video(label=f"Seed run {i + 1}", scale=1)
                        for i in range(5)]
      multi_info = gr.Markdown()

    # ---- examples (always at the bottom) ----
    gr.Markdown("---\n### Assets used in the paper (Eyeful dataset)\n"
                "Video + palette mask included — click one to jump straight "
                "to light configuration.")
    gallery = gr.Gallery(value=thumbs, columns=4, height=190,
                         label="Paper assets", allow_preview=False)
    gr.Markdown("### More scenes — input only (RE10K · robotics · ego-centric)\n"
                "No mask included: clicking loads the video and you select "
                "the lights yourself with SAM2.")
    extra_gallery = gr.Gallery(value=ex_thumbs, columns=4, height=190,
                               label="Extra scenes", allow_preview=False)

    # ------------------------------- wiring -------------------------------
    # .upload fires only on user uploads, so programmatically loading an
    # extra-scene video into video_in does not re-trigger the analysis.
    video_in.upload(
        on_upload, [video_in, res_radio],
        [stepper, analysis_md, sample_radio, continue_btn,
         mask_section, canvas, frame_slider, lights_md])
    continue_btn.click(
        on_continue, [video_in, sample_radio, res_radio],
        [stepper, mask_section, canvas, frame_slider, lights_md])

    canvas.select(core.add_click, [frame_slider, light_id, point_type],
                  [canvas, lights_md])
    frame_slider.change(core.show_frame, [frame_slider], [canvas])
    new_light_btn.click(core.new_light, None, [light_id, lights_md])
    reset_light_btn.click(core.reset_light, [light_id], [canvas, lights_md])
    clear_btn.click(core.clear_all, None, [canvas, lights_md])

    lights_done_btn.click(
        confirm_lights, None,
        [stepper, canvas, config_section, config_summary]
        + cfg_rows + cfg_labels)

    render_btn.click(render_mask, cfg_settings,
                     [stepper, ctrl_preview, render_info, relight_section])

    relight_btn.click(studio_relight,
                      [steps_slider, ae_slider, ambient_slider, seed_box],
                      [stepper, out_video, in_video, relight_info])
    relight5_btn.click(studio_relight5,
                       [steps_slider, ae_slider, ambient_slider],
                       [stepper] + multi_videos + [multi_info])

    gallery.select(
        load_example, [res_radio],
        [stepper, config_section, config_summary, ex_first, ex_mask_preview]
        + cfg_rows + cfg_labels)
    gallery.select(lambda: gr.update(visible=True), None,
                   [example_preview_row])
    extra_gallery.select(
        load_extra_example, [res_radio],
        [video_in, stepper, analysis_md, sample_radio, continue_btn,
         mask_section, canvas, frame_slider, lights_md])

  return demo


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", type=int, default=7862)
  args = parser.parse_args()

  os.makedirs(core.SESSIONS_ROOT, exist_ok=True)
  assert torch.cuda.is_available(), "CUDA GPU required"
  print(f"[studio] GPU: {torch.cuda.get_device_name(0)}", flush=True)

  core.PIPELINE = core.RelightPipeline(device_id=0)

  ui = build_studio_ui()
  ui.queue()
  print(f"[studio] launching Gradio on 0.0.0.0:{args.port} "
        f"(node: {os.uname().nodename})", flush=True)
  # example/extra-scene videos live under the assets root, outside cwd/temp —
  # gradio refuses to serve them unless explicitly allowed
  allowed = [core.ASSETS_ROOT]
  try:
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=False,
              allowed_paths=allowed)
  except OSError:
    print(f"[studio] port {args.port} busy; picking a free port...", flush=True)
    ui.launch(server_name="0.0.0.0", server_port=None, share=False,
              allowed_paths=allowed)


if __name__ == "__main__":
  main()
