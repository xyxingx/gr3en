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
"""Random relighting variations from a palette light-source mask.

Takes an input video and a palette light-source mask (mp4 video or a directory
of PNG frames; palette channel values in {0, 0.5, 1}), detects the light
sources, and generates N variation configs where each light's ON/OFF state and
color are randomized. Optionally submits the relight jobs.

Examples:
  # write aligned data + 5 random variation configs (no jobs submitted)
  python random_relight.py \
      --video /path/scene_rgb.mp4 --mask /path/scene_mask.mp4 \
      --out ./output/randoms/scene --num 5

  # same, and submit one SLURM relight job per variation
  python random_relight.py --video ... --mask ... --out ... --num 5 --submit
"""

import argparse
import math
import colorsys
import glob
import os
import random
import subprocess

import cv2
import numpy as np
import yaml

INFERENCE_DIR = os.path.abspath(os.path.dirname(__file__))
N_FRAMES = 81
PALETTE_SPECS = ['110', '101', '011', '111', '11h', '1h1', 'h11', 'hh1',
                 'h1h', '1hh', '001', '010', '100']

WAN_CKPT_DIR = os.path.join(INFERENCE_DIR, "checkpoints", "wan2.2-ti2v-5b")
GR3EN_WEIGHTS = os.path.join(INFERENCE_DIR, "checkpoints",
                             "gr3en_weights.pt")


def parse_spec(spec, hval=0.5):
    table = {'1': 1.0, '0': 0.0, 'h': float(hval)}
    return np.array([table[c] for c in spec], np.float32)


def load_frames(path):
    """Decode an mp4 OR read a directory of image frames -> list of BGR arrays."""
    if os.path.isdir(path):
        files = sorted(
            f for f in glob.glob(os.path.join(path, "*"))
            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        return [cv2.imread(f) for f in files]
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    return frames


def align_pair(vid, msk):
    """Same resolution (mask -> video res, nearest) + normalized-time sample."""
    vh, vw = vid[0].shape[:2]
    if msk[0].shape[:2] != (vh, vw):
        msk = [cv2.resize(m, (vw, vh), interpolation=cv2.INTER_NEAREST)
               for m in msk]
    vi = np.linspace(0, len(vid) - 1, N_FRAMES).round().astype(int)
    mi = np.linspace(0, len(msk) - 1, N_FRAMES).round().astype(int)
    return [vid[i] for i in vi], [msk[i] for i in mi]


def detect_specs(mask_frames, thr_px=300):
    """Detect palette lights (dataset matching rule) across probe frames."""
    probe = [mask_frames[t] for t in (0, N_FRAMES // 2, N_FRAMES - 1)]
    found = []
    for spec in PALETTE_SPECS:
        target = parse_spec(spec)
        peak = 0
        for m in probe:
            rgb = cv2.cvtColor(m, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            bright = rgb.max(axis=-1) >= 0.30
            match = (np.abs(rgb - target) <= 0.18).all(axis=-1) & bright
            peak = max(peak, int(match.sum()))
        if peak >= thr_px:
            found.append((spec, peak))
    return found


def random_light_settings(specs, rng, p_off):
    """Per spec: random ON/OFF + random color; returns (intensity, color) dicts."""
    intensity, color = {}, {}
    n_on = 0
    for spec in specs:
        if rng.random() < p_off:
            intensity[spec] = 0.5  # off
        else:
            n_on += 1
            intensity[spec] = round(  # sigmoid of a raw intensity in [1.5, 5]
                1.0 / (1.0 + math.exp(-rng.uniform(1.5, 5.0))), 4)
            hue = rng.random()
            sat = rng.uniform(0.0, 0.9)
            rgb = colorsys.hsv_to_rgb(hue, sat, 1.0)
            color[spec] = [round(c, 3) for c in rgb]
    # never generate the all-off degenerate case: force one light on
    if specs and n_on == 0:
        spec = rng.choice(specs)
        intensity[spec] = round(
            1.0 / (1.0 + math.exp(-rng.uniform(1.5, 5.0))), 4)
        color[spec] = [1.0, 1.0, 1.0]
    return intensity, color


def build_config(data_root, intensity, color):
    return {
        "task": "ti2v-5B",
        "ckpt_dir": WAN_CKPT_DIR,
        "resume_from_checkpoint": GR3EN_WEIGHTS,
        "resume_step": 0,
        "zipnerf": False,
        "start_idx": 0,
        "mask_intensity": intensity,
        "light_color": color,
        "ambient_scale": 0.5,
        "test_root": data_root + "/",
        "input_name": "frames",
        "mask_name": "mask",
        "ae_scale": 0.99,
        "frame_step": 1,
        "image_size": [352, 640],
        "max_num_frames": N_FRAMES,
        "mixed_precision": "bf16",
        "stride_min": 1, "stride_max": 1,
        "train_batch_size": 1, "dataloader_num_workers": 2,
        "gradient_accumulation_steps": 1, "learning_rate": 0.0001,
        "lr_scheduler": "cosine_with_restarts", "lr_warmup_steps": 100,
        "lr_num_cycles": 3, "max_grad_norm": 1.0,
        "enable_slicing": True, "enable_tiling": True,
        "gradient_checkpointing": True, "allow_tf32": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video (mp4 or frame dir)")
    ap.add_argument("--mask", required=True, help="palette mask (mp4 or frame dir)")
    ap.add_argument("--out", required=True, help="output root directory")
    ap.add_argument("--num", type=int, default=5, help="number of variations")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--p-off", type=float, default=0.3,
                    help="probability a light is switched OFF per variation")
    ap.add_argument("--submit", action="store_true",
                    help="sbatch one relight job per variation (1 GPU each)")
    ap.add_argument("--partition", default="delta")
    ap.add_argument("--account", default="deltausers")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    vid = load_frames(args.video)
    msk = load_frames(args.mask)
    assert vid and msk, "could not load the video and/or mask"
    print(f"video: {len(vid)} frames {vid[0].shape[:2]}, "
          f"mask: {len(msk)} frames {msk[0].shape[:2]}")
    vid, msk = align_pair(vid, msk)

    data_root = os.path.abspath(os.path.join(args.out, "data"))
    os.makedirs(os.path.join(data_root, "frames"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "mask"), exist_ok=True)
    for i, (v, m) in enumerate(zip(vid, msk)):
        cv2.imwrite(os.path.join(data_root, "frames", f"{i:05d}.png"), v)
        cv2.imwrite(os.path.join(data_root, "mask", f"{i:05d}.png"), m)
    print(f"wrote {N_FRAMES} aligned frame pairs to {data_root}")

    found = detect_specs(msk)
    specs = [s for s, _ in found]
    assert specs, "no palette lights detected in the mask!"
    print("detected lights:", ", ".join(f"'{s}' ({px}px)" for s, px in found))

    cfg_dir = os.path.join(args.out, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    commands = []
    for k in range(args.num):
        intensity, color = random_light_settings(specs, rng, args.p_off)
        cfg = build_config(data_root, intensity, color)
        cfg_path = os.path.join(cfg_dir, f"variation_{k:02d}.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        desc = ", ".join(
            f"'{s}' " + ("OFF" if intensity[s] == 0.5 else
                         f"on {intensity[s]:.2f} rgb{tuple(color[s])}")
            for s in specs)
        print(f"variation {k:02d}: {desc}")
        workdir = os.path.abspath(os.path.join(args.out, f"variation_{k:02d}"))
        commands.append(
            f"sbatch --partition={args.partition} --account={args.account} "
            f"--gres=gpu:rtx_6000:1 run_inference_slurm.sh "
            f"{os.path.abspath(cfg_path)} {workdir}")

    print("\nrelight commands:")
    for c in commands:
        print(" ", c)
    if args.submit:
        for c in commands:
            subprocess.run(c.split(), cwd=INFERENCE_DIR, check=True)
        print(f"submitted {len(commands)} jobs")
    else:
        print("(dry run: pass --submit to launch the jobs)")


if __name__ == "__main__":
    main()
