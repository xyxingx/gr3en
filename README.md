# GR3EN: Generative Relighting for 3D Environments
GR3EN is a relighting model that finetunes [Wan2.2](https://github.com/Wan-Video/Wan2.2)
for 3D-aware video generation with controllable lighting. Given input images
and light control masks, GR3EN generates relit video sequences.

## Installation

```bash
git clone https://github.com/google-deepmind/gr3en.git
cd gr3en
pip install -r requirements.txt
```

## Model Weights

Download the pretrained weights and place them in the following locations:

1. **Wan2.2 base model**: Download from [Wan2.2 HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-T2V-5B) and place in `./checkpoints/wan2.2-ti2v-5b/`
2. **GR3EN weights**: Download `gr3en_weights.pt` and place in `./checkpoints/gr3en_weights.pt`
3. **Prompt embeddings**: Download `prompt_embed.pt` and `null_prompt_embed.pt` and place in `./checkpoints/wan2.2-ti2v-5b/`

Your directory layout should look like:

```
gr3en/
├── checkpoints/
│   ├── wan2.2-ti2v-5b/
│   │   ├── prompt_embed.pt
│   │   ├── null_prompt_embed.pt
│   │   └── ... (Wan2.2 model files)
│   └── gr3en_weights.pt
├── inference/
│   └── ...
└── requirements.txt
```

## Usage

### Single-node inference

```bash
PYTHONPATH=inference torchrun --nproc_per_node=8 inference/fsdp.py \
    --model_configs_string="$(cat inference/configs/eyeful_seat.yaml)" \
    --workdir=./output \
    --enable_flash=True
```

### Configuration

Edit YAML configs in `inference/configs/` to control:

- `test_root`: Path to input data directory
- `mask_intensity`: Light source intensities (dict mapping spec IDs to values in [0.5, 1.0])
- `light_color`: RGB color per light source (dict mapping spec IDs to [R, G, B])
- `ambient_scale`: Ambient lighting scale factor
- `resume_from_checkpoint`: Path to GR3EN model checkpoint
- `start_idx`: Starting frame index (-1 for random)
- `frame_step`: Stride between sampled frames

See `inference/configs/config_fun.yaml` for a detailed example with comments.

## Citing this work

If you use GR3EN in your research, please cite:

```bibtex
@article{xing2026gr3en,
  title={GR3EN: Generative Relighting for 3D Environments},
  author={Xing, Xiaoyan and Henzler, Philipp and Hur, Junhwa and Li, Runze and Barron, Jonathan T and Srinivasan, Pratul P and Verbin, Dor},
  journal={arXiv preprint arXiv:2601.16272},
  year={2026}
}
```

## Licensing & Disclaimer

Copyright 2026 Google LLC
All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0
All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode
Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.

