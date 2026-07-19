#!/bin/bash
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
# GR3EN Relighting Studio — streamlined single-page wizard UI. ONE GPU.
#
#   sbatch run_studio_slurm.sh
#
#SBATCH --job-name=gr3en_studio
#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=gr3en_studio_%j.log
set -e

PORT=${PORT:-7862}

source /home/xxing/anaconda3/etc/profile.d/conda.sh
conda activate w_diffusers
# under sbatch, $0 is a spooled copy — use the submission directory instead
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

nvidia-smi -L
echo "node: $(hostname)"

# stage weights into node tmpfs and use the staged copies
STAGE=/tmp/gr3en_staged
export GR3EN_STAGE_CKPTS=28000
bash stage_weights.sh || echo "WARNING: staging failed; using NFS paths"
if [ -f "$STAGE/wan2.2-ti2v-5b/Wan2.2_VAE.pth" ]; then
  export WAN_CKPT_DIR="$STAGE/wan2.2-ti2v-5b"
  [ -f "$STAGE/sam2/sam2.1_hiera_large.pt" ] && export SAM2_CKPT="$STAGE/sam2/sam2.1_hiera_large.pt"
  [ -f "$STAGE/gr3en_weights_28000_full.pt" ] && export GR3EN_WEIGHTS_PT="$STAGE/gr3en_weights_28000_full.pt"
  echo "using staged weights from $STAGE"
fi

echo ""
echo "==================================================================="
echo " Tunnel from your laptop, then open http://localhost:${PORT} :"
echo "   ssh -J xxing@ivi-h1.science.uva.nl -L ${PORT}:localhost:${PORT} xxing@$(hostname)"
echo "==================================================================="
echo ""
PYTHONPATH=. python studio_app.py --port "$PORT"
