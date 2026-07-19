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
#SBATCH --job-name=gr3en_prerelease
#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=gr3en_prerelease_%j.log
set -e

source /home/xxing/anaconda3/etc/profile.d/conda.sh
conda activate w_diffusers
# under sbatch, $0 is a spooled copy — use the submission directory instead
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

nvidia-smi -L

CONFIG=${1:-configs/demo_local.yaml}
WORKDIR=${2:-./output/demo_local}
NPROC=$(nvidia-smi -L | wc -l)

# Stage weights into node-local tmpfs (first run on a node pays the NFS read
# once; later runs load from RAM in ~1 min), then point the config at the
# staged copies.
STAGE=/tmp/gr3en_staged
bash stage_weights.sh || echo "WARNING: staging failed; using NFS paths"
REPO=$(pwd)
CFG_STR=$(cat "$CONFIG")
if [ -f "$STAGE/wan2.2-ti2v-5b/Wan2.2_VAE.pth" ]; then
  # rewrite absolute repo paths first, then bare repo-relative paths
  CFG_STR=$(echo "$CFG_STR" | sed \
    -e "s|$REPO/checkpoints/wan2.2-ti2v-5b|$STAGE/wan2.2-ti2v-5b|g" \
    -e "s|$REPO/checkpoints/gr3en_weights_28000_full.pt|$STAGE/gr3en_weights_28000_full.pt|g" \
    -e "s|$REPO/checkpoints/gr3en_weights_38400_full.pt|$STAGE/gr3en_weights_38400_full.pt|g" \
    -e "s|'checkpoints/wan2.2-ti2v-5b'|'$STAGE/wan2.2-ti2v-5b'|g" \
    -e "s|\"checkpoints/gr3en_weights_28000_full.pt\"|\"$STAGE/gr3en_weights_28000_full.pt\"|g" \
    -e "s|\"checkpoints/gr3en_weights_38400_full.pt\"|\"$STAGE/gr3en_weights_38400_full.pt\"|g")
  echo "using staged weights from $STAGE"
fi

PYTHONPATH=. torchrun --nproc_per_node="$NPROC" --standalone fsdp.py \
    --model_configs_string="$CFG_STR" \
    --workdir="$WORKDIR" \
    --enable_flash=True
