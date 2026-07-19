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
# Stage GR3EN weights into node-local tmpfs (/tmp/gr3en_staged) so jobs on this
# node load them from RAM instead of the slow NFS home.
# Safe to run repeatedly: existing files are skipped (files are immutable).
# NOTE: compute nodes are diskless — /tmp is RAM-backed; a full stage holds
# ~31 GiB of node RAM until reboot.
set -e

SELF=$(cd "$(dirname "$0")" && pwd)
SRC_WAN=$SELF/checkpoints/wan2.2-ti2v-5b
SRC_MAIN=$SELF/checkpoints/gr3en_weights.pt
SRC_ALT=$SELF/checkpoints/gr3en_weights_alt.pt
SRC_SAM2=$SELF/checkpoints/sam2/sam2.1_hiera_large.pt

STAGE=${GR3EN_STAGE:-/tmp/gr3en_staged}
mkdir -p "$STAGE/wan2.2-ti2v-5b" "$STAGE/sam2"

stage_file() {  # stage_file <src> <dst>
  if [ -f "$2" ] && [ "$(stat -c%s "$2")" = "$(stat -c%s "$1")" ]; then
    echo "staged already: $2"
  else
    echo "staging $1 -> $2 ..."
    cp "$1" "$2.part" && mv "$2.part" "$2"
  fi
}

for f in "$SRC_WAN"/*; do
  stage_file "$f" "$STAGE/wan2.2-ti2v-5b/$(basename "$f")"
done
# GR3EN_STAGE_CKPTS: which fine-tuned checkpoints to stage (main|alt|all)
CKPTS=${GR3EN_STAGE_CKPTS:-all}
case "$CKPTS" in *main*|all) stage_file "$SRC_MAIN" "$STAGE/gr3en_weights.pt" ;; esac
case "$CKPTS" in *alt*|all) [ -f "$SRC_ALT" ] && stage_file "$SRC_ALT" "$STAGE/gr3en_weights_alt.pt" ;; esac
[ -f "$SRC_SAM2" ] && stage_file "$SRC_SAM2" "$STAGE/sam2/sam2.1_hiera_large.pt"

du -sh "$STAGE"
echo "STAGING_DONE on $(hostname)"
