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
from functools import partial
import gc

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._fully_shard._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fully_shard import fully_shard
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def wrap_with_fsdp2(model, cpu_offload=False):

  fsdp_kwargs = {
      "mp_policy": MixedPrecisionPolicy(
          param_dtype=torch.bfloat16,
          reduce_dtype=torch.bfloat16,
          # default is True, but there are a few variables in Wan explicitly set to float32
          # need to make this False to keep those at float32
          cast_forward_inputs=True,
      ),
  }

  if cpu_offload:
    fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

  # dit
  if getattr(model, "blocks", None) is not None:
    for block in model.blocks:
      fully_shard(block, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

  # controlnet
  elif getattr(model, "model", None) is not None:
    for block in model.model.blocks:
      fully_shard(block, **fsdp_kwargs)

    fully_shard(model.model, **fsdp_kwargs)

  # vggt
  elif getattr(model, "aggregator", None) is not None:
    for block in model.aggregator.frame_blocks:
      fully_shard(block, **fsdp_kwargs)
    for block in model.aggregator.global_blocks:
      fully_shard(block, **fsdp_kwargs)
    fully_shard(model.aggregator, **fsdp_kwargs)

  return model


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
  model = FSDP(
      module=model,
      process_group=process_group,
      sharding_strategy=sharding_strategy,
      auto_wrap_policy=partial(
          lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks
      ),
      mixed_precision=MixedPrecision(
          param_dtype=param_dtype,
          reduce_dtype=reduce_dtype,
          buffer_dtype=buffer_dtype,
      ),
      device_id=device_id,
      sync_module_states=sync_module_states,
  )
  return model


def free_model(model):
  for m in model.modules():
    if isinstance(m, FSDP):
      _free_storage(m._handle.flat_param.data)
  del model
  gc.collect()
  torch.cuda.empty_cache()


from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, set_model_state_dict, StateDictOptions


class AppState(Stateful):
  """This is a useful wrapper for checkpointing the Application State.

  Since this object is compliant with the Stateful protocol, DCP will
  automatically call state_dict/load_stat_dict as needed in the dcp.save/load
  APIs.

  Note: We take advantage of this wrapper to hande calling distributed state
  dict methods on the model
  and optimizer.
  """

  def __init__(self, model, optimizer=None):
    self.model = model
    self.optimizer = optimizer

  def state_dict(self):
    # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT

    if self.optimizer is not None:
      model_state_dict, optimizer_state_dict = get_state_dict(
          self.model, self.optimizer
      )
      return {"model": model_state_dict, "optim": optimizer_state_dict}
    else:
      model_state_dict = get_model_state_dict(self.model)
      return {"model": model_state_dict}

  def load_state_dict(self, state_dict):

    if self.optimizer is not None:
      set_state_dict(
          self.model,
          self.optimizer,
          model_state_dict=state_dict["model"],
          optim_state_dict=state_dict["optim"],
          # options=options
      )
    else:
      set_model_state_dict(
          self.model,
          model_state_dict=state_dict["model"],
          # options=options
      )
