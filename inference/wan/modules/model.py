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
import math

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops.einops import rearrange
import torch
import torch.nn as nn
from wan.modules.attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
  # preprocess
  assert dim % 2 == 0
  half = dim // 2
  position = position.type(torch.float32)

  # calculation
  sinusoid = torch.outer(
      position, torch.pow(10000, -torch.arange(half).to(position).div(half))
  )
  x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
  return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
  assert dim % 2 == 0
  freqs = torch.outer(
      torch.arange(max_seq_len),
      1.0
      / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
  )
  freqs = torch.polar(torch.ones_like(freqs), freqs)
  return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
  out_dtype = x.dtype
  n, c = x.size(2), x.size(3) // 2

  # split freqs
  freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

  # loop over samples
  output = []
  for i, (f, h, w) in enumerate(grid_sizes.tolist()):
    seq_len = f * h * w

    # precompute multipliers
    x_i = torch.view_as_complex(
        x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2).contiguous()
    )
    freqs_i = torch.cat(
        [
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    # apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    x_i = torch.cat([x_i, x[i, seq_len:]])

    # append to collection
    output.append(x_i)
  return torch.stack(output).to(out_dtype)


class WanRMSNorm(nn.Module):

  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    r"""Args:

    x(Tensor): Shape [B, L, C]
    """
    return self._norm(x.float()).type_as(x) * self.weight

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

  def __init__(self, dim, eps=1e-6, elementwise_affine=False):
    super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

  def forward(self, x):
    r"""Args:

    x(Tensor): Shape [B, L, C]
    """
    # return super().forward(x.float()).type_as(x)
    return super().forward(x)


class WanSelfAttention(nn.Module):

  def __init__(
      self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6
  ):
    assert dim % num_heads == 0
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.eps = eps

    # layers
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.o = nn.Linear(dim, dim)
    self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

  def forward(self, x, seq_lens, grid_sizes, freqs):
    r"""Args:

    x(Tensor): Shape [B, L, num_heads, C / num_heads]
    seq_lens(Tensor): Shape [B]
    grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H,
    W)
    freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
      q = self.norm_q(self.q(x)).view(b, s, n, d)
      k = self.norm_k(self.k(x)).view(b, s, n, d)
      v = self.v(x).view(b, s, n, d)
      return q, k, v

    x = x.to(torch.bfloat16)
    q, k, v = qkv_fn(x)

    x = flash_attention(
        q=rope_apply(q, grid_sizes, freqs),
        k=rope_apply(k, grid_sizes, freqs),
        v=v,
        k_lens=seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class WanCrossAttention(WanSelfAttention):

  def forward(self, x, context, context_lens):
    r"""Args:

    x(Tensor): Shape [B, L1, C]
    context(Tensor): Shape [B, L2, C]
    context_lens(Tensor): Shape [B]
    """
    b, n, d = x.size(0), self.num_heads, self.head_dim

    # compute query, key, value
    q = self.norm_q(self.q(x)).view(b, -1, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)

    # compute attention
    x = flash_attention(q, k, v, k_lens=context_lens)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class WanAttentionBlock(nn.Module):

  def __init__(
      self,
      dim,
      ffn_dim,
      num_heads,
      window_size=(-1, -1),
      qk_norm=True,
      cross_attn_norm=False,
      eps=1e-6,
  ):
    super().__init__()
    self.dim = dim
    self.ffn_dim = ffn_dim
    self.num_heads = num_heads
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.cross_attn_norm = cross_attn_norm
    self.eps = eps

    # layers
    self.norm1 = WanLayerNorm(dim, eps)
    self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
    self.norm3 = (
        WanLayerNorm(dim, eps, elementwise_affine=True)
        if cross_attn_norm
        else nn.Identity()
    )
    self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
    self.norm2 = WanLayerNorm(dim, eps)
    self.ffn = nn.Sequential(
        nn.Linear(dim, ffn_dim),
        nn.GELU(approximate='tanh'),
        nn.Linear(ffn_dim, dim),
    )

    # modulation
    self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

  def forward(
      self,
      x,
      e,
      seq_lens,
      grid_sizes,
      freqs,
      context,
      context_lens,
  ):
    r"""Args:

    x(Tensor): Shape [B, L, C]
    e(Tensor): Shape [B, L1, 6, C]
    seq_lens(Tensor): Shape [B], length of each sequence in batch
    grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H,
    W)
    freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

    # self-attention
    y = self.self_attn(
        self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
        seq_lens,
        grid_sizes,
        freqs,
    )
    # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    x = x + y * e[2].squeeze(2)

    def cross_attn_ffn(x, context, context_lens, e):
      tgt_lens = x.shape[1] // 3
      x_tgt = x[:, :tgt_lens, :]
      x_tgt = x_tgt + self.cross_attn(self.norm3(x_tgt), context, context_lens)
      x = torch.cat([x_tgt, x[:, tgt_lens:, :]], dim=1)
      y = self.ffn(
          (self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(
              torch.bfloat16
          )
      )
      with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[5].squeeze(2)
      return x

    x = x.to(torch.bfloat16)
    x = cross_attn_ffn(x, context, context_lens, e)
    return x.to(torch.bfloat16)


class Head(nn.Module):

  def __init__(self, dim, out_dim, patch_size, eps=1e-6):
    super().__init__()
    self.dim = dim
    self.out_dim = out_dim
    self.patch_size = patch_size
    self.eps = eps

    # layers
    out_dim = math.prod(patch_size) * out_dim
    self.norm = WanLayerNorm(dim, eps)
    self.head = nn.Linear(dim, out_dim)

    # modulation
    self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

  def forward(self, x, e):
    r"""Args:

    x(Tensor): Shape [B, L1, C]
    e(Tensor): Shape [B, L1, C]
    """
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
      x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
    return x


class WanModel(ModelMixin, ConfigMixin):
  r"""Wan diffusion backbone supporting both text-to-video and image-to-video."""

  ignore_for_config = [
      'patch_size',
      'cross_attn_norm',
      'qk_norm',
      'text_dim',
      'window_size',
  ]
  _no_split_modules = ['WanAttentionBlock']

  @register_to_config
  def __init__(
      self,
      model_type='t2v',
      patch_size=(1, 2, 2),
      text_len=512,
      in_dim=16,
      dim=2048,
      ffn_dim=8192,
      freq_dim=256,
      text_dim=4096,
      out_dim=16,
      num_heads=16,
      num_layers=32,
      window_size=(-1, -1),
      qk_norm=True,
      cross_attn_norm=True,
      eps=1e-6,
  ):
    r"""Initialize the diffusion model backbone.

    Args:
        model_type (`str`, *optional*, defaults to 't2v'): Model variant - 't2v'
          (text-to-video) or 'i2v' (image-to-video)
        patch_size (`tuple`, *optional*, defaults to (1, 2, 2)): 3D patch
          dimensions for video embedding (t_patch, h_patch, w_patch)
        text_len (`int`, *optional*, defaults to 512): Fixed length for text
          embeddings
        in_dim (`int`, *optional*, defaults to 16): Input video channels (C_in)
        dim (`int`, *optional*, defaults to 2048): Hidden dimension of the
          transformer
        ffn_dim (`int`, *optional*, defaults to 8192): Intermediate dimension in
          feed-forward network
        freq_dim (`int`, *optional*, defaults to 256): Dimension for sinusoidal
          time embeddings
        text_dim (`int`, *optional*, defaults to 4096): Input dimension for text
          embeddings
        out_dim (`int`, *optional*, defaults to 16): Output video channels
          (C_out)
        num_heads (`int`, *optional*, defaults to 16): Number of attention heads
        num_layers (`int`, *optional*, defaults to 32): Number of transformer
          blocks
        window_size (`tuple`, *optional*, defaults to (-1, -1)): Window size for
          local attention (-1 indicates global attention)
        qk_norm (`bool`, *optional*, defaults to True): Enable query/key
          normalization
        cross_attn_norm (`bool`, *optional*, defaults to False): Enable
          cross-attention normalization
        eps (`float`, *optional*, defaults to 1e-6): Epsilon value for
          normalization layers
    """

    super().__init__()

    assert model_type in ['t2v', 'i2v', 'ti2v']
    self.model_type = model_type

    self.patch_size = patch_size
    self.text_len = text_len
    self.in_dim = in_dim
    self.dim = dim
    self.ffn_dim = ffn_dim
    self.freq_dim = freq_dim
    self.text_dim = text_dim
    self.out_dim = out_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.cross_attn_norm = cross_attn_norm
    self.eps = eps

    # embeddings
    self.patch_embedding = nn.Conv3d(
        in_dim, dim, kernel_size=patch_size, stride=patch_size
    )
    self.text_embedding = nn.Sequential(
        nn.Linear(text_dim, dim),
        nn.GELU(approximate='tanh'),
        nn.Linear(dim, dim),
    )
    self.role_embedding = nn.Embedding(2, self.dim)
    try:
      if self.role_embedding.weight.device.type == 'meta':
        # Pick a real device; CPU is fine (FSDP will move/shard later)
        self.role_embedding.to_empty(device=torch.device('cpu'))
        nn.init.zeros_(self.role_embedding.weight)
    except AttributeError:
      pass

    self.time_embedding = nn.Sequential(
        nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
    )
    self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    # blocks
    self.blocks = nn.ModuleList([
        WanAttentionBlock(
            dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
        )
        for _ in range(num_layers)
    ])

    # head
    self.head = Head(dim, out_dim, patch_size, eps)

    # buffers (don't use register_buffer otherwise dtype will be changed in to())
    assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
    d = dim // num_heads
    self.freqs = torch.cat(
        [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ],
        dim=1,
    )

    # initialize weights
    self.init_weights()

  # def forward(self, x, t, context, seq_len, y=None, is_training=False, **pose_kwargs):
  #     device = self.patch_embedding.weight.device
  #     if self.freqs.device != device:
  #         self.freqs = self.freqs.to(device)

  #     ret_list = False
  #     if isinstance(x, list):
  #         ret_list = True
  #         x = x[0][None]
  #         if y is not None and isinstance(y, list):
  #             y = y[0][None]
  #         context = context[0][None]

  #     # --- frame counts BEFORE concat ---
  #     F_x_raw = x.shape[2]
  #     F_y_raw = y.shape[2] if y is not None else 0
  #     t_patch = self.patch_size[0]
  #     assert F_x_raw % t_patch == 0 and (F_y_raw % t_patch == 0 if y is not None else True)

  #     # --- patch-embed x and y separately; then concat along frames ---
  #     x_p = self.patch_embedding(x)                          # [B, D, F_x_p, H_p, W_p]
  #     if y is not None:
  #         y_p = self.patch_embedding(y)                      # [B, D, F_y_p, H_p, W_p]
  #         xy_p = torch.cat([x_p, y_p], dim=2)
  #         F_x_p, F_y_p = x_p.shape[2], y_p.shape[2]
  #     else:
  #         xy_p = x_p
  #         F_x_p, F_y_p = x_p.shape[2], 0

  #     # --- tokens & grid ---
  #     grid_sizes = torch.tensor(xy_p.shape[2:], dtype=torch.long, device=xy_p.device)[None]  # [1,3]=(F_p,H_p,W_p)
  #     one_frame_len = int(grid_sizes[0,1] * grid_sizes[0,2])
  #     x = xy_p.flatten(2).transpose(1, 2)                     # [B, L_total, D]
  #     B, L_total, D = x.shape
  #     F_total_p = int(grid_sizes[0,0])
  #     assert F_total_p == F_x_p + F_y_p and L_total == F_total_p * one_frame_len

  #     # --- token indices for target vs condition ---
  #     one = torch.arange(one_frame_len, device=x.device)
  #     tgt_frames  = torch.arange(0, F_x_p, device=x.device)
  #     tgt_idx     = (tgt_frames[:,None]*one_frame_len + one[None]).reshape(-1)
  #     if F_y_p > 0:
  #         cond_frames = torch.arange(F_x_p, F_x_p + F_y_p, device=x.device)
  #         cond_idx    = (cond_frames[:,None]*one_frame_len + one[None]).reshape(-1)

  #     # --- seq_lens as [B] ---
  #     seq_lens = torch.tensor([L_total] * B, dtype=torch.long, device=x.device)

  #     # --- diffusion time embeddings for L_total ---
  #     if t.dim() == 0:
  #         t = t[None].expand(B, L_total)
  #     elif t.dim() == 1:
  #         assert t.size(0) == B
  #         t = t.expand(B, L_total)
  #     else:
  #         if t.shape[1] != L_total:
  #             t = t[:, :1].expand(B, L_total)

  #     with torch.amp.autocast('cuda', dtype=torch.float32):
  #         e  = self.time_embedding(
  #             sinusoidal_embedding_1d(self.freq_dim, t.flatten()).unflatten(0, (B, L_total)).float()
  #         )                                                   # [B, L, C]
  #         e0 = self.time_projection(e).unflatten(2, (6, self.dim))  # [B, L, 6, C]

  #     # --- use existing role_embedding (0=cond, 1=target) ---
  #     role_emb = getattr(self, "role_embedding", None)
  #     if role_emb is not None:
  #         role_ids = torch.zeros(L_total, dtype=torch.long, device=x.device)
  #         role_ids[tgt_idx] = 1
  #         role_ids = role_ids[None].expand(B, -1)            # [B, L]

  #         x = x + role_emb(role_ids)

  #     # --- freeze condition tokens via e0 gates (context-only cond) ---
  #     if F_y_p > 0:
  #         cond_mask = torch.zeros(B, L_total, dtype=torch.bool, device=x.device)
  #         cond_mask[:, cond_idx] = True
  #         keep = (~cond_mask).float().unsqueeze(-1)          # [B, L, 1]
  #         # zero post-residual gates for cond tokens
  #         e0[:,:,2,:] = e0[:,:,2,:] * keep
  #         e0[:,:,5,:] = e0[:,:,5,:] * keep
  #         # (optional) neutralize pre-mods on cond tokens
  #         e0[:,:,0,:] = e0[:,:,0,:] * keep
  #         e0[:,:,1,:] = e0[:,:,1,:] * keep
  #         e0[:,:,3,:] = e0[:,:,3,:] * keep
  #         e0[:,:,4,:] = e0[:,:,4,:] * keep

  #     # --- text context ---
  #     context_lens = None
  #     text_lens = context.shape[1]
  #     text_padding = torch.zeros_like(context)[:, :1].repeat(1, self.text_len - text_lens, 1)
  #     context = torch.cat([context, text_padding], dim=1)
  #     context = self.text_embedding(context)

  #     # --- transformer ---
  #     kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
  #                 context=context, context_lens=context_lens)
  #     for block in self.blocks:
  #         if not is_training:
  #             x = block(x, **kwargs)
  #         else:
  #             x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False, **kwargs)

  #     # --- decode only target tokens ---
  #     x_tgt = x[:, tgt_idx, :]
  #     e_tgt = e[:, tgt_idx, :]
  #     x_tgt = self.head(x_tgt, e_tgt)

  #     grid_sizes_final = grid_sizes.clone()
  #     grid_sizes_final[:, 0] = F_x_p
  #     x = self.unpatchify(x_tgt, grid_sizes_final.squeeze())

  #     if ret_list:
  #         return [u.float() for u in x]
  #     return x

  def forward(
      self, x, t, context, seq_len, y=None, is_training=False, **regr_kwargs
  ):
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
      self.freqs = self.freqs.to(device)

    ret_list = False
    if isinstance(x, list):
      ret_list = True
      x = x[0][None]
      if y is not None and isinstance(y, list):
        y = y[0][None]
      context = context[0][None]

    # ---- capture original frame counts BEFORE concat ----
    F_x_raw = x.shape[2]  # frames in target video
    F_y_raw = y.shape[2] if y is not None else 0  # frames in condition video

    # ---- concat along frame dim if conditioning ----
    xy = (
        torch.cat([x, y], dim=2) if y is not None else x
    )  # [B, C, F_x+F_y, H, W]

    # ---- patchify ----
    x = self.patch_embedding(
        xy
    )  # [B, D, F_p, H_p, W_p] where F_p = (F_x+F_y)/t_patch
    grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long, device=x.device)[
        None
    ]  # [1,3]=(F_p,H_p,W_p)
    x = x.flatten(2).transpose(1, 2)  # [B, L_total, D]

    B, L_total, D = x.shape
    t_patch = self.patch_size[0]
    assert F_x_raw % t_patch == 0 and (
        F_y_raw % t_patch == 0 if y is not None else True
    ), 'Temporal dim must be divisible by t_patch'
    F_x_p = F_x_raw // t_patch
    F_y_p = F_y_raw // t_patch
    F_total_p = int(grid_sizes[0, 0])
    assert (
        F_total_p == F_x_p + F_y_p
    ), 'Frame accounting mismatch after patching'

    one_frame_len = int(grid_sizes[0, 1] * grid_sizes[0, 2])
    assert L_total == F_total_p * one_frame_len

    # ---- role / segment embedding: 0 = cond, 1 = target ----
    one = torch.arange(one_frame_len, device=x.device)
    tgt_frames = torch.arange(0, F_x_p, device=x.device)
    cond_frames = torch.arange(F_x_p, F_x_p + F_y_p, device=x.device)
    tgt_idx = (tgt_frames[:, None] * one_frame_len + one[None]).reshape(-1)
    cond_idx = (cond_frames[:, None] * one_frame_len + one[None]).reshape(-1)

    role_ids = torch.zeros(L_total, dtype=torch.long, device=x.device)
    role_ids[tgt_idx] = 1
    role_ids = role_ids[None].expand(B, -1)  # [B, L]
    x = x + self.role_embedding(role_ids)

    # ---- seq_lens must be [B] ----
    seq_lens = torch.tensor([L_total] * B, dtype=torch.long, device=x.device)

    # ---- diffusion time embeddings length must equal L_total ----
    if t.dim() == 0:
      t = t[None].expand(B, L_total)
    elif t.dim() == 1:
      assert t.size(0) == B
      t = t.expand(B, L_total)
    else:
      if t.shape[1] != L_total:
        t = t[:, :1].expand(B, L_total)
    t = t.to(dtype=torch.bfloat16)
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        .unflatten(0, (B, L_total))
        .to(dtype=torch.bfloat16)
    )  # [B, L, C]
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))  # [B, L, 6, C]

    # ---- text context (as in your code) ----
    ambient_embeds = regr_kwargs['ambient_embeds']
    ambient_embeds = self.ambient_embedding(ambient_embeds)
    # context = torch.cat([context, ambient_embeds], dim=1)
    text_lens = context.shape[1]
    text_padding = torch.zeros_like(context)[:, :1].repeat(
        1, self.text_len - text_lens, 1
    )
    context = torch.cat([context, text_padding], dim=1)
    context = self.text_embedding(context)
    context = torch.cat([context, ambient_embeds], dim=1)
    if regr_kwargs['ae_embeds'] is not None:
      ae_embeds = regr_kwargs['ae_embeds']
      ae_embeds = self.ae_embedding(ae_embeds)
      context = torch.cat([context, ae_embeds], dim=1)
    context_lens = None

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    for block in self.blocks:
      if not is_training:
        x = block(x, **kwargs)
      else:
        x = torch.utils.checkpoint.checkpoint(
            block, x, use_reentrant=False, **kwargs
        )

    # ---- decode only target tokens (recommended) ----
    x_tgt = x[:, tgt_idx, :]
    e_tgt = e[:, tgt_idx, :]
    x_tgt = self.head(x_tgt, e_tgt)

    grid_sizes_final = grid_sizes.clone()
    grid_sizes_final[:, 0] = F_x_p  # only target frames in output
    x = self.unpatchify(x_tgt, grid_sizes_final.squeeze())

    if ret_list:
      return [u.float() for u in x]
    return x

  def original_forward(
      self,
      x,
      t,
      context,
      seq_len,
      y=None,
  ):
    r"""Forward pass through the diffusion model

    Args:
        x (List[Tensor]): List of input video tensors, each with shape [C_in, F,
          H, W]
        t (Tensor): Diffusion timesteps tensor of shape [B]
        context (List[Tensor]): List of text embeddings each with shape [L, C]
        seq_len (`int`): Maximum sequence length for positional encoding
        y (List[Tensor], *optional*): Conditional video inputs for
          image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F,
            H / 8, W / 8]
    """

    """
        example of traced inputs during inference:
        x: list of len 1, x[0].shape: [16, 21, 64, 96]
        y[0].shape: [20, 21, 64, 96]
        t: [999]
        context[0].shape: [15, 4096]
        seqlen: 32256
        """

    if self.model_type == 'i2v':
      assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
      self.freqs = self.freqs.to(device)

    if y is not None:
      x = [
          torch.cat([u, v], dim=0) for u, v in zip(x, y)
      ]  # e.g. x[0].shape now = [36, 21, 64, 96]

    # embeddings
    x = [
        self.patch_embedding(u.unsqueeze(0)) for u in x
    ]  # patch embedding: conv3d 36->5120 (doesn't flatten, changes channel and halves spatial dims->5120,21,32,48)
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
    )
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])  # pads to the max seq len of 32256

    # here x.shape = [1, 32256, 5120] (not list anymore)
    # grid_sizes shape: 1,3; tensor([[21, 32, 48]])

    # time embeddings
    if t.dim() == 1:
      t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      bt = t.size(0)
      t = t.flatten()
      e = self.time_embedding(
          sinusoidal_embedding_1d(self.freq_dim, t)
          .unflatten(0, (bt, seq_len))
          .float()
      )
      e0 = self.time_projection(e).unflatten(2, (6, self.dim))
      assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # e0 shape: 1, 32256, 6, 5120

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )

    # context shape: 1, 512, 5120 (self.text_len=512)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    for block in self.blocks:
      x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)

    return [u.float() for u in x]

  # def unpatchify(self, x, grid_sizes):
  #     r"""
  #     Reconstruct video tensors from patch embeddings.

  #     Args:
  #         x (List[Tensor]):
  #             List of patchified features, each with shape [L, C_out * prod(patch_size)]
  #         grid_sizes (Tensor):
  #             Original spatial-temporal grid dimensions before patching,
  #                 shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

  #     Returns:
  #         List[Tensor]:
  #             Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
  #     """

  #     c = self.out_dim
  #     out = []
  #     for u, v in zip(x, grid_sizes.tolist()):
  #         u = u[:math.prod(v)].view(*v, *self.patch_size, c)
  #         u = torch.einsum('fhwpqrc->cfphqwr', u)
  #         u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
  #         out.append(u)
  #     return out

  def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
    grid_size = grid_size.squeeze()
    return rearrange(
        x,
        'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
        f=grid_size[0],
        h=grid_size[1],
        w=grid_size[2],
        x=self.patch_size[0],
        y=self.patch_size[1],
        z=self.patch_size[2],
    )

  def init_weights(self):
    r"""Initialize model parameters using Xavier initialization."""

    # basic init
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

    # init embeddings
    nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
    for m in self.text_embedding.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
    for m in self.time_embedding.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)

    # init output layer
    nn.init.zeros_(self.head.head.weight)

  def add_REGR_modules(self, ae_scale=None):
    """Adds experimental layers dynamically after model initialization."""
    self.ambient_embedding = nn.Sequential(
        nn.Linear(24, 1024),
        nn.Tanh(),
        nn.Linear(1024, self.text_dim),
        nn.Tanh(),
        nn.Linear(self.text_dim, self.dim),
    )
    # self.patch_embedding_con1= nn.Conv3d(
    #     self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
    # self.patch_embedding_con2= nn.Conv3d(
    #     self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
    ae_input_dim = 24 if ae_scale is not None else 16
    self.ae_embedding = nn.Sequential(
        nn.Linear(ae_input_dim, 1024),
        nn.Tanh(),
        nn.Linear(1024, self.text_dim),
        nn.Tanh(),
        nn.Linear(self.text_dim, self.dim),
    )

    # for layer in [self.patch_embedding_con1, self.patch_embedding_con2]:
    #     nn.init.zeros_(layer.weight)
    #     if layer.bias is not None:
    #         nn.init.zeros_(layer.bias)

    for layer in [self.ambient_embedding, self.ae_embedding]:
      nn.init.zeros_(layer[-1].weight)
      if layer[-1].bias is not None:
        nn.init.zeros_(layer[-1].bias)

    self._experimental_layers_added = True

    # If model is not on meta device, move new layers to current device
    device = self.patch_embedding.weight.device
    if device.type != 'meta':
      self.ambient_embedding.to(device)
      self.ae_embedding.to(device)
    print(f'Experimental layers added to WanModel, running on device: {device}')
