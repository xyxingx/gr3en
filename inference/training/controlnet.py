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
import torch
from torch import nn
from einops.einops import rearrange
import torch.nn.functional as F

class ControlNet(nn.Module):
    def __init__(self, 
                use_5b_model=False,
                use_vggt_latents=False,
                vggt_condition='seq'
                ):
        super().__init__()

        # ====pose processing (to downsample the temporal dim)====
        # pose_channels = [7,8] #[7, 32, 8]
        # if use_5b_model:
        pose_channels = [7, 384, 48] # 48 is the channel dim of 5B VAE; if I choose to do seq conditioning 
        self.poses_spatial_conv1 = nn.Sequential(
            nn.Conv2d(pose_channels[0], pose_channels[1], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, pose_channels[1]),
            nn.SiLU(),
        )
        self.poses_spatial_conv2 = nn.Sequential(
            nn.Conv2d(pose_channels[1], pose_channels[2], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(16, pose_channels[2]),
            nn.SiLU(),
        )

        self.poses_temporal_conv1 = nn.Sequential(
            nn.Conv3d(pose_channels[1], pose_channels[1], kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
            nn.GroupNorm(32, pose_channels[1]),
            nn.SiLU(),
        )
        self.poses_temporal_conv2 = nn.Sequential(
            nn.Conv3d(pose_channels[2], pose_channels[2], kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
            nn.GroupNorm(16, pose_channels[2]),
            nn.SiLU(),
        )

        # cheaper version; pose_channels = [7, 64]
        # self.poses_temporal_conv1 = nn.Sequential(
        #     nn.Conv3d(pose_channels[0], pose_channels[0], kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
        #     nn.GroupNorm(1, pose_channels[0]),
        #     nn.SiLU(),
        # )
        # self.poses_spatial_conv1 = nn.Sequential(
        #     nn.Conv2d(pose_channels[0], pose_channels[1], kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(2, pose_channels[1]),
        #     nn.SiLU(),
        # )
        # self.poses_temporal_conv2 = nn.Sequential(
        #     nn.Conv3d(pose_channels[1], pose_channels[1], kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
        #     nn.GroupNorm(2, pose_channels[1]),
        #     nn.SiLU(),
        # )



        # ====pose projection====
        init_channels = 36 if not use_5b_model else 48
        pose_channels = pose_channels[-1]
        pose_proj = nn.Conv3d(
            in_channels=init_channels+pose_channels,
            out_channels=init_channels,
            kernel_size=1,  
            bias=True
        )

        with torch.no_grad():
            # weight shape: (C_out, C_in, 1, 1, 1)
            pose_proj.weight.zero_()      # start everything at 0
            pose_proj.bias.zero_()
            # Make the first C input channels an exact identity
            for c in range(init_channels):
                pose_proj.weight[c, c, 0, 0, 0] = 1.0
        self.pose_proj = pose_proj


         # ====vggt process====
        if use_vggt_latents:
            latents_channels = 256
            out_channels = 48 # 48 is the channel dim of 5B VAE; if I choose to do seq conditioning 
            self.vggt_latents_spatial_conv1 = nn.Sequential(
                nn.Conv2d(256, latents_channels, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(32, latents_channels),
                nn.SiLU(),
            )
            self.vggt_latents_spatial_conv2 = nn.Sequential(
                nn.Conv2d(latents_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(16, out_channels),
                nn.SiLU(),
            )

            # input to temporal conv should be (b, c, f, h, w)
            self.vggt_latents_temporal_conv1 = nn.Sequential(
                nn.Conv3d(latents_channels, latents_channels, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
                nn.GroupNorm(32, latents_channels),
                nn.SiLU(),
            )
            self.vggt_latents_temporal_conv2 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
                nn.GroupNorm(16, out_channels),
                nn.SiLU(),
            )


    
    def preprocess_poses(self, poses):

        b,f,c,h,w = poses.shape
        poses = rearrange(poses, 'b f c h w -> (b f) c h w')
        
        poses = self.poses_spatial_conv1(poses)
        poses = rearrange(poses, '(b f) c h w -> b c f h w', b=b)
        
        # poses = rearrange(poses, 'b f c h w -> b c f h w', b=b)
        poses = self.poses_temporal_conv1(poses)
        poses = rearrange(poses, 'b c f h w -> (b f) c h w')
        
        poses = self.poses_spatial_conv2(poses)
        poses = rearrange(poses, '(b f) c h w -> b c f h w', b=b)
        poses = self.poses_temporal_conv2(poses)

        return poses

    def preprocess_vggt_latents(self, vggt_latents, target_shape):
        '''
        latents: B, F (10-17), 256, h, w
        '''        

        b,f,c,h,w = vggt_latents.shape
        vggt_latents = rearrange(vggt_latents, 'b f c h w -> (b f) c h w')
        vggt_latents = self.vggt_latents_spatial_conv1(vggt_latents)
        vggt_latents = F.interpolate(vggt_latents, scale_factor=0.5, mode='bilinear', align_corners=False)
        vggt_latents = rearrange(vggt_latents, '(b f) c h w -> b c f h w', b=b).contiguous()
        vggt_latents = self.vggt_latents_temporal_conv1(vggt_latents)

        vggt_latents = rearrange(vggt_latents, 'b c f h w -> (b f) c h w', b=b)
        vggt_latents = self.vggt_latents_spatial_conv2(vggt_latents)
        vggt_latents = F.interpolate(vggt_latents, size=target_shape, mode='bilinear', align_corners=False)
        vggt_latents = rearrange(vggt_latents, '(b f) c h w -> b c f h w', b=b).contiguous()
        vggt_latents = self.vggt_latents_temporal_conv2(vggt_latents)

        # b c f h w; f=1
        vggt_latents = torch.mean(vggt_latents, dim=2, keepdim=True) 

        return vggt_latents
