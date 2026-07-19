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
import numpy as np
from PIL import Image
import os
from typing import Optional, Literal
import matplotlib
import logging

def save_image_paths_to_files(image_paths, path_name, output_dir="./saved_paths"):
    """
    Save image paths from dataset to various file formats for easy reading later.
    
    Args:
        image_paths: List of image paths to save
        path_name: Name for the output file
        output_dir: Directory to save the files
    """
    import os
    import json
    import pickle
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Option 1: Save as JSON file
    json_file = os.path.join(output_dir, f"{path_name}.json")
    with open(json_file, 'w') as f:
        json.dump(image_paths, f)
    print(f"Saved {json_file}, seq len: {len(image_paths)}")

def save_video_as_gif(
    video_tensor: torch.Tensor, 
    output_path: str = 'output.gif', 
    fps: int = 12, 
    batch_idx: int = 0,
    duration: Optional[int] = None,
    input_range: Literal["01", "-11"] = "-11"
) -> None:
    """
    Save a video tensor as a GIF file.
    
    Args:
        video_tensor (torch.Tensor): Video tensor with shape (B, N, 3, H, W)
        output_path (str): Path to save the GIF file
        fps (int): Frames per second for the GIF (default: 12)
        batch_idx (int): Which batch to save (default: 0)
        duration (Optional[int]): Duration per frame in milliseconds. 
                                If None, calculated from fps.
        input_range (Literal["0_1", "-1_1"]): Input value range. 
                                            "0_1" for [0,1], "-1_1" for [-1,1] (default: "0_1")
    
    Raises:
        ValueError: If tensor shape is incorrect or values are out of range
        IndexError: If batch_idx is out of bounds
    """
    # Validate input tensor
    
    if video_tensor.dim() == 4:
        video_tensor = video_tensor[None]

    if video_tensor.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B, N, 3, H, W), got {video_tensor.dim()}D")
    
    B, N, C, H, W = video_tensor.shape
    if C != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {C}")
    
    if batch_idx >= B:
        raise IndexError(f"batch_idx {batch_idx} out of bounds for batch size {B}")
    
    # Validate input range parameter
    if input_range not in ["01", "-11"]:
        raise ValueError(f"input_range must be '0_1' or '-1_1', got '{input_range}'")
    
    # Ensure tensor is on CPU and detached
    video_tensor = video_tensor.detach().cpu()
    
    # Check and normalize value range
    if input_range == "01":
        expected_min, expected_max = 0.0, 1.0
        if video_tensor.min() < -0.1 or video_tensor.max() > 1.1:
            print(f"Warning: Tensor values outside [0,1] range. Min: {video_tensor.min():.3f}, Max: {video_tensor.max():.3f}")
        # Clamp to [0, 1] and keep as is
        video_tensor = torch.clamp(video_tensor, 0, 1)
        normalized_tensor = video_tensor
    else:  # input_range == "-1_1"
        expected_min, expected_max = -1.0, 1.0
        if video_tensor.min() < -1.1 or video_tensor.max() > 1.1:
            print(f"Warning: Tensor values outside [-1,1] range. Min: {video_tensor.min():.3f}, Max: {video_tensor.max():.3f}")
        # Clamp to [-1, 1] and normalize to [0, 1]
        video_tensor = torch.clamp(video_tensor, -1, 1)
        normalized_tensor = (video_tensor + 1.0) / 2.0  # Convert [-1,1] to [0,1]
    
    # Select batch
    video = normalized_tensor[batch_idx]  # Shape: (N, 3, H, W)
    
    # Convert to numpy and scale to [0, 255]
    video_np = (video.numpy() * 255).astype(np.uint8)
    
    # Rearrange dimensions from (N, 3, H, W) to (N, H, W, 3)
    video_np = video_np.transpose(0, 2, 3, 1)
    
    # Create PIL images
    frames = []
    for frame in video_np:
        frames.append(Image.fromarray(frame))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate duration
    if duration is None:
        duration = 1000 // fps  # Convert fps to milliseconds per frame
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"GIF saved to: {output_path} ({N} frames, {fps} fps, input_range: {input_range})")


def save_batch_videos_as_gifs(
    video_tensor: torch.Tensor,
    output_dir: str,
    prefix: str = "video",
    fps: int = 8,
    input_range: Literal["0_1", "-1_1"] = "0_1"
) -> None:
    """
    Save all videos in a batch as separate GIF files.
    
    Args:
        video_tensor (torch.Tensor): Video tensor with shape (B, N, 3, H, W)
        output_dir (str): Directory to save GIF files
        prefix (str): Prefix for GIF filenames (default: "video")
        fps (int): Frames per second for the GIFs (default: 8)
        input_range (Literal["0_1", "-1_1"]): Input value range. 
                                            "0_1" for [0,1], "-1_1" for [-1,1] (default: "0_1")
    """
    B = video_tensor.shape[0]
    
    for batch_idx in range(B):
        output_path = os.path.join(output_dir, f"{prefix}_{batch_idx:03d}.gif")
        save_video_as_gif(video_tensor, output_path, fps=fps, batch_idx=batch_idx, input_range=input_range)



def save_images_as_grid(imgs, fixed_height=256, spacing=5, max_per_row=5):
    """
    Save a grid of images with a maximum number of images per row.

    :param imgs: List of NumPy images
    :param fixed_height: Fixed height for each image in the grid
    :param spacing: Space between images in pixels
    :param max_per_row: Maximum number of images per row
    """
    row_widths = []
    row_images = []
    current_row = []

    from PIL import Image
    # Process images and organize them into rows
    for np_img in imgs:
        img = Image.fromarray(np_img)
        aspect_ratio = img.width / img.height
        new_width = int(fixed_height * aspect_ratio)
        resized_img = img.resize((new_width, fixed_height))

        if len(current_row) < max_per_row:
            current_row.append(resized_img)
        else:
            row_widths.append(sum(img.width for img in current_row) + spacing * (len(current_row) - 1))
            row_images.append(current_row)
            current_row = [resized_img]

    # Add last row
    if current_row:
        row_widths.append(sum(img.width for img in current_row) + spacing * (len(current_row) - 1))
        row_images.append(current_row)

    total_width = max(row_widths)
    total_height = fixed_height * len(row_images) + spacing * (len(row_images) - 1)

    # Create a new blank image with a white background
    grid_img = Image.new('RGB', (total_width, total_height), color='white')

    # Paste each resized image into the grid
    y_offset = 0
    for row in row_images:
        x_offset = 0
        for img in row:
            grid_img.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
        y_offset += fixed_height + spacing

    # Return the grid image
    return grid_img

def torch_batch_to_np_arr(batch, assume_neg1_pos1=False):
    '''
    Convert a torch batch of images B,3,H,W to list of np images with shape H,W,3
    Args:
        batch: torch tensor of shape B,3,H,W
        assume_neg1_pos1: if True, assumes input is in [-1,1] range and uses 127.5 * x + 128 conversion
                         if False, normalizes input to [0,1] based on min/max values
    '''
    np_imgs = []

    if assume_neg1_pos1:
        batch = torch.clamp(127.5 * batch + 128.0, 0, 255).float().cpu().numpy().transpose(0,2,3,1)
    else:
        batch = (batch - batch.min()) / (batch.max() - batch.min() + 1e-8)
        batch = (255.0 * batch).float().cpu().numpy().transpose(0,2,3,1)

    for i in range(len(batch)):
        np_imgs.append(batch[i].astype(dtype=np.uint8))

    return np_imgs

def depth_to_np_arr(depth):
    # torch batch of B,H,W

    if isinstance(depth, list):
        depth = torch.stack(depth)
    depth = depth.detach().cpu().float()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) 
    depth = depth.numpy()
    cmap = matplotlib.colormaps.get_cmap('inferno')

    np_depth = []
    for i in range(len(depth)):
        d = depth[i]
        if d.min() == d.max():
            # logging.info(f"Depth min and max are the same for {i}")
            d = np.zeros_like(d)
        image = cmap(d)[:, :, :3] * 255
        np_depth.append(image.astype(np.uint8))

    return np_depth


def save_gifs_as_grid(video_frames, gt_frames, pred_frames, output_path, fixed_height=256, spacing=5, duration=110):
    """
    Create a GIF with two or three columns: video, (optional) ground truth, and predictions.
    Each frame will show the corresponding images side by side.

    Args:
        video_frames: List of NumPy arrays for the video frames
        gt_frames: List of NumPy arrays for the ground truth depth, or None
        pred_frames: List of NumPy arrays for the predicted depth
        fixed_height: Fixed height for each image in pixels
        spacing: Space between columns in pixels
    
    Returns:
        tuple: (PIL Image object containing the animated GIF grid, 
               numpy array of shape (T, C, H, W) containing concatenated frames)
    """
    from PIL import Image
    
    frames = []
    concat_frames = []  # Will store the numpy arrays
    n_frames = len(video_frames)
    assert len(pred_frames) == n_frames, "Video and prediction frames must have same length"
    if gt_frames is not None:
        assert len(gt_frames) == n_frames, "Ground truth frames must have same length"
    
    for i in range(n_frames):
        # Convert and resize each frame
        video_img = Image.fromarray(video_frames[i])
        pred_img = Image.fromarray(pred_frames[i])
        
        # Create list of images to process
        frame_images = [video_img]
        if gt_frames is not None:
            gt_img = Image.fromarray(gt_frames[i])
            frame_images.append(gt_img)
        frame_images.append(pred_img)
        
        # Maintain aspect ratio while resizing
        resized_images = []
        for img in frame_images:
            aspect_ratio = img.width / img.height
            new_width = int(fixed_height * aspect_ratio)
            resized = img.resize((new_width, fixed_height), Image.Resampling.LANCZOS)
            resized_images.append(resized)
        
        # Calculate total width needed
        total_width = sum(img.width for img in resized_images) + spacing * (len(resized_images) - 1)
        # Create a new frame with white background
        frame = Image.new('RGB', (total_width, fixed_height), color='white')
        
        # Paste images with spacing
        x_offset = 0
        for img in resized_images:
            frame.paste(img, (x_offset, 0))
            x_offset += img.width + spacing
            
        frames.append(frame)
        
        # Create concatenated numpy array for this timestep
        np_images = [np.array(img) for img in resized_images]
        # Convert to shape (C, H, W)
        np_images = [img.transpose(2, 0, 1) for img in np_images]
        # Concatenate along width dimension
        concat_frame = np.concatenate(np_images, axis=2)  # Concatenate along W dimension
        concat_frames.append(concat_frame)
    
    # Create animated GIF
    first_frame = frames[0]
    first_frame.save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    # Stack all concatenated frames along time dimension
    concat_frames = np.stack(concat_frames, axis=0)  # Shape: (T, C, H, W)
    
    np_frames = [np.array(frame) for frame in frames]
    grid_img = save_images_as_grid(np_frames, fixed_height=fixed_height, spacing=spacing, max_per_row=1)
    
    return {'stacked_frames': concat_frames, 'grid_img': grid_img}



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# --------------------------------------------------------------------------- #
def quat_to_rotmat(q):
    """
    Convert quaternions -> 3×3 rotation matrices.

    Parameters
    ----------
    q : (N, 4) array_like
        Quaternion per row.  **Assumed order = (w, x, y, z)**.
        If yours is (x, y, z, w) just swap the columns first:
        `q = q[:, [3, 0, 1, 2]]`.

    Returns
    -------
    R : (N, 3, 3) ndarray
    """
    w, x, y, z = q.T         # shape (N,)
    N = w.size

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty((N, 3, 3), dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - wz)
    R[:, 0, 2] = 2*(xz + wy)

    R[:, 1, 0] = 2*(xy + wz)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - wx)

    R[:, 2, 0] = 2*(xz - wy)
    R[:, 2, 1] = 2*(yz + wx)
    R[:, 2, 2] = 1 - 2*(xx + yy)
    return R
# --------------------------------------------------------------------------- #

class CoordinateMap:
    """
    Builds a 2-D “world map” from (quaternions, translations).
    Z is ignored (camera height is fixed ≈ 0).
    """
    def __init__(self, poses, padding=0.5):
        """
        Parameters
        ----------
        poses : tuple(np.ndarray, np.ndarray)
            (quats, trans) where
              • quats  – shape (N, 4)
              • trans  – shape (N, 3)
        padding : float
            Margin (world units) added around the outermost points.
        """
        quats, trans = poses
        assert quats.shape[0] == trans.shape[0], "Quat / trans length mismatch"

        self.R = quat_to_rotmat(quats)           # (N, 3, 3)
        self.t = trans                           # (N, 3)

        self.xy = trans[:, :2]                   # drop fixed-Z
        xmin, ymin = self.xy.min(0) - padding
        xmax, ymax = self.xy.max(0) + padding
        self.extent = (xmin, xmax, ymin, ymax)

    # --------------------------------------------------------------------- #
    def plot_trajectory(self, start, end, *,
                        show_all=True, figsize=(6, 6),
                        title=None, arrow_every=1, savepath=None):
        """
        Identical signature & behaviour to previous version.
        """
        s, e = int(start), int(end)
        if s < 0 or e >= len(self.t) or s > e:
            raise ValueError("Indices out of range or start > end")

        fig, ax = plt.subplots(figsize=figsize)
        x0, x1, y0, y1 = self.extent
        ax.set_xlim(x0, x1);  ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.grid(True, linestyle=":", linewidth=0.5)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title(title or f"Trajectory {s} → {e}")

        if show_all:
            ax.scatter(self.xy[:, 0], self.xy[:, 1],
                       s=12, c="lightgray", label="all poses")

        traj_xy = self.xy[s:e+1]
        ax.plot(traj_xy[:, 0], traj_xy[:, 1],
                linewidth=2, c="tab:blue", label="selected path")

        # orientation arrows
        for k, idx in enumerate(range(s, e+1)):
            if k % arrow_every:            # skip unless multiple matches
                continue
            x, y = self.t[idx, :2]
            forward = self.R[idx][:, 2]    # camera +Z axis in world coords
            dx, dy = forward[0], forward[1]
            scale = 0.2 * max(x1 - x0, y1 - y0)
            ax.add_patch(
                FancyArrowPatch((x, y), (x + dx*scale, y + dy*scale),
                                arrowstyle="-|>", mutation_scale=12,
                                color="tab:orange")
            )

        ax.scatter(*traj_xy[0], s=80, c="green", zorder=5, label="start")
        ax.scatter(*traj_xy[-1], s=80, c="red", zorder=5, label="end")
        ax.legend()

        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()