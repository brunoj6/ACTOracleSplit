import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from types import SimpleNamespace
from typing import Dict, List, Optional
import time  # Add import for timestamp
import h5py  # Add import for HDF5
from datetime import datetime  # Add import for datetime formatting
from .act_policy import ACT
import copy
from PIL import Image
from argparse import Namespace
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from collections import deque
from src.models.worldmodel_coop import WorldModelCoop
from src.models.common.rotation_transformer import RotationTransformer
from src.transforms import make_transforms, make_inv_transforms
import json


# Helper for debugging visually 
import matplotlib
matplotlib.use("WebAgg", force=True) 
matplotlib.rcParams['webagg.port'] = 9999
matplotlib.rcParams['webagg.open_in_browser'] = False

import matplotlib.pyplot as plt
import numpy as np
import os

class WMRunner:
    '''
    Loads world model for inference
    '''
    def __init__(
        self,
        config_dir: str,
        config_name: str,
        ckpt_path: str,
        device: torch.device,
        camera_names: List[str],
        ckpt_dir: Optional[str] = None,
        split_ratio: Optional[float] = 0.5, # fraction of scene camera for left arm
    ):
        self.device = device
        self.camera_names = list(camera_names)
        self.ckpt_path = ckpt_path
        # Ensure split_ratio is always a float to prevent string concatenation issues
        self.split_ratio = float(split_ratio) if split_ratio is not None else 0.5

            # --- Hydra init: accept absolute config dir safely ---
        abs_cfg_dir = os.path.abspath(config_dir)

        # If Hydra was already initialized elsewhere in this process, clear it
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=abs_cfg_dir, job_name="wm", version_base="1.1")
        cfg = compose(config_name=config_name)
        self.cfg: DictConfig = cfg  # set before using it

        encoder = instantiate(self.cfg.encoder)
        decoder = instantiate(self.cfg.decoder, emb_dim=encoder.embed_dim)

        if encoder.__class__.__name__ == "VisionTransformer":
            num_patches = encoder.patch_embed.num_patches
        else:
            num_patches = (cfg.shared.img_resize_shape // encoder.patch_size) ** 2

        # ---- single-camera split handling ----
        orig_obs_keys = list(cfg.task.dataset.train_observation_keys)
        use_surveyor_split = len(orig_obs_keys) == 1
        if use_surveyor_split:
            base_key = orig_obs_keys[0]
            effective_obs_keys = [f"{base_key}_left", f"{base_key}_right"]
        else:
            base_key = None
            effective_obs_keys = orig_obs_keys
        effective_act_keys = cfg.task.dataset.train_action_keys
        predictor = instantiate(
            cfg.dynamics_predictor,
            num_cams = len(effective_obs_keys),
            num_patches=num_patches,
            embed_dim=encoder.embed_dim,
            num_heads=encoder.num_heads,
            history_length=cfg.training.latent_state_history_length
        ).to(device)
        self.wm = WorldModelCoop(
            cfg=self.cfg,
            device=device,
            encoder=encoder,
            decoder=decoder,
            dynamics_predictor=predictor,
            mode='inference',
            ablation=None,
            encoder_frozen=True
        )
        
        self.wm.load_dynamics_predictor_weights(self.ckpt_path)
        self.wm.load_decoder_weights(self.ckpt_path)

        # Store previous observation 
        self.prev_obs = None

        encoder.eval()
        decoder.eval()
        predictor.eval()


    def predict(self, next_observation):

        # First observation will run guess with empty action
        if self.prev_obs is None:
            self.prev_obs = next_observation
            self.last_action = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.])

        observation = self.prev_obs
        last_action = self.last_action

        # VALIDATION: Log input shapes and types

        # Unnormalize 14D actions first, then convert to 20D for world model
        #unnormalized_14d = self.unnormalize_actions(last_action)  # Unnormalize 14D actions using dataset stats
        actions = torch.tensor(self.convert_actions(last_action)).to(self.device).to(self.wm.dtype)  # Convert to 20D

        # Transform images
        images = observation["observation"]
        
        # VALIDATION: Check available camera keys

        # Use BGR to match WM training 
        if "left_camera" in images:
            left_bgr = images["left_camera"]["rgb"][..., ::-1]
            left_wrist = self.wm.transform(Image.fromarray(left_bgr))
        elif "cam_left_wrist" in images:
            left_bgr = images["cam_left_wrist"]["rgb"][..., ::-1]
            left_wrist = self.wm.transform(Image.fromarray(left_bgr))
        else:
            raise KeyError("Left camera not found in observation")
            
        if "right_camera" in images:
            right_bgr = images["right_camera"]["rgb"][..., ::-1]
            right_wrist = self.wm.transform(Image.fromarray(right_bgr))
        elif "cam_right_wrist" in images:
            right_bgr = images["cam_right_wrist"]["rgb"][..., ::-1]
            right_wrist = self.wm.transform(Image.fromarray(right_bgr))
        else:
            raise KeyError("Right camera not found in observation")
            
        if "head_camera" in images:
            head_bgr = images["head_camera"]["rgb"][..., ::-1]
            head_cam = self.wm.transform(Image.fromarray(head_bgr))
        elif "cam_high" in images:
            head_bgr = images["cam_high"]["rgb"][..., ::-1]
            head_cam = self.wm.transform(Image.fromarray(head_bgr))
        else:
            raise KeyError("Head camera not found in observation")

        # CORRECTED: Match camera order with world model training
        # World model expects: left_camera, head_camera, right_camera (based on cam_names)
        sim_cam_keys = ['left_camera', 'head_camera', 'right_camera']  # Match training order
        cam_map = {
            "left_camera": left_wrist,   # left camera first
            "head_camera": head_cam,     # head camera second  
            "right_camera": right_wrist, # right camera third
        }
        obs_seq = {k: cam_map[k].unsqueeze(0) for k in sim_cam_keys}  # (1,3,H,W) each

        # Resolve camera splits
        obs_seq = self.wm._maybe_inject_virtual_cams(obs_seq)
        obs_keys = list(self.wm.img_key_to_token_idx.keys())  # e.g. ['left_camera_wrist', 'head_camera_keft', 'right_camera_wrist', 'head_camera_right']

        target = self.wm.img_resize_shape
        cams_resized = [obs_seq[k] for k in obs_keys]
        # concat head camera halves
        cams_resized = [
            cams_resized[0],  # e.g., RoboTwin left-camera
            torch.concat((cams_resized[3], cams_resized[1]), dim=-1),  # e.g., RoboTwin head-camera
            cams_resized[2],  # e.g., RoboTwin right-camera
        ]
        observation_sequence = torch.stack(cams_resized, dim=0).to(device=self.wm.device, dtype=self.wm.dtype)  # (1,3,H,W) each -> (1,3,3,H,W)
        N, B, C, _, _ = observation_sequence.shape
        obs_resized = observation_sequence                        # already resized; keep name consistent

        # Apply dropouts
        obs_resized_left = obs_resized.clone()
        obs_resized_right = obs_resized.clone()
        actions_left = actions.clone()
        actions_right = actions.clone()

        # split index
        split_scene_cam_idx =  int(obs_resized.shape[-1] * self.split_ratio) # split index for head camera
        split_action_idx = actions.shape[-1] // 2

        # Set left arm dropouts (zero out right arm information)
        obs_resized_left[2, ...] = 0.0 # right camera 
        obs_resized_left[1, ...,  split_scene_cam_idx:] = 0.0 # left-split of scene-camera 
        actions_left[..., split_action_idx:] = 0.0 # right actions
        
        # Set right arm dropouts (zero out left arm information)
        obs_resized_right[0, ...] = 0.0 # left camera
        obs_resized_right[1, ...,  :split_scene_cam_idx] = 0.0 # right-split of scene-camera
        actions_right[..., :split_action_idx] = 0.0 # left actions

        # Encode images
        imgs_flat_left = obs_resized_left.view(N * B, C, target, target)
        imgs_flat_right = obs_resized_right.view(N * B, C, target, target)
        with torch.no_grad():
            z_flat_left = self.wm.encoder(imgs_flat_left)  # (N*B, P, E)
            z_flat_right = self.wm.encoder(imgs_flat_right)  # (N*B, P, E)

        # TODO: Confirm that left and right shapes match here
        # assert z_flat_left.shape == z_flat_right.shape, "Left and right latent shapes do not match"
        P, E = z_flat_left.shape[-2], z_flat_left.shape[-1]   # tokens per cam, embed dim (e.g., 256, 384)
    
        # Reshape
        z_all_left = z_flat_left.view(N, B, P, E)          # (N, B, P, E)
        z_all_right = z_flat_right.view(N, B, P, E)          # (N, B, P, E)
        z_concat_left = torch.cat([z_all_left[i] for i in range(N)], dim=1)    # (B, N*P, E)
        z_concat_right = torch.cat([z_all_right[i] for i in range(N)], dim=1)    # (B, N*P, E)

        # Include latent history
        z_hist_left = deque([z_concat_left] * (self.wm.latent_state_history_length - 1))
        z_hist_right = deque([z_concat_right] * (self.wm.latent_state_history_length - 1))
        z_curr_with_hist_left = torch.cat(list(z_hist_left) + [z_concat_left], dim=1)  # (B, H*N*P, E)
        z_curr_with_hist_right = torch.cat(list(z_hist_right) + [z_concat_right], dim=1)  # (B, H*N*P, E)

        z_pred_left = self.wm.dynamics_predictor(z_curr_with_hist_left, actions_left[:,None])   # (T, N*P, E)
        z_pred_right = self.wm.dynamics_predictor(z_curr_with_hist_right, actions_right[:,None])   # (T, N*P, E)

        mu_left, logvar_left = torch.chunk(z_pred_left, 2, dim=-1)
        mu_right, logvar_right = torch.chunk(z_pred_right, 2, dim=-1)
        
        # CORRECTED: Use dynamic token slicing based on actual camera count
        # Calculate tokens per camera dynamically
        tokens_per_cam = P  # P is the number of tokens per camera
        total_tokens = N * P  # Total tokens across all cameras
        

        # For inference, use the first timestep prediction (earliest prediction)
        z_next_left = mu_left[:, :total_tokens, :]  # All camera tokens for left arm
        z_next_right = mu_right[:, :total_tokens, :]  # All camera tokens for right arm

        # Decode images 
        recons_left, recons_right = {}, {}
        decoder_split_cam_idx = int(self.wm.cfg.shared.img_decoder_shape * self.split_ratio) # split index for decoder output

        with torch.no_grad():
            # Build left dict
            left_pred_left = self.wm.decoder(z_next_left[:, :P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            left_pred_inv = self.wm.inv_transform(left_pred_left[0].detach().cpu())
            recons_left['left_camera'] = left_pred_inv.permute(1,2,0).numpy()

            head_swapped_left = self.wm.decoder(z_next_left[:, P:2*P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            head_swapped_left_inv = self.wm.inv_transform(head_swapped_left[0].detach().cpu())

            recons_left['head_camera_right'] = head_swapped_left_inv[..., decoder_split_cam_idx:].permute(1,2,0).numpy() # (B, 3, H, W/2)
            recons_left['head_camera_left'] = head_swapped_left_inv[..., :decoder_split_cam_idx].permute(1,2,0).numpy() # (B, 3, H, W/2)

            left_pred_right = self.wm.decoder(z_next_left[:, 2*P:, :].unsqueeze(1))[0]   # (B, 3, H, W)
            left_pred_right_inv = self.wm.inv_transform(left_pred_right[0].detach().cpu())
            recons_left['right_camera'] = left_pred_right_inv.permute(1,2,0).numpy()

            # Build right dict - CORRECTED CAMERA ASSIGNMENTS
            right_pred_left = self.wm.decoder(z_next_right[:, :P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            right_pred_left_inv = self.wm.inv_transform(right_pred_left[0].detach().cpu())
            recons_right['left_camera'] = right_pred_left_inv.permute(1,2,0).numpy()

            head_swapped_right = self.wm.decoder(z_next_right[:, P:2*P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            head_swapped_right_inv = self.wm.inv_transform(head_swapped_right[0].detach().cpu())

            recons_right['head_camera_right'] = head_swapped_right_inv[..., decoder_split_cam_idx:].permute(1,2,0).numpy() # (B, 3, H, W/2)
            recons_right['head_camera_left'] = head_swapped_right_inv[..., :decoder_split_cam_idx].permute(1,2,0).numpy() # (B, 3, H, W/2)

            right_pred_right = self.wm.decoder(z_next_right[:, 2*P:, :].unsqueeze(1))[0]   # (B, 3, H, W)
            right_pred_right_inv = self.wm.inv_transform(right_pred_right[0].detach().cpu())
            recons_right['right_camera'] = right_pred_right_inv.permute(1,2,0).numpy()

        # VALIDATION: Log output shapes and ranges

        # Step observation
        self.prev_obs = next_observation
        
        return recons_left, recons_right


    def grab_dropouts(self, arm_idx):
        '''
        Grab existing dropouts indices from the model given arm index ('arm_1', 'arm_2')
        '''
        obs_starts = []
        obs_ends = []
        act_starts = []
        act_ends = []
        # Store the per-arm camera slice lists for convenience
        self._arm_cam_slices = []
        
        dropout_arm_idx = arm_idx

        g = self.wm.arm_groups[dropout_arm_idx]
        cam_slices = list(g["cams"])  # list of (s,e)
        self._arm_cam_slices.append(cam_slices)
        # We still return a start/end pair for API compatibility; you won’t use these directly.
        # (Keep placeholders—some older code may expect same-length lists.)
        if len(cam_slices) > 0:
            obs_starts.append(cam_slices[0][0])
            obs_ends.append(cam_slices[-1][1])
        else:
            obs_starts.append(0); obs_ends.append(0)
        act_starts.append(g["act"][0]); act_ends.append(g["act"][1])

        return obs_starts, obs_ends, act_starts, act_ends
        

    def convert_actions(self, action14):
        '''
        Converts 14D joint action to 20D action
        '''
        rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
        raw_vec = np.array(action14)
        raw = raw_vec.reshape(2,7)
        pos = raw[:, :3]
        rot_aa = raw[:, 3:6]
        grip = raw[:, 6:]
        rot_6d = rotation_transformer.forward(rot_aa.reshape(-1, 3)).reshape(2, 6)
        out = np.concatenate([pos, rot_6d, grip], axis=-1).astype(np.float32)  # (T,2,10)
        return out.reshape(-1, 20)
            