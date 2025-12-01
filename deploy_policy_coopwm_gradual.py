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
from collections import deque
from src.models.worldmodel_coop import WorldModelCoop
from src.models.common.rotation_transformer import RotationTransformer
from src.transforms import make_transforms
import json


# Helper for debugging visually 
import matplotlib.pyplot as plt
import numpy as np
import os


import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])[None,None,:]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])[None,None,:]

# Global flag to enable/disable camera prediction logging
ENABLE_CAMERA_PREDICTION_LOGGING = True  # Set to False to disable logging

# Global flag to enable/disable debug logging
ENABLE_DEBUG_LOGGING = True  # Set to False to disable debug output

# Global tracking for first success/failure logging
_first_success_logged = False
_first_failure_logged = False
_current_episode_images = None  # Store images from first step of current episode

def invert_dino_transforms(img):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
    std  = np.array(IMAGENET_STD).reshape(1,1,3)
    img_out = img * std + mean
    # Clip to valid range to prevent negative values from becoming zero in uint8 conversion
    img_out = np.clip(img_out, 0, 1)
    return img_out

def save_camera_predictions(observation, recons_left, recons_right, episode_num, success, output_dir="./camera_predictions"):
    """
    Save camera predictions for logging purposes.
    
    Args:
        observation: Raw observation from environment
        recons_left: Left arm predictions from world model
        recons_right: Right arm predictions from world model  
        episode_num: Episode number for naming
        success: Whether episode was successful
        output_dir: Directory to save images
    """
    global _first_success_logged, _first_failure_logged
    
    if not ENABLE_CAMERA_PREDICTION_LOGGING:
        return
    
    # Check if we should log this episode
    should_log = False
    episode_type = ""
    
    if success and not _first_success_logged:
        should_log = True
        episode_type = "success"
        _first_success_logged = True
        print(f"📸 Logging camera predictions for first successful episode {episode_num}")
    elif not success and not _first_failure_logged:
        should_log = True
        episode_type = "failure"
        _first_failure_logged = True
        print(f"📸 Logging camera predictions for first failed episode {episode_num}")
    
    if not should_log:
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract ground truth images
    gt_head = observation["observation"]["head_camera"]["rgb"]
    gt_left = observation["observation"]["left_camera"]["rgb"] 
    gt_right = observation["observation"]["right_camera"]["rgb"]
    
    # Save all camera combinations for GRADUAL approach
    camera_combinations = {
        "left_arm_oracle": {  # Left arm gets ALL ground truth
            "head_cam": gt_head,
            "left_cam": gt_left,
            "right_cam": gt_right
        },
        "right_arm_predicted": {  # Right arm gets predicted information
            "head_cam": gt_head,  # Keep GT head cam for right arm
            "left_cam": recons_right['left_camera'],  # Use predicted left cam
            "right_cam": gt_right  # Keep GT right cam for right arm
        }
    }
    
    # Save each combination
    for arm_type, cameras in camera_combinations.items():
        for cam_name, img in cameras.items():
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img_clipped = np.clip(img, 0, 1)
                img_uint8 = (img_clipped * 255).astype(np.uint8)
            else:
                img_uint8 = img
            
            # Create filename with GRADUAL approach identifier
            filename = f"episode{episode_num}_{episode_type}_gradual_{arm_type}_{cam_name}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    
    # Save metadata
    metadata = {
        "episode_num": episode_num,
        "episode_type": episode_type,
        "approach": "gradual",
        "description": "Left arm gets oracle info, right arm gets predicted info",
        "timestamp": datetime.now().isoformat(),
        "camera_combinations": list(camera_combinations.keys()),
    }
    
    metadata_file = os.path.join(output_dir, f"episode{episode_num}_{episode_type}_gradual_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved GRADUAL camera predictions to {output_dir}")

def reset_camera_logging_state():
    """Reset the global logging state - useful for new evaluation runs"""
    global _first_success_logged, _first_failure_logged, _current_episode_images
    _first_success_logged = False
    _first_failure_logged = False
    _current_episode_images = None
    print("🔄 Reset camera prediction logging state")

def log_episode_result(episode_num, success):
    """
    Log camera predictions for episode result.
    This should be called at the end of each episode.
    """
    global _first_success_logged, _first_failure_logged, _current_episode_images
    
    if not ENABLE_CAMERA_PREDICTION_LOGGING or _current_episode_images is None:
        return
    
    # Check if we should log this episode
    should_log = False
    episode_type = ""
    
    if success and not _first_success_logged:
        should_log = True
        episode_type = "success"
        _first_success_logged = True
        print(f"📸 Logging camera predictions for first successful episode {episode_num}")
    elif not success and not _first_failure_logged:
        should_log = True
        episode_type = "failure"
        _first_failure_logged = True
        print(f"📸 Logging camera predictions for first failed episode {episode_num}")
    
    if should_log:
        observation, recons_left, recons_right = _current_episode_images
        save_camera_predictions(observation, recons_left, recons_right, episode_num, success)
    
    # Clear stored images for next episode
    _current_episode_images = None

def encode_obs_left(observation, recons):
    """
    GRADUAL APPROACH: Left arm gets ALL ground truth information (oracle)
    This tests the baseline performance with full information.
    """
    # VALIDATION: Log input shapes
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [LEFT_ORACLE] Input observation keys: {list(observation['observation'].keys())}")
        print(f"🔍 [LEFT_ORACLE] Reconstruction keys: {list(recons.keys())}")
    
    # GRADUAL APPROACH: Left arm gets ALL ground truth cameras
    # This provides the baseline performance with full oracle information
    
    # Ground truth left camera
    left_cam = cv2.resize(observation["observation"]["left_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0

    # Ground truth head camera
    head_cam = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0

    # Ground truth right camera
    right_cam = cv2.resize(observation["observation"]["right_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0

    # Use left arm state
    qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]])

    result = {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }
    
    # VALIDATION: Check qpos
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [LEFT_ORACLE] QPOS shape: {len(qpos)}, range: [{min(qpos):.3f}, {max(qpos):.3f}]")
        print(f"🔍 [LEFT_ORACLE] QPOS values: {qpos}")
        print(f"🔍 [LEFT_ORACLE] APPROACH: Oracle (all ground truth cameras)")
    
    # VALIDATION: Log output shapes and ranges
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [LEFT_ORACLE] Encoded observation shapes:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"🔍 [LEFT_ORACLE]   {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"🔍 [LEFT_ORACLE]   {key}: {type(value)} with length {len(value)}")

    return result

def encode_obs_right(observation, recons):
    """
    GRADUAL APPROACH: Right arm gets predicted information for left camera only
    This tests the policy's ability to handle missing left camera information.
    """
    # VALIDATION: Log input shapes
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [RIGHT_PREDICTED] Input observation keys: {list(observation['observation'].keys())}")
        print(f"🔍 [RIGHT_PREDICTED] Reconstruction keys: {list(recons.keys())}")
    
    # GRADUAL APPROACH: Right arm gets predicted left camera, but keeps GT head and right cameras
    # This tests robustness to missing left camera information
    
    # Ground truth right camera (unchanged)
    right_cam = cv2.resize(observation["observation"]["right_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0

    # Ground truth head camera (unchanged)
    head_cam = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0

    # EXPERIMENT: Replace left camera with world model prediction
    # This tests if the right arm can handle missing left camera information
    left_cam = cv2.resize(recons['left_camera'], (640, 480), interpolation=cv2.INTER_LINEAR)
    left_cam = np.moveaxis(left_cam, -1, 0) 
    
    # Use right arm state
    qpos = (observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])

    result = {
        "head_cam": head_cam,
        "left_cam": left_cam,  # This is now predicted, not ground truth
        "right_cam": right_cam,
        "qpos": qpos,
    }
    
    # VALIDATION: Check qpos
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [RIGHT_PREDICTED] QPOS shape: {len(qpos)}, range: [{min(qpos):.3f}, {max(qpos):.3f}]")
        print(f"🔍 [RIGHT_PREDICTED] QPOS values: {qpos}")
        print(f"🔍 [RIGHT_PREDICTED] APPROACH: Predicted left camera, GT head and right cameras")
    
    # VALIDATION: Log output shapes and ranges
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [RIGHT_PREDICTED] Encoded observation shapes:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"🔍 [RIGHT_PREDICTED]   {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"🔍 [RIGHT_PREDICTED]   {key}: {type(value)} with length {len(value)}")

    return result

def get_model(usr_args):
    # Init policies
    usr_args['ckpt_name'] = usr_args['left_ckpt_name']
    left_policy = ACT(usr_args, Namespace(**usr_args))
    usr_args['ckpt_name'] = usr_args['right_ckpt_name']
    right_policy = ACT(usr_args, Namespace(**usr_args))

    # CORRECTED: Use camera names that match the observation keys
    cam_names = ["head_camera", "right_camera", "left_camera"]

    # Init world model
    wmrunner = WMRunner(
            config_dir=usr_args.get("wm_config_dir"),
            config_name=usr_args.get("wm_config_name"),
            ckpt_path=usr_args.get("wm_ckpt_path"),
            device=usr_args.get("device"),
            camera_names=cam_names,
        )
    model = SimpleNamespace(
        left_policy=left_policy,
        right_policy=right_policy,
        wmrunner=wmrunner,
    )

    return model


def eval(TASK_ENV, model, observation):
    '''
    Observation: dict holding all camera images from simulator
        {'cam_high', 'cam_left_wrist', 'cam_right_wrist'}
    TASK_ENV: environment object
    model: policy object
    '''
    global _current_episode_images
    
    # World model prediction
    last_action = observation['joint_action']['vector']
    recons_left, recons_right = model.wmrunner.predict(observation, last_action)
    
    # Store images from first step of episode for potential logging
    if _current_episode_images is None and ENABLE_CAMERA_PREDICTION_LOGGING:
        _current_episode_images = (observation, recons_left, recons_right)
    
    # GRADUAL APPROACH: Update observations with different strategies
    # Left arm gets oracle (all GT), right arm gets predicted info
    obs_left = encode_obs_left(observation, recons_left)   # Oracle approach
    obs_right = encode_obs_right(observation, recons_right)  # Predicted approach
    
    action_left = model.left_policy.get_action(obs_left, arm_flag='left')
    action_right = model.right_policy.get_action(obs_right, arm_flag='right')
    
    # VALIDATION: Check action dimensions
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [ACTIONS] Left action shape: {action_left.shape}, Right action shape: {action_right.shape}")
        print(f"🔍 [ACTIONS] Left action range: [{action_left.min():.3f}, {action_left.max():.3f}]")
        print(f"🔍 [ACTIONS] Right action range: [{action_right.min():.3f}, {action_right.max():.3f}]")
        print(f"🔍 [ACTIONS] GRADUAL APPROACH: Left=Oracle (all GT), Right=Predicted (left cam predicted)")
    
    # CORRECTED: Handle different action dimension cases
    if action_left.shape[-1] == 14 and action_right.shape[-1] == 14:
        # Each arm returns 14D actions - use left arm's left half and right arm's right half
        actions = np.concatenate([action_left[..., :7], action_right[..., 7:]], axis=-1)
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [ACTIONS] Using 14D arm actions, concatenated shape: {actions.shape}")
    elif action_left.shape[-1] == 7 and action_right.shape[-1] == 7:
        # Each arm returns 7D actions - simple concatenation
        actions = np.concatenate([action_left, action_right], axis=-1)
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [ACTIONS] Using 7D arm actions, concatenated shape: {actions.shape}")
    else:
        # Fallback - assume simple concatenation
        actions = np.concatenate([action_left, action_right], axis=-1)
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [ACTIONS] Fallback concatenation, shape: {actions.shape}")

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()

    last_action = action # Store last applied action

    # Prepare robot input images for video recording
    # Use standard robot names for compatibility with eval_policy.py
    robot_input_images = {
        'left_robot': {  # Left arm gets oracle information
            'head_cam': obs_left['head_cam'].transpose(1, 2, 0),  # Convert from CHW to HWC
            'left_cam': obs_left['left_cam'].transpose(1, 2, 0),
            'right_cam': obs_left['right_cam'].transpose(1, 2, 0)
        },
        'right_robot': {  # Right arm gets predicted information
            'head_cam': obs_right['head_cam'].transpose(1, 2, 0),  # Convert from CHW to HWC
            'left_cam': obs_right['left_cam'].transpose(1, 2, 0),
            'right_cam': obs_right['right_cam'].transpose(1, 2, 0)
        }
    }

    return observation, last_action, robot_input_images # observation


def reset_model(model):
    # Reset temporal aggregation state if enabled
    model_left = model.left_policy
    if model_left.temporal_agg:
        model_left.all_time_actions = torch.zeros([
            model_left.max_timesteps,
            model_left.max_timesteps + model_left.num_queries,
            model_left.state_dim,
        ]).to(model_left.device)
        model_left.t = 0
        print("Reset temporal aggregation state")
    else: 
        model_left.t = 0

    model_right = model.right_policy
    if model_right.temporal_agg:
        model_right.all_time_actions = torch.zeros([
            model_right.max_timesteps,
            model_right.max_timesteps + model_right.num_queries,
            model_right.state_dim,
        ]).to(model_right.device)
        model_right.t = 0
        print("Reset temporal aggregation state")
    else:
        model_right.t = 0
    
    # Reset camera prediction logging state for new evaluation run
    reset_camera_logging_state()


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
        camera_names: List[str]
    ):
        self.device = device
        self.camera_names = list(camera_names)
        self.ckpt_path = ckpt_path
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

        encoder.eval()
        decoder.eval()
        predictor.eval()

    def predict(self, observation, last_action):
        # VALIDATION: Log input shapes and types
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Input observation keys: {list(observation['observation'].keys())}")
            print(f"🔍 [WM] Last action shape: {np.array(last_action).shape}, dtype: {type(last_action)}")

        # Convert actions (14D to 20D)
        actions = torch.tensor(self.convert_actions(last_action)).to(self.device).to(self.wm.dtype)  # (T, 20)
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Converted actions shape: {actions.shape}, dtype: {actions.dtype}")

        # Transform images
        images = observation["observation"]
        
        # VALIDATION: Check available camera keys
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Available observation keys: {list(images.keys())}")
        
        # Handle different camera naming conventions
        if "left_camera" in images:
            left_wrist = self.wm.transform(Image.fromarray(images["left_camera"]["rgb"]))
        elif "cam_left_wrist" in images:
            left_wrist = self.wm.transform(Image.fromarray(images["cam_left_wrist"]["rgb"]))
        else:
            raise KeyError("Left camera not found in observation")
            
        if "right_camera" in images:
            right_wrist = self.wm.transform(Image.fromarray(images["right_camera"]["rgb"]))
        elif "cam_right_wrist" in images:
            right_wrist = self.wm.transform(Image.fromarray(images["cam_right_wrist"]["rgb"]))
        else:
            raise KeyError("Right camera not found in observation")
            
        if "head_camera" in images:
            head_cam = self.wm.transform(Image.fromarray(images["head_camera"]["rgb"]))
        elif "cam_high" in images:
            head_cam = self.wm.transform(Image.fromarray(images["cam_high"]["rgb"]))
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
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Virtual camera keys: {obs_keys}")

        target = self.wm.img_resize_shape
        cams_resized = [obs_seq[k] for k in obs_keys]
        # concat head camera halves
        cams_resized = [
            cams_resized[0],  # e.g., RoboTwin left-camera
            torch.concat((cams_resized[1], cams_resized[3]), dim=-1),  # e.g., RoboTwin head-camera
            cams_resized[2],  # e.g., RoboTwin right-camera
        ]
        observation_sequence = torch.stack(cams_resized, dim=0).to(device=self.wm.device, dtype=self.wm.dtype)  # (1,3,H,W) each -> (1,3,3,H,W)
        N, B, C, _, _ = observation_sequence.shape
        obs_resized = observation_sequence                        # already resized; keep name consistent
        
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Observation sequence shape: {observation_sequence.shape}")
            print(f"🔍 [WM] N={N}, B={B}, C={C}, target={target}")

        # Apply dropouts
        obs_resized_left = obs_resized.clone()
        obs_resized_right = obs_resized.clone()
        actions_left = actions.clone()
        actions_right = actions.clone()

        # split index
        split_scene_cam_idx = obs_resized.shape[-1] // 2 # half width
        split_action_idx = actions.shape[-1] // 2

        # CORRECTED: Remove ::2 indexing for single-timestep evaluation
        # Set left arm dropouts (zero out right arm information)
        obs_resized_left[2, ...] = 0.0 # right camera 
        obs_resized_left[1, ...,  split_scene_cam_idx:] = 0.0 # right-split of scene-camera 
        actions_left[..., split_action_idx:] = 0.0 # right actions
        
        # Set right arm dropouts (zero out left arm information)
        obs_resized_right[0, ...] = 0.0 # left camera
        obs_resized_right[1, ...,  :split_scene_cam_idx] = 0.0 # left-split of scene-camera
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
        
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Tokens per camera: {tokens_per_cam}, Total tokens: {total_tokens}")
            print(f"🔍 [WM] Prediction shapes - mu_left: {mu_left.shape}, mu_right: {mu_right.shape}")
        
        # For inference, use the first timestep prediction (earliest prediction)
        z_next_left = mu_left[:, :total_tokens, :]  # All camera tokens for left arm
        z_next_right = mu_right[:, :total_tokens, :]  # All camera tokens for right arm
        
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Selected token shapes - z_next_left: {z_next_left.shape}, z_next_right: {z_next_right.shape}")

        # Decode images 
        recons_left, recons_right = {}, {}
        decoder_split_cam_idx = self.wm.cfg.shared.img_decoder_shape // 2 # half width post decoder
        with torch.no_grad():
            # Build left dict
            left_pred_left = self.wm.decoder(z_next_left[:, :P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            recons_left['left_camera'] = invert_dino_transforms(left_pred_left.detach().cpu().squeeze(0).permute(1,2,0).numpy()) # (H, W, 3)
            head_swapped_left = self.wm.decoder(z_next_left[:, P:2*P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            left_pred_head_right = head_swapped_left[..., :decoder_split_cam_idx] # (B, 3, H, W/2)
            recons_left['head_camera_right'] = invert_dino_transforms(left_pred_head_right.detach().cpu().squeeze(0).permute(1,2,0).numpy()) # (H, W/2, 3)
            left_pred_head_left = head_swapped_left[..., decoder_split_cam_idx:] # (B, 3, H, W/2)
            recons_left['head_camera_left'] = invert_dino_transforms(left_pred_head_left.detach().cpu().squeeze(0).permute(1,2,0).numpy()) # (H, W/2, 3)
            left_pred_right = self.wm.decoder(z_next_left[:, 2*P:, :].unsqueeze(1))[0]   # (B, 3, H, W)
            recons_left['right_camera'] = invert_dino_transforms(left_pred_right.detach().cpu().squeeze(0).permute(1,2,0).numpy()) # (H, W, 3)

            # Build right dict - CORRECTED CAMERA ASSIGNMENTS
            # Right arm predicts LEFT camera using LEFT camera tokens (first P tokens)
            right_pred_left = self.wm.decoder(z_next_right[:, :P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            recons_right['left_camera'] = invert_dino_transforms(right_pred_left.detach().cpu().squeeze(0).permute(1,2,0).numpy()) # (H, W, 3) 
            
            # Right arm predicts HEAD camera using HEAD camera tokens (P to 2*P tokens)
            head_swapped_right = self.wm.decoder(z_next_right[:, P:2*P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            right_pred_head_right = head_swapped_right[..., :decoder_split_cam_idx] # (B, 3, H, W/2)
            recons_right['head_camera_right'] = invert_dino_transforms(right_pred_head_right.detach().cpu().squeeze(0).permute(1,2,0).numpy()) # (H, W/2, 3)

            right_pred_head_left = head_swapped_right[..., decoder_split_cam_idx:] # (B, 3, H, W/2)
            recons_right['head_camera_left'] = invert_dino_transforms(right_pred_head_left.detach().cpu().squeeze(0).permute(1,2,0).numpy()) # (H, W/2, 3)
            
            # Right arm predicts RIGHT camera using RIGHT camera tokens (2*P to 3*P tokens)
            right_pred_right = self.wm.decoder(z_next_right[:, 2*P:, :].unsqueeze(1))[0]   # (B, 3, H, W)
            recons_right['right_camera'] = invert_dino_transforms(right_pred_right.detach().cpu().squeeze(0).permute(1,2,0).numpy()) # (H, W, 3)

        # VALIDATION: Log output shapes and ranges
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Reconstructions completed:")
            for arm, recons in [("left", recons_left), ("right", recons_right)]:
                for cam, img in recons.items():
                    print(f"🔍 [WM]   {arm}_{cam}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
        
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
        # We still return a start/end pair for API compatibility; you won't use these directly.
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
