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



IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])[None,None,:]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])[None,None,:]

# Global flag to enable/disable camera prediction logging
ENABLE_CAMERA_PREDICTION_LOGGING = True  # Set to False to disable logging

# Global flag to enable/disable debug logging
ENABLE_DEBUG_LOGGING = False  # Set to False to disable debug output

# Global tracking for first success/failure logging
_first_success_logged = False
_first_failure_logged = False
_current_episode_images = None  # Store images from first step of current episode

def invert_dino_transforms(img):
    # TODO: replace with self.wm.inv_transform
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
    std  = np.array(IMAGENET_STD).reshape(1,1,3)
    img_out = img * std + mean
    # Clip to valid range to prevent negative values from becoming zero in uint8 conversion
    img_out = np.clip(img_out, 0, 1)
    return img_out

def _ensure_rgb(img_arr):
    """
    Ensure numpy image is RGB-ordered. Some decoders or intermediate steps
    may yield BGR; we standardize to RGB by swapping channels.
    Expects shape (H, W, 3) with values in [0,1] or [0,255].
    """
    # TODO: Remove this once it is fixed in WM training
    return  img_arr
    
    if isinstance(img_arr, np.ndarray) and img_arr.ndim == 3 and img_arr.shape[-1] == 3:
        return img_arr[..., ::-1]
    return img_arr

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
    
    # Save all camera combinations
    camera_combinations = {
        "left_arm_gt": {
            "head_cam": gt_head,
            "left_cam": gt_left,
            "right_cam": gt_right
        },
        "left_arm_pred": {
            "head_cam": gt_head,  # Use GT head cam for left arm
            "left_cam": gt_left,   # Use GT left cam for left arm
            "right_cam": recons_left['right_camera']  # Use predicted right cam
        },
        "right_arm_gt": {
            "head_cam": gt_head,
            "left_cam": gt_left,
            "right_cam": gt_right
        },
        "right_arm_pred": {
            "head_cam": gt_head,  # Use GT head cam for right arm
            "left_cam": recons_right['left_camera'],  # Use predicted left cam
            "right_cam": gt_right  # Use GT right cam for right arm
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
            
            # Create filename
            filename = f"episode{episode_num}_{episode_type}_{arm_type}_{cam_name}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    
    # Save metadata
    metadata = {
        "episode_num": episode_num,
        "episode_type": episode_type,
        "timestamp": datetime.now().isoformat(),
        "camera_combinations": list(camera_combinations.keys()),
        "description": "Camera predictions for coopwm evaluation with history"
    }
    
    metadata_file = os.path.join(output_dir, f"episode{episode_num}_{episode_type}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved camera predictions to {output_dir}")

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
    
    # VALIDATION: Log input shapes
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [LEFT] Input observation keys: {list(observation['observation'].keys())}")
        print(f"🔍 [LEFT] Reconstruction keys: {list(recons.keys())}")
    
    # Left cam is unchanged (ground truth)
    left_cam = cv2.resize(observation["observation"]["left_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0

    # CORRECTED: Create separate head camera for left arm to avoid conflicts
    # Start with ground truth head camera
    head_cam_gt = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam_gt = np.moveaxis(head_cam_gt, -1, 0) / 255.0
    
    # Create left arm's head camera: GT right half + predicted left half
    head_cam_left = head_cam_gt.copy()
    head_pred_left = cv2.resize(recons['head_camera_left'], (320, 480), interpolation=cv2.INTER_LINEAR)
    head_pred_left = np.moveaxis(head_pred_left, -1, 0) 
    head_cam_left[..., :320] = head_pred_left  # Replace left half with prediction

    # Replace right cam with predicted 
    right_cam = cv2.resize(recons['right_camera'], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = np.moveaxis(right_cam, -1, 0) 

    # Use left arm state
    qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]])

    result = {
        "head_cam": head_cam_left,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }
    
    # VALIDATION: Check qpos
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [LEFT] QPOS shape: {len(qpos)}, range: [{min(qpos):.3f}, {max(qpos):.3f}]")
        print(f"🔍 [LEFT] QPOS values: {qpos}")
    
    # VALIDATION: Log output shapes and ranges
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [LEFT] Encoded observation shapes:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"🔍 [LEFT]   {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"🔍 [LEFT]   {key}: {type(value)} with length {len(value)}")

    return result

def encode_obs_right(observation, recons):
    # VALIDATION: Log input shapes
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [RIGHT] Input observation keys: {list(observation['observation'].keys())}")
        print(f"🔍 [RIGHT] Reconstruction keys: {list(recons.keys())}")
    
    # Right cam is unchanged (ground truth)
    right_cam = cv2.resize(observation["observation"]["right_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0

    # CORRECTED: Create separate head camera for right arm to avoid conflicts
    # Start with ground truth head camera
    head_cam_gt = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam_gt = np.moveaxis(head_cam_gt, -1, 0) / 255.0
    
    # Create right arm's head camera: GT left half + predicted right half
    head_cam_right = head_cam_gt.copy()
    head_pred_right = cv2.resize(recons['head_camera_right'], (320, 480), interpolation=cv2.INTER_LINEAR)
    head_pred_right = np.moveaxis(head_pred_right, -1, 0)
    head_cam_right[..., 320:] = head_pred_right  # Replace right half with prediction

    # Replace left cam with predicted
    left_cam = cv2.resize(recons['left_camera'], (640, 480), interpolation=cv2.INTER_LINEAR)
    left_cam = np.moveaxis(left_cam, -1, 0)     
    # Use right arm state
    qpos = (observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])

    result = {
        "head_cam": head_cam_right,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }
    
    # VALIDATION: Check qpos
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [RIGHT] QPOS shape: {len(qpos)}, range: [{min(qpos):.3f}, {max(qpos):.3f}]")
        print(f"🔍 [RIGHT] QPOS values: {qpos}")
    
    # VALIDATION: Log output shapes and ranges
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [RIGHT] Encoded observation shapes:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"🔍 [RIGHT]   {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"🔍 [RIGHT]   {key}: {type(value)} with length {len(value)}")

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
    wmrunner = WMRunnerWithHistory(
            config_dir=usr_args.get("wm_config_dir"),
            config_name=usr_args.get("wm_config_name"),
            ckpt_path=usr_args.get("wm_ckpt_path"),
            device=usr_args.get("device"),
            camera_names=cam_names,
            ckpt_dir=usr_args.get("ckpt_dir"),  # Pass ckpt_dir for dataset stats
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
    
    # World model prediction with proper history handling
    recons_left, recons_right = model.wmrunner.predict(observation)
    
    # Store images from first step of episode for potential logging
    if _current_episode_images is None and ENABLE_CAMERA_PREDICTION_LOGGING:
        _current_episode_images = (observation, recons_left, recons_right)
    
    # Update observations with predictions 
    obs_left = encode_obs_left(observation, recons_left)
    obs_right = encode_obs_right(observation, recons_right)
    
    action_left = model.left_policy.get_action(obs_left, arm_flag='left')
    action_right = model.right_policy.get_action(obs_right, arm_flag='right')
    
    # VALIDATION: Check action dimensions
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [ACTIONS] Left action shape: {action_left.shape}, Right action shape: {action_right.shape}")
        print(f"🔍 [ACTIONS] Left action range: [{action_left.min():.3f}, {action_left.max():.3f}]")
        print(f"🔍 [ACTIONS] Right action range: [{action_right.min():.3f}, {action_right.max():.3f}]")
    
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

    # Prepare robot input images for video recording
    robot_input_images = {
        'left_robot': {
            'head_cam': obs_left['head_cam'].transpose(1, 2, 0),  # Convert from CHW to HWC
            'left_cam': obs_left['left_cam'].transpose(1, 2, 0),
            'right_cam': obs_left['right_cam'].transpose(1, 2, 0)
        },
        'right_robot': {
            'head_cam': obs_right['head_cam'].transpose(1, 2, 0),  # Convert from CHW to HWC
            'left_cam': obs_right['left_cam'].transpose(1, 2, 0),
            'right_cam': obs_right['right_cam'].transpose(1, 2, 0)
        }
    }

    # Feedback last computed action to world model for history
    model.wmrunner.update_action_history(actions)

    return observation, actions, robot_input_images # observation


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
    
    # Reset world model history state
    model.wmrunner.reset_history()
    
    # Reset camera prediction logging state for new evaluation run
    reset_camera_logging_state()


class WMRunnerWithHistory:
    '''
    Enhanced world model runner that properly maintains action history
    and uses both previous and current observations for better predictions.
    '''
    def __init__(
        self,
        config_dir: str,
        config_name: str,
        ckpt_path: str,
        device: torch.device,
        camera_names: List[str],
        ckpt_dir: Optional[str] = None
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

        # Initialize history tracking
        self.reset_history()

        encoder.eval()
        decoder.eval()
        predictor.eval()

    def reset_history(self):
        """Reset all history tracking for a new episode"""
        # Default action for first step (14D: [left_arm_7d, right_arm_7d])
        # Gripper actions are typically [1, 0] for open, [0, 1] for closed
        self.default_action = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.])
        
        # Action history: store last two action predictions
        self.action_history = deque([self.default_action, self.default_action], maxlen=2)
        
        # Observation history: store previous observation for history component
        self.prev_obs = None
        
        # Latent state history - CORRECTED: Match training implementation
        # History stores previous latent states, not current ones
        self.latent_history_left = None
        self.latent_history_right = None
        
        print("🔄 Reset world model history state")

    def update_action_history(self, actions):
        """Update action history with the latest action prediction"""
        # Take the last action from the sequence (most recent prediction)
        if len(actions.shape) > 1:
            latest_action = actions[-1]  # Last timestep action
        else:
            latest_action = actions
            
        self.action_history.append(latest_action)
        
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [HISTORY] Updated action history, latest action shape: {latest_action.shape}")

    def predict(self, current_observation):
        """
        Enhanced predict function that uses proper history:
        - Uses previous observation and current observation
        - Maintains action history for better temporal modeling
        - Properly initializes latent state history
        """
        
        # First observation: initialize with default action
        if self.prev_obs is None:
            self.prev_obs = current_observation
            print("🔍 [HISTORY] First observation - using default action")
            return self._predict_with_history(current_observation, self.default_action)
        
        # Use the most recent action from history
        last_action = self.action_history[-1]
        
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [HISTORY] Using action from history, shape: {last_action.shape}")
            print(f"🔍 [HISTORY] Action history length: {len(self.action_history)}")

        # Get predictions using both previous and current observations
        recons_left, recons_right = self._predict_with_history(current_observation, last_action)
        
        # Update previous observation for next step
        self.prev_obs = current_observation
        
        return recons_left, recons_right

    def _predict_with_history(self, observation, last_action):
        """
        Core prediction logic with proper history handling
        """
        # VALIDATION: Log input shapes and types
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Input observation keys: {list(observation['observation'].keys())}")
            print(f"🔍 [WM] Last action shape: {np.array(last_action).shape}, dtype: {type(last_action)}")

        # Convert 14D actions to 20D for world model
        actions = torch.tensor(self.convert_actions(last_action)).to(self.device).to(self.wm.dtype)
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Converted actions shape: {actions.shape}, dtype: {actions.dtype}")

        # Transform images
        images = observation["observation"]
        
        # VALIDATION: Check available camera keys
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Available observation keys: {list(images.keys())}")
            
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
        sim_cam_keys = ['left_camera', 'head_camera', 'right_camera']  # Match training order
        cam_map = {
            "left_camera": left_wrist,   # left camera first
            "head_camera": head_cam,     # head camera second  
            "right_camera": right_wrist, # right camera third
        }
        obs_seq = {k: cam_map[k].unsqueeze(0) for k in sim_cam_keys}  # (1,3,H,W) each

        # Resolve camera splits
        obs_seq = self.wm._maybe_inject_virtual_cams(obs_seq)
        obs_keys = list(self.wm.img_key_to_token_idx.keys())
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Virtual camera keys: {obs_keys}")

        target = self.wm.img_resize_shape
        cams_resized = [obs_seq[k] for k in obs_keys]
        # concat head camera halves
        cams_resized = [
            cams_resized[0],  # e.g., RoboTwin left-camera
            torch.concat((cams_resized[3], cams_resized[1]), dim=-1),  # e.g., RoboTwin head-camera
            cams_resized[2],  # e.g., RoboTwin right-camera
        ]
        observation_sequence = torch.stack(cams_resized, dim=0).to(device=self.wm.device, dtype=self.wm.dtype)
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

        P, E = z_flat_left.shape[-2], z_flat_left.shape[-1]   # tokens per cam, embed dim
    
        # Reshape
        z_all_left = z_flat_left.view(N, B, P, E)          # (N, B, P, E)
        z_all_right = z_flat_right.view(N, B, P, E)          # (N, B, P, E)
        z_concat_left = torch.cat([z_all_left[i] for i in range(N)], dim=1)    # (B, N*P, E)
        z_concat_right = torch.cat([z_all_right[i] for i in range(N)], dim=1)    # (B, N*P, E)

        # CORRECTED: Match training implementation exactly
        # Initialize history with copies of current state if first timestep
        if self.latent_history_left is None:
            # First timestep: initialize history with copies of current state
            self.latent_history_left = deque([z_concat_left] * (self.wm.latent_state_history_length - 1))
            self.latent_history_right = deque([z_concat_right] * (self.wm.latent_state_history_length - 1))
            if ENABLE_DEBUG_LOGGING:
                print(f"🔍 [WM] Initialized history with {self.wm.latent_state_history_length - 1} copies of current state")
        
        # Build history-augmented state: current state first, then history (matches training)
        z_curr_with_hist_left = torch.cat((z_concat_left, *self.latent_history_left), dim=-2)
        z_curr_with_hist_right = torch.cat((z_concat_right, *self.latent_history_right), dim=-2)

        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] History lengths - left: {len(self.latent_history_left)}, right: {len(self.latent_history_right)}")
            print(f"🔍 [WM] Combined history shapes - left: {z_curr_with_hist_left.shape}, right: {z_curr_with_hist_right.shape}")

        # Predict next latent states
        z_pred_left = self.wm.dynamics_predictor(z_curr_with_hist_left, actions_left[:,None])
        z_pred_right = self.wm.dynamics_predictor(z_curr_with_hist_right, actions_right[:,None])

        mu_left, logvar_left = torch.chunk(z_pred_left, 2, dim=-1)
        mu_right, logvar_right = torch.chunk(z_pred_right, 2, dim=-1)
        
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

        # CORRECTED: Update history exactly like training (pop oldest, appendleft current)
        if self.wm.latent_state_history_length > 1:
            self.latent_history_left.pop()
            self.latent_history_left.appendleft(z_concat_left)
            self.latent_history_right.pop()
            self.latent_history_right.appendleft(z_concat_right)

        # Decode images 
        recons_left, recons_right = {}, {}
        decoder_split_cam_idx = self.wm.cfg.shared.img_decoder_shape // 2 # half width post decoder
        with torch.no_grad():
            # Build left dict
            left_pred_left = self.wm.decoder(z_next_left[:, :P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            left_pred_inv = self.wm.inv_transform(left_pred_left[0].detach().cpu())
            recons_left['left_camera'] = left_pred_inv.permute(1,2,0).numpy()

            head_swapped_left = self.wm.decoder(z_next_left[:, P:2*P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            head_swapped_left_inv = self.wm.inv_transform(head_swapped_left[0].detach().cpu())
            recons_left['head_camera_right'] = head_swapped_left_inv.permute(1,2,0).numpy()

            head_swapped_left_left = head_swapped_left[..., :decoder_split_cam_idx] # (B, 3, H, W/2)
            head_swapped_left_left_inv = self.wm.inv_transform(head_swapped_left_left[0].detach().cpu())
            recons_left['head_camera_left'] = head_swapped_left_left_inv.permute(1,2,0).numpy()

            left_pred_right = self.wm.decoder(z_next_left[:, 2*P:, :].unsqueeze(1))[0]   # (B, 3, H, W)
            left_pred_right_inv = self.wm.inv_transform(left_pred_right[0].detach().cpu())
            recons_left['right_camera'] = left_pred_right_inv.permute(1,2,0).numpy()

            # Build right dict - CORRECTED CAMERA ASSIGNMENTS
            right_pred_left = self.wm.decoder(z_next_right[:, :P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            right_pred_left_inv = self.wm.inv_transform(right_pred_left[0].detach().cpu())
            recons_right['left_camera'] = right_pred_left_inv.permute(1,2,0).numpy()

            head_swapped_right = self.wm.decoder(z_next_right[:, P:2*P, :].unsqueeze(1))[0]   # (B, 3, H, W)
            head_swapped_right_inv = self.wm.inv_transform(head_swapped_right[0].detach().cpu())
            recons_right['head_camera_right'] = head_swapped_right_inv.permute(1,2,0).numpy()

            head_swapped_right_left = head_swapped_right[..., :decoder_split_cam_idx] # (B, 3, H, W/2)
            head_swapped_right_left_inv = self.wm.inv_transform(head_swapped_right_left[0].detach().cpu())
            recons_right['head_camera_left'] = head_swapped_right_left_inv.permute(1,2,0).numpy()

            right_pred_right = self.wm.decoder(z_next_right[:, 2*P:, :].unsqueeze(1))[0]   # (B, 3, H, W)
            right_pred_right_inv = self.wm.inv_transform(right_pred_right[0].detach().cpu())
            recons_right['right_camera'] = right_pred_right_inv.permute(1,2,0).numpy()

        # VALIDATION: Log output shapes and ranges
        if ENABLE_DEBUG_LOGGING:
            print(f"🔍 [WM] Reconstructions completed:")
            for arm, recons in [("left", recons_left), ("right", recons_right)]:
                for cam, img in recons.items():
                    print(f"🔍 [WM]   {arm}_{cam}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")

        return recons_left, recons_right

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
