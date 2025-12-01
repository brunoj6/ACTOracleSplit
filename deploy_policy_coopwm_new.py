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
from .wmrunner import WMRunner


# Helper for debugging visually 
import matplotlib
matplotlib.use("WebAgg", force=True) 
matplotlib.rcParams['webagg.port'] = 9999
matplotlib.rcParams['webagg.open_in_browser'] = False

import matplotlib.pyplot as plt
import numpy as np
import os

# Params for splitting head camera image
HEAD_W = 640
HEAD_H = 480
# Default split ratio (can be overridden via usr_args)
DEFAULT_HEAD_SPLIT_RATIO = 0.4

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
        "description": "Camera predictions for coopwm evaluation"
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

def encode_obs_left(observation, recons, split_ratio=None):
    if split_ratio is None:
        split_ratio = DEFAULT_HEAD_SPLIT_RATIO
    
    # Calculate split indices based on ratio
    head_split_left = int(HEAD_W * split_ratio)
    
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
    head_pred_left = cv2.resize(recons['head_camera_left'], (head_split_left, 480), interpolation=cv2.INTER_LINEAR)
    head_pred_left = np.moveaxis(head_pred_left, -1, 0) 
    head_cam_left[..., :head_split_left] = head_pred_left  # Replace left half with prediction

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

def encode_obs_right(observation, recons, split_ratio=None):
    if split_ratio is None:
        split_ratio = DEFAULT_HEAD_SPLIT_RATIO
    
    # Calculate split indices based on ratio
    head_split_left = int(HEAD_W * split_ratio)
    head_split_right = HEAD_W - head_split_left
    
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
    head_pred_right = cv2.resize(recons['head_camera_right'], (head_split_right, 480), interpolation=cv2.INTER_LINEAR)
    head_pred_right = np.moveaxis(head_pred_right, -1, 0)
    head_cam_right[..., head_split_left:] = head_pred_right  # Replace right section with prediction

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

    # Get split ratio from args or use default, ensure it's a float
    split_ratio = float(usr_args.get("wm_split_ratio", DEFAULT_HEAD_SPLIT_RATIO))

    # Init world model
    wmrunner = WMRunner(
            config_dir=usr_args.get("wm_config_dir"),
            config_name=usr_args.get("wm_config_name"),
            ckpt_path=usr_args.get("wm_ckpt_path"),
            device=usr_args.get("device"),
            camera_names=cam_names,
            ckpt_dir=usr_args.get("ckpt_dir"),  # Pass ckpt_dir for dataset stats
            split_ratio=split_ratio,  # Pass split ratio from args
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
    
    # 
    # Apply dataset transforms 
    # new_frames = {}
    # for cam in observation['observation']:
    #     frames = observation['observation'][cam]['rgb']
    #     new_frames[cam] = [
    #         model.wm.transforms(
    #             Image.fromarray(fm)
    #         )
    #         for idx, fm in enumerate(frames)
    #     ]

    # World model prediction
    #print(f'Last Action: {last_action}')
    #print(f'State: {last_action}')
    recons_left, recons_right = model.wmrunner.predict(observation)
    
    # Store images from first step of episode for potential logging
    if _current_episode_images is None and ENABLE_CAMERA_PREDICTION_LOGGING:
        _current_episode_images = (observation, recons_left, recons_right)
    
    # Get split ratio from model's wmrunner
    split_ratio = model.wmrunner.split_ratio
    
    # Update observations with predictions 
    obs_left = encode_obs_left(observation, recons_left, split_ratio=split_ratio)
    obs_right = encode_obs_right(observation, recons_right, split_ratio=split_ratio)
    #show_imgs(obs_left, obs_right, out_dir="./debug_imgs", step_idx=42)
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

    # Feedback last computed action
    model.wmrunner.last_action = actions

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
    
    # Reset camera prediction logging state for new evaluation run
    reset_camera_logging_state()



