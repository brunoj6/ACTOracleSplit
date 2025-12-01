#!/usr/bin/env python3
"""
Fine-tuned Policy Deployment Script
===================================

This script deploys fine-tuned ACTOracleSplit policies that have been trained
with partial camera information for improved robustness.

Key Features:
- Loads fine-tuned checkpoints from curriculum learning
- Supports different fine-tuning approaches (partial, world model, curriculum)
- Maintains compatibility with existing evaluation pipeline
- Clear logging to identify fine-tuned vs original policies

Usage:
    python deploy_policy_finetuned.py --approach curriculum --level level3
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from types import SimpleNamespace
from typing import Dict, List, Optional
import time
import h5py
from datetime import datetime
from .act_policy import ACT
import copy
from PIL import Image
from argparse import Namespace
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from collections import deque
from src.models.worldmodel_coop3 import WorldModelCoop
from src.models.common.rotation_transformer import RotationTransformer
from src.transforms import make_transforms
import json

# Import world model runner from original script
from .deploy_policy_coopwm import WMRunner

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])[None,None,:]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])[None,None,:]

# Global flag to enable/disable camera prediction logging
ENABLE_CAMERA_PREDICTION_LOGGING = True

# Global flag to enable/disable debug logging
ENABLE_DEBUG_LOGGING = True

# Global tracking for first success/failure logging
_first_success_logged = False
_first_failure_logged = False
_current_episode_images = None

def invert_dino_transforms(img):
    """Invert DINO transforms for image reconstruction"""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
    std  = np.array(IMAGENET_STD).reshape(1,1,3)
    img_out = img * std + mean
    img_out = np.clip(img_out, 0, 1)
    return img_out

def save_camera_predictions(observation, recons_left, recons_right, episode_num, success, output_dir="./camera_predictions"):
    """
    Save camera predictions for fine-tuned policy logging.
    Includes approach and level information in filenames.
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
    
    # Save camera combinations for fine-tuned policy
    camera_combinations = {
        "left_arm_finetuned": {
            "head_cam": gt_head,
            "left_cam": gt_left,
            "right_cam": gt_right
        },
        "right_arm_finetuned": {
            "head_cam": gt_head,
            "left_cam": recons_right['left_camera'],
            "right_cam": gt_right
        }
    }
    
    # Save each combination with fine-tuning identifier
    for arm_type, cameras in camera_combinations.items():
        for cam_name, img in cameras.items():
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img_clipped = np.clip(img, 0, 1)
                img_uint8 = (img_clipped * 255).astype(np.uint8)
            else:
                img_uint8 = img
            
            # Create filename with fine-tuning identifier
            filename = f"episode{episode_num}_{episode_type}_finetuned_{arm_type}_{cam_name}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    
    # Save metadata
    metadata = {
        "episode_num": episode_num,
        "episode_type": episode_type,
        "approach": "finetuned",
        "description": "Fine-tuned policy evaluation",
        "timestamp": datetime.now().isoformat(),
        "camera_combinations": list(camera_combinations.keys()),
    }
    
    metadata_file = os.path.join(output_dir, f"episode{episode_num}_{episode_type}_finetuned_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved fine-tuned camera predictions to {output_dir}")

def reset_camera_logging_state():
    """Reset the global logging state for new evaluation runs"""
    global _first_success_logged, _first_failure_logged, _current_episode_images
    _first_success_logged = False
    _first_failure_logged = False
    _current_episode_images = None
    print("🔄 Reset camera prediction logging state")

def log_episode_result(episode_num, success):
    """Log camera predictions for episode result"""
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
    FINETUNED APPROACH: Left arm gets ground truth cameras
    This tests the fine-tuned policy's performance with full information.
    """
    # VALIDATION: Log input shapes
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [LEFT_FINETUNED] Input observation keys: {list(observation['observation'].keys())}")
        print(f"🔍 [LEFT_FINETUNED] Reconstruction keys: {list(recons.keys())}")
    
    # FINETUNED APPROACH: Left arm gets ground truth cameras
    # This tests the fine-tuned policy's performance with full information
    
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
        print(f"🔍 [LEFT_FINETUNED] QPOS shape: {len(qpos)}, range: [{min(qpos):.3f}, {max(qpos):.3f}]")
        print(f"🔍 [LEFT_FINETUNED] QPOS values: {qpos}")
        print(f"🔍 [LEFT_FINETUNED] APPROACH: Fine-tuned policy with ground truth cameras")
    
    # VALIDATION: Log output shapes and ranges
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [LEFT_FINETUNED] Encoded observation shapes:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"🔍 [LEFT_FINETUNED]   {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"🔍 [LEFT_FINETUNED]   {key}: {type(value)} with length {len(value)}")

    return result

def encode_obs_right(observation, recons):
    """
    FINETUNED APPROACH: Right arm gets predicted left camera + ground truth head and right cameras
    This tests the fine-tuned policy's robustness to missing left camera information.
    """
    # VALIDATION: Log input shapes
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [RIGHT_FINETUNED] Input observation keys: {list(observation['observation'].keys())}")
        print(f"🔍 [RIGHT_FINETUNED] Reconstruction keys: {list(recons.keys())}")
    
    # FINETUNED APPROACH: Right arm gets predicted left camera, but keeps GT head and right cameras
    # This tests the fine-tuned policy's robustness to missing left camera information
    
    # Ground truth right camera (unchanged)
    right_cam = cv2.resize(observation["observation"]["right_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0

    # Ground truth head camera (unchanged)
    head_cam = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0

    # EXPERIMENT: Replace left camera with world model prediction
    # This tests if the fine-tuned right arm can handle missing left camera information
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
        print(f"🔍 [RIGHT_FINETUNED] QPOS shape: {len(qpos)}, range: [{min(qpos):.3f}, {max(qpos):.3f}]")
        print(f"🔍 [RIGHT_FINETUNED] QPOS values: {qpos}")
        print(f"🔍 [RIGHT_FINETUNED] APPROACH: Fine-tuned policy with predicted left camera")
    
    # VALIDATION: Log output shapes and ranges
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [RIGHT_FINETUNED] Encoded observation shapes:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"🔍 [RIGHT_FINETUNED]   {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"🔍 [RIGHT_FINETUNED]   {key}: {type(value)} with length {len(value)}")

    return result

def get_model(usr_args):
    """
    Load fine-tuned model with specified approach and level.
    
    Args:
        usr_args: Dictionary containing model configuration
                 - finetuned_approach: 'curriculum', 'world_model', or 'partial'
                 - finetuned_level: 'level1', 'level2', 'level3', or 'final'
                 - finetuned_checkpoint_dir: Directory containing fine-tuned checkpoints
    """
    
    # Extract fine-tuning configuration
    approach = usr_args.get('finetuned_approach', 'curriculum')
    level = usr_args.get('finetuned_level', 'final')
    checkpoint_dir = usr_args.get('finetuned_checkpoint_dir', 'act_ckpt/finetuned')
    
    print(f"🔄 Loading fine-tuned model:")
    print(f"   Approach: {approach}")
    print(f"   Level: {level}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    
    # Determine checkpoint paths based on approach and level
    if approach == 'curriculum':
        if level == 'final':
            left_ckpt_path = os.path.join(checkpoint_dir, 'left_policy_finetuned.ckpt')
            right_ckpt_path = os.path.join(checkpoint_dir, 'right_policy_finetuned.ckpt')
        else:
            left_ckpt_path = os.path.join(checkpoint_dir, level, 'left_policy.ckpt')
            right_ckpt_path = os.path.join(checkpoint_dir, level, 'right_policy.ckpt')
    elif approach == 'world_model':
        left_ckpt_path = os.path.join(checkpoint_dir, 'left_policy_wm_finetuned.ckpt')
        right_ckpt_path = os.path.join(checkpoint_dir, 'right_policy_wm_finetuned.ckpt')
    else:
        # Default to curriculum final
        left_ckpt_path = os.path.join(checkpoint_dir, 'left_policy_finetuned.ckpt')
        right_ckpt_path = os.path.join(checkpoint_dir, 'right_policy_finetuned.ckpt')
    
    # Verify checkpoint files exist
    if not os.path.exists(left_ckpt_path):
        raise FileNotFoundError(f"Left checkpoint not found: {left_ckpt_path}")
    if not os.path.exists(right_ckpt_path):
        raise FileNotFoundError(f"Right checkpoint not found: {right_ckpt_path}")
    
    print(f"✅ Left checkpoint: {left_ckpt_path}")
    print(f"✅ Right checkpoint: {right_ckpt_path}")
    
    # Load original policy architecture
    usr_args['ckpt_name'] = usr_args['left_ckpt_name']
    left_policy = ACT(usr_args, Namespace(**usr_args))
    usr_args['ckpt_name'] = usr_args['right_ckpt_name']
    right_policy = ACT(usr_args, Namespace(**usr_args))
    
    # Load fine-tuned weights
    print("🔄 Loading fine-tuned weights...")
    left_state_dict = torch.load(left_ckpt_path, map_location=usr_args.get("device", "cuda:0"))
    right_state_dict = torch.load(right_ckpt_path, map_location=usr_args.get("device", "cuda:0"))
    
    left_policy.load_state_dict(left_state_dict)
    right_policy.load_state_dict(right_state_dict)
    
    print("✅ Fine-tuned weights loaded successfully")
    
    # Initialize world model if needed
    wmrunner = None
    if usr_args.get("use_wm", False):
        # CORRECTED: Use camera names that match the observation keys
        cam_names = ["head_camera", "right_camera", "left_camera"]
        
        wmrunner = WMRunner(
            config_dir=usr_args.get("wm_config_dir"),
            config_name=usr_args.get("wm_config_name"),
            ckpt_path=usr_args.get("wm_ckpt_path"),
            device=usr_args.get("device"),
            camera_names=cam_names,
        )
        print("✅ World model runner initialized")
    
    model = SimpleNamespace(
        left_policy=left_policy,
        right_policy=right_policy,
        wmrunner=wmrunner,
        finetuned_approach=approach,
        finetuned_level=level,
    )

    return model

def eval(TASK_ENV, model, observation):
    """
    Evaluation function for fine-tuned policy.
    Same interface as original but with fine-tuning logging.
    """
    global _current_episode_images
    
    # World model prediction if available
    recons_left, recons_right = None, None
    if model.wmrunner:
        last_action = observation['joint_action']['vector']
        recons_left, recons_right = model.wmrunner.predict(observation, last_action)
        
        # Store images from first step of episode for potential logging
        if _current_episode_images is None and ENABLE_CAMERA_PREDICTION_LOGGING:
            _current_episode_images = (observation, recons_left, recons_right)
    
    # Update observations with predictions 
    obs_left = encode_obs_left(observation, recons_left or {})
    obs_right = encode_obs_right(observation, recons_right or {})
    
    action_left = model.left_policy.get_action(obs_left, arm_flag='left')
    action_right = model.right_policy.get_action(obs_right, arm_flag='right')
    
    # VALIDATION: Check action dimensions
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 [ACTIONS] Left action shape: {action_left.shape}, Right action shape: {action_right.shape}")
        print(f"🔍 [ACTIONS] Left action range: [{action_left.min():.3f}, {action_left.max():.3f}]")
        print(f"🔍 [ACTIONS] Right action range: [{action_right.min():.3f}, {action_right.max():.3f}]")
        print(f"🔍 [ACTIONS] FINETUNED APPROACH: {model.finetuned_approach} - {model.finetuned_level}")
    
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
    robot_input_images = {
        'left_robot': {  # Left arm gets ground truth information
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

    return observation, last_action, robot_input_images

def reset_model(model):
    """Reset model state for new evaluation run"""
    
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
    
    print(f"🔄 Reset fine-tuned model: {model.finetuned_approach} - {model.finetuned_level}")




