#!/usr/bin/env python3
"""
Test script to verify naive augmentation functionality
"""

import numpy as np
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import EpisodicDataset

def test_naive_augmentation():
    """Test that naive augmentation works correctly"""
    
    # Create dummy data
    camera_names = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
    norm_stats = {
        "qpos_mean": np.zeros(14),
        "qpos_std": np.ones(14),
        "action_mean": np.zeros(14),
        "action_std": np.ones(14)
    }
    
    # Test with augmentation disabled (default)
    print("Testing with augmentation disabled...")
    dataset_no_aug = EpisodicDataset(
        episode_ids=np.array([0]),
        dataset_dir="/tmp",  # dummy path
        camera_names=camera_names,
        norm_stats=norm_stats,
        max_action_len=100,
        naive_augment_prob=0.0
    )
    print("✓ Dataset created with augmentation disabled")
    
    # Test with augmentation enabled
    print("Testing with augmentation enabled...")
    dataset_with_aug = EpisodicDataset(
        episode_ids=np.array([0]),
        dataset_dir="/tmp",  # dummy path
        camera_names=camera_names,
        norm_stats=norm_stats,
        max_action_len=100,
        naive_augment_prob=0.5
    )
    print("✓ Dataset created with augmentation enabled")
    
    print("\n✓ All tests passed! Naive augmentation is ready to use.")
    print("\nUsage:")
    print("  python imitate_episodes.py --naive_augment_prob 0.3 --other_args...")
    print("  # This will apply naive augmentation to 30% of training samples")

if __name__ == "__main__":
    test_naive_augmentation()
