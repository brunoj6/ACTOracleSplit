#!/usr/bin/env python3
"""
Script to verify ACT policy normalization by comparing raw HDF5 data 
with dataset_stats.pkl normalization values.

This script will:
1. Load a dataset_stats.pkl file
2. Load an episode HDF5 file 
3. Apply the same normalization as act_policy.py
4. Show the comparison between raw and normalized values
"""

import h5py
import pickle
import numpy as np
import argparse
import os
from pathlib import Path

def load_dataset_stats(stats_path):
    """Load the dataset statistics from pickle file"""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    print("=== Dataset Statistics ===")
    print(f"Action mean shape: {stats['action_mean'].shape}")
    print(f"Action std shape: {stats['action_std'].shape}")
    print(f"Qpos mean shape: {stats['qpos_mean'].shape}")
    print(f"Qpos std shape: {stats['qpos_std'].shape}")
    
    print(f"\nAction mean: {stats['action_mean']}")
    print(f"Action std: {stats['action_std']}")
    print(f"Qpos mean: {stats['qpos_mean']}")
    print(f"Qpos std: {stats['qpos_std']}")
    
    return stats

def load_episode_data(hdf5_path):
    """Load episode data from HDF5 file with correct structure"""
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        print(f"\n=== HDF5 File Structure ===")
        print(f"Top-level keys: {list(f.keys())}")
        
        # Check joint_action structure
        if 'joint_action' in f:
            joint_action_group = f['joint_action']
            print(f"Joint action keys: {list(joint_action_group.keys())}")
            
            # Load individual components
            left_arm = joint_action_group['left_arm'][()]  # Shape: (T, 6)
            left_gripper = joint_action_group['left_gripper'][()]  # Shape: (T,) - needs reshaping
            right_arm = joint_action_group['right_arm'][()]  # Shape: (T, 6)
            right_gripper = joint_action_group['right_gripper'][()]  # Shape: (T,) - needs reshaping
            
            print(f"Left arm shape: {left_arm.shape}")
            print(f"Left gripper shape: {left_gripper.shape}")
            print(f"Right arm shape: {right_arm.shape}")
            print(f"Right gripper shape: {right_gripper.shape}")
            
            # Reshape gripper data to 2D for concatenation
            left_gripper = left_gripper.reshape(-1, 1)  # Shape: (T, 1)
            right_gripper = right_gripper.reshape(-1, 1)  # Shape: (T, 1)
            
            print(f"After reshaping - Left gripper shape: {left_gripper.shape}")
            print(f"After reshaping - Right gripper shape: {right_gripper.shape}")
            
            # Combine into 14D qpos (joint positions)
            qpos = np.concatenate([
                left_arm, left_gripper,
                right_arm, right_gripper
            ], axis=1)  # Shape: (T, 14)
            
            # For actions, we'll use the same data since this represents joint positions
            # In the simulator, joint_action represents the current joint state
            actions = qpos.copy()
            
        else:
            raise KeyError("'joint_action' group not found in HDF5 file")
        
        print(f"\n=== Episode Data ===")
        print(f"Qpos shape: {qpos.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Qpos range: [{qpos.min():.3f}, {qpos.max():.3f}]")
        print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Show breakdown by component
        print(f"\n=== Component Breakdown ===")
        print(f"Left arm range: [{left_arm.min():.3f}, {left_arm.max():.3f}]")
        print(f"Left gripper range: [{left_gripper.min():.3f}, {left_gripper.max():.3f}]")
        print(f"Right arm range: [{right_arm.min():.3f}, {right_arm.max():.3f}]")
        print(f"Right gripper range: [{right_gripper.min():.3f}, {right_gripper.max():.3f}]")
        
        return qpos, actions

def apply_normalization(data, mean, std, data_type="unknown"):
    """Apply normalization using the same formula as act_policy.py"""
    normalized = (data - mean) / std
    
    print(f"\n=== {data_type} Normalization ===")
    print(f"Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"Normalized mean: {normalized.mean():.6f}")
    print(f"Normalized std: {normalized.std():.6f}")
    
    return normalized

def apply_denormalization(normalized_data, mean, std, data_type="unknown"):
    """Apply denormalization using the same formula as act_policy.py"""
    denormalized = normalized_data * std + mean
    
    print(f"\n=== {data_type} Denormalization ===")
    print(f"Normalized range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
    print(f"Denormalized range: [{denormalized.min():.3f}, {denormalized.max():.3f}]")
    print(f"Denormalized mean: {denormalized.mean():.6f}")
    print(f"Denormalized std: {denormalized.std():.6f}")
    
    return denormalized

def compare_arm_specific_normalization(qpos, actions, stats, arm_flag="left"):
    """Compare normalization for specific arm (left or right)"""
    print(f"\n=== {arm_flag.upper()} ARM SPECIFIC NORMALIZATION ===")
    
    if arm_flag == "left":
        qpos_slice = slice(0, 7)
        action_slice = slice(0, 7)
    else:
        qpos_slice = slice(7, 14)
        action_slice = slice(7, 14)
    
    # Extract arm-specific data
    qpos_arm = qpos[:, qpos_slice]
    actions_arm = actions[:, action_slice]
    
    # Extract arm-specific stats
    qpos_mean_arm = stats['qpos_mean'][qpos_slice]
    qpos_std_arm = stats['qpos_std'][qpos_slice]
    action_mean_arm = stats['action_mean'][action_slice]
    action_std_arm = stats['action_std'][action_slice]
    
    print(f"{arm_flag} arm qpos range: [{qpos_arm.min():.3f}, {qpos_arm.max():.3f}]")
    print(f"{arm_flag} arm actions range: [{actions_arm.min():.3f}, {actions_arm.max():.3f}]")
    
    # Apply normalization
    qpos_norm = apply_normalization(qpos_arm, qpos_mean_arm, qpos_std_arm, f"{arm_flag} arm qpos")
    actions_norm = apply_normalization(actions_arm, action_mean_arm, action_std_arm, f"{arm_flag} arm actions")
    
    # Test denormalization
    qpos_denorm = apply_denormalization(qpos_norm, qpos_mean_arm, qpos_std_arm, f"{arm_flag} arm qpos")
    actions_denorm = apply_denormalization(actions_norm, action_mean_arm, action_std_arm, f"{arm_flag} arm actions")
    
    # Check reconstruction accuracy
    qpos_error = np.abs(qpos_arm - qpos_denorm).max()
    actions_error = np.abs(actions_arm - actions_denorm).max()
    
    print(f"\n{arm_flag} arm reconstruction errors:")
    print(f"Qpos max error: {qpos_error:.8f}")
    print(f"Actions max error: {actions_error:.8f}")
    
    return qpos_norm, actions_norm

def main():
    parser = argparse.ArgumentParser(description='Verify ACT policy normalization')
    parser.add_argument('--stats_path', type=str, required=True,
                        help='Path to dataset_stats.pkl file')
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='Path to episode HDF5 file (e.g., episode0.hdf5)')
    parser.add_argument('--episode_idx', type=int, default=0,
                        help='Episode index to analyze (default: 0)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ACT POLICY NORMALIZATION VERIFICATION")
    print("=" * 80)
    
    # Load dataset statistics
    stats = load_dataset_stats(args.stats_path)
    
    # Load episode data
    qpos, actions = load_episode_data(args.hdf5_path)
    
    # Show first few timesteps of raw data
    print(f"\n=== FIRST 3 TIMESTEPS (RAW DATA) ===")
    print("Qpos (first 3 timesteps):")
    print(qpos[:3])
    print("\nActions (first 3 timesteps):")
    print(actions[:3])
    
    # Apply full normalization
    print(f"\n=== FULL NORMALIZATION (14D) ===")
    qpos_norm_full = apply_normalization(qpos, stats['qpos_mean'], stats['qpos_std'], "Full qpos")
    actions_norm_full = apply_normalization(actions, stats['action_mean'], stats['action_std'], "Full actions")
    
    # Test denormalization
    qpos_denorm_full = apply_denormalization(qpos_norm_full, stats['qpos_mean'], stats['qpos_std'], "Full qpos")
    actions_denorm_full = apply_denormalization(actions_norm_full, stats['action_mean'], stats['action_std'], "Full actions")
    
    # Check reconstruction accuracy
    qpos_error_full = np.abs(qpos - qpos_denorm_full).max()
    actions_error_full = np.abs(actions - actions_denorm_full).max()
    
    print(f"\nFull reconstruction errors:")
    print(f"Qpos max error: {qpos_error_full:.8f}")
    print(f"Actions max error: {actions_error_full:.8f}")
    
    # Test arm-specific normalization
    qpos_norm_left, actions_norm_left = compare_arm_specific_normalization(qpos, actions, stats, "left")
    qpos_norm_right, actions_norm_right = compare_arm_specific_normalization(qpos, actions, stats, "right")
    
    # Show normalized data for first few timesteps
    print(f"\n=== FIRST 3 TIMESTEPS (NORMALIZED DATA) ===")
    print("Normalized qpos (first 3 timesteps):")
    print(qpos_norm_full[:3])
    print("\nNormalized actions (first 3 timesteps):")
    print(actions_norm_full[:3])
    
    print(f"\n=== SUMMARY ===")
    print(f"✓ Dataset stats loaded from: {args.stats_path}")
    print(f"✓ Episode data loaded from: {args.hdf5_path}")
    print(f"✓ Raw action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"✓ Normalized action range: [{actions_norm_full.min():.3f}, {actions_norm_full.max():.3f}]")
    print(f"✓ Reconstruction accuracy: {actions_error_full:.8f} max error")
    
    if actions_error_full < 1e-6:
        print("✓ Normalization/denormalization is working correctly!")
    else:
        print("⚠ Warning: Large reconstruction error detected!")

if __name__ == "__main__":
    main()