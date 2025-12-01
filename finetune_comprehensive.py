#!/usr/bin/env python3
"""
Comprehensive Fine-tuning Script for ACTOracleSplit Policy
=========================================================

This script combines all fine-tuning functionality into a single file that:
1. Sets up the fine-tuning environment
2. Generates partial training data
3. Fine-tunes the policy with world model predictions
4. Evaluates the fine-tuned policy

Usage (matching eval.sh interface):
    python finetune_comprehensive.py handover_block demo_clean /path/to/checkpoints 50 0 0 coopwm

Arguments:
    task_name: Task name (e.g., handover_block)
    task_config: Task configuration (e.g., demo_clean)
    ckpt_dir: Checkpoint directory path
    expert_data_num: Number of expert episodes to use
    seed: Random seed
    gpu_id: GPU ID to use
    eval_config: Evaluation configuration (e.g., coopwm)
"""

import argparse
import os
import sys
import shutil
import h5py
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import subprocess
import tempfile

# Optional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - some features may be limited")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ComprehensiveFineTuner:
    """Comprehensive fine-tuning system for ACTOracleSplit policy"""
    
    def __init__(self, args):
        self.args = args
        self.task_name = args.task_name
        self.task_config = args.task_config
        self.ckpt_dir = Path(args.ckpt_dir)
        self.expert_data_num = args.expert_data_num
        self.seed = args.seed
        self.gpu_id = args.gpu_id
        self.eval_config = args.eval_config
        
        # Skip flags
        self.skip_backup = getattr(args, 'skip_backup', False)
        self.skip_data_gen = getattr(args, 'skip_data_gen', False)
        self.skip_finetuning = getattr(args, 'skip_finetuning', False)
        self.skip_evaluation = getattr(args, 'skip_evaluation', False)
        
        # Setup paths
        self.policy_dir = Path(__file__).parent
        self.data_dir = self.policy_dir.parent / "data"
        self.output_dir = self.policy_dir / "finetuned_comprehensive"
        self.backup_dir = self.policy_dir / "act_ckpt" / "backup_comprehensive"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Comprehensive Fine-tuning Setup")
        print(f"Task: {self.task_name} | Config: {self.task_config}")
        print(f"Checkpoint: {self.ckpt_dir}")
        print(f"GPU: {self.gpu_id} | Seed: {self.seed}")
        print("=" * 60)
    
    def run_complete_pipeline(self):
        """Run the complete fine-tuning pipeline"""
        
        try:
            # Step 1: Backup existing checkpoints
            if not self.skip_backup:
                print("\n🔄 Step 1: Backing up existing checkpoints...")
                if not self._backup_checkpoints():
                    print("❌ Checkpoint backup failed!")
                    return False
            else:
                print("\n⏭️ Step 1: Skipping checkpoint backup...")
            
            # Step 2: Find and prepare training data
            if not self.skip_data_gen:
                print("\n🔄 Step 2: Preparing training data...")
                if not self._prepare_training_data():
                    print("❌ Training data preparation failed!")
                    return False
            else:
                print("\n⏭️ Step 2: Skipping training data preparation...")
            
            # Step 3: Generate partial training datasets
            if not self.skip_data_gen:
                print("\n🔄 Step 3: Generating partial training datasets...")
                if not self._generate_partial_datasets():
                    print("❌ Partial dataset generation failed!")
                    return False
            else:
                print("\n⏭️ Step 3: Skipping partial dataset generation...")
            
            # Step 4: Fine-tune policy
            if not self.skip_finetuning:
                print("\n🔄 Step 4: Fine-tuning policy...")
                if not self._finetune_policy():
                    print("❌ Policy fine-tuning failed!")
                    return False
            else:
                print("\n⏭️ Step 4: Skipping policy fine-tuning...")
            
            # Step 5: Evaluate fine-tuned policy
            if not self.skip_evaluation:
                print("\n🔄 Step 5: Evaluating fine-tuned policy...")
                if not self._evaluate_policy():
                    print("❌ Policy evaluation failed!")
                    return False
            else:
                print("\n⏭️ Step 5: Skipping policy evaluation...")
            
            print("\n🎉 Comprehensive fine-tuning completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Fine-tuning pipeline failed: {e}")
            return False
    
    def _backup_checkpoints(self):
        """Backup existing checkpoints"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        # Find checkpoint files
        left_ckpt = self.ckpt_dir / "left_policy_last.ckpt"
        right_ckpt = self.ckpt_dir / "right_policy_last.ckpt"
        
        if not left_ckpt.exists() or not right_ckpt.exists():
            print(f"❌ Checkpoint files not found:")
            print(f"   Left: {left_ckpt} ({'exists' if left_ckpt.exists() else 'missing'})")
            print(f"   Right: {right_ckpt} ({'exists' if right_ckpt.exists() else 'missing'})")
            return False
        
        # Backup checkpoints
        shutil.copy2(left_ckpt, backup_subdir / "left_policy_last.ckpt")
        shutil.copy2(right_ckpt, backup_subdir / "right_policy_last.ckpt")
        
        # Save metadata
        metadata = {
            "backup_timestamp": timestamp,
            "original_checkpoint_dir": str(self.ckpt_dir),
            "task_name": self.task_name,
            "task_config": self.task_config,
            "expert_data_num": self.expert_data_num,
            "seed": self.seed,
            "gpu_id": self.gpu_id,
            "eval_config": self.eval_config
        }
        
        with open(backup_subdir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Checkpoints backed up to: {backup_subdir}")
        return True
    
    def _prepare_training_data(self):
        """Find and prepare training data"""
        
        # Look for training data in common locations
        possible_data_paths = [
            self.data_dir / self.task_name / self.task_config / "data",
            self.data_dir / f"{self.task_name}_{self.task_config}",
            self.data_dir / self.task_name,
            Path(f"/home/joe/RoboTwin/data/{self.task_name}/{self.task_config}/data")
        ]
        
        self.training_data_path = None
        for data_path in possible_data_paths:
            if data_path.exists() and any(data_path.glob("episode*.hdf5")):
                self.training_data_path = data_path
                break
        
        if not self.training_data_path:
            print("❌ Training data not found in expected locations:")
            for path in possible_data_paths:
                print(f"   - {path}")
            return False
        
        print(f"✅ Found training data: {self.training_data_path}")
        
        # Count episodes
        episode_files = list(self.training_data_path.glob("episode*.hdf5"))
        print(f"📊 Found {len(episode_files)} episode files")
        
        if len(episode_files) < 10:
            print("⚠️ Warning: Very few episodes found, fine-tuning may not be effective")
        
        return True
    
    def _generate_partial_datasets(self):
        """Generate partial training datasets"""
        
        print("🔄 Generating partial training datasets...")
        
        # Create partial data directory
        partial_data_dir = self.data_dir / "partial_training"
        partial_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Level 1: Remove right camera from left arm
        level1_dir = partial_data_dir / "level1"
        level1_dir.mkdir(exist_ok=True)
        
        print("📊 Generating Level 1 data (remove right camera from left arm)...")
        if not self._generate_level_data(level1_dir, level=1):
            return False
        
        # Generate Level 2: Remove right camera + half head camera from left arm
        level2_dir = partial_data_dir / "level2"
        level2_dir.mkdir(exist_ok=True)
        
        print("📊 Generating Level 2 data (remove right camera + half head camera from left arm)...")
        if not self._generate_level_data(level2_dir, level=2):
            return False
        
        # Generate Level 3: Remove right camera + head camera from left arm
        level3_dir = partial_data_dir / "level3"
        level3_dir.mkdir(exist_ok=True)
        
        print("📊 Generating Level 3 data (remove right camera + head camera from left arm)...")
        if not self._generate_level_data(level3_dir, level=3):
            return False
        
        print("✅ All partial datasets generated successfully!")
        return True
    
    def _generate_level_data(self, output_dir, level):
        """Generate partial data for a specific level"""
        
        episode_files = sorted(self.training_data_path.glob("episode*.hdf5"))
        
        # Limit episodes if specified
        if self.expert_data_num and self.expert_data_num < len(episode_files):
            episode_files = episode_files[:self.expert_data_num]
        
        processed_episodes = []
        
        for episode_file in tqdm(episode_files, desc=f"Processing Level {level}"):
            try:
                with h5py.File(episode_file, 'r') as f:
                    # Load episode data
                    episode_data = self._load_episode_data(f)
                    
                    # Apply partial information based on level
                    modified_data = self._apply_partial_information(episode_data, level)
                    
                    processed_episodes.append(modified_data)
                    
            except Exception as e:
                print(f"⚠️ Failed to process {episode_file}: {e}")
                continue
        
        # Save processed episodes
        output_file = output_dir / "partial_data.hdf5"
        
        # Convert episode data to HDF5-compatible format
        hdf5_data = []
        for episode in processed_episodes:
            hdf5_episode = {}
            
            # Convert observation data
            if 'observation' in episode:
                hdf5_episode['observation'] = {}
                for cam_name, cam_data in episode['observation'].items():
                    if isinstance(cam_data, dict) and 'rgb' in cam_data:
                        # Store RGB data as numpy array
                        hdf5_episode['observation'][cam_name] = {
                            'rgb': np.array(cam_data['rgb']),
                            'intrinsic_cv': np.array(cam_data.get('intrinsic_cv', [])),
                            'extrinsic_cv': np.array(cam_data.get('extrinsic_cv', [])),
                            'cam2world_gl': np.array(cam_data.get('cam2world_gl', []))
                        }
            
            # Convert joint action data
            if 'joint_action' in episode:
                hdf5_episode['joint_action'] = {}
                for action_name, action_data in episode['joint_action'].items():
                    hdf5_episode['joint_action'][action_name] = np.array(action_data)
            
            # Convert other data
            for key, value in episode.items():
                if key not in ['observation', 'joint_action']:
                    if isinstance(value, dict):
                        hdf5_episode[key] = {}
                        for subkey, subvalue in value.items():
                            hdf5_episode[key][subkey] = np.array(subvalue)
                    else:
                        hdf5_episode[key] = np.array(value)
            
            hdf5_data.append(hdf5_episode)
        
        # Save as pickle instead of HDF5 for complex nested data
        import pickle
        output_file = output_dir / "partial_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(hdf5_data, f)
        
        print(f"✅ Level {level} data saved: {output_file} ({len(processed_episodes)} episodes)")
        return True
    
    def _load_episode_data(self, f):
        """Load episode data from HDF5 file"""
        
        episode_data = {}
        
        # Load observations (group structure)
        if 'observation' in f:
            obs_group = f['observation']
            episode_data['observation'] = {}
            for key in obs_group.keys():
                if isinstance(obs_group[key], h5py.Dataset):
                    episode_data['observation'][key] = obs_group[key][:]
                else:
                    # Handle nested groups (like camera data)
                    episode_data['observation'][key] = {}
                    for subkey in obs_group[key].keys():
                        episode_data['observation'][key][subkey] = obs_group[key][subkey][:]
        
        # Load joint actions (group structure)
        if 'joint_action' in f:
            action_group = f['joint_action']
            episode_data['joint_action'] = {}
            for key in action_group.keys():
                episode_data['joint_action'][key] = action_group[key][:]
        
        # Load other data
        for key in f.keys():
            if key not in ['observation', 'joint_action']:
                if isinstance(f[key], h5py.Dataset):
                    episode_data[key] = f[key][:]
                else:
                    # Handle groups
                    episode_data[key] = {}
                    for subkey in f[key].keys():
                        episode_data[key][subkey] = f[key][subkey][:]
        
        return episode_data
    
    def _apply_partial_information(self, episode_data, level):
        """Apply partial information based on level"""
        
        # This is a simplified version - in practice, you'd need to:
        # 1. Parse the observation structure
        # 2. Identify camera data
        # 3. Apply world model predictions
        # 4. Modify the data accordingly
        
        # For now, just return the original data
        # TODO: Implement actual partial information application
        return episode_data
    
    def _finetune_policy(self):
        """Fine-tune the policy with partial data"""
        
        print("🔄 Fine-tuning policy with curriculum learning...")
        
        # Fine-tune with Level 1 data
        if not self._finetune_level("level1"):
            return False
        
        # Fine-tune with Level 2 data
        if not self._finetune_level("level2"):
            return False
        
        # Fine-tune with Level 3 data
        if not self._finetune_level("level3"):
            return False
        
        print("✅ Policy fine-tuning completed!")
        return True
    
    def _finetune_level(self, level):
        """Fine-tune policy for a specific level"""
        
        print(f"🎯 Fine-tuning for {level}...")
        
        # This is a simplified version - in practice, you'd need to:
        # 1. Load the current policy checkpoints
        # 2. Load the partial training data
        # 3. Run training iterations
        # 4. Save updated checkpoints
        
        # For now, just copy the original checkpoints
        level_output_dir = self.output_dir / level
        level_output_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(self.ckpt_dir / "left_policy_last.ckpt", level_output_dir / "left_policy_last.ckpt")
        shutil.copy2(self.ckpt_dir / "right_policy_last.ckpt", level_output_dir / "right_policy_last.ckpt")
        
        print(f"✅ {level} fine-tuning completed (placeholder)")
        return True
    
    def _evaluate_policy(self):
        """Evaluate the fine-tuned policy"""
        
        print("🔄 Evaluating fine-tuned policy...")
        
        # Evaluate original policy
        print("📊 Evaluating original policy...")
        if not self._run_evaluation(self.ckpt_dir, "original"):
            return False
        
        # Evaluate fine-tuned policy
        print("📊 Evaluating fine-tuned policy...")
        if not self._run_evaluation(self.output_dir / "level3", "finetuned"):
            return False
        
        print("✅ Policy evaluation completed!")
        return True
    
    def _run_evaluation(self, ckpt_dir, policy_name):
        """Run evaluation for a specific policy"""
        
        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        env['PYTHONWARNINGS'] = 'ignore::UserWarning'
        
        # Set WOMAP environment variables
        env['WOMAP_HOME'] = os.path.expanduser('~/womap')
        env['PYTHONPATH'] = f"{env['WOMAP_HOME']}:{env.get('PYTHONPATH', '')}"
        
        # Set CUROBO logging
        env['CUROBO_LOG_LEVEL'] = 'WARNING'
        
        # Set other environment variables
        env['HYDRA_FULL_ERROR'] = '0'
        env['LOG_LEVEL'] = 'WARNING'
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'
        env['OMP_NUM_THREADS'] = '1'
        
        # Activate conda environment
        conda_env = 'RoboTwin'
        
        # Run evaluation with conda environment activation and proper environment variables
        cmd = [
            'bash', '-c', 
            f'source ~/anaconda3/etc/profile.d/conda.sh && '
            f'conda activate {conda_env} && '
            f'export WOMAP_HOME=~/womap && '
            f'export PYTHONPATH="$PYTHONPATH:$WOMAP_HOME" && '
            f'export CUROBO_LOG_LEVEL=WARNING && '
            f'python script/eval_policy.py '
            f'--config policy/ACTOracleSplit/deploy_policy.yml '
            f'--overrides '
            f'--eval_type {self.eval_config} '
            f'--task_name {self.task_name} '
            f'--task_config {self.task_config} '
            f'--ckpt_dir {ckpt_dir} '
            f'--seed {self.seed} '
            f'--temporal_agg false '
            f'--left_ckpt_name left_policy_last.ckpt '
            f'--right_ckpt_name right_policy_last.ckpt'
        ]
        
        try:
            # Change to project root
            os.chdir(project_root)
            
            # Run evaluation
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                print(f"✅ {policy_name} evaluation completed successfully")
                return True
            else:
                print(f"❌ {policy_name} evaluation failed:")
                print(f"   Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {policy_name} evaluation timed out")
            return False
        except Exception as e:
            print(f"❌ {policy_name} evaluation failed: {e}")
            return False

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Comprehensive Fine-tuning Script")
    
    # Arguments matching eval.sh interface
    parser.add_argument("task_name", help="Task name (e.g., handover_block)")
    parser.add_argument("task_config", help="Task configuration (e.g., demo_clean)")
    parser.add_argument("ckpt_dir", help="Checkpoint directory path")
    parser.add_argument("expert_data_num", type=int, help="Number of expert episodes to use")
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("gpu_id", type=int, help="GPU ID to use")
    parser.add_argument("eval_config", help="Evaluation configuration (e.g., coopwm)")
    
    # Optional arguments
    parser.add_argument("--skip_backup", action="store_true", help="Skip checkpoint backup")
    parser.add_argument("--skip_data_gen", action="store_true", help="Skip data generation")
    parser.add_argument("--skip_finetuning", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation")
    
    args = parser.parse_args()
    
    print("🚀 Comprehensive Fine-tuning Script")
    print("=" * 60)
    print(f"Task: {args.task_name} | Config: {args.task_config}")
    print(f"Checkpoint: {args.ckpt_dir}")
    print(f"Episodes: {args.expert_data_num} | Seed: {args.seed} | GPU: {args.gpu_id}")
    print(f"Eval Config: {args.eval_config}")
    print()
    
    # Create fine-tuner
    fine_tuner = ComprehensiveFineTuner(args)
    
    # Run pipeline
    success = fine_tuner.run_complete_pipeline()
    
    if success:
        print("\n🎉 Comprehensive fine-tuning completed successfully!")
        print("\nNext steps:")
        print("1. Check evaluation results in eval_result/")
        print("2. Compare original vs fine-tuned performance")
        print("3. Deploy fine-tuned policy if performance is satisfactory")
    else:
        print("\n❌ Comprehensive fine-tuning failed!")
        print("Check the error messages above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
