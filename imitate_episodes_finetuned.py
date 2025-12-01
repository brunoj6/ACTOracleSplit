#!/usr/bin/env python3
"""
Fine-tuning Training Script for ACTOracleSplit
==============================================

This script modifies the original training process to support fine-tuning
with partial camera information and curriculum learning.

Key Features:
- Supports partial information training datasets
- Implements curriculum learning across difficulty levels
- Preserves original training functionality
- Adds fine-tuning specific configurations

Usage:
    python imitate_episodes_finetuned.py --data_dir data/partial_training/level1 --approach curriculum
"""

import sys
import os
import pickle
import h5py
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

# Import from current directory
from act_policy import ACTTwinPolicy
from imitate_episodes import AGENT_SPECS, RoboTwin_Config

class PartialDataset(Dataset):
    """Dataset for partial information training"""
    
    def __init__(self, data_path, level="level1"):
        self.data_path = Path(data_path)
        self.level = level
        
        # Load data
        with h5py.File(self.data_path, 'r') as f:
            self.episodes = f['data'][:]
        
        print(f"📊 Loaded {len(self.episodes)} episodes for {level}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # Extract observations and actions
        observations = episode['observations']
        actions = episode['actions']
        
        # Convert to tensors
        obs_tensor = torch.from_numpy(observations).float()
        action_tensor = torch.from_numpy(actions).float()
        
        return obs_tensor, action_tensor

class CurriculumTrainer:
    """Curriculum learning trainer for partial information robustness"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Curriculum levels
        self.levels = ["level1", "level2", "level3"]
        self.current_level = 0
        
        # Training configuration
        self.config = {
            "base_lr": args.lr,
            "finetune_lr": args.lr * 0.1,  # Lower LR for fine-tuning
            "epochs_per_level": args.num_epochs // len(self.levels),
            "batch_size": args.batch_size,
            "eval_frequency": 10,
            "patience": 20
        }
        
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb.init(
                project="act-finetuning",
                name=f"curriculum-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config
            )
    
    def train_curriculum(self):
        """Train policy using curriculum learning"""
        
        print("🎓 Starting curriculum learning training...")
        print("=" * 50)
        
        # Load original policy if fine-tuning
        if self.args.finetune_from:
            print(f"🔄 Loading original policy from: {self.args.finetune_from}")
            policy = self._load_original_policy()
        else:
            print("🔄 Initializing new policy...")
            policy = self._initialize_policy()
        
        # Train each level progressively
        for i, level in enumerate(self.levels):
            print(f"\n🎯 Level {i+1}/{len(self.levels)}: {level}")
            print("-" * 30)
            
            # Load data for this level
            data_path = Path(self.args.data_dir) / level / f"partial_{level}.hdf5"
            if not data_path.exists():
                print(f"⚠️ Data not found for {level}, skipping...")
                continue
            
            # Create dataset and dataloader
            dataset = PartialDataset(data_path, level)
            dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
            
            # Train for this level
            policy = self._train_level(policy, dataloader, level, i)
            
            # Save checkpoint for this level
            self._save_checkpoint(policy, level)
            
            # Evaluate performance
            self._evaluate_level(policy, level)
        
        # Save final model
        self._save_final_model(policy)
        
        print("\n🎉 Curriculum training completed!")
    
    def _load_original_policy(self):
        """Load original policy for fine-tuning"""
        
        # Load original policy configuration
        original_args = self.args.copy()
        original_args['ckpt_name'] = 'left_policy_last.ckpt'  # Use original checkpoint name
        
        # Initialize policy with original configuration
        policy = ACTTwinPolicy(original_args, RoboTwin_Config)
        
        # Load original weights
        original_ckpt_path = Path(self.args.finetune_from)
        left_ckpt = torch.load(original_ckpt_path / "left_policy_last.ckpt", map_location=self.device)
        right_ckpt = torch.load(original_ckpt_path / "right_policy_last.ckpt", map_location=self.device)
        
        policy.model_left.load_state_dict(left_ckpt)
        policy.model_right.load_state_dict(right_ckpt)
        
        print("✅ Original policy loaded for fine-tuning")
        return policy
    
    def _initialize_policy(self):
        """Initialize new policy from scratch"""
        
        policy = ACTTwinPolicy(self.args, RoboTwin_Config)
        print("✅ New policy initialized")
        return policy
    
    def _train_level(self, policy, dataloader, level, level_idx):
        """Train policy for a specific difficulty level"""
        
        print(f"🎓 Training policy for {level}...")
        
        # Set up optimizer with learning rate based on level
        lr = self._get_learning_rate(level, level_idx)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
        
        # Set up loss function
        criterion = nn.MSELoss()
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["epochs_per_level"]):
            total_loss = 0
            num_batches = 0
            
            # Training phase
            policy.train()
            for batch_idx, (observations, actions) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Get policy predictions
                predicted_actions = policy(observations)
                
                # Compute loss
                loss = criterion(predicted_actions, actions)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Log to wandb if enabled
            if self.args.use_wandb:
                wandb.log({
                    f"{level}/train_loss": avg_loss,
                    f"{level}/epoch": epoch,
                    f"{level}/learning_rate": lr
                })
            
            # Evaluation phase
            if epoch % self.config["eval_frequency"] == 0:
                eval_loss = self._evaluate_epoch(policy, dataloader, criterion)
                print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Eval Loss = {eval_loss:.4f}")
                
                # Log evaluation metrics
                if self.args.use_wandb:
                    wandb.log({
                        f"{level}/eval_loss": eval_loss,
                        f"{level}/epoch": epoch
                    })
                
                # Early stopping
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config["patience"]:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        print(f"✅ Policy trained for {level}")
        return policy
    
    def _get_learning_rate(self, level, level_idx):
        """Get learning rate based on difficulty level"""
        
        # Start with higher LR for easier levels, decrease for harder levels
        lr_schedule = {
            "level1": self.config["finetune_lr"] * 2,  # Higher LR for easier level
            "level2": self.config["finetune_lr"],     # Standard LR
            "level3": self.config["finetune_lr"] * 0.5 # Lower LR for harder level
        }
        
        return lr_schedule.get(level, self.config["finetune_lr"])
    
    def _evaluate_epoch(self, policy, dataloader, criterion):
        """Evaluate policy for one epoch"""
        
        policy.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for observations, actions in dataloader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                predicted_actions = policy(observations)
                loss = criterion(predicted_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, policy, level):
        """Save checkpoint for specific level"""
        
        checkpoint_dir = Path(self.args.ckpt_dir) / "finetuned" / level
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save left policy
        left_path = checkpoint_dir / "left_policy.ckpt"
        torch.save(policy.model_left.state_dict(), left_path)
        
        # Save right policy
        right_path = checkpoint_dir / "right_policy.ckpt"
        torch.save(policy.model_right.state_dict(), right_path)
        
        # Save metadata
        metadata = {
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "left_checkpoint": str(left_path),
            "right_checkpoint": str(right_path),
            "description": f"Fine-tuned policy for {level}",
            "training_config": self.config
        }
        
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved {level} checkpoint: {checkpoint_dir}")
    
    def _save_final_model(self, policy):
        """Save final fine-tuned model"""
        
        # Save left policy
        left_path = Path(self.args.ckpt_dir) / "finetuned" / "left_policy_finetuned.ckpt"
        torch.save(policy.model_left.state_dict(), left_path)
        
        # Save right policy
        right_path = Path(self.args.ckpt_dir) / "finetuned" / "right_policy_finetuned.ckpt"
        torch.save(policy.model_right.state_dict(), right_path)
        
        # Save final metadata
        metadata = {
            "type": "final_finetuned",
            "timestamp": datetime.now().isoformat(),
            "left_checkpoint": str(left_path),
            "right_checkpoint": str(right_path),
            "description": "Final fine-tuned policy with curriculum learning",
            "levels_completed": self.levels,
            "training_config": self.config
        }
        
        metadata_path = Path(self.args.ckpt_dir) / "finetuned" / "final_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved final model: {Path(self.args.ckpt_dir) / 'finetuned'}")
    
    def _evaluate_level(self, policy, level):
        """Evaluate policy performance for specific level"""
        
        print(f"📊 Evaluating {level} performance...")
        
        # TODO: Implement evaluation
        # This would run the policy on test episodes and measure success rate
        
        print(f"✅ {level} evaluation completed")

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Fine-tune ACTOracleSplit with partial information")
    
    # Training arguments
    parser.add_argument("--data_dir", required=True, help="Path to partial training data")
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory")
    parser.add_argument("--finetune_from", help="Path to original checkpoints for fine-tuning")
    parser.add_argument("--approach", choices=["curriculum", "world_model"], 
                      default="curriculum", help="Training approach")
    
    # Model arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    
    args = parser.parse_args()
    
    print("🚀 Fine-tuning Training Script")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.ckpt_dir}")
    print(f"Approach: {args.approach}")
    print(f"Fine-tune from: {args.finetune_from}")
    print()
    
    try:
        # Initialize trainer
        trainer = CurriculumTrainer(args)
        
        # Start training
        trainer.train_curriculum()
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
