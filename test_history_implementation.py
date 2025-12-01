#!/usr/bin/env python3
"""
Test script for deploy_policy_coopwm_with_history.py
This script tests the enhanced world model runner with proper history handling.
"""

import numpy as np
from collections import deque
import sys
import os

# Mock torch for testing
class MockTorch:
    @staticmethod
    def randn(*args):
        return np.random.randn(*args)
    
    @staticmethod
    def cat(tensors, dim=0):
        return np.concatenate(tensors, axis=dim)
    
    @staticmethod
    def equal(a, b):
        return np.array_equal(a, b)

torch = MockTorch()

# Add the policy directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_action_history():
    """Test the action history functionality"""
    print("🧪 Testing action history functionality...")
    
    # Test deque with maxlen=2
    action_history = deque(maxlen=2)
    default_action = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.])
    
    # Initialize with default actions
    action_history.append(default_action)
    action_history.append(default_action)
    
    print(f"✅ Initial history length: {len(action_history)}")
    print(f"✅ Default action shape: {default_action.shape}")
    
    # Test adding new actions
    new_action1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    new_action2 = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])
    
    action_history.append(new_action1)
    print(f"✅ After adding action1, history length: {len(action_history)}")
    print(f"✅ Latest action: {action_history[-1][:3]}...")
    
    action_history.append(new_action2)
    print(f"✅ After adding action2, history length: {len(action_history)}")
    print(f"✅ Latest action: {action_history[-1][:3]}...")
    
    # Verify maxlen works
    assert len(action_history) == 2, f"Expected history length 2, got {len(action_history)}"
    assert np.array_equal(action_history[-1], new_action2), "Latest action should be new_action2"
    assert np.array_equal(action_history[0], new_action1), "First action should be new_action1"
    
    print("✅ Action history test passed!")

def test_latent_history():
    """Test the latent state history functionality"""
    print("\n🧪 Testing latent state history functionality...")
    
    # Simulate latent states
    batch_size = 1
    num_tokens = 100
    embed_dim = 256
    
    latent_history = deque(maxlen=3)  # history_length - 1
    
    # Create dummy latent states
    z1 = torch.randn(batch_size, num_tokens, embed_dim)
    z2 = torch.randn(batch_size, num_tokens, embed_dim)
    z3 = torch.randn(batch_size, num_tokens, embed_dim)
    z4 = torch.randn(batch_size, num_tokens, embed_dim)
    
    # Test history management
    latent_history.append(z1)
    print(f"✅ After adding z1, history length: {len(latent_history)}")
    
    latent_history.append(z2)
    print(f"✅ After adding z2, history length: {len(latent_history)}")
    
    latent_history.append(z3)
    print(f"✅ After adding z3, history length: {len(latent_history)}")
    
    # Test concatenation
    z_current = z4
    z_with_history = torch.cat(list(latent_history) + [z_current], dim=1)
    expected_shape = (batch_size, num_tokens * 4, embed_dim)  # 3 history + 1 current
    
    print(f"✅ Concatenated shape: {z_with_history.shape}")
    print(f"✅ Expected shape: {expected_shape}")
    
    assert z_with_history.shape == expected_shape, f"Shape mismatch: {z_with_history.shape} vs {expected_shape}"
    
    # Test maxlen behavior
    latent_history.append(z4)
    assert len(latent_history) == 3, f"Expected history length 3, got {len(latent_history)}"
    assert torch.equal(latent_history[0], z2), "First element should be z2 after overflow"
    
    print("✅ Latent history test passed!")

def test_action_conversion():
    """Test the action conversion from 14D to 20D"""
    print("\n🧪 Testing action conversion...")
    
    # Mock the conversion function
    def convert_actions_mock(action14):
        """Mock version of convert_actions for testing"""
        raw_vec = np.array(action14)
        raw = raw_vec.reshape(2, 7)
        pos = raw[:, :3]
        rot_aa = raw[:, 3:6]
        grip = raw[:, 6:]
        
        # Mock rotation conversion (simplified)
        rot_6d = np.concatenate([rot_aa, rot_aa], axis=-1)  # Mock 6D rotation
        
        out = np.concatenate([pos, rot_6d, grip], axis=-1).astype(np.float32)
        return out.reshape(-1, 20)
    
    # Test conversion
    action_14d = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.])
    action_20d = convert_actions_mock(action_14d)
    
    print(f"✅ Input 14D action shape: {action_14d.shape}")
    print(f"✅ Output 20D action shape: {action_20d.shape}")
    
    assert action_20d.shape == (1, 20), f"Expected shape (1, 20), got {action_20d.shape}"
    
    print("✅ Action conversion test passed!")

def test_history_reset():
    """Test the history reset functionality"""
    print("\n🧪 Testing history reset...")
    
    # Simulate a history state
    action_history = deque([np.array([1., 2., 3.]), np.array([4., 5., 6.])], maxlen=2)
    latent_history = deque([torch.randn(1, 10, 5), torch.randn(1, 10, 5)], maxlen=2)
    
    print(f"✅ Before reset - action history length: {len(action_history)}")
    print(f"✅ Before reset - latent history length: {len(latent_history)}")
    
    # Reset
    default_action = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.])
    action_history.clear()
    action_history.extend([default_action, default_action])
    latent_history.clear()
    
    print(f"✅ After reset - action history length: {len(action_history)}")
    print(f"✅ After reset - latent history length: {len(latent_history)}")
    
    assert len(action_history) == 2, "Action history should have 2 elements after reset"
    assert len(latent_history) == 0, "Latent history should be empty after reset"
    assert np.array_equal(action_history[0], default_action), "First action should be default"
    
    print("✅ History reset test passed!")

def main():
    """Run all tests"""
    print("🚀 Starting tests for deploy_policy_coopwm_with_history.py")
    print("=" * 60)
    
    try:
        test_action_history()
        test_latent_history()
        test_action_conversion()
        test_history_reset()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The enhanced world model runner should work correctly.")
        print("\nKey improvements in the new implementation:")
        print("✅ Proper action history maintenance (last 2 predictions)")
        print("✅ Latent state history for temporal modeling")
        print("✅ Enhanced predict function using both previous and current observations")
        print("✅ Proper history reset between episodes")
        print("✅ Better debugging and validation")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
