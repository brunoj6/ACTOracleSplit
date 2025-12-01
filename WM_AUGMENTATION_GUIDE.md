# World Model Augmentation Guide

## Overview

This guide explains how world model augmentation works during training and how to use the probe to visualize the augmentation process.

## How World Model Augmentation Works

### 1. Training-Time Process (in `utils.py`)

The world model augmentation mimics what happens during evaluation in `deploy_policy_coopwm.py`:

**During Training:**
1. **Trigger**: When `wm_augment_enabled=True` and random chance < `wm_augment_prob`
2. **Load World Model**: Lazy initialization of `TrainingWMRunner` (avoids CUDA multiprocessing issues)
3. **Create Observation**: Convert dataset sample to world model's expected format
4. **Get Predictions**: World model predicts next timestep camera views from two perspectives:
   - **Left arm perspective**: Sees left camera, predicts right camera and right half of head camera
   - **Right arm perspective**: Sees right camera, predicts left camera and left half of head camera
5. **Apply Augmentation**: Replace ground truth camera views with predictions
6. **Return to Training**: Augmented images go to the training pipeline

### 2. Evaluation-Time Process (in `deploy_policy_coopwm.py`)

**During Evaluation:**
1. Get current observation from environment
2. World model predicts next camera views for both arms
3. Each arm policy receives its own augmented observation:
   - Left arm: Uses ground truth left cam + left half head cam, plus predicted right cam + right half head cam
   - Right arm: Uses ground truth right cam + right half head cam, plus predicted left cam + left half head cam
4. Policies each compute their actions
5. Actions are combined and applied to environment

### 3. Key Differences

- **Training**: Augmentations are applied to dataset samples before batching
- **Evaluation**: Augmentations are applied on-the-fly during rollout
- **Training**: Randomly chooses left or right robot perspective each time
- **Evaluation**: Simultaneously provides both perspectives to respective arm policies

## The Augmentation Logic

### Camera Replacements

**For Left Robot (during training):**
- ✅ Keep: `cam_left_wrist` (ground truth)
- ❌ Replace: `cam_right_wrist` → predicted right camera
- ✅ Keep: Left half of `cam_high` (ground truth)
- ❌ Replace: Right half of `cam_high` → predicted right half

**For Right Robot (during training):**
- ❌ Replace: `cam_left_wrist` → predicted left camera
- ✅ Keep: `cam_right_wrist` (ground truth)
- ❌ Replace: Left half of `cam_high` → predicted left half
- ✅ Keep: Right half of `cam_high` (ground truth)

### Why Split Head Camera?

The head camera is split because it provides overhead view of both workspaces. Each robot arm can see its own half of the table, but not the other half. The world model predicts the occluded half to simulate the other robot's perspective.

## Using the Probe

### Setup

The probe automatically saves visualization images when world model augmentation is enabled. To enable it:

```python
wm_augment_enabled=True,
wm_augment_prob=0.5,  # Adjust probability as needed
wm_config_dir="/path/to/wm/config",
wm_config_name="config_name",
wm_ckpt_path="/path/to/wm/checkpoint.pt"
```

### Output

When training with world model augmentation enabled, the probe saves:

**Location**: `{dataset_dir}/wm_augment_probe/`

**Files for each augmentation** (first 5):
- `aug1_cam_right_wrist_leftrobot.png` - Side-by-side original vs predicted right camera (left robot view)
- `aug1_cam_high_leftrobot.png` - Side-by-side original vs predicted head camera (left robot view)
- `aug1_summary_leftrobot.png` - Grid showing all cameras (original top row, augmented bottom row)
- `aug1_metadata.json` - Metadata about the augmentation

**For right robot augmentations:**
- `aug2_cam_left_wrist_rightrobot.png` - Predicted left camera
- `aug2_cam_high_rightrobot.png` - Predicted head camera half
- `aug2_summary_rightrobot.png` - Full summary

### Reading the Probe Output

1. **Individual Camera Comparisons**: 
   - Left half = Ground truth from dataset
   - Right half = World model prediction
   
2. **Summary Images**:
   - Top row: Original ground truth cameras
   - Bottom row: Augmented cameras (with predictions)
   - Labels indicate which camera is which

3. **Metadata**:
   - `augmentation_index`: Which augmentation (1-5)
   - `robot_type`: "left" or "right" perspective
   - `timestamp`: When saved
   - `camera_names`: List of camera order

### What to Look For

**Good Augmentation:**
- Predictions should be plausible (objects present, reasonable positions)
- Visual quality may be slightly degraded but recognizable
- Predictions should make sense given the left/right robot perspective

**Bad Augmentation:**
- Complete black images or noise
- Objects in impossible positions
- Severe artifacts or distortion
- Predictions that don't match the scene semantics

## Verification Checklist

When running training with world model augmentation:

- ✅ Check that probe images are being saved
- ✅ Verify at least 5 augmentations are saved (both left and right perspectives)
- ✅ Inspect summary images to ensure predictions look reasonable
- ✅ Confirm that the correct cameras are being replaced (not all cameras show predictions)
- ✅ Check that head camera predictions only affect the appropriate half

## Troubleshooting

**No probe images saved:**
- Check if `wm_augment_enabled=True`
- Verify `wm_augment_prob > 0`
- Check world model initialization succeeded (no errors in console)

**Predictions are black/empty:**
- World model may not be loading weights correctly
- Check `wm_ckpt_path` is correct
- Verify world model checkpoint contains decoder weights

**Only some cameras show predictions:**
- This is expected! Only specific cameras should be replaced based on left/right robot perspective

## Integration with Training

The world model augmentation runs **before** images are normalized and sent to the training loop. This means:

1. Original images are loaded from HDF5
2. World model augmentation may occur (with probability `wm_augment_prob`)
3. Augmented images are used for the rest of training
4. The model trains on a mixture of ground truth and predicted views

This simulates the evaluation-time scenario where robots must rely on world model predictions for occluded camera views.
