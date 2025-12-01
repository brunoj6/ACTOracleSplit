# World Model Augmentation for ACTOracleSplit Training

This document describes the new world model augmentation functionality added to the ACTOracleSplit training pipeline.

## Overview

The world model augmentation feature allows training the ACTOracleSplit policy with synthetic camera predictions from a pre-trained world model. This helps the policy learn to handle scenarios where certain camera views are occluded or unavailable, similar to the naive augmentation but using more realistic predictions.

## Features

- **Probabilistic Augmentation**: Apply world model predictions with a configurable probability
- **Per-Arm Perspectives**: Randomly choose left or right robot perspective for augmentation
- **Camera Replacement**: Replace specific camera views with world model predictions:
  - Left robot: Replace right camera and right half of head camera
  - Right robot: Replace left camera and left half of head camera
- **Graceful Fallback**: If world model fails to load or predict, training continues with original images
- **Compatible with Naive Augmentation**: Can be used alongside existing Gaussian blur augmentation

## Usage

### 1. Training Script (train.sh)

Add the following parameters to your training script:

```bash
--wm_augment_enabled \
--wm_augment_prob 0.3 \
--wm_config_dir /path/to/wm/config \
--wm_config_name config \
--wm_ckpt_path /path/to/wm/checkpoint.ckpt \
```

### 2. Command Line Arguments

The following new arguments are available in `imitate_episodes.py`:

- `--wm_augment_enabled`: Enable world model augmentation during training
- `--wm_augment_prob`: Probability of applying world model augmentation (0.0-1.0, default: 0.1)
- `--wm_config_dir`: Directory containing world model config files
- `--wm_config_name`: World model config name (default: "config")
- `--wm_ckpt_path`: Path to world model checkpoint

### 3. Example Training Command

```bash
python imitate_episodes.py \
    --task_name sim-transfer-cube-v0 \
    --ckpt_dir ./act_ckpt/act-ora-split-transfer-cube \
    --policy_class ACTOracleSplit \
    --kl_weight 1 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 12 \
    --dim_feedforward 4096 \
    --num_epochs 10000 \
    --lr 2e-4 \
    --seed 42 \
    --wandb \
    --wandb_project robotwin-act-transfer-cube \
    --wandb_run_name ora-split_transfer-cube_wm_aug \
    --naive_augment_prob 0.2 \
    --naive_augment_blur_std 2 \
    --wm_augment_enabled \
    --wm_augment_prob 0.3 \
    --wm_config_dir /path/to/wm/config \
    --wm_config_name config \
    --wm_ckpt_path /path/to/wm/checkpoint.ckpt
```

## Implementation Details

### TrainingWMRunner Class

A simplified world model runner (`TrainingWMRunner`) handles:
- World model initialization using Hydra configuration
- Camera image preprocessing and transformation
- Per-arm perspective prediction with proper dropout
- Action conversion from 14D to 20D format
- Temporal consistency maintenance

### EpisodicDataset Integration

The `EpisodicDataset` class now supports:
- World model runner initialization during dataset creation
- Probabilistic application of world model predictions
- Image resizing and tensor conversion for predictions
- Error handling and graceful fallback

### Augmentation Logic

1. **Random Selection**: Choose left or right robot perspective (50/50 probability)
2. **Observation Creation**: Build observation dictionary with camera images and joint states
3. **World Model Prediction**: Generate predictions for both arm perspectives
4. **Image Replacement**: Replace specific camera views based on chosen perspective
5. **Action Update**: Update world model with current action for next prediction

## Requirements

- Pre-trained world model checkpoint
- World model configuration files
- Compatible camera naming convention:
  - `cam_high` for head camera
  - `cam_left_wrist` for left wrist camera  
  - `cam_right_wrist` for right wrist camera

## Error Handling

The implementation includes comprehensive error handling:
- World model loading failures disable augmentation automatically
- Prediction failures continue training with original images
- Missing parameters disable augmentation with warning messages
- Invalid camera configurations are handled gracefully

## Testing

Run the test script to verify the integration:

```bash
python test_wm_augmentation.py
```

This will test:
- Dataset creation with augmentation disabled
- Dataset creation with missing parameters
- Dataset creation with dummy parameters (graceful failure)

## Performance Considerations

- World model predictions add computational overhead during training
- Consider using lower `wm_augment_prob` values (0.1-0.3) to balance augmentation benefits with training speed
- World model is loaded lazily (only when first needed) to avoid CUDA multiprocessing issues
- DataLoader uses `num_workers=0` when world model augmentation is enabled to prevent CUDA context conflicts
- Predictions are cached within the world model runner for temporal consistency

## CUDA Multiprocessing Fix

The implementation includes fixes for CUDA multiprocessing issues:

1. **Multiprocessing Start Method**: Set to 'spawn' to avoid CUDA context sharing issues
2. **Lazy Initialization**: World model runner is initialized only when first needed, not during dataset creation
3. **Single-threaded Data Loading**: Uses `num_workers=0` when world model augmentation is enabled
4. **Error Handling**: Graceful fallback if CUDA initialization fails

This ensures compatibility with PyTorch's DataLoader multiprocessing while using CUDA-enabled world models.

## Troubleshooting

### Common Issues

1. **"Failed to initialize world model"**: Check that config directory and checkpoint path are correct
2. **"World model augmentation failed"**: Usually indicates missing dependencies or incompatible world model format
3. **"Camera not found in observation"**: Verify camera naming matches expected convention

### Debug Mode

Enable debug logging by setting `ENABLE_DEBUG_LOGGING = True` in the `TrainingWMRunner` class to see detailed prediction information.

## Future Enhancements

- Support for different world model architectures
- Configurable prediction horizons
- Integration with other augmentation techniques
- Performance optimizations for faster prediction
