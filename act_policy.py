from copy import deepcopy
from curobo.curobolib import opt
import torch.nn as nn
import os
import torch
import numpy as np
import pickle
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except:
    from .detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
import IPython

e = IPython.embed

class ACTPolicy(nn.Module):

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, RoboTwin_Config)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class _TwinOptim:
    def __init__(self, opt_left: torch.optim.Optimizer, opt_right: torch.optim.Optimizer):
        self.opt_left = opt_left
        self.opt_right = opt_right
    def zero_grad(self):
        self.opt_left.zero_grad()
        self.opt_right.zero_grad()
    def step(self):
        self.opt_left.step()
        self.opt_right.step()
    @property
    def param_groups(self):
        return self.opt_left.param_groups + self.opt_right.param_groups


class ACTTwinPolicy(nn.Module):
    """
    Decentralized twin ACT:
      - Each arm gets ONLY its own 7-D qpos slice and its own camera subset.
      - Can optionally split a shared camera (e.g., 'cam_high') into left/right halves
        and assign halves to arms (including cross-assign).
      - Supports alias names for convenience (e.g., 'left_wrist' -> 'cam_left_wrist').

    Required in args_override (policy_config):
      - "camera_names":          full camera order used by get_image()  (e.g., ["cam_high","cam_right_wrist","cam_left_wrist"])
      - "left_camera_names":     list like ["left_wrist", "fixed"] or exact names
      - "right_camera_names":    list like ["right_wrist", "fixed"] or exact names
    Optional:
      - "camera_aliases":        dict mapping aliases to real dataset names
      - "split_cam": {
            "name": "cam_high",
            "left_half_to":  "left" or "right",
            "right_half_to": "left" or "right"
        }
    """

    LEFT_SLICE  = slice(0, 7)
    RIGHT_SLICE = slice(7, 14)

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        from copy import deepcopy
        self.kl_weight = args_override["kl_weight"]

        # ---- camera config / aliases
        all_cams   = list(args_override.get("camera_names", []))
        left_cams  = list(args_override.get("left_camera_names", []))
        right_cams = list(args_override.get("right_camera_names", []))
        aliases    = dict(args_override.get("camera_aliases", {}))
        assert all_cams and left_cams and right_cams, \
            "[ACTTwinPolicy] Provide 'camera_names', 'left_camera_names', 'right_camera_names'."

        # resolve aliases
        def _resolve(name): return aliases.get(name, name)

        left_cams_res  = [_resolve(n) for n in left_cams]
        right_cams_res = [_resolve(n) for n in right_cams]

        # Build index maps (ignore any entry that's not actually present)
        def _idx_map(sub):
            idxs = []
            for s in sub:
                if s in all_cams:
                    idxs.append(all_cams.index(s))
            return idxs

        self._all_cams      = all_cams
        self._left_cam_idx  = _idx_map(left_cams_res)
        self._right_cam_idx = _idx_map(right_cams_res)

        # ---- optional split camera (define these BEFORE using them)
        split_cfg = args_override.get("split_cam", None)
        self._split_idx = None
        self._split_to  = None
        if split_cfg:
            split_name = _resolve(split_cfg.get("name", ""))
            assert split_name in all_cams, (
                f"[ACTTwinPolicy] split_cam.name '{split_name}' not in camera_names {all_cams}"
            )
            self._split_idx = all_cams.index(split_name)
            l_to = split_cfg.get("left_half_to", "right")
            r_to = split_cfg.get("right_half_to", "left")
            assert l_to in ["left", "right"] and r_to in ["left", "right"], \
                "[ACTTwinPolicy] split_cam: left_half_to/right_half_to must be 'left' or 'right'"
            self._split_to = (l_to, r_to)
            print(f"[ACTTwin] Splitting '{split_name}': left_half→{l_to}, right_half→{r_to}")

        # ---- drop the FULL split camera from base lists (so only halves are fed)
        self._drop_full_split_cam = bool(args_override.get("drop_full_split_cam", True))
        if self._split_idx is not None and self._drop_full_split_cam:
            before_L = list(self._left_cam_idx)
            before_R = list(self._right_cam_idx)
            self._left_cam_idx  = [i for i in self._left_cam_idx  if i != self._split_idx]
            self._right_cam_idx = [i for i in self._right_cam_idx if i != self._split_idx]
            if before_L != self._left_cam_idx or before_R != self._right_cam_idx:
                print(f"[ACTTwin] Dropped full '{self._all_cams[self._split_idx]}' "
                    f"from base per-arm cams (split halves will be appended).")

        # ---- ensure each arm gets at least one camera after the above
        def _arm_has_any(arm):
            if arm == "left":
                return (len(self._left_cam_idx) > 0) or (self._split_idx is not None and ("left" in self._split_to))
            else:
                return (len(self._right_cam_idx) > 0) or (self._split_idx is not None and ("right" in self._split_to))

        if not _arm_has_any("left"):
            raise AssertionError(f"[ACTTwinPolicy] No cameras mapped to LEFT arm. full={all_cams}, "
                                f"left={left_cams}→{left_cams_res}, split_to={self._split_to}")
        if not _arm_has_any("right"):
            raise AssertionError(f"[ACTTwinPolicy] No cameras mapped to RIGHT arm. full={all_cams}, "
                                f"right={right_cams}→{right_cams_res}, split_to={self._split_to}")

        # ---- build two ACT models (per-arm)
        left_args  = deepcopy(args_override)
        right_args = deepcopy(args_override)

        # IMPORTANT: keep 14-D outputs to match current training/eval code
        left_args.setdefault("action_dim", 7)
        right_args.setdefault("action_dim", 7)

        # Restrict per-arm "declared" cameras to their own (split halves appended at runtime)
        left_args["camera_names"]  = [all_cams[i] for i in self._left_cam_idx]
        right_args["camera_names"] = [all_cams[i] for i in self._right_cam_idx]

        model_left,  opt_left  = build_ACT_model_and_optimizer(left_args,  RoboTwin_Config)
        model_right, opt_right = build_ACT_model_and_optimizer(right_args, RoboTwin_Config)
        self.model_left  = model_left
        self.model_right = model_right
        self.optimizer   = _TwinOptim(opt_left, opt_right)

        self.num_queries = self.model_left.num_queries
        assert self.model_right.num_queries == self.num_queries, "num_queries mismatch (left vs right)"

        self._img_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        # (optional) sanity that heads are 14-D if the builder exposes it
        try:
            out_dim_L = getattr(self.model_left,  "action_dim", None)
            out_dim_R = getattr(self.model_right, "action_dim", None)
            if out_dim_L is not None: assert out_dim_L == 14, f"Left head action_dim={out_dim_L}, expected 14"
            if out_dim_R is not None: assert out_dim_R == 14, f"Right head action_dim={out_dim_R}, expected 14"
        except Exception:
            pass

        print(f"[ACTTwin] KL={self.kl_weight}, Q={self.num_queries}")
        print(f"[ACTTwin] Full cams:   {self._all_cams}")
        print(f"[ACTTwin] Left cams:   {[self._all_cams[i] for i in self._left_cam_idx]}")
        print(f"[ACTTwin] Right cams:  {[self._all_cams[i] for i in self._right_cam_idx]}")


    def _normalize_images(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, K, C, H, W]
        B, K, C, H, W = image.shape
        x = image.reshape(B * K, C, H, W)
        x = self._img_norm(x)
        return x.reshape(B, K, C, H, W)

    def _select_cameras(self, image: torch.Tensor, cam_idx: list) -> torch.Tensor:
        # Select subset along K dimension (may be empty)
        if len(cam_idx) == 0:
            # create an empty slice with K=0 to start; we'll append split halves
            B, K, C, H, W = image.shape
            return image[:, :0, ...]
        return image[:, cam_idx, ...]  # [B, K_sel, C, H, W]

    def _zero_other_half(self, x14: torch.Tensor, which: str) -> torch.Tensor:
        """x14 [...,14] -> zero-out the other arm's 7 dims in-place view-safe manner."""
        x = x14.clone()
        if which == "left":
            x[..., 7:14] = 0.0
        else:
            x[..., 0:7] = 0.0
        return x

    def _to14_from7(self, x7: torch.Tensor, which: str) -> torch.Tensor:
        """x7 [...,7] -> [...,14] by placing into the correct half and zeroing the other."""
        zeros = torch.zeros_like(x7)
        return torch.cat([x7, zeros], dim=-1) if which == "left" else torch.cat([zeros, x7], dim=-1)

    def _append_split_halves(self, image_all: torch.Tensor, image_left: torch.Tensor, image_right: torch.Tensor):
        """
        If split_cam is configured, grab that frame from image_all (by index),
        crop left/right halves, then bilinear-resize each half back to (H,W),
        and append to whichever arm(s) split config specifies.
        """
        if self._split_idx is None:
            return image_left, image_right

        # shared: [B, C, H, W] (already normalized)
        shared = image_all[:, self._split_idx, ...]
        B, C, H, W = shared.shape
        mid = W // 2

        # Crop halves
        left_half  = shared[..., :mid]      # [B, C, H, mid]
        right_half = shared[..., mid:]      # [B, C, H, W-mid]

        # Resize halves back to (H, W) (no black bars)
        # F.interpolate expects NCHW
        left_resized  = F.interpolate(left_half,  size=(H, W), mode="bilinear", align_corners=False)   # [B,C,H,W]
        right_resized = F.interpolate(right_half, size=(H, W), mode="bilinear", align_corners=False)   # [B,C,H,W]

        # Add as separate "cameras"
        left_resized  = left_resized.unsqueeze(1)   # [B,1,C,H,W]
        right_resized = right_resized.unsqueeze(1)  # [B,1,C,H,W]

        l_to, r_to = self._split_to  # e.g., ('right','left')

        if l_to == "left":
            image_left = torch.cat([image_left, left_resized], dim=1)
        else:
            image_right = torch.cat([image_right, left_resized], dim=1)

        if r_to == "left":
            image_left = torch.cat([image_left, right_resized], dim=1)
        else:
            image_right = torch.cat([image_right, right_resized], dim=1)

        return image_left, image_right


    # --- forward ---

    def __call__(self, qpos, image, actions=None, is_pad=None):
        """
        qpos:    [B, ..., 14]
        image:   [B, K_all, C, H, W]
        actions: [B, Q, 14] (train) / None (inference)
        is_pad:  [B, Q] (bool) (train)
        """
        env_state = None
        # Normalize once, then slice / split
        image = self._normalize_images(image)
        image_left  = self._select_cameras(image,  self._left_cam_idx)   # [B, K_L, C, H, W] (K_L can be 0)
        image_right = self._select_cameras(image,  self._right_cam_idx)  # [B, K_R, C, H, W] (K_R can be 0)
        image_left, image_right = self._append_split_halves(image, image_left, image_right)

        # Slice qpos to per-arm 7-D
        if qpos.dim() == 2:
            qpos_left  = qpos[..., self.LEFT_SLICE]   # [B, 7]
            qpos_right = qpos[..., self.RIGHT_SLICE]  # [B, 7]
        else:
            qpos_left  = qpos[..., self.LEFT_SLICE]   # [B, T, 7]
            qpos_right = qpos[..., self.RIGHT_SLICE]  # [B, T, 7]

        # TRAIN
        if actions is not None:
            actions14 = actions[:, :self.num_queries]       # [B,Q,14]
            is_pad     = is_pad[:,  :self.num_queries]      # [B,Q]

            # Per-arm inputs: qpos 14-D with other half zeroed
            if qpos.dim() == 2:
                qpos_left14  = self._zero_other_half(qpos, "left")   # [B,14]
                qpos_right14 = self._zero_other_half(qpos, "right")  # [B,14]
            else:
                # if a time axis ever appears, same idea applies
                qpos_left14  = self._zero_other_half(qpos, "left")
                qpos_right14 = self._zero_other_half(qpos, "right")

            # Teacher forcing targets to 14-D as-is to keep timing/shape,
            # but we zero-out the other half so no leak of targets either.
            tgt_left14  = self._zero_other_half(actions14, "left")    # [B,Q,14]
            tgt_right14 = self._zero_other_half(actions14, "right")   # [B,Q,14]

            # Forward per arm
            aL_hat14, _, (muL, logvarL) = self.model_left(qpos_left14,  image_left,  env_state, tgt_left14,  is_pad)
            aR_hat14, _, (muR, logvarR) = self.model_right(qpos_right14, image_right, env_state, tgt_right14, is_pad)

            # Compute loss only on each arm's slice
            tgt_left7   = actions14[..., self.LEFT_SLICE]            # [B,Q,7]
            tgt_right7  = actions14[..., self.RIGHT_SLICE]           # [B,Q,7]
            aL_hat7     = aL_hat14[..., self.LEFT_SLICE]             # [B,Q,7]
            aR_hat7     = aR_hat14[..., self.RIGHT_SLICE]            # [B,Q,7]

            l1_left_all  = F.l1_loss(tgt_left7,  aL_hat7, reduction="none")
            l1_right_all = F.l1_loss(tgt_right7, aR_hat7, reduction="none")
            l1_left  = (l1_left_all  * ~is_pad.unsqueeze(-1)).mean()
            l1_right = (l1_right_all * ~is_pad.unsqueeze(-1)).mean()

            kldL, _, _ = kl_divergence(muL, logvarL)
            kldR, _, _ = kl_divergence(muR, logvarR)

            loss = l1_left + l1_right + self.kl_weight * (kldL[0] + kldR[0])
            return {
                "l1_left":  l1_left,
                "l1_right": l1_right,
                "kl_left":  kldL[0],
                "kl_right": kldR[0],
                "loss":     loss,
            }

        # INFERENCE
        # zero the other half in inputs, no actions provided
        if qpos.dim() == 2:
            qpos_left14  = self._zero_other_half(qpos, "left")    # [B,14]
            qpos_right14 = self._zero_other_half(qpos, "right")   # [B,14]
        else:
            qpos_left14  = self._zero_other_half(qpos, "left")
            qpos_right14 = self._zero_other_half(qpos, "right")

        aL_hat14, _, _ = self.model_left(qpos_left14,  image_left,  env_state)   # [B,Q,14]
        aR_hat14, _, _ = self.model_right(qpos_right14, image_right, env_state)  # [B,Q,14]

        # Fuse: take left slice from left head, right slice from right head
        out_left7  = aL_hat14[..., self.LEFT_SLICE]               # [B,Q,7]
        out_right7 = aR_hat14[..., self.RIGHT_SLICE]              # [B,Q,7]
        a_full = torch.cat([out_left7, out_right7], dim=-1)       # [B,Q,14]
        return a_full


    def configure_optimizers(self):
        return self.optimizer

class ACTOracleSplitPolicy(nn.Module):
    """
    Decentralized Info Sharing ACT:
      - Trains a single policy
      - Uses all camera information
      - Predicts a single 7 dim action
    """

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        from copy import deepcopy

        args_override.setdefault("action_dim", 7)

        model, optimizer = build_ACT_model_and_optimizer(args_override,  RoboTwin_Config)
        self.model = model
        self.optimizer = optimizer
    
        self.kl_weight = args_override["kl_weight"]

    # --- forward ---
    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class ACT:

    def __init__(self, args_override=None, RoboTwin_Config=None):
        if args_override is None:
            args_override = {
                "kl_weight": 0.1,  # Default value, can be overridden
                "device": "cuda:0",
            }
        self.policy = ACTOracleSplitPolicy(args_override, RoboTwin_Config)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg = args_override.get("temporal_agg", False)
        if isinstance(self.temporal_agg, str):
            self.temporal_agg = self.temporal_agg.lower() in ["true", "1", "yes"]
        self.num_queries = args_override["chunk_size"]
        self.state_dim = RoboTwin_Config.action_dim  # Standard joint dimension for bimanual robot
        self.max_timesteps = 3000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")

        self.t = 0  # Current timestep

        # Load statistics for normalization
        ckpt_dir = args_override.get("ckpt_dir", "")
        if ckpt_dir:
            # Load dataset stats for normalization
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: Could not find stats file at {stats_path}")
                self.stats = None

            # Load policy weights
            ckpt_path = os.path.join(ckpt_dir, args_override.get("ckpt_name", "policy_best.ckpt")) #os.path.join(ckpt_dir, "policy_best.ckpt")
            print("current pwd:", os.getcwd())
            if os.path.exists(ckpt_path):
                loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                print(f"Warning: Could not find policy checkpoint at {ckpt_path}")
        else:
            self.stats = None

    def pre_process(self, qpos, arm_flag=None):
        """Normalize input joint positions"""
        if self.stats is not None:
            if arm_flag == "left":
                stats = (qpos - self.stats["qpos_mean"][:7]) / self.stats["qpos_std"][:7]
            elif arm_flag == "right":
                stats = (qpos - self.stats["qpos_mean"][7:14]) / self.stats["qpos_std"][7:14]
            else:
                stats = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
            return stats
        return qpos
    
    def post_process(self, action, arm_flag=None):
        """Denormalize model outputs"""
        if self.stats is not None:
            if arm_flag == "left":
                action = action * self.stats["action_std"][:7] + self.stats["action_mean"][:7]
            elif arm_flag == "right":
                action = action * self.stats["action_std"][7:14] + self.stats["action_mean"][7:14]
            else:
                action = action * self.stats["action_std"] + self.stats["action_mean"]
            # return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def get_action(self, obs=None, arm_flag="left"):
        if obs is None:
            return None

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy, arm_flag=arm_flag)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras
        curr_images = []
        camera_names = ["head_cam", "left_cam", "right_cam"]
        for cam_name in camera_names:
            curr_images.append(obs[cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Use same weighting factor as in imitate_episodes.py
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1))

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action, arm_flag=arm_flag)
        self.t += 1
        return action