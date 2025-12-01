# test_wmrunner_from_h5.py
import os, sys, h5py, numpy as np
from PIL import Image
import torch

# --- repo paths / imports ---
sys.path.append(os.path.abspath("."))  # so "policy.ACTOracleSplit" resolves
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir

from deploy_policy import WMRunner

# ======= EDIT THESE =======
H5_PATH = "/home/joe/RoboTwin/policy/ACT/processed_data/sim-handover_block/demo_clean-50/episode_0.hdf5"
CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]  # order must match training
WOMAP_HOME = os.path.expanduser("~/womap")
WM_CONFIG_NAME = "train_robotwin_handover_block"  # hydra config name
WM_ROOT = os.path.join(WOMAP_HOME, "logs/handover_block")
WM_CKPT = "test_run_2025_09_11_12_34_26-latest.pth.tar"
OVERRIDES = ["shared.img_resize_shape=224", "training.latent_state_history_length=1"]
OUT_DIR = "wmrunner_h5_test_out"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ==========================

def _to_chw(x):
    """Accept HxWx3 or 3xHxW uint8/float; return float32 CHW in [0,1]."""
    x = np.asarray(x)
    if x.ndim == 4:  # T,H,W,3 -> grab first frame
        x = x[0]
    if x.shape[0] == 3 and x.ndim == 3:  # 3,H,W
        chw = x.astype(np.float32) / 255.0
    else:  # H,W,3
        chw = np.transpose(x, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(chw)

def _load_first_frames(h5_path, camera_names):
    """
    Tries a few common layouts:
      - /images/<cam>
      - /observation/images/<cam>
      - /images (group with datasets named by cam)
    Returns dict cam->CHW torch (float32 [0,1]).
    """
    out = {}
    with h5py.File(h5_path, "r") as f:
        def exists(path):
            try:
                f[path]
                return True
            except KeyError:
                return False
        # first try /images/<cam>
        for cam in camera_names:
            ds_path = f"/images/{cam}"
            if exists(ds_path):
                out[cam] = _to_chw(f[ds_path][...])
        if len(out) == len(camera_names):
            return out

        # try /observation/images/<cam>
        out.clear()
        for cam in camera_names:
            ds_path = f"/observations/images/{cam}"
            if exists(ds_path):
                out[cam] = _to_chw(f[ds_path][...])
        if len(out) == len(camera_names):
            return out

        # try /images group with datasets inside
        if exists("/images"):
            grp = f["/images"]
            for cam in camera_names:
                if cam in grp:
                    out[cam] = _to_chw(grp[cam][...])
            if len(out) == len(camera_names):
                return out

        # last resort: print keys to help adjust names
        print("[HDF5] Could not find expected camera datasets. Available keys:")
        def _recurse(g, prefix=""):
            for k in g.keys():
                path = f"{prefix}/{k}"
                try:
                    if isinstance(g[k], h5py.Group):
                        print(f"[G] {path}")
                        _recurse(g[k], path)
                    else:
                        d = g[k]
                        print(f"[D] {path} shape={d.shape} dtype={d.dtype}")
                except Exception:
                    print(f"[?] {path}")
        _recurse(f, "")
        raise KeyError("Camera datasets not found; adjust CAMERA_NAMES or loader paths.")
    return out

def main():
    # 1) Load first frame per camera from HDF5
    cam2chw = _load_first_frames(H5_PATH, CAMERA_NAMES)

    # 2) Build [1, N, C, H, W] tensor in your training camera order
    chws = [cam2chw[c] for c in CAMERA_NAMES]  # torch CHW
    H, W = chws[0].shape[1], chws[0].shape[2]
    imgs = torch.stack(chws, dim=0).unsqueeze(0)  # [1,N,C,H,W]

    with h5py.File(H5_PATH, "r") as f:
        if "/action" in f:
            action = np.asarray(f["/action"][0])                    # (14,)
        elif "joint_action/vector" in f:
            action = np.asarray(f["joint_action/vector"][0])        # (14,)
        else:
            raise KeyError("No 14-D action found; expected '/action' or 'joint_action/vector'")


    # 3) Hydra compose using WoMap configs directory
    GlobalHydra.instance().clear()
    conf_dir = os.path.join(WOMAP_HOME, "configs")
    if not os.path.isdir(conf_dir):
        raise FileNotFoundError(f"WOMAP configs not found at {conf_dir}")
    initialize_config_dir(config_dir=conf_dir, version_base="1.1")
    cfg = compose(config_name=WM_CONFIG_NAME, overrides=OVERRIDES)

    # 4) Init WMRunner
    wmr = WMRunner(cfg=cfg,
                   device=DEVICE,
                   camera_names=CAMERA_NAMES,
                   model_root_path=WM_ROOT,
                   ckpt_name=WM_CKPT)

    # 5) Inference (no action feedback)
    preds = wmr.step(imgs, action_14=action)  # dict cam -> (C,H,W) torch
    # 6) Save inputs and predictions
    os.makedirs(OUT_DIR, exist_ok=True)
    for cam, chw in cam2chw.items():
        np_img = (chw.clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)
        Image.fromarray(np_img).save(os.path.join(OUT_DIR, f"in_{cam}.png"))
    for cam, rec in preds.items():
        np_img = (rec.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(np_img).save(os.path.join(OUT_DIR, f"pred_{cam}.png"))

    print(f"Saved inputs & preds to {OUT_DIR}")

if __name__ == "__main__":
    main()
