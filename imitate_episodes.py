import os
import sys
# Set rendering backend for MuJoCo
#os.environ["MUJOCO_GL"] = "egl"

import torch
import numpy as np
import pickle
import argparse
import cv2

###################### headless servers ####################
import matplotlib
#matplotlib.use("Agg")
###################### headless servers ####################
import imageio.v2 as imageio

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from act_policy import ACTPolicy, ACTTwinPolicy, ACTOracleSplitPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

from agentspec import AGENT_SPECS  # needs: {"left":{"camera_names":[...]}, "right":{...}}

import IPython
e = IPython.embed



# ---- remove W&B flags from sys.argv so downstream parsers (DETR) don't see them ----
_WANDB_FLAGS = {
    "--wandb", "--wandb_project", "--wandb_run_name",
    "--wandb_entity", "--wandb_mode", "--wandb_group", "--wandb_tags"
}

def _wb_log_rollout_visuals(policy, config, wb, steps=150, tag_prefix="val"):
    """Run a short rollout on current policy and log a video + per-arm joint traces."""
    if wb is None:
        return

    import numpy as np
    import matplotlib.pyplot as plt
    from sim_env import make_sim_env
    from einops import rearrange

    ckpt_dir      = config["ckpt_dir"]
    task_name     = config["task_name"]
    camera_names  = config["camera_names"]
    policy_class  = config["policy_class"]  # "ACT" or "ACTTWIN"
    policy_class  = policy_class.upper()
    H = int(config.get("eval_render_h", 480))
    W = int(config.get("eval_render_w", 640))
    onscreen_cam = config.get("eval_render_cam", "angle")
    frame_stride = int(config.get("eval_frame_stride", 10))

    # stats for (de)norm
    import pickle, os
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    pre  = lambda q: (q - stats["qpos_mean"]) / stats["qpos_std"]
    post = lambda a: a * stats["action_std"] + stats["action_mean"]

    def _get_image(ts):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    env = make_sim_env(task_name)
    ts = env.reset()

    frames = []
    qpos_hist = []
    cmd_hist  = []

    policy_was_training = policy.training
    policy.eval()
    with torch.inference_mode():
        for t in range(min(steps, config["episode_len"])):
            frame = env._physics.render(height=H, width=W, camera_id=onscreen_cam)
            if (t % frame_stride) == 0:
                frames.append(frame.copy())

            obs = ts.observation
            qpos_np = np.array(obs["qpos"], dtype=np.float32)
            qpos_hist.append(qpos_np)

            qpos = torch.from_numpy(pre(qpos_np)).float().cuda().unsqueeze(0)
            img  = _get_image(ts)

            if t % config["policy_config"]["num_queries"] == 0:
                all_actions = policy(qpos, img)
            raw = all_actions[:, t % config["policy_config"]["num_queries"]]
            act = post(raw.squeeze(0).cpu().numpy()).astype(np.float32)
            cmd_hist.append(act)

            # step
            if policy_class == "ACTTWIN":
                ts = env.step(act)
            else:
                # single-arm case: place into correct half (assume left by default)
                full = qpos_np.copy()
                full[:7] = act
                ts = env.step(full.astype(np.float32))

    if policy_was_training:
        policy.train()

    # ---- log video directly from numpy (T,H,W,3)
    import numpy as np
    if frames:
        vid_arr = np.stack(frames, axis=0)  # uint8 (T,H,W,3)
        fps = max(1, int(round(1.0 / (DT * frame_stride))))
        wb.log({f"{tag_prefix}/video": wb.Video(vid_arr, fps=fps, format="gif")})

    # ---- log per-arm joint traces
    qpos_hist = np.stack(qpos_hist, axis=0) if len(qpos_hist) else np.zeros((0,14), np.float32)
    cmd_hist  = np.stack(cmd_hist,  axis=0) if len(cmd_hist)  else np.zeros((0,14), np.float32)
    T = min(qpos_hist.shape[0], cmd_hist.shape[0])
    qpos_hist = qpos_hist[:T]
    cmd_hist  = cmd_hist[:T]

    def _plot_arm(arm):
        fig, axes = plt.subplots(7, 1, figsize=(8, 10), sharex=True)
        sl = slice(0,7) if arm == "left" else slice(7,14)
        for j in range(7):
            axes[j].plot(qpos_hist[:, sl][:, j], label="observed")
            axes[j].plot(cmd_hist[:,  sl][:, j], linestyle="--", label="commanded")
            axes[j].set_ylabel(f"j{j}")
        axes[-1].set_xlabel("timestep")
        axes[0].legend(loc="upper right")
        fig.tight_layout()
        return fig

    try:
        figL = _plot_arm("left")
        wb.log({f"{tag_prefix}/left_traces": wb.Image(figL)})
        plt.close(figL)
    except Exception as ex:
        print(f"[wandb] left traces plot failed: {ex}")

    try:
        figR = _plot_arm("right")
        wb.log({f"{tag_prefix}/right_traces": wb.Image(figR)})
        plt.close(figR)
    except Exception as ex:
        print(f"[wandb] right traces plot failed: {ex}")

def _strip_from_sys_argv(flags):
    """Remove flags (and their values if present) from sys.argv in-place."""
    new = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        tok = sys.argv[i]
        key = tok.split("=", 1)[0]
        if key in flags:
            # skip this token and any following non-flag tokens (values; tags may be multiple)
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                i += 1
            continue
        new.append(tok)
        i += 1
    sys.argv = new
# -------------------- W&B helpers --------------------

def _wb_init(args, run_config):
    if not args.get("wandb", False):
        return None
    try:
        import wandb
        run = wandb.init(
            project=args["wandb_project"],
            name=args["wandb_run_name"],
            entity=args["wandb_entity"],
            mode=args["wandb_mode"],
            group=args["wandb_group"],
            tags=args["wandb_tags"],
            config=run_config,
        )
        return wandb
    except Exception as ex:
        print(f"[wandb] init failed: {ex}")
        return None


def _wb_log_scalars(wb, prefix, metrics: dict, step: int):
    if wb is None:
        return
    payload = {}
    for k, v in metrics.items():
        try:
            val = v.item() if hasattr(v, "item") else float(v)
        except Exception:
            continue
        payload[f"{prefix}/{k}"] = val
    if payload:
        wb.log(payload, step=step)


def _grad_global_norm(policy):
    total = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            g = p.grad.data
            total += (g.norm(2) ** 2).item()
    return total ** 0.5


# -------------------- Core --------------------

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]
    
    # ---- WM augmentation config
    enable_wm_augmentation = args["enable_wm_augmentation"]
    wm_config = {
        "config_dir": args.get("wm_config_dir"),
        "config_name": args.get("wm_config_name"),
        "ckpt_path": args.get("wm_ckpt_path"),
        "ckpt_dir": args.get("wm_ckpt_dir"),
    }
    wm_augmentation_prob = args.get("wm_augmentation_prob", 0.0)
    # Ensure split_ratio is always a float to prevent string concatenation issues
    wm_split_ratio = float(args.get("wm_split_ratio", 0.4))

    # ---- Task config
    is_sim = True
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]  # full order used by get_image()

    # ---- Agent config (only used for single-arm runs)
    agent_name = args.get("agent_name", "left")
    
    state_dim = 7
    # ---- Policy config
    lr_backbone = 1e-5
    backbone = "resnet18"

    if policy_class in ["ACT", "ACTTWIN", "ACTOracleSplit"]:
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,  # full order used by get_image()
            # Optional image dropout flags
            "image_dropout_enabled": bool(args.get("image_dropout_enabled", False)),
            "image_dropout_prob": float(args.get("image_dropout_prob", 0.1)),
        }
        if policy_class == "ACTTWIN":
            # decentralized heads: per-arm cam lists from AGENT_SPECS
            policy_config['drop_full_split_cam'] = True
            policy_config["left_camera_names"]  = AGENT_SPECS["left"]["camera_names"]
            policy_config["right_camera_names"] = AGENT_SPECS["right"]["camera_names"]
            policy_config["split_cam"] = {
                "name": "cam_high",           # the dataset camera to split
                "left_half_to": "right",      # send left half to right arm (your request)
                "right_half_to": "left"       # send right half to left arm
            }
            policy_config["camera_aliases"] = {
                "left_wrist":  "cam_left_wrist",
                "right_wrist": "cam_right_wrist",
                "fixed":       "cam_high",
            }
        elif policy_class == "ACTOracleSplit":
            # decentralized heads: per-arm cam lists from AGENT_SPECS
            policy_config['drop_full_split_cam'] = False
            policy_config["left_camera_names"]  = camera_names
            policy_config["right_camera_names"] = camera_names

            policy_config["camera_aliases"] = {
                "left_wrist":  "cam_left_wrist",
                "right_wrist": "cam_right_wrist",
                "fixed":       "cam_high",
            }

    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,      # 7 for single-arm, 14 for ACTTWIN
        "full_state_dim": 14,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,  # full camera list for get_image()
        "real_robot": not is_sim,
        "agent_name": agent_name,
        "wandb": args.get("wandb", False),
    }

    config["eval_save_n"] = 3            # how many episodes to save
    config["eval_render_h"] = 480
    config["eval_render_w"] = 640
    config["eval_render_cam"] = "angle"  # camera_id used for visualization
    config["eval_frame_stride"] = 10      # keep every k-th frame in GIF to shrink size


    # ---- W&B
    run_config_for_wandb = {**config, "policy_config": policy_config}
    wb = _wb_init(args, run_config_for_wandb)
    _strip_from_sys_argv(_WANDB_FLAGS)

    if is_eval:
        ckpt_names = [f"policy_best.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, wb=wb)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        return

    # ---- Data
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # ---------- debug visualization BEFORE training ----------
    try:
        debug_visualize_sample(
            train_dataloader,
            stats,
            ckpt_dir,
            wb=wb,
            tag_prefix="train_sample",
            max_cams=len(camera_names),
        )
    except Exception as ex:
        print(f"[debug_visualize_sample] failed on train_dataloader: {ex}")

    try:
        debug_visualize_sample(
            val_dataloader,
            stats,
            ckpt_dir,
            wb=wb,
            tag_prefix="val_sample",
            max_cams=len(camera_names),
        )
    except Exception as ex:
        print(f"[debug_visualize_sample] failed on val_dataloader: {ex}")

    # ---- Initialize WM for augmentation if enabled
    wm_runner = None
    wm_probe_dir = None
    if enable_wm_augmentation and wm_config.get("config_dir"):
        from wmrunner import WMRunner
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🌐 Initializing WM for augmentation on device: {device}")
        wm_runner = WMRunner(
            config_dir=wm_config["config_dir"],
            config_name=wm_config["config_name"],
            ckpt_path=wm_config["ckpt_path"],
            device=device,
            camera_names=["head_camera", "right_camera", "left_camera"],
            ckpt_dir=wm_config.get("ckpt_dir"),
            split_ratio=wm_split_ratio,
        )
        print(f"✅ WM initialized, will augment with probability {wm_augmentation_prob}")
        
        # Create probe directory for saving augmentation visualizations
        wm_probe_dir = save_wm_augmentation_comparison(ckpt_dir, camera_names)
    
    # ---- Train
    best_ckpt_info_left, best_ckpt_info_right = train_bc(
        train_dataloader, val_dataloader, config, wb=wb, 
        wm_runner=wm_runner, wm_augmentation_prob=wm_augmentation_prob, 
        enable_wm_augmentation=enable_wm_augmentation, stats=stats,
        wm_probe_dir=wm_probe_dir, camera_names=camera_names
    )
    best_epoch_left, min_val_loss_left, best_state_dict_left = best_ckpt_info_left
    best_epoch_right, min_val_loss_right, best_state_dict_right = best_ckpt_info_right

    # save best checkpoint
    ckpt_path_left = os.path.join(ckpt_dir, f"left_policy_best.ckpt")
    torch.save(best_state_dict_left, ckpt_path_left)
    print(f"Best ckpt, val loss {min_val_loss_left:.6f} @ epoch {best_epoch_left} saved to {ckpt_path_left}")
    ckpt_path_right = os.path.join(ckpt_dir, f"right_policy_best.ckpt")
    torch.save(best_state_dict_right, ckpt_path_right)
    print(f"Best ckpt, val loss {min_val_loss_right:.6f} @ epoch {best_epoch_right} saved to {ckpt_path_right}")



def assemble_full_action_from_agent(agent_action_7, full_qpos_14, agent="left"):
    full = full_qpos_14.copy()
    if agent == "left":
        full[0:7]  = agent_action_7  # left 6 + left grip
    else:
        full[7:14] = agent_action_7  # right 6 + right grip
    return full


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    elif policy_class == "ACTTWIN":
        return ACTTwinPolicy(policy_config)   # decentralized twin
    elif policy_class == "ACTOracleSplit":
        return ACTOracleSplitPolicy(policy_config)   # decentralized info
    elif policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError


def make_optimizer(policy_class, policy):
    # Twin returns a wrapper that steps both
    return policy.configure_optimizers()


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(
    config,
    ckpt_name: str = None,              # ignored for split; kept for API compat
    save_episode: bool = True,
    wb=None,
):
    """
    Evaluates two per-arm policies that both observe all images but output their own 7-DoF actions.
    Expects checkpoints:
        - <ckpt_dir>/left_policy_best.ckpt
        - <ckpt_dir>/right_policy_best.ckpt
    You can override via config:
        config["ckpt_left_name"], config["ckpt_right_name"]
    """
    set_seed(1000)

    ckpt_dir      = config["ckpt_dir"]
    policy_class  = config["policy_class"]     # should be your split class (e.g., "ACTOracleSplit" or "ACT")
    camera_names  = config["camera_names"]
    task_name     = config["task_name"]
    temporal_agg  = bool(config.get("temporal_agg", False))
    max_timesteps = int(config["episode_len"])
    onscreen_cam  = config.get("eval_render_cam", "angle")
    H             = int(config.get("eval_render_h", 480))
    W             = int(config.get("eval_render_w", 640))
    frame_stride  = int(config.get("eval_frame_stride", 1))
    save_n        = int(config.get("eval_save_n", 3))

    # ---- ckpt names (defaults)
    ckpt_left_name  = config.get("ckpt_left_name",  "left_policy_best.ckpt")
    ckpt_right_name = config.get("ckpt_right_name", "right_policy_best.ckpt")
    ckpt_left_path  = os.path.join(ckpt_dir, ckpt_left_name)
    ckpt_right_path = os.path.join(ckpt_dir, ckpt_right_name)

    # ---- stats
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    # Per-arm pre/post using sliced stats
    def pre_left(qpos14: np.ndarray):
        return (qpos14[:7]  - stats["qpos_mean"][:7])   / stats["qpos_std"][:7]
    def pre_right(qpos14: np.ndarray):
        return (qpos14[7: ] - stats["qpos_mean"][7:])  / stats["qpos_std"][7:]

    def post_left(a7: np.ndarray):
        return a7 * stats["action_std"][:7] + stats["action_mean"][:7]
    def post_right(a7: np.ndarray):
        return a7 * stats["action_std"][7:] + stats["action_mean"][7:]

    # ---- helpers
    def get_image(ts, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)                       # [N,C,H,W]
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  # [1,N,C,H,W]
        return curr_image

    # ---- make and load policies (two copies of same class/config)
    policy_config = config["policy_config"]
    left_policy  = make_policy(policy_class, policy_config).cuda().eval()
    right_policy = make_policy(policy_class, policy_config).cuda().eval()

    left_status  = left_policy.load_state_dict(torch.load(ckpt_left_path))
    right_status = right_policy.load_state_dict(torch.load(ckpt_right_path))
    print(f"[eval] Loaded left:  {ckpt_left_path}\n{left_status}")
    print(f"[eval] Loaded right: {ckpt_right_path}\n{right_status}")

    # ---- env
    from sim_env import make_sim_env
    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward

    # ---- ACT chunking / temporal agg
    num_queries    = int(policy_config["num_queries"])
    query_frequency = 1 if temporal_agg else num_queries  # when aggregating, we emit every step

    # ---- rollouts
    os.makedirs(os.path.join(ckpt_dir, "eval_rollouts"), exist_ok=True)
    num_rollouts = 50
    episode_returns = []
    highest_rewards = []

    for rollout_id in range(num_rollouts):
        # task sampling (kept from your original)
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())

        ts = env.reset()
        frames_for_gif = [] if rollout_id < save_n else None

        # temporal aggregation buffers (if used)
        if temporal_agg:
            all_time_left  = torch.zeros([max_timesteps, max_timesteps + num_queries, 7],  device="cuda")
            all_time_right = torch.zeros([max_timesteps, max_timesteps + num_queries, 7],  device="cuda")

        rewards = []
        image_list = []
        qpos_hist  = []
        cmd_hist   = []  # [T,14] commanded

        with torch.inference_mode():
            for t in range(max_timesteps):
                # render/log
                frame = env._physics.render(height=H, width=W, camera_id=onscreen_cam)
                if frames_for_gif is not None and (t % frame_stride) == 0:
                    frames_for_gif.append(frame.copy())

                # obs → qpos & images
                obs         = ts.observation
                qpos_np_14  = np.array(obs["qpos"], dtype=np.float32)     # [14]
                qpos_hist.append(qpos_np_14.copy())
                image_list.append(obs.get("images", {"main": obs["image"]}))  # for save_videos

                # per-arm normalized qpos tensors
                qpos_L = torch.from_numpy(pre_left(qpos_np_14)).float().cuda().unsqueeze(0)   # [1,7]
                qpos_R = torch.from_numpy(pre_right(qpos_np_14)).float().cuda().unsqueeze(0)  # [1,7]
                imgs   = get_image(ts, camera_names)  # [1,N,C,H,W], same for both policies

                # query both policies when needed
                if (t % num_queries) == 0:
                    # left/right produce [1, Q, 7]
                    all_left_actions  = left_policy(qpos_L, imgs)
                    all_right_actions = right_policy(qpos_R, imgs)

                if temporal_agg:
                    # populate diagonal bands and aggregate for current step t
                    all_time_left[[t],  t:t+num_queries]  = all_left_actions[0]
                    all_time_right[[t], t:t+num_queries]  = all_right_actions[0]

                    acts_L = all_time_left[:,  t]   # [k,7] for populated k
                    acts_R = all_time_right[:, t]   # [k,7]
                    populated_L = torch.any(acts_L != 0, dim=1)
                    populated_R = torch.any(acts_R != 0, dim=1)
                    acts_L = acts_L[populated_L]
                    acts_R = acts_R[populated_R]
                    if acts_L.numel() == 0 or acts_R.numel() == 0:
                        # fall back to current-slot
                        L_now = all_left_actions[:, t % num_queries]   # [1,7]
                        R_now = all_right_actions[:, t % num_queries]  # [1,7]
                    else:
                        kL = acts_L.size(0); kR = acts_R.size(0)
                        wL = torch.from_numpy(np.exp(-0.01 * np.arange(kL))).float().cuda()
                        wR = torch.from_numpy(np.exp(-0.01 * np.arange(kR))).float().cuda()
                        wL = (wL / (wL.sum() + 1e-8)).unsqueeze(1)
                        wR = (wR / (wR.sum() + 1e-8)).unsqueeze(1)
                        L_now = (acts_L * wL).sum(dim=0, keepdim=True)  # [1,7]
                        R_now = (acts_R * wR).sum(dim=0, keepdim=True)  # [1,7]
                else:
                    # pick slot inside the current query block
                    L_now = all_left_actions[:,  t % num_queries]    # [1,7]
                    R_now = all_right_actions[:, t % num_queries]    # [1,7]

                # to numpy and de-normalize per arm
                L_now_np = L_now.squeeze(0).detach().cpu().numpy()
                R_now_np = R_now.squeeze(0).detach().cpu().numpy()
                L_cmd = post_left(L_now_np)    # [7]
                R_cmd = post_right(R_now_np)   # [7]

                # compose 14-D action
                full_cmd = np.zeros((14,), dtype=np.float32)
                full_cmd[:7]  = L_cmd
                full_cmd[7:]  = R_cmd
                cmd_hist.append(full_cmd.copy())

                # step env
                ts = env.step(full_cmd.astype(np.float32))
                rewards.append(ts.reward)

        rewards_arr = np.array(rewards, dtype=np.float32)
        ep_return   = float(np.sum(rewards_arr[rewards_arr == rewards_arr]))  # guard NaNs
        episode_returns.append(ep_return)
        ep_highest  = float(np.nanmax(rewards_arr))
        highest_rewards.append(ep_highest)

        print(f"Rollout {rollout_id}  episode_return={ep_return}, "
              f"episode_highest_reward={ep_highest}, env_max_reward={env_max_reward}, "
              f"Success: {ep_highest == env_max_reward}")

        # W&B: per-rollout
        if wb is not None:
            wb.log({"eval/episode_return": ep_return,
                    "eval/episode_highest_reward": ep_highest})

        # Save mp4 per rollout
        if save_episode:
            video_path = os.path.join(ckpt_dir, f"video{rollout_id}.mp4")
            try:
                save_videos(image_list, DT, video_path=video_path)
                if wb is not None and rollout_id < 3:
                    wb.log({"eval/video": wb.Video(video_path, fps=int(1/DT), format="mp4")})
            except Exception as ex:
                print(f"[eval] video write/log failed: {ex}")

        # Optional gif (frames_for_gif)
        if frames_for_gif is not None:
            gif_path = os.path.join(ckpt_dir, "eval_rollouts", f"rollout_{rollout_id}.gif")
            try:
                imageio.mimsave(gif_path, frames_for_gif, duration=DT * frame_stride)
                if wb is not None:
                    wb.log({"eval/video_gif": wb.Video(gif_path)})
            except Exception as ex:
                print(f"[eval] writing/logging GIF failed: {ex}")

        # Per-arm traces figure (observed vs commanded)
        try:
            qpos_hist_np = np.stack(qpos_hist, axis=0)   # [T,14]
            cmd_hist_np  = np.stack(cmd_hist,  axis=0)   # [T,14]
            T = min(qpos_hist_np.shape[0], cmd_hist_np.shape[0])
            qpos_hist_np = qpos_hist_np[:T]; cmd_hist_np = cmd_hist_np[:T]

            def _plot_arm(arm):
                fig, axes = plt.subplots(7, 1, figsize=(8, 10), sharex=True)
                sl = slice(0,7) if arm == "left" else slice(7,14)
                for j in range(7):
                    axes[j].plot(qpos_hist_np[:, sl][:, j], label="observed")
                    axes[j].plot(cmd_hist_np[:,  sl][:, j], linestyle="--", label="commanded")
                    axes[j].set_ylabel(f"j{j}")
                axes[-1].set_xlabel("timestep")
                axes[0].legend(loc="upper right")
                fig.tight_layout()
                return fig

            if wb is not None and rollout_id < 3:
                figL = _plot_arm("left");  wb.log({"eval/left_traces":  wb.Image(figL)});  plt.close(figL)
                figR = _plot_arm("right"); wb.log({"eval/right_traces": wb.Image(figR)}); plt.close(figR)
        except Exception as ex:
            print(f"[eval] trace plotting failed: {ex}")

    # ---- aggregate metrics
    highest_rewards = np.array(highest_rewards, dtype=np.float32)
    success_rate = float(np.mean(highest_rewards == env_max_reward))
    avg_return   = float(np.mean(episode_returns))

    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(int(env_max_reward) + 1):
        ge_r = int((highest_rewards >= r).sum())
        summary_str += f"Reward >= {r}: {ge_r}/{num_rollouts} = {100.0*ge_r/num_rollouts:.1f}%\n"
    print(summary_str)

    # save summary
    result_file_name = "result_split_policies.txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns)); f.write("\n\n"); f.write(repr(highest_rewards.tolist()))

    if wb is not None:
        wb.log({"eval/success_rate": success_rate, "eval/avg_return": avg_return})

    return success_rate, avg_return



def apply_wm_augmentation_to_batch(batch_data, wm_runner, wm_augmentation_prob, stats, camera_names):
    """
    Apply WM augmentation to a training batch with given probability.
    Works like deploy_policy_coopwm.py - augments images with WM predictions.
    
    Args:
        batch_data: Tuple of (images, qpos, actions, is_pad)
        wm_runner: WMRunner instance or None
        wm_augmentation_prob: Probability of augmentation
        stats: Dataset stats for normalization
        camera_names: List of camera names
    
    Returns:
        Augmented batch_data tuple
    """
    if wm_runner is None or wm_augmentation_prob <= 0:
        return batch_data
    
    # Decide if this batch should be augmented
    if np.random.rand() >= wm_augmentation_prob:
        return batch_data
    
    images, qpos, actions, is_pad = batch_data
    B = images.shape[0]
    
    # Decide which samples in batch to augment (50/50 probability per sample)
    augment_mask = np.random.rand(B) < 0.5
    if not augment_mask.any():
        return batch_data
    
    images_aug = images.clone()
    
    for b in range(B):
        if not augment_mask[b]:
            continue
        
        try:
            # Convert sample to WM observation format
            sample_images = images[b]  # [K, C, H, W] in [0, 1]
            sample_qpos = qpos[b]  # [14] normalized
            sample_actions = actions[b]  # [T, 14] normalized
            
            # Get device from sample tensors
            device = sample_qpos.device
            
            # Denormalize for WM - ensure all tensors are on the same device
            qpos_std_tensor = torch.from_numpy(stats["qpos_std"]).float().to(device)
            qpos_mean_tensor = torch.from_numpy(stats["qpos_mean"]).float().to(device)
            qpos_denorm = sample_qpos * qpos_std_tensor + qpos_mean_tensor
            
            # Find first non-padded action
            sample_is_pad = is_pad[b]
            first_valid_action_idx = 0
            for i in range(len(sample_is_pad)):
                if not sample_is_pad[i]:
                    first_valid_action_idx = i
                    break
            
            action_std_tensor = torch.from_numpy(stats["action_std"]).float().to(device)
            action_mean_tensor = torch.from_numpy(stats["action_mean"]).float().to(device)
            action_denorm = sample_actions[first_valid_action_idx] * action_std_tensor + action_mean_tensor
            
            # Get as 14D for WM
            action_14d = action_denorm.cpu().numpy()
            qpos_14d = qpos_denorm.cpu().numpy()
            
            # Create WM observation dict (convert to format WM expects)
            obs_dict = {"observation": {}}
            for i, cam_name in enumerate(camera_names):
                # Convert from [C, H, W] to [H, W, C] uint8
                img_tensor = sample_images[i].permute(1, 2, 0).cpu()  # [H, W, C]
                img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                # WM expects RGB in observation dict
                obs_dict["observation"][cam_name] = {"rgb": img_np}
            
            # Set WM internal state for this sample (each training sample is independent)
            wm_runner.prev_obs = obs_dict
            wm_runner.last_action = action_14d
            
            # Get WM predictions
            recons_left, recons_right = wm_runner.predict(obs_dict)
            
            # Decide which arm to augment (50/50)
            augment_arm = np.random.choice(['left', 'right'])
            
            # Apply augmentation
            if augment_arm == 'left':
                images_aug[b] = _apply_left_arm_wm_augmentation(sample_images, recons_left, camera_names)
            else:
                images_aug[b] = _apply_right_arm_wm_augmentation(sample_images, recons_right, camera_names)
                
        except Exception as e:
            # If augmentation fails, keep original images
            if np.random.rand() < 0.01:  # Log occasionally to avoid spam
                print(f"WM augmentation failed: {e}")
            pass
    
    return images_aug, qpos, actions, is_pad


def _apply_left_arm_wm_augmentation(images, recons_left, camera_names):
    """
    Apply left arm augmentation (like encode_obs_left in deploy_policy_coopwm.py):
    - Keep left camera as GT
    - Replace right camera with WM prediction
    - Replace left half of head camera with WM prediction, keep right half as GT
    """
    
    left_idx = camera_names.index("cam_left_wrist")
    right_idx = camera_names.index("cam_right_wrist")
    head_idx = camera_names.index("cam_high")
    
    device = images.device
    aug_images = images.clone()
    
    # Replace right camera with WM prediction
    right_pred = recons_left['right_camera']  # [H, W, 3] in [0, 1]
    C, H, W = aug_images[right_idx].shape
    right_pred_resized = cv2.resize((right_pred * 255).astype(np.uint8), (W, H))
    right_pred_tensor = torch.from_numpy(right_pred_resized).permute(2, 0, 1).float() / 255.0
    aug_images[right_idx] = right_pred_tensor.to(device)
    
    # Replace left half of head camera
    head_gt = images[head_idx]  # [C, H, W]
    head_pred_left = recons_left['head_camera_left']  # [H, W_pred, 3]
    C, H, W = head_gt.shape
    head_pred_resized = cv2.resize((head_pred_left * 255).astype(np.uint8), (W // 2, H))
    head_pred_tensor = torch.from_numpy(head_pred_resized).permute(2, 0, 1).float() / 255.0
    # Concatenate: pred (left half) + GT (right half)
    aug_images[head_idx] = torch.cat([head_pred_tensor.to(device), head_gt[:, :, W//2:]], dim=-1)
    
    return aug_images


def _apply_right_arm_wm_augmentation(images, recons_right, camera_names):
    """
    Apply right arm augmentation (like encode_obs_right in deploy_policy_coopwm.py):
    - Keep right camera as GT
    - Replace left camera with WM prediction
    - Replace right half of head camera with WM prediction, keep left half as GT
    """
    
    left_idx = camera_names.index("cam_left_wrist")
    right_idx = camera_names.index("cam_right_wrist")
    head_idx = camera_names.index("cam_high")
    
    device = images.device
    aug_images = images.clone()
    
    # Replace left camera with WM prediction
    left_pred = recons_right['left_camera']  # [H, W, 3] in [0, 1]
    C, H, W = aug_images[left_idx].shape
    left_pred_resized = cv2.resize((left_pred * 255).astype(np.uint8), (W, H))
    left_pred_tensor = torch.from_numpy(left_pred_resized).permute(2, 0, 1).float() / 255.0
    aug_images[left_idx] = left_pred_tensor.to(device)
    
    # Replace right half of head camera
    head_gt = images[head_idx]
    head_pred_right = recons_right['head_camera_right']  # [H, W_pred, 3]
    C, H, W = head_gt.shape
    head_pred_resized = cv2.resize((head_pred_right * 255).astype(np.uint8), (W // 2, H))
    head_pred_tensor = torch.from_numpy(head_pred_resized).permute(2, 0, 1).float() / 255.0
    # Concatenate: GT (left half) + pred (right half)
    aug_images[head_idx] = torch.cat([head_gt[:, :, :W//2], head_pred_tensor.to(device)], dim=-1)
    
    return aug_images


def save_wm_augmentation_comparison(ckpt_dir, camera_names, num_samples=3):
    """
    Save comparison images showing ground truth vs augmented (with WM predictions).
    Samples from the training dataloader and shows what WM augmentation produces.
    """

    probe_dir = os.path.join(ckpt_dir, "wm_augment_probe")
    os.makedirs(probe_dir, exist_ok=True)
    
    print(f"\n🔍 Saving WM augmentation samples to {probe_dir}")
    print(f"   (This will be called during training to visualize augmentation effects)")
    
    # This function will be called with actual batch data during training
    # For now, just return the directory path
    return probe_dir


def visualize_wm_augmentation_batch(batch_data, wm_runner, wm_augmentation_prob, stats, camera_names, save_dir):
    """
    Visualize a single batch by comparing GT vs augmented images.
    Saves to save_dir.
    """
    import matplotlib.gridspec as gridspec
    
    if wm_runner is None or wm_augmentation_prob <= 0:
        return
    
    images, qpos, actions, is_pad = batch_data
    B = images.shape[0]
    
    # Only process first sample in batch to avoid too many images
    b = 0
    
    try:
        sample_images = images[b]  # [K, C, H, W] in [0, 1]
        sample_qpos = qpos[b]
        sample_actions = actions[b]
        
        # Get device from sample tensors
        device = sample_qpos.device
        
        # Denormalize for WM - ensure all tensors are on the same device
        qpos_std_tensor = torch.from_numpy(stats["qpos_std"]).float().to(device)
        qpos_mean_tensor = torch.from_numpy(stats["qpos_mean"]).float().to(device)
        qpos_denorm = sample_qpos * qpos_std_tensor + qpos_mean_tensor
        
        # Find first non-padded action
        sample_is_pad = is_pad[b]
        first_valid_action_idx = 0
        for i in range(len(sample_is_pad)):
            if not sample_is_pad[i]:
                first_valid_action_idx = i
                break
        
        action_std_tensor = torch.from_numpy(stats["action_std"]).float().to(device)
        action_mean_tensor = torch.from_numpy(stats["action_mean"]).float().to(device)
        action_denorm = sample_actions[first_valid_action_idx] * action_std_tensor + action_mean_tensor
        action_14d = action_denorm.cpu().numpy()
        
        # Create WM observation
        obs_dict = {"observation": {}}
        for i, cam_name in enumerate(camera_names):
            img_tensor = sample_images[i].permute(1, 2, 0).cpu()
            img_np = (img_tensor.numpy() * 255).astype(np.uint8)
            obs_dict["observation"][cam_name] = {"rgb": img_np}
        
        # Set WM state
        wm_runner.prev_obs = obs_dict
        wm_runner.last_action = action_14d
        
        # Get WM predictions
        recons_left, recons_right = wm_runner.predict(obs_dict)
        
        # Create both augmentations for visualization
        aug_left = _apply_left_arm_wm_augmentation(sample_images, recons_left, camera_names)
        aug_right = _apply_right_arm_wm_augmentation(sample_images, recons_right, camera_names)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 6 * len(camera_names)))
        gs = gridspec.GridSpec(len(camera_names), 4, figure=fig, wspace=0.05, hspace=0.2)
        
        for cam_idx, cam_name in enumerate(camera_names):
            # Ground Truth
            img_gt = sample_images[cam_idx].permute(1, 2, 0).cpu().numpy()
            img_gt = np.clip(img_gt, 0, 1)
            
            ax = fig.add_subplot(gs[cam_idx, 0])
            ax.imshow(img_gt)
            ax.axis('off')
            ax.set_title('GT' if cam_idx == 0 else '', fontsize=12, fontweight='bold')
            
            # Left augmentation
            img_left = aug_left[cam_idx].permute(1, 2, 0).cpu().numpy()
            img_left = np.clip(img_left, 0, 1)
            
            ax = fig.add_subplot(gs[cam_idx, 1])
            ax.imshow(img_left)
            ax.axis('off')
            ax.set_title('Left Aug' if cam_idx == 0 else '', fontsize=12, fontweight='bold')
            
            # Right augmentation
            img_right = aug_right[cam_idx].permute(1, 2, 0).cpu().numpy()
            img_right = np.clip(img_right, 0, 1)
            
            ax = fig.add_subplot(gs[cam_idx, 2])
            ax.imshow(img_right)
            ax.axis('off')
            ax.set_title('Right Aug' if cam_idx == 0 else '', fontsize=12, fontweight='bold')
            
            # Difference (from first augmentation)
            diff = np.abs(img_gt - img_left)
            ax = fig.add_subplot(gs[cam_idx, 3])
            ax.imshow(diff, cmap='hot')
            ax.axis('off')
            max_diff = diff.max()
            ax.set_title(f'Diff (max={max_diff:.3f})' if cam_idx == 0 else f'{max_diff:.3f}', fontsize=10)
        
        # Add camera name labels on the left
        for cam_idx, cam_name in enumerate(camera_names):
            fig.text(0.01, 0.5 - (cam_idx * 0.3), cam_name, rotation=90, 
                    ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Save
        import time
        timestamp = int(time.time())
        save_path = os.path.join(save_dir, f"wm_augment_sample_{timestamp}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved WM augmentation comparison to {save_path}")
        
    except Exception as e:
        print(f"   ⚠️  Failed to save WM augmentation comparison: {e}")
        import traceback
        traceback.print_exc()


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    device = next(policy.parameters()).device  # Get device from policy
    image_data, qpos_data, action_data, is_pad = (
        image_data.to(device),
        qpos_data.to(device),
        action_data.to(device),
        is_pad.to(device),
    )
    return policy(qpos_data, image_data, action_data, is_pad)


def eval_current_policy_for_wandb(policy, config, wb, num_rollouts=1, max_steps=None, tag_prefix="val"):
    """Mini rollout from the current in-memory policy; logs video + per-arm joint plots to W&B."""
    if wb is None:
        return

    from sim_env import make_sim_env
    set_seed(1234)

    H = int(config.get("eval_render_h", 480))
    W = int(config.get("eval_render_w", 640))
    onscreen_cam = config.get("eval_render_cam", "angle")
    frame_stride = int(config.get("eval_frame_stride", 10))
    task_name = config["task_name"]
    camera_names = config["camera_names"]

    # pull lambdas & stats
    ckpt_dir = config["ckpt_dir"]
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    pre_process  = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    env = make_sim_env(task_name)
    max_timesteps = config["episode_len"] if max_steps is None else min(max_steps, config["episode_len"])
    query_frequency = config["policy_config"]["num_queries"]
    temporal_agg = config["temporal_agg"]

    # local renderer
    def _get_image(ts, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    for ridx in range(num_rollouts):
        ts = env.reset()
        frames = []
        qpos_hist = []
        cmd_hist  = []

        with torch.inference_mode():
            for t in range(max_timesteps):
                frame = env._physics.render(height=H, width=W, camera_id=onscreen_cam)
                if (t % frame_stride) == 0:
                    frames.append(frame.copy())

                obs = ts.observation
                qpos_np = np.array(obs["qpos"], dtype=np.float32)
                qpos_hist.append(qpos_np)

                qpos = torch.from_numpy(pre_process(qpos_np)).float().cuda().unsqueeze(0)
                img  = _get_image(ts, camera_names)

                if t % query_frequency == 0:
                    all_actions = policy(qpos, img)  # ACTTwin-> [1,Q,14]
                act = all_actions[:, t % query_frequency]
                act = act.squeeze(0).cpu().numpy()
                act = post_process(act).astype(np.float32)
                cmd_hist.append(act)

                ts = env.step(act)

        # save mp4
        video_path = os.path.join(ckpt_dir, f"wb_val_rollout_{ridx}.mp4")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(video_path, frames, fps=int(1.0 / (DT * frame_stride)))
            wb.log({f"{tag_prefix}/video": wb.Video(video_path, fps=int(1.0/(DT*frame_stride)), format="mp4")})
        except Exception as ex:
            print(f"[wandb] video write/log failed: {ex}")

        # per-arm joint traces (observed vs commanded)
        qpos_hist   = np.stack(qpos_hist, axis=0)           # [T,14]
        cmd_hist    = np.stack(cmd_hist,  axis=0)           # [T,14]
        T = min(qpos_hist.shape[0], cmd_hist.shape[0])
        qpos_hist = qpos_hist[:T]
        cmd_hist  = cmd_hist[:T]

        def _plot_arm(arm, color_obs=None, color_cmd=None):
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(7, 1, figsize=(8, 10), sharex=True)
            sl = slice(0,7) if arm == "left" else slice(7,14)
            for j in range(7):
                axes[j].plot(qpos_hist[:, sl][..., j], label="observed")
                axes[j].plot(cmd_hist[:,  sl][..., j], linestyle="--", label="commanded")
                axes[j].set_ylabel(f"j{j}")
            axes[-1].set_xlabel("timestep")
            axes[0].legend(loc="upper right")
            fig.tight_layout()
            return fig

        figL = _plot_arm("left")
        figR = _plot_arm("right")
        wb.log({f"{tag_prefix}/left_traces": wb.Image(figL),
                f"{tag_prefix}/right_traces": wb.Image(figR)})
        plt.close(figL); plt.close(figR)


def train_bc(train_dataloader, val_dataloader, config, wb=None, wm_runner=None, wm_augmentation_prob=0.0, enable_wm_augmentation=False, stats=None, wm_probe_dir=None, camera_names=None):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    if camera_names is None:
        camera_names = config["camera_names"]

    config_copy = config.copy()
    config_copy.pop("policy_config", None)  # avoid nesting

    policy_config = {**policy_config, **config}  # BC not BC+RL

    set_seed(seed)

    policy_left = make_policy(policy_class, policy_config)
    policy_right = make_policy(policy_class, policy_config)
    policy_left.cuda()
    policy_right.cuda()
    optimizer_left = make_optimizer(policy_class, policy_left)
    optimizer_right = make_optimizer(policy_class, policy_right)

    # ---------- Resume support (robust) ----------
    start_epoch = 0

    # TODO: Refactor
    save_ckpt_every_n = 2000  # epochs
 
    train_history_left, train_history_right = [], []
    validation_history_left, validation_history_right = [], []
    min_val_loss_left, min_val_loss_right = np.inf, np.inf
    best_ckpt_info_left, best_ckpt_info_right = None, None
    
    # ---- Save WM augmentation visualization at start
    if enable_wm_augmentation and wm_runner is not None and wm_probe_dir is not None:
        try:
            print("\n📸 Capturing WM augmentation visualization...")
            # Get a batch from the training dataloader
            for data in train_dataloader:
                visualize_wm_augmentation_batch(
                    data, wm_runner, wm_augmentation_prob, stats, camera_names, wm_probe_dir
                )
                break  # Only do this once
        except Exception as e:
            print(f"⚠️  Failed to capture WM augmentation visualization: {e}")
            import traceback
            traceback.print_exc()
    
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"): #for epoch in tqdm(range(num_epochs)):
        # print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy_left.eval()
            policy_right.eval()
            epoch_dicts_left, epoch_dicts_right = [], []
            for batch_idx, data in enumerate(val_dataloader):
                qpos_left = data[1][...,:7]
                qpos_right = data[1][...,7:]
                actions_left = data[2][...,:7] 
                actions_right = data[2][...,7:] 
                data_left = (data[0], qpos_left, actions_left, data[3])
                data_right = (data[0], qpos_right, actions_right, data[3])
                # Split data into left and right actions
                forward_dict_left = forward_pass(data_left, policy_left)
                forward_dict_right = forward_pass(data_right, policy_right)
                epoch_dicts_left.append(forward_dict_left)
                epoch_dicts_right.append(forward_dict_right)

            epoch_summary_left = compute_dict_mean(epoch_dicts_left)
            epoch_summary_right = compute_dict_mean(epoch_dicts_right)

            validation_history_left.append(epoch_summary_left)
            validation_history_right.append(epoch_summary_right)

            epoch_val_loss_left = epoch_summary_left["loss"]
            epoch_val_loss_right = epoch_summary_right["loss"]
            if epoch_val_loss_left < min_val_loss_left:
                min_val_loss_left = epoch_val_loss_left
                best_ckpt_info_left = (epoch, min_val_loss_left, deepcopy(policy_left.state_dict()))
            if epoch_val_loss_right < min_val_loss_right:
                min_val_loss_right = epoch_val_loss_right
                best_ckpt_info_right = (epoch, min_val_loss_right, deepcopy(policy_right.state_dict()))


        # print(f"Left Val loss:   {epoch_val_loss_left:.5f}")
        # print(f"Right Val loss:   {epoch_val_loss_right:.5f}")
        summary_string = ""
        for k, v in epoch_summary_left.items():
            summary_string += f"{k}: {v.item():.3f} "
        summary_string += " | "
        for k, v in epoch_summary_right.items():
            summary_string += f"{k}_right: {v.item():.3f} "

        # print(summary_string)

        # W&B: validation scalars (will include per-arm keys if ACTTWIN)
        if wb is not None:
            _wb_log_scalars(wb, "val", epoch_summary_left, step=epoch)
            _wb_log_scalars(wb, "val", epoch_summary_right, step=epoch)

        # training
        policy_left.train()
        optimizer_left.zero_grad()
        policy_right.train()
        optimizer_right.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            # Apply WM augmentation if enabled
            if enable_wm_augmentation and wm_runner is not None:
                data = apply_wm_augmentation_to_batch(data, wm_runner, wm_augmentation_prob, stats, camera_names)
            
            qpos_left = data[1][...,:7]
            qpos_right = data[1][...,7:]
            actions_left = data[2][...,:7] 
            actions_right = data[2][...,7:] 
            data_left = (data[0], qpos_left, actions_left, data[3])
            data_right = (data[0], qpos_right, actions_right, data[3])
            forward_dict_left = forward_pass(data_left, policy_left)
            forward_dict_right = forward_pass(data_right, policy_right)

            loss_left = forward_dict_left["loss"]
            loss_right = forward_dict_right["loss"]
            loss_left.backward()
            loss_right.backward()
            optimizer_left.step()
            optimizer_right.step()
            optimizer_left.zero_grad()
            optimizer_right.zero_grad()
            train_history_left.append(detach_dict(forward_dict_left))
            train_history_right.append(detach_dict(forward_dict_right))

        # epoch train summary
        epoch_summary_left = compute_dict_mean(train_history_left[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_summary_right = compute_dict_mean(train_history_right[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss_left = epoch_summary_left["loss"]
        epoch_train_loss_right = epoch_summary_right["loss"]
        # print(f"Train loss left: {epoch_train_loss_left:.5f}")
        # print(f"Train loss right: {epoch_train_loss_right:.5f}")
        summary_string = ""
        for k, v in epoch_summary_left.items():
            summary_string += f"{k}: {v.item():.3f} "
        summary_string += " | "
        for k, v in epoch_summary_right.items():
            summary_string += f"{k}_right: {v.item():.3f} "

        # print(summary_string)

        # W&B: training scalars (per-arm keys appear automatically for ACTTWIN)
        if wb is not None:
            payload = {"train/loss_left": epoch_train_loss_left.item(), "train/loss_right": epoch_train_loss_right.item()}
            # opt grad norm (from last step of epoch)
            try:
                gnorm_left = _grad_global_norm(policy_left)
                gnorm_right = _grad_global_norm(policy_right)
                payload["train/grad_norm_left"] = gnorm_left
                payload["train/grad_norm_right"] = gnorm_right
            except Exception:
                pass
            # log per-arm if present
            for k in ["l1_left", "l1_right", "kl_left", "kl_right", "l1", "mse"]:
                if k in epoch_summary_left:
                    payload[f"train/{k}_left"] = epoch_summary_left[k].item()
                if k in epoch_summary_right:
                    payload[f"train/{k}_right"] = epoch_summary_right[k].item()
            wb.log(payload, step=epoch)


        if epoch % save_ckpt_every_n == 0:
            ckpt_path_left = os.path.join(ckpt_dir, f"left_policy_last_{epoch}.ckpt")
            torch.save(policy_left.state_dict(), ckpt_path_left)
            ckpt_path_right = os.path.join(ckpt_dir, f"right_policy_last_{epoch}.ckpt")
            torch.save(policy_right.state_dict(), ckpt_path_right)

            best_epoch_left, min_val_loss_left, best_state_dict_left = best_ckpt_info_left
            ckpt_path_left = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch_left}_seed_{seed}_{epoch}.ckpt")
            torch.save(best_state_dict_left, ckpt_path_left)
            best_epoch_right, min_val_loss_right, best_state_dict_right = best_ckpt_info_right
            ckpt_path_right = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch_right}_seed_{seed}_{epoch}.ckpt")
            torch.save(best_state_dict_right, ckpt_path_right)


    ckpt_path_left = os.path.join(ckpt_dir, f"left_policy_last.ckpt")
    torch.save(policy_left.state_dict(), ckpt_path_left)
    ckpt_path_right = os.path.join(ckpt_dir, f"right_policy_last.ckpt")
    torch.save(policy_right.state_dict(), ckpt_path_right)

    best_epoch_left, min_val_loss_left, best_state_dict_left = best_ckpt_info_left
    ckpt_path_left = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch_left}_seed_{seed}.ckpt")
    torch.save(best_state_dict_left, ckpt_path_left)
    best_epoch_right, min_val_loss_right, best_state_dict_right = best_ckpt_info_right
    ckpt_path_right = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch_right}_seed_{seed}.ckpt")
    torch.save(best_state_dict_right, ckpt_path_right)

    print(f"Training finished:\nSeed {seed}, val loss left {min_val_loss_left:.6f} at epoch {best_epoch_left}")
    print(f"Training finished:\nSeed {seed}, val loss right {min_val_loss_right:.6f} at epoch {best_epoch_right}")


    # save training curves
    plot_history(train_history_left, validation_history_left, num_epochs, ckpt_dir, seed)
    plot_history(train_history_right, validation_history_right, num_epochs, ckpt_dir, seed)

    return best_ckpt_info_left, best_ckpt_info_right


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Optional for single-arm runs; ignored by ACTTWIN
    parser.add_argument("--agent_name", action="store", type=str,
                        choices=["left", "right"], required=False, default="left")

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        choices=["ACT", "ACTTWIN", "ACTOracleSplit", "CNNMLP"],
        required=True,
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT/ACTTWIN
    parser.add_argument("--kl_weight", action="store", type=float, help="KL Weight", required=False, default=1.0)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False, default=10)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False, default=256)
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
        default=512,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="robotwin-act", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    
    # World model augmentation
    parser.add_argument("--enable_wm_augmentation", action="store_true", help="Enable world model augmentation during training")
    parser.add_argument("--wm_augmentation_prob", type=float, default=0.0, help="Probability of applying world model augmentation to a batch")
    parser.add_argument("--wm_config_dir", type=str, default=None, help="Directory containing world model config files")
    parser.add_argument("--wm_config_name", type=str, default=None, help="World model config name")
    parser.add_argument("--wm_ckpt_path", type=str, default=None, help="Path to world model checkpoint")
    parser.add_argument("--wm_ckpt_dir", type=str, default=None, help="Directory containing world model checkpoints")
    parser.add_argument("--wm_split_ratio", type=float, default=0.4, help="Split ratio for head camera (fraction for left arm)")

    main(vars(parser.parse_args()))
