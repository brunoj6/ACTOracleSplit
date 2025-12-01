# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)                 # will be overridden
    parser.add_argument("--lr_backbone", default=1e-5, type=float)        # will be overridden
    parser.add_argument("--batch_size", default=2, type=int)              # not used
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)                # not used
    parser.add_argument("--lr_drop", default=200, type=int)               # not used
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")

    # Model parameters
    parser.add_argument("--backbone", default="resnet18", type=str, help="CNN backbone")
    parser.add_argument("--dilation", action="store_true",
                        help="Replace stride with dilation in last conv block (DC5)")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"))
    parser.add_argument("--camera_names", default=[], type=list, help="List of camera names")

    # Transformer
    parser.add_argument("--enc_layers", default=4, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--pre_norm", action="store_true")

    # Segmentation
    parser.add_argument("--masks", action="store_true")

    # Mirror args used elsewhere to avoid KeyError; we won't parse CLI here
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", type=str)          # was required when using CLI
    parser.add_argument("--policy_class", type=str)      # was required when using CLI
    parser.add_argument("--task_name", type=str)         # was required when using CLI
    parser.add_argument("--seed", type=int)              # was required when using CLI
    parser.add_argument("--num_epochs", type=int)        # was required when using CLI
    parser.add_argument("--kl_weight", type=float)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--temporal_agg", action="store_true")
    return parser


def _namespace_from_overrides(args_override: dict) -> SimpleNamespace:
    """
    Build an args namespace from parser defaults + overrides, without calling parse_args().
    """
    parser = get_args_parser()
    defaults = {}
    for a in parser._actions:
        # skip help action; keep everything else (even those previously 'required')
        if getattr(a, "dest", None) and a.dest != "help":
            defaults[a.dest] = a.default
    # Merge overrides
    merged = {**defaults, **(args_override or {})}

    # Minimal required fields for model builders (raise early if missing)
    must_have = ["ckpt_dir", "policy_class", "task_name", "seed", "num_epochs"]
    missing = [k for k in must_have if merged.get(k) is None]
    if missing:
        raise ValueError(f"[DETR args] Missing required keys in overrides: {missing}. "
                         f"Pass them via your caller (e.g., imitate_episodes or deploy shim).")

    return SimpleNamespace(**merged)


def build_ACT_model_and_optimizer(args_override, RoboTwin_Config=None):
    """
    Create ACT-style model + optimizer using provided overrides or a pre-built Namespace.
    No CLI parse here.
    """
    if RoboTwin_Config is not None:
        args = RoboTwin_Config
    else:
        args = _namespace_from_overrides(args_override or {})

    print("build_ACT_model_and_optimizer", args)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    """
    Create CNN+MLP model + optimizer using overrides. No CLI parse here.
    """
    args = _namespace_from_overrides(args_override or {})

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer
