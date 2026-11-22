"""Unified local training script for SO-101 (SmolVLA / ACT / FC MLP / CNN).

Examples:
    python local/train_local.py --arch smolvla --dataset so101_pick_teabag --batch-size 16 --steps 25000
    python local/train_local.py --arch smolvla --dataset so101_pick_cube --batch-size 16 --steps 25000
    python local/train_local.py --arch act --dataset so101_pick_teabag --batch-size 16 --steps 15000
    python local/train_local.py --arch fc --dataset so101_pick_cube
"""

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Local SO-101 training script (SmolVLA / ACT / FC / CNN)")
    parser.add_argument(
        "--arch",
        required=True,
        choices=["smolvla", "act", "fc", "cnn"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset folder name under ./datasets/ (e.g. so101_pick_teabag, so101_pick_cube).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Multiple dataset folder names under ./datasets/ for multi-task training.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to local dataset directory (default: ./datasets/<--dataset> or ./dataset).",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot dataset repo_id identifier (default: local/<dataset_name>).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root directory (default: ./outputs/<arch>/<dataset_name> for ./datasets/*, else ./outputs/<arch>).",
    )
    parser.add_argument("--chunk", type=int, default=50, help="Action chunk horizon.")
    parser.add_argument("--steps", type=int, default=15000, help="Total training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 16 smolvla/act, 32 fc/cnn).")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate (default: 1e-4 smolvla, 1e-5 act, 3e-4 fc/cnn).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for fc/cnn.")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps for act/fc/cnn.")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="Gradient clipping norm (for fc/cnn; smolvla/act use config preset).",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--save-freq", type=int, default=500, help="Checkpoint save frequency (steps).")
    parser.add_argument("--milestone-freq", type=int, default=5000, help="Milestone checkpoint frequency (steps).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and start fresh.")
    parser.add_argument("--no-trim", action="store_true", help="Disable head/tail idle frame trimming.")
    parser.add_argument("--no-trim-tail", action="store_true", help="Disable trailing idle frame trimming only.")
    parser.add_argument("--move-thresh", type=float, default=3.0, help="State delta threshold for idle trimming.")
    parser.add_argument("--keep-lead-frames", type=int, default=10, help="Frames to keep before first movement.")
    parser.add_argument("--keep-tail-frames", type=int, default=10, help="Frames to keep after last movement.")
    parser.add_argument(
        "--drop-longest-episodes",
        type=int,
        default=0,
        help="Exclude the N longest-duration episodes (default: 0, set >0 to enable).",
    )
    parser.add_argument(
        "--exclude-episodes",
        default="",
        help="Comma-separated episode indices to exclude manually (e.g., 3,12).",
    )
    parser.add_argument("--video-backend", default="pyav", help="LeRobot video decoding backend.")

    # SmolVLA options
    parser.add_argument("--smolvla-base", default="lerobot/smolvla_base", help="Pretrained SmolVLA base model.")
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 before Hub imports.")

    # FC options
    parser.add_argument("--fc-no-images", action="store_true", help="Train FC policy using only observation.state.")
    parser.add_argument("--fc-image-size", type=int, default=32, help="Square image resize for FC policy.")
    parser.add_argument("--fc-dropout", type=float, default=0.1, help="Dropout rate for FC policy.")

    # CNN options
    parser.add_argument("--cnn-image-size", type=int, default=128, help="Square image resize for CNN policy.")
    parser.add_argument("--share-camera-encoder", action="store_true", help="Share CNN encoder across cameras.")
    parser.add_argument("--cnn-dropout", type=float, default=0.1, help="Dropout rate for CNN policy.")

    return parser.parse_args()


def validate_args(args, batch_size):
    for name, val in (
        ("--chunk", args.chunk),
        ("--steps", args.steps),
        ("--save-freq", args.save_freq),
        ("--milestone-freq", args.milestone_freq),
        ("--batch-size", batch_size),
    ):
        if val <= 0:
            raise ValueError(f"{name} 必須為正整數，目前為 {val}。")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers 不可為負數，目前為 {args.num_workers}。")
    if args.drop_longest_episodes < 0 or args.keep_lead_frames < 0 or args.keep_tail_frames < 0:
        raise ValueError("--drop-longest-episodes、--keep-lead-frames 與 --keep-tail-frames 不可為負數。")
    if args.fc_image_size <= 0 or args.cnn_image_size <= 0:
        raise ValueError("--fc-image-size 與 --cnn-image-size 必須為正整數。")


def promote_trainable_params_to_fp32(policy):
    """Cast all trainable (requires_grad) parameters to FP32 so AdamW does not suffer BF16 rounding erasure."""
    import torch

    promoted = 0
    for p in policy.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            promoted += 1
    trainable_dtypes = {str(p.dtype) for p in policy.parameters() if p.requires_grad}
    print(f"Trainable parameter dtypes: {trainable_dtypes} (promoted {promoted} tensors from low precision to float32)")


def build_smolvla_components(args, ds, device, resume_dir):
    import huggingface_hub.constants as hf_constants
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not args.offline and resume_dir is None:
        snapshot_download(args.smolvla_base)
        tmp_policy = SmolVLAPolicy.from_pretrained(args.smolvla_base)
        del tmp_policy
        gc.collect()
        os.environ["HF_HUB_OFFLINE"] = "1"
        hf_constants.HF_HUB_OFFLINE = True

    if resume_dir is not None:
        print(f"--> Resuming SmolVLA from checkpoint: {resume_dir}")
        cfg = PreTrainedConfig.from_pretrained(resume_dir)
        cfg.pretrained_path = resume_dir
        cfg.device = device
        if hasattr(cfg, "chunk_size") and cfg.chunk_size != args.chunk:
            raise ValueError(
                f"Checkpoint chunk_size ({cfg.chunk_size}) 與 --chunk ({args.chunk}) 不一致，"
                f"請調整 --chunk={cfg.chunk_size} 或使用 --no-resume。"
            )
        policy = make_policy(cfg, ds_meta=ds.meta)
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            pretrained_path=resume_dir,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    else:
        print(f"--> Starting fresh SmolVLA fine-tune from: {args.smolvla_base}")
        cfg = PreTrainedConfig.from_pretrained(args.smolvla_base)
        cfg.pretrained_path = args.smolvla_base
        cfg.device = device
        cfg.input_features = None
        if hasattr(cfg, "chunk_size") and cfg.chunk_size != args.chunk:
            cfg.chunk_size = args.chunk
            if hasattr(cfg, "n_action_steps") and cfg.n_action_steps > args.chunk:
                cfg.n_action_steps = args.chunk
        policy = make_policy(cfg, ds_meta=ds.meta)
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            pretrained_path=args.smolvla_base,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "normalizer_processor": {
                    "features": {**policy.config.input_features, **policy.config.output_features},
                    "norm_map": policy.config.normalization_mapping,
                    "stats": ds.meta.stats,
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "features": policy.config.output_features,
                    "norm_map": policy.config.normalization_mapping,
                    "stats": ds.meta.stats,
                },
            },
        )

    promote_trainable_params_to_fp32(policy)

    lr = args.lr if args.lr is not None else 1e-4
    opt_cfg = policy.config.get_optimizer_preset()
    opt_cfg.lr = lr
    grad_clip_norm = opt_cfg.grad_clip_norm
    optimizer = opt_cfg.build(policy.get_optim_params())

    sched_cfg = policy.config.get_scheduler_preset()
    sched_cfg.peak_lr = lr
    scheduler = sched_cfg.build(optimizer, num_training_steps=args.steps)
    return policy, preprocess, postprocess, optimizer, scheduler, grad_clip_norm


def build_act_components(args, ds, device, resume_dir):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    if resume_dir is not None:
        print(f"--> Resuming ACT from checkpoint: {resume_dir}")
        cfg = PreTrainedConfig.from_pretrained(resume_dir)
        cfg.pretrained_path = resume_dir
        cfg.device = device
        if hasattr(cfg, "chunk_size") and cfg.chunk_size != args.chunk:
            raise ValueError(
                f"Checkpoint chunk_size ({cfg.chunk_size}) 與 --chunk ({args.chunk}) 不一致，"
                f"請調整 --chunk={cfg.chunk_size} 或使用 --no-resume。"
            )
        policy = make_policy(cfg, ds_meta=ds.meta)
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            pretrained_path=resume_dir,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    else:
        print("--> Starting fresh ACT training (ResNet18 + Transformer CVAE)")
        cfg = ACTConfig(
            chunk_size=args.chunk,
            n_action_steps=args.chunk,
            device=device,
        )
        policy = make_policy(cfg, ds_meta=ds.meta)
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            dataset_stats=ds.meta.stats,
        )

    lr = args.lr if args.lr is not None else 1e-5
    opt_cfg = policy.config.get_optimizer_preset()
    opt_cfg.lr = lr
    if hasattr(policy.config, "optimizer_lr"):
        policy.config.optimizer_lr = lr
    if hasattr(policy.config, "optimizer_lr_backbone"):
        policy.config.optimizer_lr_backbone = lr
    grad_clip_norm = getattr(opt_cfg, "grad_clip_norm", args.grad_clip_norm)
    optimizer = opt_cfg.build(policy.get_optim_params())

    sched_cfg = policy.config.get_scheduler_preset()
    if sched_cfg is not None:
        if hasattr(sched_cfg, "peak_lr"):
            sched_cfg.peak_lr = lr
        scheduler = sched_cfg.build(optimizer, num_training_steps=args.steps)
    else:
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                return float(current_step + 1) / float(max(1, args.warmup_steps))
            progress = float(current_step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return policy, preprocess, postprocess, optimizer, scheduler, grad_clip_norm


def build_custom_components(args, ds, device, resume_dir, load_custom_policy_from_checkpoint, FCActionPolicy, CNNActionPolicy):
    import torch

    state_dim = int(ds.meta.shapes["observation.state"][0])
    action_dim = int(ds.meta.shapes["action"][0])
    camera_keys = list(ds.meta.camera_keys)

    if resume_dir is not None:
        print(f"--> Resuming {args.arch.upper()} from checkpoint: {resume_dir}")
        policy = load_custom_policy_from_checkpoint(resume_dir, device)
        if policy.chunk_size != args.chunk:
            raise ValueError(
                f"Checkpoint chunk_size ({policy.chunk_size}) 與 --chunk ({args.chunk}) 不一致，"
                f"請調整 --chunk={policy.chunk_size} 或使用 --no-resume。"
            )
    elif args.arch == "fc":
        print("--> Starting fresh training with FCActionPolicy (MLP)")
        policy = FCActionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=args.chunk,
            camera_keys=camera_keys,
            image_size=(args.fc_image_size, args.fc_image_size),
            use_images=not args.fc_no_images,
            hidden_dims=[1024, 512, 512],
            dropout=args.fc_dropout,
            stats=ds.meta.stats,
        ).to(device)
    else:
        print("--> Starting fresh training with CNNActionPolicy")
        policy = CNNActionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=args.chunk,
            camera_keys=camera_keys,
            image_size=(args.cnn_image_size, args.cnn_image_size),
            conv_channels=[32, 64, 128, 256],
            img_feat_dim=256,
            state_feat_dim=64,
            head_dims=[512, 512],
            share_camera_encoder=args.share_camera_encoder,
            dropout=args.cnn_dropout,
            stats=ds.meta.stats,
        ).to(device)

    lr = args.lr if args.lr is not None else 3e-4
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=args.weight_decay)

    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step + 1) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return policy, None, None, optimizer, scheduler, args.grad_clip_norm


def save_checkpoint(
    arch,
    policy,
    preprocess,
    postprocess,
    optimizer,
    scheduler,
    losses,
    checkpoint_root,
    target_dirs,
    current_step,
    required_files,
    check_processor_states,
    expected_arch,
    build_training_state_payload,
    sync_dir_atomically,
    verify_checkpoint_dir,
):
    import torch

    if isinstance(target_dirs, str):
        target_dirs = [target_dirs]
    staging_dir = os.path.join(checkpoint_root, "_ckpt_staging")
    shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, exist_ok=True)
    state_payload = build_training_state_payload(current_step, optimizer, scheduler, losses)

    try:
        if arch in ("smolvla", "act"):
            policy.save_pretrained(staging_dir)
            preprocess.save_pretrained(staging_dir)
            postprocess.save_pretrained(staging_dir)
        else:
            with open(os.path.join(staging_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(policy.get_config(), f, indent=2, ensure_ascii=False)
            torch.save(policy.state_dict(), os.path.join(staging_dir, "model.pt"))

        torch.save(state_payload, os.path.join(staging_dir, "training_state.pt"))
        if not verify_checkpoint_dir(
            staging_dir,
            required_files,
            check_processor_states=check_processor_states,
            expected_arch=expected_arch,
        ):
            raise RuntimeError(f"暫存 checkpoint 驗證失敗：{staging_dir}")

        for dst_dir in target_dirs:
            sync_dir_atomically(staging_dir, dst_dir)
            if not verify_checkpoint_dir(
                dst_dir,
                required_files,
                check_processor_states=check_processor_states,
                expected_arch=expected_arch,
            ):
                raise RuntimeError(f"目標 checkpoint 驗證失敗：{dst_dir}")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def merge_dataset_stats(stats_list):
    """Merge normalization statistics across multiple LeRobotDatasets by sample count."""
    import copy
    import numpy as np

    merged = {}
    for key in stats_list[0].keys():
        if not all(key in s for s in stats_list):
            continue
        first = stats_list[0][key]
        if "mean" not in first or "std" not in first or "count" not in first:
            merged[key] = copy.deepcopy(first)
            continue
        counts = [np.asarray(s[key]["count"], dtype=np.float64) for s in stats_list]
        means = [np.asarray(s[key]["mean"], dtype=np.float64) for s in stats_list]
        stds = [np.asarray(s[key]["std"], dtype=np.float64) for s in stats_list]
        mins = [np.asarray(s[key]["min"], dtype=np.float64) for s in stats_list]
        maxs = [np.asarray(s[key]["max"], dtype=np.float64) for s in stats_list]

        total_count = np.sum(counts, axis=0)
        safe_count = np.maximum(total_count, 1.0)
        comb_mean = np.sum([c * m for c, m in zip(counts, means)], axis=0) / safe_count
        comb_var = np.sum([c * (s ** 2 + (m - comb_mean) ** 2) for c, m, s in zip(counts, means, stds)], axis=0) / safe_count
        comb_std = np.sqrt(np.maximum(comb_var, 0.0))
        merged[key] = {
            "min": np.min(mins, axis=0),
            "max": np.max(maxs, axis=0),
            "mean": comb_mean,
            "std": comb_std,
            "count": total_count,
        }
    return merged


def main():
    args = parse_args()
    is_lerobot_policy = args.arch in ("smolvla", "act")
    batch_size = args.batch_size if args.batch_size is not None else (16 if is_lerobot_policy else 32)
    validate_args(args, batch_size)

    # Set HF_HUB_OFFLINE before importing lerobot / huggingface_hub
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from local_policies import CNNActionPolicy, FCActionPolicy, load_custom_policy_from_checkpoint
    from local_utils import (
        CUSTOM_REQUIRED_FILES,
        SMOVLA_REQUIRED_FILES,
        build_training_state_payload,
        build_trimmed_subset,
        find_latest_checkpoint,
        recover_atomic_checkpoints,
        resolve_dataset_root,
        restore_rng_state,
        save_loss_plot,
        sync_dir_atomically,
        verify_checkpoint_dir,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch: {torch.__version__} | arch: {args.arch}")

    specified_sources = sum(bool(x) for x in (args.dataset, args.datasets, args.dataset_dir))
    if specified_sources > 1:
        raise ValueError("不可同時指定 --dataset、--datasets 與 --dataset-dir，請擇一使用。")

    if args.datasets:
        dataset_roots = [resolve_dataset_root(PROJECT_ROOT / "datasets" / name) for name in args.datasets]
        output_subdir = "multitask"
    else:
        if args.dataset:
            raw_dataset_dir = PROJECT_ROOT / "datasets" / args.dataset
        elif args.dataset_dir:
            raw_dataset_dir = Path(args.dataset_dir)
        else:
            raw_dataset_dir = PROJECT_ROOT / "dataset"
        single_root = resolve_dataset_root(raw_dataset_dir)
        dataset_roots = [single_root]
        dataset_name = Path(single_root).name
        output_subdir = dataset_name if (args.dataset or Path(single_root).parent.name == "datasets") else None

    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve() / args.arch
    elif output_subdir:
        output_root = PROJECT_ROOT / "outputs" / args.arch / output_subdir
    else:
        output_root = PROJECT_ROOT / "outputs" / args.arch
    checkpoint_root = str(output_root / "checkpoints")
    latest_dir = os.path.join(checkpoint_root, "latest")
    os.makedirs(checkpoint_root, exist_ok=True)

    required_files = SMOVLA_REQUIRED_FILES if is_lerobot_policy else CUSTOM_REQUIRED_FILES
    expected_arch = "fc_mlp" if args.arch == "fc" else ("cnn" if args.arch == "cnn" else None)
    milestone_prefix = f"{args.arch}_pick_step"

    recover_atomic_checkpoints(checkpoint_root)

    ds_pairs = []
    all_tasks = []
    for ds_root_i in dataset_roots:
        name_i = Path(ds_root_i).name
        repo_id_i = args.repo_id or f"local/{name_i}"
        ds_meta_i = LeRobotDatasetMetadata(repo_id_i, root=ds_root_i)
        fps_i = ds_meta_i.fps
        delta_timestamps_i = {"action": [i / fps_i for i in range(args.chunk)]}
        ds_i = LeRobotDataset(
            repo_id_i,
            root=ds_root_i,
            delta_timestamps=delta_timestamps_i,
            video_backend=args.video_backend,
        )
        ds_tasks_i = list(ds_i.meta.tasks.index) if getattr(ds_i.meta, "tasks", None) is not None else []
        all_tasks.extend(ds_tasks_i)
        ds_pairs.append((ds_i, ds_root_i))
        print(f"Dataset root: {ds_root_i} | repo_id: {repo_id_i} | {ds_i.num_episodes} eps, {ds_i.num_frames} frames @ {fps_i} FPS")

    ds = ds_pairs[0][0]
    if len(ds_pairs) > 1:
        ds.meta.stats = merge_dataset_stats([d.meta.stats for d, _ in ds_pairs])
    dataset_tasks = list(dict.fromkeys(all_tasks))
    print(f"Output root: {output_root}")
    print("Dataset cameras:", ds.meta.camera_keys)
    if args.arch == "smolvla" or dataset_tasks:
        print("Dataset tasks:", dataset_tasks)

    if args.no_resume:
        existing = [n for n in os.listdir(checkpoint_root) if not n.startswith("_")]
        if existing:
            print(
                f"Warning: --no-resume is set, ignoring existing checkpoints in {checkpoint_root}: {existing}"
            )
        resume_dir, resume_ckpt = None, None
    else:
        resume_dir, resume_ckpt = find_latest_checkpoint(
            checkpoint_root,
            required_files,
            check_processor_states=is_lerobot_policy,
            expected_arch=expected_arch,
        )

    if args.arch == "smolvla":
        policy, preprocess, postprocess, optimizer, scheduler, grad_clip_norm = build_smolvla_components(
            args, ds, device, resume_dir
        )
    elif args.arch == "act":
        policy, preprocess, postprocess, optimizer, scheduler, grad_clip_norm = build_act_components(
            args, ds, device, resume_dir
        )
    else:
        policy, preprocess, postprocess, optimizer, scheduler, grad_clip_norm = build_custom_components(
            args,
            ds,
            device,
            resume_dir,
            load_custom_policy_from_checkpoint,
            FCActionPolicy,
            CNNActionPolicy,
        )

    print(f"{args.arch.upper()} ready:", round(sum(p.numel() for p in policy.parameters()) / 1e6, 2), "M params")

    start_step = 0
    losses = []
    if resume_ckpt is not None:
        start_step = int(resume_ckpt["step"])
        losses = list(resume_ckpt.get("losses", []))
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        for p in policy.parameters():
            if p in optimizer.state:
                for k, v in optimizer.state[p].items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point() and k != "step" and v.dtype != p.dtype:
                        optimizer.state[p][k] = v.to(dtype=p.dtype)
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        restore_rng_state(resume_ckpt.get("rng_state"))
        print(f"Restored training state at step {start_step}/{args.steps} (loaded {len(losses)} loss records)")

    last_completed_step = start_step
    completed = False
    interrupted = False

    if start_step >= args.steps:
        print(f"Training already completed ({start_step}/{args.steps} steps). Skipping loop.")
        completed = True
    else:
        manual_excludes = [
            int(x.strip()) for x in args.exclude_episodes.split(",") if x.strip()
        ]
        use_filter = (
            (not args.no_trim)
            or args.drop_longest_episodes > 0
            or bool(manual_excludes)
        )
        if use_filter:
            subsets = [
                build_trimmed_subset(
                    ds_i,
                    ds_root_i,
                    move_thresh=args.move_thresh,
                    keep_lead_frames=args.keep_lead_frames,
                    keep_tail_frames=args.keep_tail_frames,
                    trim_head=not args.no_trim,
                    trim_tail=(not args.no_trim and not args.no_trim_tail),
                    drop_longest_n=args.drop_longest_episodes,
                    exclude_episodes=manual_excludes,
                )
                for ds_i, ds_root_i in ds_pairs
            ]
        else:
            subsets = [ds_i for ds_i, _ in ds_pairs]
        train_ds = subsets[0] if len(subsets) == 1 else torch.utils.data.ConcatDataset(subsets)
        if len(train_ds) < batch_size:
            raise ValueError(
                f"可用訓練樣本數 ({len(train_ds)}) 小於 batch_size ({batch_size})，"
                "請調小 --batch-size 或關閉 --no-trim。"
            )
        loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda" and args.num_workers > 0),
            persistent_workers=(args.num_workers > 0),
        )
        if args.arch == "smolvla" or dataset_tasks:
            print(f"Starting training from step {start_step}/{args.steps} | task(s): {dataset_tasks}")
        policy.train()
        it = iter(loader)
        start_t = time.time()

        try:
            for step in range(start_step + 1, args.steps + 1):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)

                if preprocess is not None:
                    batch = preprocess(batch)

                out = policy(batch)
                loss = out[0] if isinstance(out, (tuple, list)) else out["loss"]

                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
                optimizer.step()
                scheduler.step()

                loss_val = loss.detach().item()
                losses.append(loss_val)
                last_completed_step = step
                elapsed_min = (time.time() - start_t) / 60.0

                if step % 200 == 0 or step == start_step + 1:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(f"step {step:5d}/{args.steps} | loss {loss_val:.4f} | lr {cur_lr:.2e} | elapsed {elapsed_min:.1f}m")

                save_targets = []
                if step % args.save_freq == 0 or step == args.steps:
                    save_targets.append(latest_dir)
                if step % args.milestone_freq == 0 or step == args.steps:
                    save_targets.append(os.path.join(checkpoint_root, f"{milestone_prefix}_{step}"))

                if save_targets:
                    save_checkpoint(
                        args.arch,
                        policy,
                        preprocess,
                        postprocess,
                        optimizer,
                        scheduler,
                        losses,
                        checkpoint_root,
                        save_targets,
                        step,
                        required_files,
                        is_lerobot_policy,
                        expected_arch,
                        build_training_state_payload,
                        sync_dir_atomically,
                        verify_checkpoint_dir,
                    )
                    for t_dir in save_targets:
                        print(f"--> Checkpoint synced & verified at step {step}: {t_dir}")
            completed = True
        except KeyboardInterrupt:
            interrupted = True
            if last_completed_step > start_step:
                save_checkpoint(
                    args.arch,
                    policy,
                    preprocess,
                    postprocess,
                    optimizer,
                    scheduler,
                    losses,
                    checkpoint_root,
                    [latest_dir],
                    last_completed_step,
                    required_files,
                    is_lerobot_policy,
                    expected_arch,
                    build_training_state_payload,
                    sync_dir_atomically,
                    verify_checkpoint_dir,
                )
                print(f"\n[Interrupted] Saved & verified progress at step {last_completed_step}/{args.steps} to {latest_dir}")
            else:
                print("\n[Interrupted] Stopped before completing a new step.")
        except Exception:
            if last_completed_step > start_step:
                try:
                    save_checkpoint(
                        args.arch,
                        policy,
                        preprocess,
                        postprocess,
                        optimizer,
                        scheduler,
                        losses,
                        checkpoint_root,
                        [latest_dir],
                        last_completed_step,
                        required_files,
                        is_lerobot_policy,
                        expected_arch,
                        build_training_state_payload,
                        sync_dir_atomically,
                        verify_checkpoint_dir,
                    )
                    print(f"\n[Error Recovery] Saved progress at step {last_completed_step}/{args.steps} to {latest_dir}")
                except Exception as save_err:
                    print(f"\n[Error Recovery] Emergency save failed: {save_err}")
            raise

    if args.arch == "smolvla":
        ylabel = "flow-matching loss"
    elif args.arch == "act":
        ylabel = "ACT loss (L1 + KL)"
    else:
        ylabel = "masked MSE loss (normalized)"
    save_loss_plot(
        losses,
        ylabel=ylabel,
        title=f"{args.arch.upper()} Local Training Loss (SO-101)",
        out_path=str(output_root / "loss_curve.png"),
    )

    if interrupted:
        return

    if not is_lerobot_policy:
        sample = ds[0]
        sample_batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}
        pred_chunk = policy.predict_action(sample_batch)
        print("Sanity check - predicted action chunk shape:", tuple(pred_chunk.shape))
        print("Sanity check - first predicted step:", pred_chunk[0, 0].detach().cpu().numpy().round(3))
    elif args.arch in ("act", "smolvla"):
        policy.eval()
        policy.reset()
        sample = ds[0]
        obs_batch = {
            k: v.unsqueeze(0)
            for k, v in sample.items()
            if k.startswith("observation.") and isinstance(v, torch.Tensor)
        }
        if args.arch == "smolvla":
            obs_batch["task"] = sample.get("task", "Pick up the stylus from the stand")
        with torch.no_grad():
            pred_action = postprocess(policy.select_action(preprocess(obs_batch)))
        print("Sanity check - predicted action shape:", tuple(pred_action.shape))
        print("Sanity check - first predicted step:", pred_action.squeeze(0).detach().cpu().numpy().round(3))

    if completed:
        target_step = max(args.steps, start_step)
        final_milestone = os.path.join(checkpoint_root, f"{milestone_prefix}_{target_step}")
        candidates = [final_milestone, latest_dir]
        if resume_dir is not None:
            candidates.append(resume_dir)
        verify_target = next(
            (
                c
                for c in candidates
                if verify_checkpoint_dir(
                    c,
                    required_files,
                    check_processor_states=is_lerobot_policy,
                    expected_arch=expected_arch,
                )
            ),
            None,
        )
        if verify_target is None:
            raise RuntimeError(f"最終檢查失敗，找不到完整模型檔案：{candidates[0]}")
        weight_name = "model.safetensors" if is_lerobot_policy else "model.pt"
        weights_mb = os.path.getsize(os.path.join(verify_target, weight_name)) / (1024 * 1024)
        print(f"[Verified] 模型已完整儲存至：{verify_target} ({weight_name}: {weights_mb:.1f} MB)")


if __name__ == "__main__":
    main()
