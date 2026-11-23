"""SmolVLA policy pool with zero-latency / low-VRAM hot-swapping for multi-stage tasks.

Supports three preload modes tailored to different deployment hardware:
- ``gpu_full``: Keeps all unique SmolVLA models resident on GPU (< 0.05 ms pointer switch).
  Ideal for RTX 3060 (12 GB VRAM) and AMD Ryzen AI Max+ 395.
- ``gpu_experts``: Keeps one shared frozen VLM backbone on GPU + all trainable expert
  weight sets in VRAM (< 2 ms device-to-device copy).
- ``cpu_pinned``: Keeps one shared frozen VLM backbone on GPU + expert weight sets in
  CPU pinned RAM (~15-25 ms H2D copy). Fallback for <= 4-6 GB VRAM laptops.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deploy_local import PolicyRunner, detect_arch  # noqa: E402
from local_utils import DEPLOY_LEROBOT_REQUIRED_FILES, verify_checkpoint_dir  # noqa: E402

VALID_PRELOAD_MODES = ("auto", "gpu_full", "gpu_experts", "cpu_pinned")


def detect_optimal_preload_mode(device: str, requested_mode: str = "auto") -> str:
    """Resolve ``preload_mode`` from user request and available GPU VRAM."""
    if requested_mode not in VALID_PRELOAD_MODES:
        raise ValueError(
            f"Unsupported preload_mode={requested_mode!r}; expected one of {VALID_PRELOAD_MODES}."
        )
    if requested_mode != "auto":
        return requested_mode

    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gb >= 8.0:
                return "gpu_full"
            if total_vram_gb >= 3.5:
                return "gpu_experts"
        except Exception:
            pass
    return "cpu_pinned"


class MockPolicyRunner:
    """Lightweight mock runner for offline dry-run verification without external checkpoints."""

    def __init__(
        self,
        stage_key: str,
        ckpt_dir: str,
        device: str = "cpu",
        n_action_steps: int = 35,
        action_ema: float = 1.0,
    ) -> None:
        self.stage_key = stage_key
        self.arch = "smolvla"
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.n_action_steps = n_action_steps
        self.action_ema = action_ema
        self.camera_keys = [
            "observation.images.wrist",
            "observation.images.side_45",
        ]
        self.state_dim = 6
        self.action_dim = 6
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def select_action(self, obs_frame: dict[str, Any]) -> torch.Tensor:
        self._step += 1
        state = obs_frame.get("observation.state")
        if isinstance(state, torch.Tensor):
            base = state.detach().clone().to(dtype=torch.float32)
            if base.ndim == 1:
                base = base.unsqueeze(0)
        else:
            base = torch.zeros((1, self.state_dim), dtype=torch.float32)

        action = base.clone()
        # Simulate smooth joint motion + realistic gripper open/close behavior
        if "pick" in self.stage_key:
            # Close gripper after initial approach
            action[:, 5] = 8.0 if self._step > 3 else 45.0
            action[:, 1] = -10.0 + min(self._step, 10) * 1.5
        elif "put" in self.stage_key:
            # Hold closed during transport, then open gripper to drop into cup
            action[:, 5] = 48.0 if self._step >= 5 else 8.0
            action[:, 0] = 20.0
        elif "shake" in self.stage_key:
            # Periodic wrist oscillation
            sign = 1.0 if (self._step % 2 == 0) else -1.0
            action[:, 4] = 15.0 * sign
        return action


class SmolVLAPolicyPool:
    """Preload and rapidly switch among multiple SmolVLA task policies."""

    def __init__(
        self,
        checkpoint_map: dict[str, str],
        device: str = "cuda",
        preload_mode: str = "auto",
        n_action_steps: int = 35,
        action_ema: float = 1.0,
        mock_models: bool = False,
    ) -> None:
        if not checkpoint_map:
            raise ValueError("checkpoint_map cannot be empty.")

        self.device = device
        self.mock_models = bool(mock_models)
        self.preload_mode = detect_optimal_preload_mode(device, preload_mode)
        self.n_action_steps = int(n_action_steps)
        self.action_ema = float(action_ema)

        # Resolve checkpoint paths and deduplicate identical checkpoints (e.g. put_cup used twice)
        self.key_to_path: dict[str, str] = {}
        self.unique_paths: list[str] = []
        seen_paths: set[str] = set()
        for key, raw_path in checkpoint_map.items():
            resolved = str(Path(raw_path).expanduser().resolve())
            self.key_to_path[key] = resolved
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                self.unique_paths.append(resolved)

        self.active_key: str | None = None
        self.active_runner: PolicyRunner | MockPolicyRunner | None = None

        # Mode-specific storage
        self._runners_by_path: dict[str, PolicyRunner | MockPolicyRunner] = {}
        self._resident_runner: PolicyRunner | None = None
        self._expert_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._processors_cache: dict[str, tuple[Any, Any]] = {}

        if self.mock_models:
            self._init_mock_pool()
        else:
            self._validate_all_checkpoints()
            if self.preload_mode == "gpu_full":
                self._init_gpu_full_pool()
            else:
                self._init_shared_backbone_pool()

        first_key = next(iter(self.key_to_path))
        self.activate(first_key)

    @property
    def camera_keys(self) -> list[str]:
        assert self.active_runner is not None
        return list(self.active_runner.camera_keys)

    @property
    def state_dim(self) -> int:
        assert self.active_runner is not None
        return int(self.active_runner.state_dim)

    def _init_mock_pool(self) -> None:
        for key, ckpt_path in self.key_to_path.items():
            if ckpt_path not in self._runners_by_path:
                self._runners_by_path[ckpt_path] = MockPolicyRunner(
                    stage_key=key,
                    ckpt_dir=ckpt_path,
                    device=self.device,
                    n_action_steps=self.n_action_steps,
                    action_ema=self.action_ema,
                )

    def _validate_all_checkpoints(self) -> None:
        for ckpt_dir in self.unique_paths:
            arch = detect_arch(ckpt_dir, "auto")
            if arch != "smolvla":
                raise ValueError(
                    f"SmolVLAPolicyPool 僅支援 SmolVLA 模型，但 {ckpt_dir} 為 {arch!r}。"
                )
            if not verify_checkpoint_dir(
                ckpt_dir,
                DEPLOY_LEROBOT_REQUIRED_FILES,
                check_processor_states=True,
                expected_type="smolvla",
            ):
                raise FileNotFoundError(
                    f"SmolVLA checkpoint 驗證失敗或不完整：{ckpt_dir}"
                )

    def _init_gpu_full_pool(self) -> None:
        base_cameras: list[str] | None = None
        base_state_dim: int | None = None
        for ckpt_dir in self.unique_paths:
            runner = PolicyRunner(
                arch="smolvla",
                ckpt_dir=ckpt_dir,
                device=self.device,
                n_action_steps=self.n_action_steps,
                action_ema=self.action_ema,
            )
            if base_cameras is None:
                base_cameras = list(runner.camera_keys)
                base_state_dim = runner.state_dim
            else:
                if list(runner.camera_keys) != base_cameras or runner.state_dim != base_state_dim:
                    raise ValueError(
                        f"Checkpoint 相機或狀態維度不一致：{ckpt_dir} "
                        f"({runner.camera_keys}, state_dim={runner.state_dim}) vs "
                        f"({base_cameras}, state_dim={base_state_dim})"
                    )
            self._runners_by_path[ckpt_dir] = runner

    def _init_shared_backbone_pool(self) -> None:
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.torch import load_file

        base_ckpt = self.unique_paths[0]
        self._resident_runner = PolicyRunner(
            arch="smolvla",
            ckpt_dir=base_ckpt,
            device=self.device,
            n_action_steps=self.n_action_steps,
            action_ema=self.action_ema,
        )
        base_cameras = list(self._resident_runner.camera_keys)
        base_state_dim = self._resident_runner.state_dim
        base_sd = load_file(os.path.join(base_ckpt, "model.safetensors"), device="cpu")

        target_param_names = {name for name, _ in self._resident_runner.policy.named_parameters()}
        target_buffer_names = {name for name, _ in self._resident_runner.policy.named_buffers()}
        known_names = target_param_names | target_buffer_names

        pin_cpu = self.preload_mode == "cpu_pinned" and str(self.device).startswith("cuda") and torch.cuda.is_available()
        move_to_gpu = self.preload_mode == "gpu_experts"

        for ckpt_dir in self.unique_paths:
            if ckpt_dir == base_ckpt:
                self._processors_cache[ckpt_dir] = (
                    self._resident_runner.preprocess,
                    self._resident_runner.postprocess,
                )
                ckpt_sd = base_sd
            else:
                # Validate feature compatibility & build dedicated pre/post processors
                from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

                cfg = SmolVLAConfig.from_pretrained(ckpt_dir)
                cam_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
                st_dim = cfg.input_features["observation.state"].shape[0]
                if cam_keys != base_cameras or st_dim != base_state_dim:
                    raise ValueError(
                        f"Checkpoint 相機或狀態維度不一致：{ckpt_dir} "
                        f"({cam_keys}, state_dim={st_dim}) vs ({base_cameras}, state_dim={base_state_dim})"
                    )
                pre, post = make_pre_post_processors(
                    cfg,
                    pretrained_path=ckpt_dir,
                    preprocessor_overrides={"device_processor": {"device": str(self.device)}},
                )
                self._processors_cache[ckpt_dir] = (pre, post)
                ckpt_sd = load_file(os.path.join(ckpt_dir, "model.safetensors"), device="cpu")

            cached_tensors: dict[str, torch.Tensor] = {}
            for k, tensor in ckpt_sd.items():
                if k not in known_names:
                    continue
                is_vlm = k.startswith("model.vlm_with_expert.vlm.")
                if is_vlm and ckpt_dir != base_ckpt:
                    # Only cache VLM weights if this checkpoint modified frozen VLM layers
                    base_t = base_sd.get(k)
                    if base_t is not None and torch.equal(tensor, base_t):
                        continue
                elif is_vlm and ckpt_dir == base_ckpt:
                    # Check lazily if any other checkpoint changed this VLM key
                    continue

                t_cpu = tensor.detach().contiguous()
                if move_to_gpu:
                    cached_tensors[k] = t_cpu.to(self.device)
                elif pin_cpu:
                    cached_tensors[k] = t_cpu.pin_memory()
                else:
                    cached_tensors[k] = t_cpu

            self._expert_cache[ckpt_dir] = cached_tensors

        # Ensure any VLM tensor modified in any secondary checkpoint is also cached in base_ckpt
        if len(self.unique_paths) > 1:
            extra_vlm_keys: set[str] = set()
            for ckpt_dir in self.unique_paths[1:]:
                for k in self._expert_cache[ckpt_dir]:
                    if k.startswith("model.vlm_with_expert.vlm."):
                        extra_vlm_keys.add(k)
            for k in extra_vlm_keys:
                t_cpu = base_sd[k].detach().contiguous()
                if move_to_gpu:
                    self._expert_cache[base_ckpt][k] = t_cpu.to(self.device)
                elif pin_cpu:
                    self._expert_cache[base_ckpt][k] = t_cpu.pin_memory()
                else:
                    self._expert_cache[base_ckpt][k] = t_cpu

    def warmup(self, task_text: str = "Pick up the cube") -> None:
        """Run a 1-step synthetic forward pass on each resident policy to compile GPU kernels."""
        if self.mock_models:
            return
        runners = (
            list(self._runners_by_path.values())
            if self.preload_mode == "gpu_full"
            else ([self._resident_runner] if self._resident_runner is not None else [])
        )
        if not runners:
            return

        cam_keys = runners[0].camera_keys
        st_dim = runners[0].state_dim
        dummy_obs: dict[str, Any] = {
            "observation.state": torch.zeros((1, st_dim), dtype=torch.float32, device=self.device),
            "task": task_text,
        }
        for cam_key in cam_keys:
            dummy_obs[cam_key] = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=self.device)

        for r in runners:
            r.reset()
            r.select_action(dummy_obs)
            r.reset()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def activate(self, stage_key: str) -> tuple[PolicyRunner | MockPolicyRunner, float]:
        """Switch active policy to ``stage_key`` and return ``(runner, switch_ms)``."""
        if stage_key not in self.key_to_path:
            raise KeyError(f"Unknown stage_key={stage_key!r}; available={list(self.key_to_path.keys())}")

        ckpt_dir = self.key_to_path[stage_key]
        t0 = time.perf_counter()

        if self.mock_models or self.preload_mode == "gpu_full":
            runner = self._runners_by_path[ckpt_dir]
            runner.reset()
            self.active_runner = runner
            self.active_key = stage_key
            switch_ms = (time.perf_counter() - t0) * 1000.0
            return runner, switch_ms

        assert self._resident_runner is not None
        prev_ckpt = self.key_to_path.get(self.active_key) if self.active_key else None
        if prev_ckpt != ckpt_dir:
            named_params = dict(self._resident_runner.policy.named_parameters())
            named_buffers = dict(self._resident_runner.policy.named_buffers())
            tensors = self._expert_cache[ckpt_dir]
            with torch.no_grad():
                for name, src in tensors.items():
                    if name in named_params:
                        named_params[name].data.copy_(src, non_blocking=True)
                    elif name in named_buffers:
                        named_buffers[name].data.copy_(src, non_blocking=True)
            pre, post = self._processors_cache[ckpt_dir]
            self._resident_runner.preprocess = pre
            self._resident_runner.postprocess = post
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

        self._resident_runner.reset()
        self.active_runner = self._resident_runner
        self.active_key = stage_key
        switch_ms = (time.perf_counter() - t0) * 1000.0
        return self._resident_runner, switch_ms

    def get_vram_mb(self) -> dict[str, float]:
        """Return current and peak CUDA/ROCm VRAM allocation in MB."""
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "peak_mb": 0.0, "reserved_mb": 0.0}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "peak_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        }
