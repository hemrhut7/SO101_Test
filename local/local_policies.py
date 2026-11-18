"""Local training policy architectures (FC/MLP and CNN) for SO-101 action chunking."""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_stat_tensor(stats, key, stat_name, default_val, shape):
    """Extract a statistics tensor from LeRobot metadata stats with fallback defaults."""
    if stats is not None and key in stats and stat_name in stats[key]:
        t = torch.as_tensor(stats[key][stat_name], dtype=torch.float32)
        return t.reshape(shape)
    return torch.full(shape, float(default_val), dtype=torch.float32)


class BaseChunkActionPolicy(nn.Module):
    """Shared normalization, loss computation, and inference wrapper for local policies."""

    def __init__(self, state_dim, action_dim, chunk_size, camera_keys, image_size, stats=None):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.camera_keys = list(camera_keys)
        self.image_size = tuple(image_size)

        self.register_buffer(
            "state_mean",
            _extract_stat_tensor(stats, "observation.state", "mean", 0.0, (1, self.state_dim)),
        )
        self.register_buffer(
            "state_std",
            _extract_stat_tensor(stats, "observation.state", "std", 1.0, (1, self.state_dim)).clamp(min=1e-6),
        )
        self.register_buffer(
            "action_mean",
            _extract_stat_tensor(stats, "action", "mean", 0.0, (1, 1, self.action_dim)),
        )
        self.register_buffer(
            "action_std",
            _extract_stat_tensor(stats, "action", "std", 1.0, (1, 1, self.action_dim)).clamp(min=1e-6),
        )

        for idx, cam in enumerate(self.camera_keys):
            cam_mean = _extract_stat_tensor(stats, cam, "mean", 0.5, (1, 3, 1, 1))
            cam_std = _extract_stat_tensor(stats, cam, "std", 0.5, (1, 3, 1, 1)).clamp(min=1e-6)
            self.register_buffer(f"img_mean_{idx}", cam_mean)
            self.register_buffer(f"img_std_{idx}", cam_std)

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action):
        return (action - self.action_mean) / self.action_std

    def unnormalize_action(self, norm_action):
        return norm_action * self.action_std + self.action_mean

    def _prepare_state(self, batch):
        dev = self.state_mean.device
        state = batch["observation.state"].to(device=dev, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim == 3:
            state = state[:, -1]
        return self.normalize_state(state)

    def _prepare_camera_image(self, batch, idx, cam):
        dev = self.state_mean.device
        raw_img = batch[cam]
        is_uint8 = raw_img.dtype == torch.uint8
        img = raw_img.to(device=dev, dtype=torch.float32)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        elif img.ndim == 5:
            img = img[:, -1]
        if is_uint8:
            img = img / 255.0
        img = F.interpolate(img, size=self.image_size, mode="bilinear", align_corners=False)
        img_mean = getattr(self, f"img_mean_{idx}")
        img_std = getattr(self, f"img_std_{idx}")
        return (img - img_mean) / img_std

    def predict_normalized(self, batch):
        raise NotImplementedError

    @torch.no_grad()
    def predict_action(self, batch):
        """Predict unnormalized action chunk in eval mode without mutating caller state."""
        was_training = self.training
        self.eval()
        try:
            pred_norm = self.predict_normalized(batch)
            return self.unnormalize_action(pred_norm)
        finally:
            self.train(was_training)

    def forward(self, batch):
        pred_norm = self.predict_normalized(batch)
        target = batch["action"].to(device=pred_norm.device, dtype=torch.float32)
        target_norm = self.normalize_action(target)

        loss = F.mse_loss(pred_norm, target_norm, reduction="none")
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            valid_mask = (~action_is_pad.to(device=pred_norm.device, dtype=torch.bool)).unsqueeze(-1)
            loss = (loss * valid_mask).sum() / valid_mask.expand_as(loss).sum().clamp(min=1)
        else:
            loss = loss.mean()
        return {"loss": loss}


class FCActionPolicy(BaseChunkActionPolicy):
    """Pure Fully Connected Network (MLP) action chunking policy."""

    def __init__(
        self,
        state_dim,
        action_dim,
        chunk_size,
        camera_keys,
        image_size=(32, 32),
        use_images=True,
        hidden_dims=(1024, 512, 512),
        dropout=0.1,
        stats=None,
    ):
        super().__init__(state_dim, action_dim, chunk_size, camera_keys, image_size, stats=stats)
        self.use_images = bool(use_images)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)

        img_flat_dim = (
            len(self.camera_keys) * 3 * self.image_size[0] * self.image_size[1]
            if self.use_images
            else 0
        )
        in_dim = self.state_dim + img_flat_dim

        layers = []
        cur_dim = in_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(cur_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            cur_dim = h_dim
        layers.append(nn.Linear(cur_dim, self.chunk_size * self.action_dim))
        self.net = nn.Sequential(*layers)

    def get_config(self):
        return {
            "arch": "fc_mlp",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "camera_keys": self.camera_keys,
            "image_size": list(self.image_size),
            "use_images": self.use_images,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
        }

    def predict_normalized(self, batch):
        feats = [self._prepare_state(batch)]
        if self.use_images:
            for idx, cam in enumerate(self.camera_keys):
                img = self._prepare_camera_image(batch, idx, cam)
                feats.append(img.flatten(start_dim=1))
        x = torch.cat(feats, dim=-1)
        return self.net(x).view(-1, self.chunk_size, self.action_dim)


class ConvVisionEncoder(nn.Module):
    """Multi-layer CNN vision encoder preserving a 4x4 spatial feature grid."""

    def __init__(self, conv_channels=(32, 64, 128, 256), out_dim=256):
        super().__init__()
        blocks = []
        in_ch = 3
        for out_ch in conv_channels:
            blocks.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(self.pool(self.backbone(x)))


class CNNActionPolicy(BaseChunkActionPolicy):
    """Multi-camera CNN vision backbone + state fusion action chunking policy."""

    def __init__(
        self,
        state_dim,
        action_dim,
        chunk_size,
        camera_keys,
        image_size=(128, 128),
        conv_channels=(32, 64, 128, 256),
        img_feat_dim=256,
        state_feat_dim=64,
        head_dims=(512, 512),
        share_camera_encoder=False,
        dropout=0.1,
        stats=None,
    ):
        super().__init__(state_dim, action_dim, chunk_size, camera_keys, image_size, stats=stats)
        self.conv_channels = list(conv_channels)
        self.img_feat_dim = int(img_feat_dim)
        self.state_feat_dim = int(state_feat_dim)
        self.head_dims = list(head_dims)
        self.share_camera_encoder = bool(share_camera_encoder)
        self.dropout = float(dropout)

        if self.share_camera_encoder:
            shared_enc = ConvVisionEncoder(self.conv_channels, self.img_feat_dim)
            self.vision_encoders = nn.ModuleList([shared_enc])
        else:
            self.vision_encoders = nn.ModuleList([
                ConvVisionEncoder(self.conv_channels, self.img_feat_dim)
                for _ in self.camera_keys
            ])

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.state_feat_dim),
            nn.LayerNorm(self.state_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.state_feat_dim, self.state_feat_dim),
            nn.LayerNorm(self.state_feat_dim),
            nn.ReLU(inplace=True),
        )

        fusion_in_dim = len(self.camera_keys) * self.img_feat_dim + self.state_feat_dim
        head_layers = []
        cur_dim = fusion_in_dim
        for h_dim in self.head_dims:
            head_layers.append(nn.Linear(cur_dim, h_dim))
            head_layers.append(nn.LayerNorm(h_dim))
            head_layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0:
                head_layers.append(nn.Dropout(self.dropout))
            cur_dim = h_dim
        head_layers.append(nn.Linear(cur_dim, self.chunk_size * self.action_dim))
        self.action_head = nn.Sequential(*head_layers)

    def get_config(self):
        return {
            "arch": "cnn",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "camera_keys": self.camera_keys,
            "image_size": list(self.image_size),
            "conv_channels": self.conv_channels,
            "img_feat_dim": self.img_feat_dim,
            "state_feat_dim": self.state_feat_dim,
            "head_dims": self.head_dims,
            "share_camera_encoder": self.share_camera_encoder,
            "dropout": self.dropout,
        }

    def predict_normalized(self, batch):
        feats = []
        for idx, cam in enumerate(self.camera_keys):
            img = self._prepare_camera_image(batch, idx, cam)
            enc = self.vision_encoders[0] if self.share_camera_encoder else self.vision_encoders[idx]
            feats.append(enc(img))

        feats.append(self.state_encoder(self._prepare_state(batch)))
        fused = torch.cat(feats, dim=-1)
        return self.action_head(fused).view(-1, self.chunk_size, self.action_dim)


def load_custom_policy_from_checkpoint(ckpt_dir, device):
    """Instantiate FCActionPolicy or CNNActionPolicy from a saved checkpoint directory."""
    with open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    arch = cfg.get("arch")
    if arch == "fc_mlp":
        policy = FCActionPolicy(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            chunk_size=cfg["chunk_size"],
            camera_keys=cfg["camera_keys"],
            image_size=tuple(cfg["image_size"]),
            use_images=cfg["use_images"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
            stats=None,
        )
    elif arch == "cnn":
        policy = CNNActionPolicy(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            chunk_size=cfg["chunk_size"],
            camera_keys=cfg["camera_keys"],
            image_size=tuple(cfg["image_size"]),
            conv_channels=cfg["conv_channels"],
            img_feat_dim=cfg["img_feat_dim"],
            state_feat_dim=cfg["state_feat_dim"],
            head_dims=cfg["head_dims"],
            share_camera_encoder=cfg["share_camera_encoder"],
            dropout=cfg["dropout"],
            stats=None,
        )
    else:
        raise ValueError(f"Unsupported custom checkpoint arch: {arch}")

    policy = policy.to(device)
    state_dict = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device, weights_only=True)
    policy.load_state_dict(state_dict, strict=True)
    return policy
