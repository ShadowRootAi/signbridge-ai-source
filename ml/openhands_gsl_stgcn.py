"""Local inference for the released OpenHands Greek ST-GCN checkpoint.

Network and preprocessing follow AI4Bharat/OpenHands:
https://github.com/AI4Bharat/OpenHands/blob/main/openhands/models/encoder/graph/st_gcn.py
https://github.com/AI4Bharat/OpenHands/blob/main/openhands/datasets/pose_transforms.py
"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import nn


STGCN_ARCHIVE_SHA256 = "13f5d52bb052da8e6496f69c9bea810342f02495e1495753683a6f789d35c97a"
METADATA_ARCHIVE_SHA256 = "c25babb27efe114b92832f49b110763d8e641aca2cf6432c98ff5369ce787597"
POSE_POINTS = (0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
               58, 59, 62, 63, 66, 67, 70, 71, 74)


class _CheckpointCallbackMetadata:
    """Inert stand-in for a serialized Lightning callback key."""


class _EarlyStoppingMetadata:
    """Inert stand-in for a serialized Lightning callback key."""


class _GraphConvolution(nn.Module):
    def __init__(self, input_width, output_width, partitions):
        super().__init__()
        self.output_width = output_width
        self.partitions = partitions
        self.conv = nn.Conv2d(input_width, output_width * partitions, 1)

    def forward(self, x, adjacency):
        projected = self.conv(x)
        batch, _, frames, joints = projected.shape
        projected = projected.reshape(batch, self.partitions, self.output_width, frames, joints)
        return torch.einsum("nkctv,kvw->nctw", projected, adjacency).contiguous()


class _TemporalGraphBlock(nn.Module):
    def __init__(self, input_width, output_width, partitions, stride, first):
        super().__init__()
        self.gcn = _GraphConvolution(input_width, output_width, partitions)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(output_width),
            nn.ReLU(),
            nn.Conv2d(output_width, output_width, (9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(output_width),
        )
        if first:
            self.residual = None
        elif stride == 1 and input_width == output_width:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(input_width, output_width, 1, stride=(stride, 1)),
                nn.BatchNorm2d(output_width),
            )

    def forward(self, x, adjacency):
        temporal = self.tcn(self.gcn(x, adjacency))
        if self.residual is not None:
            temporal = temporal + self.residual(x)
        return torch.relu(temporal)


class _Encoder(nn.Module):
    def __init__(self, adjacency):
        super().__init__()
        self.register_buffer("A", adjacency)
        self.data_bn = nn.BatchNorm1d(54)
        widths = (2, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256)
        self.st_gcn_networks = nn.ModuleList([
            _TemporalGraphBlock(start, end, len(adjacency), 2 if index in (4, 7) else 1, index == 0)
            for index, (start, end) in enumerate(zip(widths, widths[1:]))
        ])
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones_like(adjacency)) for _ in self.st_gcn_networks
        ])

    def forward(self, x):
        batch, channels, frames, joints = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, joints * channels, frames)
        x = self.data_bn(x).reshape(batch, joints, channels, frames).permute(0, 2, 3, 1).contiguous()
        for block, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = block(x, self.A * importance)
        return x.mean(dim=(2, 3))


class _STGCN(nn.Module):
    def __init__(self, adjacency, classes):
        super().__init__()
        self.encoder = _Encoder(adjacency)
        self.decoder = nn.Sequential()
        self.decoder.add_module("classifier", nn.Linear(256, classes))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def prepare_stgcn_pose(raw):
    """Convert original MediaPipe (T, 75, 3) to OpenHands (1, 2, T, 27)."""
    keypoints = torch.as_tensor(raw, dtype=torch.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (75, 3) or keypoints.shape[0] == 0:
        raise ValueError("Expected a nonempty T × 75 × 3 MediaPipe clip")
    xy = keypoints[:, POSE_POINTS, :2]
    if not torch.isfinite(xy).all():
        raise ValueError("OpenHands pose contains non-finite coordinates")
    left_shoulder, right_shoulder = xy[:, 3], xy[:, 4]
    scale = (left_shoulder - right_shoulder).norm(dim=1).mean()
    if scale > 0:
        center = ((left_shoulder + right_shoulder) / 2).mean(dim=0)
        xy = (xy - center) / scale
    return xy.permute(2, 0, 1).unsqueeze(0).contiguous()


def load_openhands_gsl_stgcn(cache_dir):
    """Load the verified public checkpoint with no executable pickle fallback."""
    cache_dir = Path(cache_dir)
    checkpoint_archive = (cache_dir / "gsl_stgcn.zip").read_bytes()
    if sha256(checkpoint_archive).hexdigest() != STGCN_ARCHIVE_SHA256:
        raise ValueError("OpenHands ST-GCN archive checksum mismatch")
    with ZipFile(BytesIO(checkpoint_archive)) as archive:
        checkpoint = BytesIO(archive.read("gsl/st_gcn/epoch=52-step=57981.ckpt"))
    callbacks = [
        (_CheckpointCallbackMetadata, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"),
        (_EarlyStoppingMetadata, "pytorch_lightning.callbacks.early_stopping.EarlyStopping"),
    ]
    with torch.serialization.safe_globals(callbacks):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    if not all(key.startswith("model.") and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Unexpected ST-GCN checkpoint entries")
    state = {key.removeprefix("model."): value for key, value in state.items()}

    metadata_archive = (cache_dir / "gsl_metadata.zip").read_bytes()
    if sha256(metadata_archive).hexdigest() != METADATA_ARCHIVE_SHA256:
        raise ValueError("OpenHands GSL metadata checksum mismatch")
    with ZipFile(BytesIO(metadata_archive)) as archive:
        class_text = archive.read("GSL_split/GSL_isolated/iso_classes.csv").decode("utf-8")
    labels = [line.strip() for line in class_text.splitlines() if line.strip()]
    if len(labels) != 310 or len(set(labels)) != 310:
        raise ValueError("Expected 310 distinct OpenHands Greek glosses in checkpoint order")
    model = _STGCN(state["encoder.A"], len(labels))
    model.load_state_dict(state, strict=True)
    return model.eval(), labels
