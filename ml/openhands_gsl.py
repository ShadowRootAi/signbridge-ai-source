"""Standalone inference for AI4Bharat OpenHands Greek isolated-sign weights.

Source and model attribution: Selvaraj et al., ACL 2022, https://aclanthology.org/2022.acl-long.150/.
The compatible tensor operations below follow AI4Bharat/OpenHands (Apache 2.0).
"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import nn


LSTM_ARCHIVE = "gsl_lstm.zip"
METADATA_ARCHIVE = "gsl_metadata.zip"
LSTM_SHA256 = "eea777414207393db225f24e6e6377746114e019f6471d73998c6ae564c440ff"
METADATA_SHA256 = "c25babb27efe114b92832f49b110763d8e641aca2cf6432c98ff5369ce787597"
KEYPOINTS = (0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74)


class _LegacyTrainingCallback:
    """Inert stand-in for class names in this release's training metadata."""


class _TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)

    def forward(self, hidden):
        last = hidden[:, -1]
        scores = torch.bmm(self.fc1(hidden), last.unsqueeze(-1)).squeeze(-1)
        context = torch.bmm(scores.softmax(dim=-1).unsqueeze(1), hidden).squeeze(1)
        return self.fc2(torch.cat((context, last), dim=-1)).tanh()


class OpenHandsGreekLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # The upstream model passed [B,T,54] to its default (batch_first=False)
        # LSTM. Preserve the checkpoint's semantics and score clips one by one.
        self.rnn = nn.LSTM(54, 128, num_layers=4, bidirectional=True)
        self.attn_block = _TemporalAttention()
        self.fc = nn.Linear(256, 310)

    def forward(self, clips):
        """Return [B,310] logits for preprocessed [B,T,54] clips.

        Use B=1 for independent predictions. B>1 mixes examples as upstream did.
        """
        hidden, _ = self.rnn(clips)
        return self.fc(self.attn_block(hidden))


def prepare_pose(keypoints):
    """Convert [T,75,2 or 3] (body, left hand, right hand) to [T,54]."""
    pose = torch.as_tensor(keypoints, dtype=torch.float32)
    if pose.ndim != 3 or pose.shape[0] < 1 or pose.shape[1] != 75 or pose.shape[2] not in (2, 3):
        raise ValueError("OpenHands expects nonempty [time,75,2 or 3] landmarks")
    selected = pose[:, KEYPOINTS, :2]
    if not torch.isfinite(selected).all():
        raise ValueError("OpenHands input contains nonfinite selected landmarks")

    left, right = selected[:, 3], selected[:, 4]
    mean_width = ((left - right).square().sum(dim=-1).sqrt()).mean()
    if mean_width > 0:
        center = ((left + right) / 2).mean(dim=0)
        selected = (selected - center) * (1 / mean_width)
    return selected.transpose(1, 2).reshape(-1, 54)


def _read_verified_member(path, expected_sha256, member):
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Unexpected OpenHands archive SHA-256: {path}")
    with ZipFile(BytesIO(payload)) as archive:
        return archive.read(member)


def load_openhands_gsl(cache_dir):
    """Load the two verified release archives. Return (eval model, gloss labels)."""
    cache = Path(cache_dir)
    labels_data = _read_verified_member(cache / METADATA_ARCHIVE, METADATA_SHA256, "GSL_split/GSL_isolated/iso_classes.csv")
    labels = tuple(line.strip() for line in labels_data.decode("utf-8").splitlines() if line.strip())
    if len(labels) != 310 or len(set(labels)) != 310:
        raise ValueError("The official OpenHands Greek class map must contain 310 distinct labels")

    weights = _read_verified_member(cache / LSTM_ARCHIVE, LSTM_SHA256, "gsl/lstm/epoch=107-step=118043.ckpt")
    callback_globals = [
        (_LegacyTrainingCallback, "pytorch_lightning.callbacks.early_stopping.EarlyStopping"),
        (_LegacyTrainingCallback, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"),
    ]
    with torch.serialization.safe_globals(callback_globals):
        checkpoint = torch.load(BytesIO(weights), map_location="cpu", weights_only=True)
    prefix = "model.decoder."
    if any(not name.startswith(prefix) for name in checkpoint["state_dict"]):
        raise ValueError("Unexpected parameters in the official OpenHands Greek checkpoint")
    model = OpenHandsGreekLSTM()
    model.load_state_dict({name.removeprefix(prefix): value for name, value in checkpoint["state_dict"].items()}, strict=True)
    return model.eval(), labels
