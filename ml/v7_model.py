"""Landmark models that share the browser's fixed 48 x 224 input contract."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from train import TemporalClassifier, fit_normalization


def coordinate_mask(x):
    left = x[:, :, 220:221] >= .999
    right = x[:, :, 221:222] >= .999
    pose = x[:, :, 222:223] >= .999
    blocks = ((left & pose, 42), (left, 42), (right & pose, 42), (right, 42), (pose, 18))
    if isinstance(x, torch.Tensor):
        return torch.cat([mask.expand(-1, -1, width) for mask, width in blocks], dim=2)
    return np.concatenate([np.repeat(mask, width, axis=2) for mask, width in blocks], axis=2)


def fit_motion_normalization(x):
    values = x[:, :, :186].astype(np.float64)
    valid = coordinate_mask(x)
    n = valid.sum(axis=(0, 1)).clip(min=1)
    mean = (values * valid).sum(axis=(0, 1)) / n
    variance = (np.square(values - mean) * valid).sum(axis=(0, 1)) / n
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), .025).astype(np.float32)


class MaskedMotionClassifier(nn.Module):
    """Encode hand/pose positions and changes between genuinely observed points."""

    def __init__(self, mean, std, channels=40):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean)[None, None, :])
        self.register_buffer('std', torch.as_tensor(std)[None, None, :])
        self.project = nn.Conv1d(372, channels, 1)
        self.temporal = nn.ModuleList(nn.Conv1d(channels, channels, 5, padding=2 * d, dilation=d) for d in (1, 2, 4))
        self.dropout = nn.Dropout(.12)
        self.classifier = nn.Linear(channels * 2, 4)

    def forward(self, landmarks):
        mask = coordinate_mask(landmarks)
        position = torch.clamp((landmarks[:, :, :186] - self.mean) / self.std, -8., 8.) * mask
        velocity = (position[:, 1:] - position[:, :-1]) * (mask[:, 1:] & mask[:, :-1])
        velocity = torch.cat((torch.zeros_like(position[:, :1]), velocity), dim=1)
        signal = self.project(torch.cat((position, velocity), dim=2).transpose(1, 2)).relu()
        for convolution in self.temporal:
            signal = (signal + self.dropout(convolution(signal).relu())) * .70710678118
        return self.classifier(torch.cat((signal.mean(dim=2), signal.amax(dim=2)), dim=1))


def make_model(architecture, fit_x):
    if architecture == 'tcn':
        mean, std, keep, _ = fit_normalization(fit_x, 'signbridge-landmarks-xy-v1')
        return TemporalClassifier(mean, std, 4, 40, keep)
    if architecture == 'masked_motion':
        return MaskedMotionClassifier(*fit_motion_normalization(fit_x))
    raise ValueError(f'Unknown architecture {architecture}')
