#! /usr/bin/env python
"""CosPlaceExtractor's new `model=` hook (M2-1a step 1).

Verifies the query-side preprocessing (ToTensor -> Resize(512,512) -> ImageNet
normalize) is unchanged and that passing a model skips initialize_vpr_model,
so the map-building side (sgm) and the online query side stay in the same
embedding space.

Run: pytest tests/test_vpr_descriptor_model.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "third_party" / "litevloc_code" / "python"))

from vpr_descriptor import CosPlaceExtractor  # noqa: E402

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _RecordingModel(nn.Module):
    """Fake VPR model: records its input, returns a fixed 256-d descriptor."""

    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return torch.arange(256, dtype=torch.float32).unsqueeze(0)


def test_model_hook_preprocessing_matches_sgm_keyframe_side():
    model = _RecordingModel()
    extractor = CosPlaceExtractor(device="cpu", model=model)

    white_rgb = np.full((480, 640, 3), 255, dtype=np.uint8)
    desc = extractor.extract(white_rgb)

    assert model.last_input is not None
    assert tuple(model.last_input.shape) == (1, 3, 512, 512)

    expected = (1.0 - _IMAGENET_MEAN) / _IMAGENET_STD  # white pixel, ToTensor -> 1.0, then normalized
    for c in range(3):
        np.testing.assert_allclose(
            model.last_input[0, c].numpy(), np.full((512, 512), expected[c], dtype=np.float32), atol=1e-4
        )

    assert desc.shape == (256,)
    assert desc.dtype == np.float32


def test_model_hook_skips_initialize_vpr_model(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("initialize_vpr_model must not be called when model= is given")

    monkeypatch.setitem(
        sys.modules,
        "utils.utils_vpr_method",
        type(sys)("utils.utils_vpr_method"),
    )
    sys.modules["utils.utils_vpr_method"].initialize_vpr_model = _fail

    model = _RecordingModel()
    extractor = CosPlaceExtractor(device="cpu", model=model)
    desc = extractor.extract(np.zeros((16, 16, 3), dtype=np.uint8))
    assert desc.shape == (256,)
