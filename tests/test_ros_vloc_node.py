#! /usr/bin/env python
"""ros_vloc_node.py: offline CosPlace loading + the desc_fn hook (M2-1a step 1).

Run: pytest tests/test_ros_vloc_node.py
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "third_party" / "litevloc_code" / "python"))

import ros_vloc_node  # noqa: E402


def test_load_cosplace_local_raises_when_cache_missing(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match=str(tmp_path / "gmberton_cosplace_main")):
        ros_vloc_node.load_cosplace_local("ResNet18", 256, "cpu")


def test_make_desc_fn_rejects_non_cosplace_method():
    args = SimpleNamespace(vpr_method="netvlad", vpr_backbone="VGG16", vpr_descriptors_dimension=4096, device="cpu")
    with pytest.raises(RuntimeError, match="netvlad"):
        ros_vloc_node.make_desc_fn(args)


def test_make_desc_fn_returns_working_extractor(monkeypatch):
    captured = {}

    def fake_load_cosplace_local(backbone, dim, device):
        captured["backbone"] = backbone
        captured["dim"] = dim
        captured["device"] = device
        return _FakeModel()

    monkeypatch.setattr(ros_vloc_node, "load_cosplace_local", fake_load_cosplace_local)

    args = SimpleNamespace(vpr_method="cosplace", vpr_backbone="ResNet18", vpr_descriptors_dimension=256, device="cpu")
    desc_fn = ros_vloc_node.make_desc_fn(args)

    assert captured == {"backbone": "ResNet18", "dim": 256, "device": "cpu"}

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    desc = desc_fn(rgb)
    assert desc.shape == (256,)
    assert desc.dtype == np.float32


class _FakeModel:
    """Stands in for a torch.nn.Module without importing torch at module scope."""

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        import torch

        return torch.arange(256, dtype=torch.float32).unsqueeze(0)
