#! /usr/bin/env python
"""Thin entry point: run litevloc's ros_loc_pipeline with the opennavmap CosPlace
descriptor hook, so the online query side and the mapping side (sgm) compute VPR
descriptors with the exact same model and preprocessing.

Loads the CosPlace weights from the local torch.hub cache (no network access);
see load_cosplace_local below for why.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

_PY = Path(__file__).resolve().parent  # opennavmap/python
_LITEVLOC_PY = _PY.parent / "third_party" / "litevloc_code" / "python"
for _p in (str(_PY), str(_LITEVLOC_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vpr_descriptor import CosPlaceExtractor  # noqa: E402


def load_cosplace_local(backbone: str, dim: int, device: str) -> Any:
    """Load CosPlace from the local torch.hub cache only -- never touches the network.

    `torch.hub.load("gmberton/cosplace", ...)` first queries GitHub for the repo's
    default branch; without internet (or behind a broken proxy) that call hangs or
    errors. The weights and repo code are already in the torch.hub cache on this
    machine (scene_graph_mapping's perception/embedders/vpr.py loads the same way),
    so load straight from there with source="local".
    """
    import torch

    cache_dir = Path(torch.hub.get_dir()) / "gmberton_cosplace_main"
    if not cache_dir.is_dir():
        raise RuntimeError(
            f"CosPlace torch.hub cache not found at {cache_dir}. "
            'Run `torch.hub.load("gmberton/cosplace", "get_trained_model")` once on a '
            "machine with network access to populate this cache, then copy it over."
        )
    return torch.hub.load(
        str(cache_dir), "get_trained_model", backbone=backbone, fc_output_dim=int(dim), source="local"
    )


def make_desc_fn(args: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
    """Build the ros_loc_pipeline desc_fn hook from parsed CLI args."""
    if args.vpr_method != "cosplace":
        raise RuntimeError(
            f"ros_vloc_node only supports --vpr_method cosplace (offline loading), got: {args.vpr_method}"
        )
    model = load_cosplace_local(args.vpr_backbone, args.vpr_descriptors_dimension, args.device)
    extractor = CosPlaceExtractor(
        method=args.vpr_method,
        backbone=args.vpr_backbone,
        descriptors_dimension=args.vpr_descriptors_dimension,
        device=args.device,
        model=model,
    )
    return extractor.extract


if __name__ == "__main__":
    import ros_loc_pipeline

    ros_loc_pipeline.main(make_desc_fn=make_desc_fn)
