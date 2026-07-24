"""Shared CosPlace VPR descriptor extractor (map-realm, opennavmap fork).

Single source of truth for the keyframe/query VPR preprocessing so the build side
(``sim_builder``) and the query side (``core/goal/image_goal``) produce descriptors
in the SAME embedding space -- otherwise stored-descriptor retrieval breaks.

Heavy imports (torch/torchvision + the litevloc VPR model) are deferred to
``__init__``/methods so importing this module stays cheap.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import numpy as np

_ONM = Path(__file__).resolve().parent.parent  # third_party/opennavmap
_LITEVLOC_PY = _ONM / "third_party" / "litevloc_code" / "python"


class CosPlaceExtractor:
    """CosPlace VPR feature extractor: HxWx3 uint8 RGB -> 1-D float32 descriptor."""

    def __init__(
        self,
        method: str = "cosplace",
        backbone: str = "ResNet18",
        descriptors_dimension: int = 256,
        device: str = "cuda",
    ) -> None:
        if str(_LITEVLOC_PY) not in sys.path:
            sys.path.insert(0, str(_LITEVLOC_PY))
        import torchvision.transforms as tvf
        from utils.utils_vpr_method import initialize_vpr_model

        self.device = device
        self.descriptors_dimension = int(descriptors_dimension)
        self._model = initialize_vpr_model(method, backbone, self.descriptors_dimension, device)
        # ImageNet-normalized 512x512 -- MUST match both build and query sides.
        self._transform = tvf.Compose(
            [
                tvf.ToTensor(),
                tvf.Resize((512, 512), antialias=True),
                tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @classmethod
    def from_config(cls, cfg_image_goal_vpr: Any, device: str = "cuda") -> "CosPlaceExtractor":
        """Build from the ``image_goal.vpr`` block of configs/goal.yaml."""
        return cls(
            method=str(cfg_image_goal_vpr.method),
            backbone=str(cfg_image_goal_vpr.backbone),
            descriptors_dimension=int(cfg_image_goal_vpr.descriptors_dimension),
            device=device,
        )

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """Extract one descriptor (shape ``(descriptors_dimension,)``, float32)."""
        import torch

        tensor = self._transform(np.asarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            desc = self._model(tensor)
        return desc.detach().cpu().numpy().astype("float32").reshape(-1)

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract a stack of descriptors (shape ``(N, descriptors_dimension)``)."""
        return np.stack([self.extract(img) for img in images]).astype("float32")
