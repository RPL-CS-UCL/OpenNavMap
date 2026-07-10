from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

STAGE_LOAD_SUBMAP = "load_submap"
STAGE_DESCRIPTOR = "descriptor"
STAGE_VPR = "vpr"
STAGE_SEQUENCE_MATCHING = "sequence_matching"
STAGE_GV = "gv"
STAGE_METRIC_LOC = "metric_loc"
STAGE_CULLING = "culling"
STAGE_PGO_BEFORE = "pgo_before"
STAGE_PGO_AFTER = "pgo_after"
STAGE_MERGED = "merged"
STAGE_STATUS = "status"


@dataclass(frozen=True)
class ArtifactRefs:
    dmatrix: Optional[Path] = None
    current_image: Optional[Path] = None
    query_image: Optional[Path] = None
    reference_image: Optional[Path] = None
    matching_image: Optional[Path] = None
    keyframe_culling: Optional[Path] = None
    pose_graph_initial: Optional[Path] = None
    pose_graph_refined: Optional[Path] = None
    stage_image: Optional[Path] = None


@dataclass(frozen=True)
class MapMergeVizEvent:
    event_type: str
    merge_step: int
    stage: str
    submap_id: str
    keyframe_id: int
    payload: Dict[str, Any] = field(default_factory=dict)
    artifact_refs: ArtifactRefs = field(default_factory=ArtifactRefs)
    timestamp: Optional[float] = None


def w2c_vec_to_camera_position(pose_vec: np.ndarray) -> np.ndarray:
    """Convert W2C vec7 [qw, qx, qy, qz, tx, ty, tz] to camera center."""
    rotation = Rotation.from_quat(
        [pose_vec[1], pose_vec[2], pose_vec[3], pose_vec[0]]
    ).as_matrix()
    translation = pose_vec[4:7]
    return -rotation.T @ translation


def compute_axis_scale(positions: np.ndarray) -> Tuple[float, float]:
    """Return visible world-axis length and radius for a scene extent."""
    if positions.size == 0:
        return 2.0, 0.03
    extent = float(np.linalg.norm(np.ptp(positions, axis=0)))
    return max(extent * 0.08, 2.0), max(extent * 0.001, 0.03)
