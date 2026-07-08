from pathlib import Path

import numpy as np

from visualization.map_merge_viz_events import (
    ArtifactRefs,
    MapMergeVizEvent,
    STAGE_SEQUENCE_MATCHING,
    compute_axis_scale,
    w2c_vec_to_camera_position,
)


def test_w2c_vec_to_camera_position_uses_inverse_translation() -> None:
    pose_vec = np.array([1.0, 0.0, 0.0, 0.0, -2.0, 3.0, -4.0])

    position = w2c_vec_to_camera_position(pose_vec)

    np.testing.assert_allclose(position, np.array([2.0, -3.0, 4.0]))


def test_axis_scale_grows_with_scene_extent() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    axis_length, axis_radius = compute_axis_scale(positions)

    assert axis_length == 8.0
    assert axis_radius == 0.1


def test_event_accepts_artifact_refs() -> None:
    event = MapMergeVizEvent(
        event_type="sequence_match_result",
        merge_step=2,
        stage=STAGE_SEQUENCE_MATCHING,
        submap_id="4",
        keyframe_id=17,
        payload={"query_row": 17, "reference_row": 3},
        artifact_refs=ArtifactRefs(
            dmatrix=Path("dmatrix.png"), current_image=Path("seq/000017.color.jpg")
        ),
    )

    assert event.artifact_refs.dmatrix == Path("dmatrix.png")
    assert event.artifact_refs.current_image == Path("seq/000017.color.jpg")
