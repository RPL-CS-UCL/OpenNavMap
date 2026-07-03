from pathlib import Path

from PIL import Image

from visualization.map_merge_rerun_writer import MapMergeRerunWriter
from visualization.map_merge_viz_events import ArtifactRefs, MapMergeVizEvent


class FakeRerun:
    def __init__(self) -> None:
        self.logged = []

    @staticmethod
    def EncodedImage(**kwargs):
        return ("encoded_image", kwargs)

    @staticmethod
    def Image(array):
        return ("image", array)

    def log(self, entity_path, archetype) -> None:
        self.logged.append((entity_path, archetype))


def test_log_artifacts_uses_encoded_image_for_current_keyframe(tmp_path: Path) -> None:
    image_path = tmp_path / "000000.color.jpg"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)
    event = MapMergeVizEvent(
        event_type="node_observed",
        merge_step=0,
        stage="vio_pose",
        submap_id="0",
        keyframe_id=0,
        artifact_refs=ArtifactRefs(current_image=image_path),
    )
    rr = FakeRerun()

    MapMergeRerunWriter()._log_artifacts(rr, event)

    assert rr.logged == [
        (
            "/evidence/current_keyframe_image",
            (
                "encoded_image",
                {"path": image_path, "media_type": "image/jpeg"},
            ),
        )
    ]
