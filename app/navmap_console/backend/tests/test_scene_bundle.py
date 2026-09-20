# tests/test_scene_bundle.py
import numpy as np
import pytest


def test_scene_bundle_round_trip():
    from navmap_console.readers.scene_bundle import decode_scene_bundle, encode_scene_bundle

    arrays = {
        "node_id": np.arange(5, dtype=np.uint32),
        "node_pos": np.arange(15, dtype=np.float32).reshape(5, 3) * 0.5,
        "node_step": np.array([0, 0, 1, 1, 2], dtype=np.uint16),
        "node_flags": np.array([1, 0, 3, 0, 1], dtype=np.uint8),  # odd byte count exercises padding
        "edge_odom": np.zeros((0, 2), dtype=np.uint32),
        "name_with_ünicode": np.array([7], dtype=np.uint8),
    }
    data = encode_scene_bundle(arrays)
    assert data[:4] == b"NMSB" and len(data) % 4 == 0
    out = decode_scene_bundle(data)
    assert list(out) == list(arrays)
    for key, arr in arrays.items():
        assert out[key].dtype == arr.dtype and out[key].shape == arr.shape
        np.testing.assert_array_equal(out[key], arr)


def test_scene_bundle_rejects_bad_input():
    from navmap_console.readers.scene_bundle import decode_scene_bundle, encode_scene_bundle

    with pytest.raises(ValueError):
        decode_scene_bundle(b"NOPE" + b"\0" * 8)
    with pytest.raises(TypeError):
        encode_scene_bundle({"x": np.zeros(2, dtype=np.float64)})
