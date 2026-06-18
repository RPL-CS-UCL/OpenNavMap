from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

from benchmark_map_merge.hloc_sfm_merger import (
    _fundamental_inlier_count,
    _geometric_verify_pairs,
)


def _grid_points(count: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(20, dtype=np.float32), np.arange(10, dtype=np.float32))
    points = np.column_stack([x.ravel(), y.ravel()])[:count]
    return points * 10.0 + np.array([50.0, 40.0], dtype=np.float32)


def _translated_points(points: np.ndarray, offset_x: float) -> np.ndarray:
    return points + np.array([offset_x, 5.0], dtype=np.float32)


def _projected_points(count: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    points_3d = rng.uniform([-2.0, -1.5, 4.0], [2.0, 1.5, 8.0], size=(count, 3))
    query = points_3d[:, :2] / points_3d[:, 2:3]
    db_points_3d = points_3d - np.array([0.6, 0.0, 0.0])
    db = db_points_3d[:, :2] / db_points_3d[:, 2:3]
    query_pixels = query * 240.0 + np.array([320.0, 240.0])
    db_pixels = db * 240.0 + np.array([320.0, 240.0])
    return query_pixels.astype(np.float32), db_pixels.astype(np.float32)


def _outlier_points(count: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(20.0, 400.0, size=(count, 2)).astype(np.float32)


def _write_keypoints(features_path: Path, image_name: str, keypoints: np.ndarray) -> None:
    with h5py.File(features_path, "a") as h5_file:
        h5_file.create_dataset(f"{image_name}/keypoints", data=keypoints)


def _write_matches(matches_path: Path, query_name: str, db_name: str, matches0: np.ndarray) -> None:
    query_key = query_name.replace("/", "-")
    db_key = db_name.replace("/", "-")
    with h5py.File(matches_path, "a") as h5_file:
        h5_file.create_dataset(f"{query_key}/{db_key}/matches0", data=matches0)


def _write_pairs(pairs_path: Path, pairs: Tuple[Tuple[str, str], ...]) -> None:
    pairs_path.write_text("".join(f"{query} {db}\n" for query, db in pairs))


def test_geometric_verify_pairs_drops_query_below_threshold(tmp_path: Path) -> None:
    features_path = tmp_path / "features.h5"
    matches_path = tmp_path / "matches.h5"
    loc_pairs_path = tmp_path / "pairs-loc.txt"
    out_pairs_path = tmp_path / "pairs-verified.txt"
    query = "inc1/seq/000000.color.jpg"
    db = "seq/000000.color.jpg"
    keypoints = _grid_points(99)

    _write_keypoints(features_path, query, keypoints)
    _write_keypoints(features_path, db, _translated_points(keypoints, offset_x=30.0))
    _write_matches(matches_path, query, db, np.arange(99, dtype=np.int32))
    _write_pairs(loc_pairs_path, ((query, db),))

    stats = _geometric_verify_pairs(loc_pairs_path, features_path, matches_path, out_pairs_path)

    assert out_pairs_path.read_text() == ""
    assert stats == {
        "num_query_total": 1,
        "num_query_kept": 0,
        "num_query_dropped": 1,
        "num_pairs_total": 1,
        "num_pairs_written": 0,
        "num_total_matches": 99,
        "pairs_detail": [],
        "min_inliers": 100,
    }


def test_geometric_verify_pairs_reranks_by_fundamental_inliers(tmp_path: Path) -> None:
    features_path = tmp_path / "features.h5"
    matches_path = tmp_path / "matches.h5"
    loc_pairs_path = tmp_path / "pairs-loc.txt"
    out_pairs_path = tmp_path / "pairs-verified.txt"
    query = "inc1/seq/000001.color.jpg"
    low_inlier_db = "seq/000000.color.jpg"
    high_inlier_db = "seq/000001.color.jpg"
    query_keypoints, high_db_keypoints = _projected_points(120)
    low_db_keypoints = high_db_keypoints.copy()
    low_db_keypoints[:80] = _outlier_points(80)

    _write_keypoints(features_path, query, query_keypoints)
    _write_keypoints(features_path, low_inlier_db, low_db_keypoints)
    _write_keypoints(features_path, high_inlier_db, high_db_keypoints)
    _write_matches(matches_path, query, low_inlier_db, np.arange(120, dtype=np.int32))
    _write_matches(matches_path, query, high_inlier_db, np.arange(120, dtype=np.int32))
    _write_pairs(loc_pairs_path, ((query, low_inlier_db), (query, high_inlier_db)))
    weak_inliers = _fundamental_inlier_count(
        query_keypoints,
        low_db_keypoints,
        np.arange(120, dtype=np.int32),
    )
    strong_inliers = _fundamental_inlier_count(
        query_keypoints,
        high_db_keypoints,
        np.arange(120, dtype=np.int32),
    )

    stats = _geometric_verify_pairs(loc_pairs_path, features_path, matches_path, out_pairs_path)

    assert strong_inliers >= 100
    assert strong_inliers - weak_inliers >= 40
    assert out_pairs_path.read_text().splitlines() == [
        f"{query} {high_inlier_db}",
        f"{query} {low_inlier_db}",
    ]
    assert stats == {
        "num_query_total": 1,
        "num_query_kept": 1,
        "num_query_dropped": 0,
        "num_pairs_total": 2,
        "num_pairs_written": 2,
        "num_total_matches": 240,
        "pairs_detail": [
            {
                "query": query,
                "db": high_inlier_db,
                "feat_matches": 120,
                "f_inliers": strong_inliers,
            },
            {
                "query": query,
                "db": low_inlier_db,
                "feat_matches": 120,
                "f_inliers": weak_inliers,
            },
        ],
        "min_inliers": 100,
    }
