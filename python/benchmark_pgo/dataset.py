"""Download and load classic 3D pose-graph benchmark datasets in g2o format."""
import urllib.request
from pathlib import Path
from typing import Tuple

import gtsam

DEFAULT_CACHE_DIR = Path("/Titan/dataset/data_opennavmap/g2o_benchmark")

# GTSAM 4.2 only ships toy-sized g2o files, so the classic 3D pose-graph
# benchmarks are mirrored from the SE-Sync repository.
DATASET_URLS = {
    "sphere2500": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/sphere2500.g2o",
    "parking-garage": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/parking-garage.g2o",
    "torus3D": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/torus3D.g2o",
    "cubicle": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/cubicle.g2o",
    "smallGrid3D": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/smallGrid3D.g2o",
}


def fetch(name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return the local path to dataset `name`, downloading it on first use."""
    if name not in DATASET_URLS:
        raise KeyError(f"Unknown dataset '{name}', expected one of {sorted(DATASET_URLS)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.g2o"
    if not path.exists():
        urllib.request.urlretrieve(DATASET_URLS[name], str(path))
    return path


def load(path: Path) -> Tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
    """Read a 3D g2o file into a factor graph and its initial estimate."""
    return gtsam.readG2o(str(path), True)
