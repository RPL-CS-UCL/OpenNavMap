"""Pre-download LightGlue weights for disk and superpoint features.

Run once before using hloc_disk_dilg method:
    /root/miniconda3/envs/opennavmap/bin/python python/benchmark_map_merge/download_weights.py
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIGHTGLUE_DIR = str(
    _REPO_ROOT / "third_party" / "vismatch" / "vismatch" / "third_party" / "LightGlue"
)
if _LIGHTGLUE_DIR not in sys.path:
    sys.path.insert(0, _LIGHTGLUE_DIR)

from lightglue import LightGlue


def download_weights(features: str) -> None:
    print(f"Loading LightGlue weights for: {features} ...")
    model = LightGlue(features=features).eval()
    print(f"  OK — weights cached.")
    del model


if __name__ == "__main__":
    for feat in ["superpoint", "disk"]:
        download_weights(feat)
    print("Done.")
