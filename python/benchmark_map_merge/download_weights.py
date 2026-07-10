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
    try:
        model = LightGlue(features=features).eval()
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Check network connection or pre-download weights manually.")
        return
    print(f"  OK — weights cached.")
    del model


def download_netvlad_weights() -> None:
    """Pre-download NetVLAD (VGG16-NetVLAD-Pitts30K) weights for hloc_sfm_netvlad_splg."""
    import sys
    from pathlib import Path
    _HLOC_DIR = str(
        Path(__file__).resolve().parents[2]
        / "third_party" / "pose_estimation_models" / "estimator"
        / "third_party" / "Hierarchical-Localization"
    )
    if _HLOC_DIR not in sys.path:
        sys.path.insert(0, _HLOC_DIR)

    print("Loading NetVLAD weights (VGG16-NetVLAD-Pitts30K) ...")
    try:
        from hloc.extractors.netvlad import NetVLAD
        model = NetVLAD({"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}).eval()
        print("  OK — NetVLAD weights cached.")
        del model
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Check network connection.")


if __name__ == "__main__":
    for feat in ["superpoint", "disk"]:
        download_weights(feat)
    download_netvlad_weights()
    print("Done.")
