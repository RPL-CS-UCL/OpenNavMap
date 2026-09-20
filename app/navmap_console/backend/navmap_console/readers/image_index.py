"""Node id -> image path across step directories, plus cached thumbnails and side-by-side pair images."""
from pathlib import Path
from typing import Dict, Optional, Sequence

from PIL import Image

JPEG_QUALITY = 85
PAIR_GAP = 4  # px between the two images of a pair


class ImageIndex:
    def __init__(self, sources: Sequence[Path]) -> None:
        self.sources = [Path(s) for s in sources]
        self._map: Optional[Dict[int, Path]] = None

    def refresh(self) -> None:
        from map_merge_pack import build_image_index  # python/map_merge_pack.py, numpy only

        self._map = build_image_index(self.sources)

    def path(self, node_id: int) -> Optional[Path]:
        if self._map is None:
            self.refresh()
        assert self._map is not None
        return self._map.get(int(node_id))


def _resized(src: Path, width: int) -> Image.Image:
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width != width:
            im = im.resize((width, max(1, round(im.height * width / im.width))), Image.BILINEAR)
        return im.copy()


def thumbnail(src: Path, width: int, cache_dir: Path) -> Path:
    out = cache_dir / f"{src.stem}_{src.stat().st_size}_w{width}.jpg"
    if out.is_file():
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    _resized(src, width).save(tmp, "JPEG", quality=JPEG_QUALITY)
    tmp.replace(out)
    return out


def pair_image(left: Path, right: Path, width: int, cache_dir: Path) -> Path:
    out = cache_dir / f"pair_{left.stem}_{right.stem}_w{width}.jpg"
    if out.is_file():
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    a, b = _resized(left, width), _resized(right, width)
    canvas = Image.new("RGB", (width * 2 + PAIR_GAP, max(a.height, b.height)), (24, 24, 24))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (width + PAIR_GAP, 0))
    tmp = out.with_suffix(".tmp.jpg")
    canvas.save(tmp, "JPEG", quality=JPEG_QUALITY)
    tmp.replace(out)
    return out
