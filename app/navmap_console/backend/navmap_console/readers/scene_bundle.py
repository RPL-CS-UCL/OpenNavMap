"""scene.bin container ("NMSB"), mirrored by frontend/src/api/scene-bundle.ts.

Layout: b"NMSB" | u32 version | u32 count | count x array, where each array is
u16 name_len | name (utf-8) | u8 dtype | u8 ndim | u32 shape[ndim] | pad to 4 | data | pad to 4.
dtype codes: 0 = uint8, 1 = uint16, 2 = uint32, 3 = float32. Everything little-endian.
"""
import struct
from typing import Dict

import numpy as np

MAGIC = b"NMSB"
VERSION = 1
_CODES = {np.dtype("uint8"): 0, np.dtype("uint16"): 1, np.dtype("uint32"): 2, np.dtype("float32"): 3}
_DTYPES = {code: dtype for dtype, code in _CODES.items()}


def _pad(n: int) -> int:
    return (4 - n % 4) % 4


def encode_scene_bundle(arrays: Dict[str, np.ndarray]) -> bytes:
    out = bytearray(MAGIC + struct.pack("<II", VERSION, len(arrays)))
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        native = arr.dtype.newbyteorder("=")
        if native not in _CODES:
            raise TypeError(f"{name}: unsupported dtype {arr.dtype} (use uint8/uint16/uint32/float32)")
        arr = arr.astype(native.newbyteorder("<"), copy=False)
        encoded = name.encode("utf-8")
        out += struct.pack("<H", len(encoded)) + encoded
        out += struct.pack("<BB", _CODES[native], arr.ndim)
        out += struct.pack("<%dI" % arr.ndim, *arr.shape)
        out += b"\0" * _pad(len(out))
        data = arr.tobytes()
        out += data + b"\0" * _pad(len(data))
    return bytes(out)


def decode_scene_bundle(data: bytes) -> Dict[str, np.ndarray]:
    if len(data) < 12 or data[:4] != MAGIC:
        raise ValueError("not an NMSB scene bundle")
    version, count = struct.unpack_from("<II", data, 4)
    if version != VERSION:
        raise ValueError(f"unsupported scene bundle version {version}")
    pos = 12
    out: Dict[str, np.ndarray] = {}
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", data, pos)
        pos += 2
        name = data[pos:pos + name_len].decode("utf-8")
        pos += name_len
        code, ndim = struct.unpack_from("<BB", data, pos)
        pos += 2
        shape = struct.unpack_from("<%dI" % ndim, data, pos)
        pos += 4 * ndim
        pos += _pad(pos)
        dtype = _DTYPES[code].newbyteorder("<")
        n = int(np.prod(shape)) if ndim else 1
        out[name] = np.frombuffer(data, dtype=dtype, count=n, offset=pos).reshape(shape).astype(_DTYPES[code])
        nbytes = n * dtype.itemsize
        pos += nbytes + _pad(nbytes)
    return out
