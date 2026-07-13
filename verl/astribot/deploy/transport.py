from __future__ import annotations

import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray): return {"__ndarray__": True, "dtype": str(value.dtype), "shape": value.shape, "data": value.tobytes()}
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def _decode(value):
    if "__ndarray__" in value: return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(value["shape"]).copy()
    return value


def pack(value) -> bytes: return msgpack.packb(value, default=_encode, use_bin_type=True)
def unpack(value: bytes): return msgpack.unpackb(value, object_hook=_decode, raw=False)
