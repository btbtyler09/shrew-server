"""Minimal stand-in for `olefile.OleFileIO`, just enough for `app.msg_extract`.

Carries the same shape the parser touches:

  - ``listdir(streams=True, storages=False)`` -> list[list[str]]
  - ``openstream(path)`` -> file-like context manager with ``.read()``
  - ``exists(path)`` -> bool
  - context-manager protocol (``with FakeOle(...) as f:``)

Streams are supplied as a flat ``{tuple_path: bytes}`` map. Storages are
inferred from the path prefixes (any non-leaf component implicitly exists
as a storage).
"""

from __future__ import annotations

import io
import struct
from typing import Mapping, Sequence


def make_substg_name(prop_id: int, prop_type: int) -> str:
    """Return the CFB stream name for the given MAPI tag (e.g. 0x0037, 0x001F).
    The hex is uppercase to match Outlook's convention.
    """
    return f"__substg1.0_{prop_id:04X}{prop_type:04X}"


def make_properties_stream(
    header_size: int,
    long_props: Mapping[int, int] = (),
    time_props: Mapping[int, int] = (),
) -> bytes:
    """Build a synthetic ``__properties_version1.0`` payload.

    ``long_props``  maps prop_id -> int32 value (PT_LONG, type 0x0003).
    ``time_props``  maps prop_id -> 100ns FILETIME (PT_TIME, type 0x0040).
    """
    out = bytearray(b"\x00" * header_size)
    for prop_id, value in dict(long_props).items():
        entry = struct.pack("<HHI", 0x0003, prop_id, 0)  # type, id, flags
        entry += struct.pack("<I", value & 0xFFFFFFFF)
        entry += b"\x00\x00\x00\x00"
        out += entry
    for prop_id, value in dict(time_props).items():
        entry = struct.pack("<HHI", 0x0040, prop_id, 0)
        entry += struct.pack("<Q", value)
        out += entry
    return bytes(out)


class _StreamCM:
    def __init__(self, raw: bytes):
        self._buf = io.BytesIO(raw)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._buf.close()

    def read(self, *args, **kwargs):
        return self._buf.read(*args, **kwargs)


class FakeOle:
    def __init__(self, streams: Mapping[Sequence[str], bytes]):
        self._streams: dict[tuple[str, ...], bytes] = {
            tuple(k): v for k, v in streams.items()
        }
        # Derive set of storage paths from stream parents.
        self._storages: set[tuple[str, ...]] = set()
        for path in self._streams:
            for i in range(1, len(path)):
                self._storages.add(path[:i])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def listdir(self, streams: bool = True, storages: bool = False):
        out: list[list[str]] = []
        if streams:
            out.extend([list(p) for p in self._streams])
        if storages:
            out.extend([list(p) for p in self._storages])
        return out

    def openstream(self, path):
        key = tuple(path)
        if key not in self._streams:
            raise OSError(f"stream not found: {path!r}")
        return _StreamCM(self._streams[key])

    def exists(self, path):
        key = tuple(path)
        return key in self._streams or key in self._storages
