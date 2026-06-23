"""Tests for the in-house .msg parser.

These tests verify the actual bug being fixed: PT_STRING8 properties must
be decoded using the message codepage, not UTF-16LE. We substitute a
``FakeOle`` for ``olefile.OleFileIO`` so tests don't depend on hand-built
CFB binaries — ``olefile`` itself is a third-party library that we trust
to correctly read CFB streams.
"""

from __future__ import annotations

import struct
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from app import msg_extract
from tests._fake_ole import FakeOle, make_properties_stream, make_substg_name


@contextmanager
def patched_olefile(fake: FakeOle):
    """Make ``olefile.OleFileIO(path)`` return our FakeOle, ignoring path."""
    with patch.object(msg_extract.olefile, "OleFileIO", lambda _path: fake):
        yield


# ---------------------------------------------------------------------------
# Pure unit tests for the leaf decoders
# ---------------------------------------------------------------------------

def test_parse_stream_name_valid():
    assert msg_extract._parse_stream_name("__substg1.0_1000001F") == (0x1000, 0x001F)
    assert msg_extract._parse_stream_name("__substg1.0_0037001E") == (0x0037, 0x001E)


def test_parse_stream_name_invalid():
    assert msg_extract._parse_stream_name("__properties_version1.0") is None
    assert msg_extract._parse_stream_name("__substg1.0_short") is None
    assert msg_extract._parse_stream_name("__substg1.0_ZZZZ001F") is None


def test_decode_unicode_strips_trailing_null():
    raw = "helloé".encode("utf-16-le") + b"\x00\x00"
    assert msg_extract._decode_string(raw, 0x001F, "cp1252") == "helloé"


def test_decode_string8_uses_codepage_not_utf16():
    """The bug: ASCII/Windows-1252 bytes were being decoded as UTF-16LE.

    With the fix, PT_STRING8 (0x001E) must honor the codepage. ``cafe`` with
    an accented e in cp1252 is 0xE9, in UTF-16LE it would be `c\\x00a\\x00...`.
    """
    raw = b"caf\xe9"
    assert msg_extract._decode_string(raw, 0x001E, "cp1252") == "café"


def test_decode_string8_falls_back_when_codec_unknown():
    raw = b"hello \xe9"
    out = msg_extract._decode_string(raw, 0x001E, "cp99999")
    # Fallback decodes as cp1252 → 0xE9 maps to é.
    assert "é" in out


# ---------------------------------------------------------------------------
# Codepage discovery from __properties_version1.0
# ---------------------------------------------------------------------------

def test_read_codepage_from_internet_cpid():
    fake = FakeOle({
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 1252},
        ),
    })
    assert msg_extract._read_codepage(fake, []) == "cp1252"


def test_read_codepage_from_message_codepage_when_no_internet():
    fake = FakeOle({
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FFD: 1250},
        ),
    })
    assert msg_extract._read_codepage(fake, []) == "cp1250"


def test_read_codepage_internet_preferred_over_message():
    """When both PR_INTERNET_CPID and PR_MESSAGE_CODEPAGE are set, prefer
    PR_INTERNET_CPID (it's authoritative for the message body bytes)."""
    fake = FakeOle({
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 1252, 0x3FFD: 28591},
        ),
    })
    assert msg_extract._read_codepage(fake, []) == "cp1252"


def test_read_codepage_fallback_when_missing():
    fake = FakeOle({})  # no properties stream at all
    assert msg_extract._read_codepage(fake, []) == "cp1252"


def test_codepage_iso_8859_alias():
    """Codepage 28591 must map to the ISO-8859-1 codec, not "cp28591"
    (which doesn't exist in Python)."""
    fake = FakeOle({
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 28591},
        ),
    })
    assert msg_extract._read_codepage(fake, []) == "iso8859_1"


def test_codepage_utf8_alias():
    fake = FakeOle({
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 65001},
        ),
    })
    assert msg_extract._read_codepage(fake, []) == "utf-8"


# ---------------------------------------------------------------------------
# End-to-end via read_msg() with patched olefile
# ---------------------------------------------------------------------------

def test_unicode_body_and_subject():
    """All properties stored as PT_UNICODE (0x001F) — the happy path."""
    streams = {
        (make_substg_name(0x0037, 0x001F),): "Quarterly Report".encode("utf-16-le"),
        (make_substg_name(0x1000, 0x001F),): "Hello world".encode("utf-16-le"),
        (make_substg_name(0x0C1A, 0x001F),): "Alice".encode("utf-16-le"),
        (make_substg_name(0x0E04, 0x001F),): "bob@example.com".encode("utf-16-le"),
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["subject"] == "Quarterly Report"
    assert out["body"] == "Hello world"
    assert out["sender"] == "Alice"
    assert out["to"] == "bob@example.com"
    assert "�" not in out["body"]


def test_string8_body_with_cp1252_codepage():
    """The regression case: body stored as PT_STRING8 with cp1252 bytes.

    Before the fix these bytes were UTF-16-LE-decoded, producing mojibake.
    """
    body_bytes = "café — naïve".encode("cp1252")
    streams = {
        (make_substg_name(0x1000, 0x001E),): body_bytes,
        (make_substg_name(0x0037, 0x001E),): b"Re: caf\xe9",
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 1252},
        ),
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["body"] == "café — naïve"
    assert out["subject"] == "Re: café"
    # The smoking-gun check: no UTF-16 mojibake byte pattern.
    assert "\x00" not in out["body"]


def test_mixed_string8_and_unicode_props():
    """Subject is PT_UNICODE, body is PT_STRING8 with codepage. Each must
    decode under its own type — this is exactly what the old library
    couldn't do (it picked one canonical type per property ID)."""
    streams = {
        (make_substg_name(0x0037, 0x001F),): "Project — Status".encode("utf-16-le"),
        (make_substg_name(0x1000, 0x001E),): "Résumé attached".encode("cp1252"),
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 1252},
        ),
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["subject"] == "Project — Status"
    assert out["body"] == "Résumé attached"


def test_prefers_unicode_when_both_variants_exist():
    """If a file (unusually) stores both 001E and 001F for the same prop,
    prefer the Unicode one — it's authoritative."""
    streams = {
        (make_substg_name(0x0037, 0x001F),): "Unicode subject".encode("utf-16-le"),
        (make_substg_name(0x0037, 0x001E),): b"different bytes",
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["subject"] == "Unicode subject"


def test_recipients_with_string8_names():
    """Recipient display names stored as PT_STRING8 in a __recip_ folder."""
    streams = {
        ("__recip_version1.0_#00000000", make_substg_name(0x3001, 0x001E)): "Bjørn".encode("cp1252"),
        ("__recip_version1.0_#00000000", make_substg_name(0x39FE, 0x001F)): "bjorn@example.no".encode("utf-16-le"),
        ("__recip_version1.0_#00000001", make_substg_name(0x3001, 0x001F)): "Carlos".encode("utf-16-le"),
        ("__recip_version1.0_#00000001", make_substg_name(0x3003, 0x001E)): b"carlos@example.com",
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 1252},
        ),
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    names = [r["name"] for r in out["recipients"]]
    emails = [r["email"] for r in out["recipients"]]
    assert "Bjørn" in names
    assert "Carlos" in names
    assert "bjorn@example.no" in emails
    assert "carlos@example.com" in emails


def test_attachments_string8_filenames():
    streams = {
        ("__attach_version1.0_#00000000", make_substg_name(0x3707, 0x001E)): "café.pdf".encode("cp1252"),
        ("__attach_version1.0_#00000001", make_substg_name(0x3704, 0x001F)): "report.docx".encode("utf-16-le"),
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 1252},
        ),
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["attachments"] == ["café.pdf", "report.docx"]


def test_attachment_prefers_long_filename():
    streams = {
        ("__attach_version1.0_#00000000", make_substg_name(0x3704, 0x001F)): "short.txt".encode("utf-16-le"),
        ("__attach_version1.0_#00000000", make_substg_name(0x3707, 0x001F)): "long_descriptive_name.txt".encode("utf-16-le"),
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["attachments"] == ["long_descriptive_name.txt"]


def test_html_body_utf8():
    html_bytes = b"<html><body><p>caf\xc3\xa9</p></body></html>"
    streams = {
        (make_substg_name(0x1013, 0x0102),): html_bytes,
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["html"] is not None
    assert "café" in out["html"]


def test_sent_date_filetime():
    # 2023-01-15 12:34:56 UTC = filetime 133186195000000000 ticks (100ns since 1601-01-01).
    # Use a known datetime and derive its filetime.
    import datetime as _dt
    target = _dt.datetime(2023, 1, 15, 12, 34, 56, tzinfo=_dt.timezone.utc)
    delta = target - _dt.datetime(1601, 1, 1, tzinfo=_dt.timezone.utc)
    filetime = int(delta.total_seconds() * 10_000_000)
    streams = {
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, time_props={0x0039: filetime},
        ),
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["sent_date"] is not None
    assert out["sent_date"].startswith("2023-01-15T12:34:56")


def test_empty_message_does_not_crash():
    with patched_olefile(FakeOle({})):
        out = msg_extract.read_msg("ignored.msg")
    assert out["subject"] is None
    assert out["body"] is None
    assert out["recipients"] == []
    assert out["attachments"] == []


def test_missing_codepage_with_string8_falls_back_to_cp1252():
    """If the file has PT_STRING8 fields but no codepage hint, default to
    cp1252 — the most common Western encoding."""
    streams = {
        (make_substg_name(0x1000, 0x001E),): b"Caf\xe9 latte",
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["body"] == "Café latte"


# ---------------------------------------------------------------------------
# Regression simulating the original bug
# ---------------------------------------------------------------------------

def test_regression_string8_bytes_not_decoded_as_utf16():
    """If the parser ever regresses to UTF-16-LE-decoding PT_STRING8 bytes,
    the result will contain interleaved null bytes / replacement chars
    instead of the original text. This test enforces the opposite."""
    text = "Hello — world"
    raw = text.encode("cp1252")
    streams = {
        (make_substg_name(0x1000, 0x001E),): raw,
        ("__properties_version1.0",): make_properties_stream(
            header_size=32, long_props={0x3FDE: 1252},
        ),
    }
    with patched_olefile(FakeOle(streams)):
        out = msg_extract.read_msg("ignored.msg")
    assert out["body"] == text
    # If it had been UTF-16-LE decoded, we'd see "H\x00e\x00l..." pattern.
    assert "\x00" not in out["body"]
