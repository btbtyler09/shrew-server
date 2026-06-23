"""Outlook .msg (MS-OXMSG / CFB) reader that honors MAPI property type suffixes.

The previous implementation delegated to `msg_parser.MsOxMessage`, which
decides each property's encoding from a hardcoded canonical-type table
instead of the type suffix actually present in the stream name. That makes
it decode PT_STRING8 (0x001E) streams as UTF-16LE — garbling any message
that was stored as ANSI/codepage bytes.

We only need a handful of fields, so we read the CFB container directly
with `olefile` and dispatch by the type bytes in the stream name:

  __substg1.0_<id:4 hex><type:4 hex>

  type == 001F  -> PT_UNICODE,  decode as UTF-16LE
  type == 001E  -> PT_STRING8,  decode using the message codepage
  type == 0102  -> PT_BINARY,   raw bytes (HTML body lives here)

The codepage is read out of __properties_version1.0 (PR_INTERNET_CPID
0x3FDE, falling back to PR_MESSAGE_CODEPAGE 0x3FFD, then Windows-1252).
"""

from __future__ import annotations

import logging
import struct
from datetime import datetime, timedelta, timezone
from typing import Optional

import olefile

logger = logging.getLogger("shrew.msg_extract")

# MAPI property IDs we care about (16-bit IDs without type suffix).
PR_SUBJECT = 0x0037
PR_CLIENT_SUBMIT_TIME = 0x0039
PR_SENT_REPRESENTING_NAME = 0x0042
PR_SENDER_NAME = 0x0C1A
PR_SENDER_EMAIL_ADDRESS = 0x0C1F
PR_DISPLAY_TO = 0x0E04
PR_DISPLAY_CC = 0x0E03
PR_MESSAGE_DELIVERY_TIME = 0x0E06
PR_BODY = 0x1000
PR_HTML = 0x1013
PR_DISPLAY_NAME = 0x3001
PR_EMAIL_ADDRESS = 0x3003
PR_ATTACH_FILENAME = 0x3704
PR_ATTACH_LONG_FILENAME = 0x3707
PR_INTERNET_CPID = 0x3FDE
PR_MESSAGE_CODEPAGE = 0x3FFD
PR_SMTP_ADDRESS = 0x39FE

# MAPI property type codes (last 4 hex chars of stream name).
PT_STRING8 = 0x001E
PT_UNICODE = 0x001F
PT_BINARY = 0x0102
PT_LONG = 0x0003
PT_TIME = 0x0040

DEFAULT_CODEPAGE = "cp1252"

# Windows codepage ID -> Python codec name. Python accepts "cp1252" etc.
# directly for most Windows code pages, but a few have different names.
_CODEPAGE_ALIASES = {
    20127: "ascii",
    20866: "koi8_r",
    21866: "koi8_u",
    28591: "iso8859_1",
    28592: "iso8859_2",
    28593: "iso8859_3",
    28594: "iso8859_4",
    28595: "iso8859_5",
    28596: "iso8859_6",
    28597: "iso8859_7",
    28598: "iso8859_8",
    28599: "iso8859_9",
    28603: "iso8859_13",
    28605: "iso8859_15",
    65000: "utf-7",
    65001: "utf-8",
}


def _codepage_to_codec(cpid: int) -> str:
    if cpid in _CODEPAGE_ALIASES:
        return _CODEPAGE_ALIASES[cpid]
    return f"cp{cpid}"


def _parse_stream_name(name: str) -> Optional[tuple[int, int]]:
    """Parse `__substg1.0_<id><type>` (8 hex chars) into (prop_id, prop_type)."""
    if not name.startswith("__substg1.0_"):
        return None
    suffix = name[len("__substg1.0_"):]
    if len(suffix) < 8:
        return None
    try:
        prop_id = int(suffix[0:4], 16)
        prop_type = int(suffix[4:8], 16)
    except ValueError:
        return None
    return prop_id, prop_type


def _decode_string(raw: bytes, prop_type: int, codepage: str) -> str:
    """Decode a PT_UNICODE or PT_STRING8 stream to text."""
    if not raw:
        return ""
    if prop_type == PT_UNICODE:
        text = raw.decode("utf-16-le", errors="replace")
    elif prop_type == PT_STRING8:
        try:
            text = raw.decode(codepage, errors="replace")
        except LookupError:
            logger.warning("Unknown codepage %r; falling back to %s",
                           codepage, DEFAULT_CODEPAGE)
            text = raw.decode(DEFAULT_CODEPAGE, errors="replace")
    else:
        # Best-effort: try utf-8 then latin-1.
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
    return text.rstrip("\x00")


def _read_codepage(ole, scope: list[str]) -> str:
    """Find the codepage for PT_STRING8 decoding in this storage scope.

    Scans __properties_version1.0 (16-byte entries) for PR_INTERNET_CPID
    first, then PR_MESSAGE_CODEPAGE. Falls back to cp1252.
    """
    path = scope + ["__properties_version1.0"]
    if not ole.exists(path):
        return DEFAULT_CODEPAGE
    try:
        with ole.openstream(path) as stream:
            data = stream.read()
    except OSError:
        return DEFAULT_CODEPAGE

    # The header size varies by storage kind (top message 32, embedded 24,
    # attachment/recipient 8). Rather than guess, scan all 16-byte aligned
    # positions for the two tags. The tag occupies the first 4 bytes of an
    # entry as LE: type[2] + id[2].
    targets = {
        struct.pack("<HH", PT_LONG, PR_INTERNET_CPID): None,
        struct.pack("<HH", PT_LONG, PR_MESSAGE_CODEPAGE): None,
    }
    for off in range(0, len(data) - 15, 8):
        head = data[off:off + 4]
        if head in targets and targets[head] is None:
            # Property value occupies bytes 8..16; first 4 are the LONG.
            cpid = struct.unpack_from("<I", data, off + 8)[0]
            if cpid:
                targets[head] = cpid

    cpid = (targets[struct.pack("<HH", PT_LONG, PR_INTERNET_CPID)]
            or targets[struct.pack("<HH", PT_LONG, PR_MESSAGE_CODEPAGE)])
    if not cpid:
        return DEFAULT_CODEPAGE
    return _codepage_to_codec(cpid)


def _read_filetime(ole, scope: list[str], tag_id: int) -> Optional[datetime]:
    """Read a PT_TIME property out of __properties_version1.0."""
    path = scope + ["__properties_version1.0"]
    if not ole.exists(path):
        return None
    try:
        with ole.openstream(path) as stream:
            data = stream.read()
    except OSError:
        return None

    target = struct.pack("<HH", PT_TIME, tag_id)
    for off in range(0, len(data) - 15, 8):
        if data[off:off + 4] == target:
            # FILETIME = 100ns intervals since 1601-01-01 UTC.
            ft = struct.unpack_from("<Q", data, off + 8)[0]
            if not ft:
                return None
            return datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(
                microseconds=ft // 10
            )
    return None


def _read_property_streams(
    ole, scope: list[str]
) -> dict[int, dict[int, bytes]]:
    """Enumerate __substg1.0_* streams in `scope` and return
    {prop_id: {prop_type: raw_bytes}}.
    """
    out: dict[int, dict[int, bytes]] = {}
    for entry in ole.listdir(streams=True, storages=False):
        # entry is a list of path components: [..., stream_name]
        if entry[:-1] != scope:
            continue
        parsed = _parse_stream_name(entry[-1])
        if not parsed:
            continue
        prop_id, prop_type = parsed
        try:
            with ole.openstream(entry) as stream:
                raw = stream.read()
        except OSError:
            continue
        out.setdefault(prop_id, {})[prop_type] = raw
    return out


def _get_text(props: dict[int, dict[int, bytes]], prop_id: int,
              codepage: str) -> Optional[str]:
    """Look up a string property, preferring PT_UNICODE if both variants exist."""
    type_map = props.get(prop_id)
    if not type_map:
        return None
    # Prefer Unicode when available (it's authoritative and unambiguous).
    for prop_type in (PT_UNICODE, PT_STRING8):
        if prop_type in type_map:
            text = _decode_string(type_map[prop_type], prop_type, codepage)
            if text:
                return text
    return None


def _get_binary(props: dict[int, dict[int, bytes]],
                prop_id: int) -> Optional[bytes]:
    type_map = props.get(prop_id)
    if not type_map:
        return None
    return type_map.get(PT_BINARY)


def _scope_iter(ole, prefix: str) -> list[list[str]]:
    """Return the list of top-level storage scopes whose name starts with
    `prefix` (e.g. '__recip_version1.0_' or '__attach_version1.0_'), sorted.
    """
    scopes: list[list[str]] = []
    seen: set[str] = set()
    for entry in ole.listdir(streams=True, storages=True):
        if not entry:
            continue
        head = entry[0]
        if head.startswith(prefix) and head not in seen:
            seen.add(head)
            scopes.append([head])
    scopes.sort(key=lambda s: s[0])
    return scopes


def _decode_html_body(raw: bytes, codepage: str) -> str:
    """PR_HTML is stored as PT_BINARY; the bytes are the HTML payload in
    either the message codepage or whatever charset its own <meta> declares.
    Try utf-8 first, then the message codepage, then latin-1.
    """
    if not raw:
        return ""
    for codec in ("utf-8", codepage, "latin-1"):
        try:
            return raw.decode(codec).rstrip("\x00")
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace").rstrip("\x00")


def read_msg(path: str) -> dict:
    """Parse a .msg file into a dict of plain Python values.

    Returns keys: subject, sender, to, cc, sent_date, body, html,
    recipients (list of {name, email}), attachments (list of filenames).
    """
    with olefile.OleFileIO(path) as ole:
        codepage = _read_codepage(ole, [])
        props = _read_property_streams(ole, [])

        subject = _get_text(props, PR_SUBJECT, codepage)
        sender = (_get_text(props, PR_SENDER_NAME, codepage)
                  or _get_text(props, PR_SENT_REPRESENTING_NAME, codepage)
                  or _get_text(props, PR_SENDER_EMAIL_ADDRESS, codepage))
        display_to = _get_text(props, PR_DISPLAY_TO, codepage)
        display_cc = _get_text(props, PR_DISPLAY_CC, codepage)

        sent_dt = (_read_filetime(ole, [], PR_CLIENT_SUBMIT_TIME)
                   or _read_filetime(ole, [], PR_MESSAGE_DELIVERY_TIME))
        sent_date = sent_dt.isoformat() if sent_dt else None

        body_text = _get_text(props, PR_BODY, codepage)
        html_bytes = _get_binary(props, PR_HTML)
        html_text = _decode_html_body(html_bytes, codepage) if html_bytes else None

        recipients = []
        for scope in _scope_iter(ole, "__recip_version1.0_"):
            rcp_props = _read_property_streams(ole, scope)
            rcp_codepage = _read_codepage(ole, scope) or codepage
            name = _get_text(rcp_props, PR_DISPLAY_NAME, rcp_codepage)
            email = (_get_text(rcp_props, PR_SMTP_ADDRESS, rcp_codepage)
                     or _get_text(rcp_props, PR_EMAIL_ADDRESS, rcp_codepage))
            if name or email:
                recipients.append({"name": name, "email": email})

        attachments: list[str] = []
        for scope in _scope_iter(ole, "__attach_version1.0_"):
            att_props = _read_property_streams(ole, scope)
            att_codepage = _read_codepage(ole, scope) or codepage
            filename = (_get_text(att_props, PR_ATTACH_LONG_FILENAME, att_codepage)
                        or _get_text(att_props, PR_ATTACH_FILENAME, att_codepage))
            if filename:
                attachments.append(filename)

    return {
        "subject": subject,
        "sender": sender,
        "to": display_to,
        "cc": display_cc,
        "sent_date": sent_date,
        "body": body_text,
        "html": html_text,
        "recipients": recipients,
        "attachments": attachments,
    }
