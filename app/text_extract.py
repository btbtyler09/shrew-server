"""Text-family input extraction.

Reads txt/md/rtf/html/csv/eml/msg/xlsx/xls/ods into a markdown string for
the Stage 3 pipeline, bypassing rasterization and VLM transcription.
"""

import csv
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger("shrew.text_extract")

CSV_MAX_ROWS = 10000
_HTML_MARKERS = re.compile(r"<(html|body|div|p|br|table|a\s|span|h[1-6])", re.IGNORECASE)


def _html_or_text_to_markdown(content: str) -> str:
    """Convert a body string to markdown. If it looks like HTML, parse + convert.
    Otherwise return as-is (already plain text).
    """
    if not content:
        return ""
    if _HTML_MARKERS.search(content):
        from bs4 import BeautifulSoup
        from markdownify import markdownify

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return markdownify(str(soup), heading_style="ATX").strip()
    return content.strip()


def extract_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def extract_md(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def extract_rtf(path: str, output_dir: str) -> str:
    """Convert RTF to plain text via LibreOffice, then read the result."""
    basename = os.path.basename(path)
    logger.info(f"Converting {basename} to text via LibreOffice")

    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "txt",
             "--outdir", output_dir, path],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "LibreOffice is not installed. "
            "Install it to process RTF documents: apt install libreoffice"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"LibreOffice conversion timed out after 120s for {basename}"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice RTF conversion failed for {basename}: {result.stderr}"
        )

    stem = os.path.splitext(basename)[0]
    txt_path = os.path.join(output_dir, f"{stem}.txt")
    if not os.path.exists(txt_path):
        raise RuntimeError(
            f"LibreOffice did not produce expected text at {txt_path}"
        )
    return Path(txt_path).read_text(encoding="utf-8", errors="replace")


def extract_html(path: str) -> str:
    """Parse HTML, strip scripts/styles, convert to markdown."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return markdownify(str(soup), heading_style="ATX").strip()


def extract_csv(path: str) -> str:
    """Render CSV as a single GFM markdown table.

    First row is treated as the header. Rows beyond CSV_MAX_ROWS are dropped
    with a warning.
    """
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= CSV_MAX_ROWS:
                logger.warning(
                    f"CSV exceeds {CSV_MAX_ROWS} rows; truncating "
                    f"({os.path.basename(path)})"
                )
                break
            rows.append(row)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    header = rows[0] + [""] * (width - len(rows[0]))
    body = [r + [""] * (width - len(r)) for r in rows[1:]]

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ").strip()

    lines = [
        "| " + " | ".join(esc(c) for c in header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)


def _format_email_preamble(
    subject: str | None,
    sender: str | None,
    to: str | None,
    cc: str | None,
    date: str | None,
    attachments: list[str],
) -> str:
    lines = []
    if subject:
        lines.append(f"**Subject:** {subject}")
    if sender:
        lines.append(f"**From:** {sender}")
    if to:
        lines.append(f"**To:** {to}")
    if cc:
        lines.append(f"**Cc:** {cc}")
    if date:
        lines.append(f"**Date:** {date}")
    if attachments:
        lines.append(f"**Attachments:** {', '.join(attachments)}")
    return "\n".join(lines)


def extract_eml(path: str) -> str:
    """Parse an .eml MIME message into a markdown preamble + body."""
    from email import policy
    from email.parser import BytesParser

    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    body_part = msg.get_body(preferencelist=("plain", "html"))
    if body_part is None:
        body_text = ""
    else:
        content = body_part.get_content()
        if body_part.get_content_type() == "text/html":
            body_text = _html_or_text_to_markdown(content)
        else:
            body_text = content.strip()

    attachments = [
        a.get_filename() for a in msg.iter_attachments() if a.get_filename()
    ]
    preamble = _format_email_preamble(
        msg.get("subject"),
        msg.get("from"),
        msg.get("to"),
        msg.get("cc"),
        msg.get("date"),
        attachments,
    )
    return f"{preamble}\n\n{body_text}".strip() if preamble else body_text


def extract_msg(path: str) -> str:
    """Parse a .msg Outlook message into a markdown preamble + body."""
    from msg_parser import MsOxMessage

    msg = MsOxMessage(path)

    recipients_to = msg.to
    if not recipients_to and msg.recipients:
        recipients_to = ", ".join(
            r.EmailAddress or r.DisplayName or ""
            for r in msg.recipients
            if (getattr(r, "RecipientType", None) or "TO") == "TO"
        ).strip(", ")

    attachments = [
        a.Filename for a in (msg.attachments or []) if getattr(a, "Filename", None)
    ]
    body = _html_or_text_to_markdown(msg.body or "")
    preamble = _format_email_preamble(
        msg.subject,
        msg.sender,
        recipients_to,
        msg.cc,
        msg.sent_date,
        attachments,
    )
    return f"{preamble}\n\n{body}".strip() if preamble else body


def extract_text(path: str, file_type: str, output_dir: str) -> str:
    """Dispatch to the right extractor for text/csv inputs.

    `file_type` is accepted for caller compatibility but dispatch is by
    extension so the mapping is uniform.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return extract_csv(path)
    if ext == ".txt":
        return extract_txt(path)
    if ext == ".md":
        return extract_md(path)
    if ext == ".rtf":
        return extract_rtf(path, output_dir)
    if ext in {".html", ".htm"}:
        return extract_html(path)
    if ext == ".eml":
        return extract_eml(path)
    if ext == ".msg":
        return extract_msg(path)
    if ext in {".xlsx", ".xls", ".ods"}:
        from .spreadsheet_extract import extract_spreadsheet
        return extract_spreadsheet(path, output_dir)
    raise ValueError(f"extract_text: unsupported extension {ext}")
