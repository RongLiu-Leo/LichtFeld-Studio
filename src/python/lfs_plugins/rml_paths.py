# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Path helpers for RmlUI image sources."""

from pathlib import Path
from urllib.parse import quote

_RML_PATH_SAFE_CHARS = "/:._-~"


def encode_rml_query_path(path: Path | str) -> str:
    return quote(str(path), safe=_RML_PATH_SAFE_CHARS)


def _has_windows_drive(path: str) -> bool:
    return len(path) >= 3 and path[0].isalpha() and path[1] == ":" and path[2] == "/"


def rml_image_source(path: Path | str, drive_source: Path | str | None = None) -> str:
    text = str(path).replace("\\", "/")
    if (
        text.startswith("/")
        and not text.startswith("//")
        and not _has_windows_drive(text)
        and drive_source is not None
    ):
        drive_text = str(drive_source).replace("\\", "/")
        if _has_windows_drive(drive_text):
            text = drive_text[:2] + text

    encoded = quote(text, safe=_RML_PATH_SAFE_CHARS)
    if _has_windows_drive(text):
        return f"file:///{encoded}"
    if text.startswith("//"):
        return f"file:{encoded}"
    if text.startswith("/"):
        return f"file://{encoded}"
    return encoded
