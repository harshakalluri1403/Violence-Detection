#!/usr/bin/env python
"""Extract a zip whose member names may be too long for the filesystem.

    python scripts/unzip_safe.py archive.zip out_dir

Over-long or non-ASCII basenames are replaced by a short hash so labels
(taken from parent folder names) are preserved.
"""
import hashlib
import os
import re
import shutil
import sys
import zipfile

src, dst = sys.argv[1], sys.argv[2]
renamed = 0
with zipfile.ZipFile(src) as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = info.filename.replace("\\", "/").split("/")
        base = parts[-1]
        stem, ext = os.path.splitext(base)
        if len(base.encode()) > 120 or re.search(r"[^\x20-\x7e]", base):
            base = "h" + hashlib.md5(info.filename.encode("utf-8", "surrogateescape")).hexdigest()[:12] + ext
            renamed += 1
        out = os.path.join(dst, *parts[:-1], base)
        if os.path.exists(out) and os.path.getsize(out) == info.file_size:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with zf.open(info) as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
print(f"extracted {src} -> {dst} ({renamed} renamed)")
