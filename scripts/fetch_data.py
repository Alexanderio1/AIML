"""Fetch the pinned author CSV, preserving TLS and checking its SHA-256."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
metadata = json.loads((ROOT / "data/source.json").read_text(encoding="utf-8"))
destination = ROOT / "data" / metadata["file"]
if destination.exists():
    content = destination.read_bytes()
else:
    with urllib.request.urlopen(metadata["downloaded_from"], timeout=90) as response:
        content = response.read()
actual = hashlib.sha256(content).hexdigest()
if actual != metadata["sha256"]:
    raise ValueError(f"CSV checksum mismatch: {actual}. Existing file was not overwritten.")
if not destination.exists():
    temporary = destination.with_suffix(".csv.tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
print(f"Verified {destination.name}: {actual}")
