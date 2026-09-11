"""Final repository hygiene check. Run from any working directory."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT", Path(__file__).resolve().parents[1])).resolve()
TEXT_EXTS = {".py", ".md", ".html", ".js", ".json", ".cfg", ".toml", ".txt", ".bat", ".cmd", ".ps1"}
SKIP = {".git", "models", "dataset", "dataset1024", "outputs", "results", "node_modules", "__pycache__"}
PATTERNS = [
    ("token", r"gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|r8_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"),
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("personal path", r"[A-Za-z]:[\\/]+(?:Users|Download)[\\/]+(?!<|\.\.\.)|/Users/(?!<|\.\.\.)[A-Za-z0-9._-]+/|/home/(?!<|\.\.\.)[A-Za-z0-9._-]+/"),
    ("site id", r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
]

def tracked_and_untracked():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return [ROOT / x for x in sorted(set(result.stdout.splitlines()))]

print("== sensitive scan ==")
found = []
for path in tracked_and_untracked():
    if not path.is_file() or path.suffix.lower() not in TEXT_EXTS or any(p in SKIP for p in path.relative_to(ROOT).parts):
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for name, pattern in PATTERNS:
        if re.search(pattern, text): found.append((name, path.relative_to(ROOT)))
print("CLEAN" if not found else "HITS: " + repr(found[:10]))
if found: sys.exit(1)
print("== manifest ==")
print(json.dumps({"root": "<PROJECT_ROOT>", "scanned": True}, ensure_ascii=False))
