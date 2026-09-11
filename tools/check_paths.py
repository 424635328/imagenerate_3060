"""Repository hygiene checks; exits non-zero on personal paths or secrets."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTS = {".py", ".md", ".html", ".js", ".json", ".cfg", ".toml", ".txt", ".bat", ".cmd", ".ps1", ".yml", ".yaml"}
SKIP_PARTS = {".git", ".netlify", "models", "dataset", "dataset1024", "outputs", "results", "node_modules", "__pycache__"}
PATTERNS = {
    "personal absolute path": re.compile(r"(?:[A-Za-z]:[\\/]+(?:Users|Download|home)[\\/]+(?!<|\.\.\.)|/home/(?!<|\.\.\.)[A-Za-z0-9._-]+|/Users/(?!<|\.\.\.)[A-Za-z0-9._-]+)"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|r8_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"),
}


def files_to_scan():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    for name in sorted(set(result.stdout.splitlines())):
        p = ROOT / name
        if not p.is_file() or p.suffix.lower() not in TEXT_EXTS:
            continue
        if any(part in SKIP_PARTS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-eol", action="store_true", help="also reject CRLF in repository text files")
    args = parser.parse_args()
    failures = 0
    for path in files_to_scan():
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                print(f"FAIL {name}: {path.relative_to(ROOT)}")
                failures += 1
        if args.check_eol and "\r\n" in text and path.suffix.lower() not in {".bat", ".cmd", ".ps1"}:
            print(f"FAIL CRLF: {path.relative_to(ROOT)}")
            failures += 1
    if failures:
        print(f"Hygiene check failed: {failures} issue(s)")
        return 1
    print(f"Hygiene check passed: {sum(1 for _ in files_to_scan())} text files scanned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
