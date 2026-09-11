"""Static checks for the site/ frontend.

Catches the mistakes that a plain syntax check cannot:

1. JavaScript referencing an element id that ``index.html`` does not define
   (silent ``null`` at runtime).
2. ``index.html`` or a module pointing at a file that is not in ``site/``.
3. CSS regressions: an ``animation``/``transition`` on ``.lb-stage img`` whose
   ``transform`` is owned by the lightbox zoom engine, and ``animation`` names
   with no matching ``@keyframes`` (silent no-op).

Exit code is non-zero on any problem so it can gate a commit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
HTML = SITE / "index.html"
JS_DIR = SITE / "js"
CSS_DIR = SITE / "css"

ID_CALL = re.compile(r"\bel\(\s*'([A-Za-z0-9_]+)'\s*\)|getElementById\(\s*'([A-Za-z0-9_]+)'\s*\)")
TEMPLATE_ID = re.compile(r"\bel\(`")
HTML_ID = re.compile(r"\bid=\"([A-Za-z0-9_]+)\"")
IMPORTS = re.compile(r"from\s+'(\./[A-Za-z0-9_./-]+)'")
LOCAL_ASSET = re.compile(r"(?:href|src)=\"(/[A-Za-z0-9_./-]+)\"")
# ids that are created dynamically by JavaScript, not present in the HTML shell
DYNAMIC_IDS = {"mainImage", "compareBox", "compareTop", "compareHandle"}

KEYFRAMES = re.compile(r"@keyframes\s+([\w-]+)")
ANIM_VALUE = re.compile(r"animation(?:-name)?\s*:\s*([^;}]+)")
# selector chain ending in `.lb-stage … img { body }` (no nested braces in between)
LB_STAGE_IMG = re.compile(r"\.lb-stage[^{}]{0,200}\bimg\b[^{}]{0,120}\{([^{}]*)\}")


def check_css(problems: list[str]) -> None:
    """Guard the two CSS regressions that have actually bitten this project."""
    css_files = [SITE / "styles.css", SITE / "extra.css"]
    if CSS_DIR.exists():
        css_files += sorted(CSS_DIR.glob("*.css"))
    css_files = [p for p in css_files if p.exists()]

    names: set[str] = set()
    for path in css_files:
        names.update(KEYFRAMES.findall(path.read_text(encoding="utf-8")))

    for path in css_files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(SITE).as_posix()
        # 1. animation names must resolve to a @keyframes somewhere in site/
        for body in ANIM_VALUE.findall(text):
            # strip function bodies (cubic-bezier(.2, .8, .3, 1)) before comma split
            for token in [t.strip().split()[0] for t in re.sub(r"\([^)]*\)", "", body).split(",")]:
                if token and token != "none" and token not in names:
                    problems.append(f"FAIL animation '{token}' in {rel} has no @keyframes")
        # 2. .lb-stage img transform belongs to the zoom engine:
        #    a fill-mode animation overrides the inline transform forever, and a
        #    transform transition longer than the engine's own .09s micro-smoothing
        #    makes every zoom step visibly laggy.
        for body in LB_STAGE_IMG.findall(text):
            has_fill = "animation" in body and re.search(r"\b(?:both|forwards)\b", body)
            if has_fill:
                problems.append(f"FAIL {rel}: .lb-stage img has a fill-mode animation "
                                f"(overrides lightbox zoom transform)")
            tdecl = re.search(r"transition\s*:\s*([^;}]+)", body)
            if tdecl and "transform" in tdecl.group(1):
                dur = max([float(s or 0) + float(ms or 0) / 1000
                           for s, ms in re.findall(r"([\d.]+)s|(\d+)ms", tdecl.group(1))]
                          or [0.0])
                if dur > 0.15:
                    problems.append(f"FAIL {rel}: .lb-stage img has a {dur}s transform "
                                    f"transition (conflicts with lightbox zoom engine)")


def main() -> int:
    problems: list[str] = []
    if not HTML.exists():
        print("FAIL missing site/index.html")
        return 1
    html = HTML.read_text(encoding="utf-8")
    html_ids = set(HTML_ID.findall(html))

    js_files = sorted(JS_DIR.glob("*.js"))
    if not js_files:
        problems.append("FAIL no JavaScript modules found in site/js")

    for path in js_files:
        text = path.read_text(encoding="utf-8")
        for first, second in ID_CALL.findall(text):
            name = first or second
            if name not in html_ids and name not in DYNAMIC_IDS:
                problems.append(f"FAIL unknown element id '{name}' referenced in {path.name}")
        for target in IMPORTS.findall(text):
            if not (path.parent / target).resolve().exists():
                problems.append(f"FAIL broken import '{target}' in {path.name}")

    for asset in set(LOCAL_ASSET.findall(html)):
        if not (SITE / asset.lstrip("/")).exists():
            problems.append(f"FAIL missing asset {asset} referenced by index.html")

    if "type=\"module\"" not in html:
        problems.append("FAIL index.html must load the entry point as an ES module")

    check_css(problems)

    for problem in problems:
        print(problem)
    if problems:
        print(f"Frontend check failed: {len(problems)} issue(s)")
        return 1
    css_count = len([p for p in [SITE / "styles.css", SITE / "extra.css"] if p.exists()]) \
        + (len(list(CSS_DIR.glob("*.css"))) if CSS_DIR.exists() else 0)
    print(f"Frontend check passed: {len(html_ids)} ids, {len(js_files)} modules, {css_count} css")
    return 0


if __name__ == "__main__":
    sys.exit(main())
