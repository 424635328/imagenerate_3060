"""check_eol.py — 行尾门禁：发现混合/错误的行尾就 exit 1（可用于 CI 与 pre-commit）。

规则与 .gitattributes 一致：
  * 文本文件（py/js/html/css/md/json/toml/cfg/yml/txt/svg）：仅 LF
  * Windows 启动器（bat/cmd/ps1）：仅 CRLF
  * 二进制与生成物目录跳过
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT  # noqa: E402

LF_EXT = {'.py', '.js', '.html', '.htm', '.css', '.md', '.json', '.toml', '.cfg',
          '.yml', '.yaml', '.txt', '.svg', '.gitignore', '.gitattributes', ''}
CRLF_EXT = {'.bat', '.cmd', '.ps1'}
SKIP_DIRS = {'.git', 'models', 'dataset', 'dataset1024', 'outputs', 'results',
             'node_modules', '__pycache__', '.netlify', '.venv'}


def scan() -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in LF_EXT and ext not in CRLF_EXT:
                continue
            path = Path(root) / name
            try:
                data = path.read_bytes()
            except OSError:
                continue
            crlf = data.count(b'\r\n')
            bare_lf = data.count(b'\n') - crlf
            rel = path.relative_to(ROOT).as_posix()
            if ext in CRLF_EXT:
                if bare_lf:
                    problems.append((rel, f'CRLF={crlf} bare-LF={bare_lf} (期望全 CRLF)'))
            else:
                if crlf:
                    problems.append((rel, f'CRLF={crlf} LF={bare_lf} (期望全 LF)'))
    return problems


def main() -> int:
    problems = scan()
    if not problems:
        print('EOL check passed: no mixed line endings')
        return 0
    print(f'EOL check failed: {len(problems)} file(s) with wrong line endings')
    for rel, detail in problems[:40]:
        print(f'   {rel}: {detail}')
    if len(problems) > 40:
        print(f'   … {len(problems) - 40} more')
    print('\nFix with: python tools/normalize_eol.py')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
