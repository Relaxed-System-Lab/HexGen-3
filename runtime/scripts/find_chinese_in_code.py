#!/usr/bin/env python3
"""
Find lines containing Chinese in *comments* (and docstrings), for i18n cleanup.
Skips: token strings (e.g. </think>), chat template content, test data, pad/stop tokens.

Usage:
  python scripts/find_chinese_in_code.py              # summary by file + first 50 lines
  python scripts/find_chinese_in_code.py -n 200      # show up to 200 lines
  python scripts/find_chinese_in_code.py -o out.txt  # write to file
  python scripts/find_chinese_in_code.py --all       # include 3rdparty
  python scripts/find_chinese_in_code.py --dir python
"""
import argparse
import re
import sys
from pathlib import Path

# CJK Unified Ideographs only (exclude punctuation that matches token chars)
CJK_RANGE = re.compile(r"[\u4e00-\u9fff]")

CODE_SUFFIXES = {".py", ".sh", ".cpp", ".h", ".cc", ".c", ".hpp", ".cu", ".cuh", ".rs"}

DEFAULT_DIRS = ["python", "afd", "benchmark", "scripts", "sgl-kernel", "sgl-router", "test"]

# Lines matching these are skipped (token names, template markers, test strings)
SKIP_PATTERNS = [
    r"^\s*(fim_|pad_token|stop_str|sep\s*=|bot_token|eot_token)\s*=",  # token/config
    r"^\s*[\"'].*[\u4e00-\u9fff].*[\"']\s*[,\)]",   # string literal only
    r"assertIn\s*\(\s*[\"']",                        # test assert string
    r"insert\s*\(\s*[\"']",                          # test data like insert("你好"
    r"tree\.insert|prefix_match",                    # test tree
    r"worker_urls|log_dir|url:\s*[\"']|reason:\s*[\"']",  # Rust test fixtures
    r"unicode_text\s*=|location.*Tokyo|Paris",       # i18n test strings
    r"[\"'].*北京.*[\"']|[\"'].*上海.*[\"']|arg_value.*北京|tool_call.*北京",  # example cities in templates
    r"霍格沃茨|霍比特人|你好嗎|你好喔|你心情好嗎|霍格:|霍比:",  # test/example or hex-dump comments
]
SKIP_RE = re.compile("|".join(SKIP_PATTERNS))


def should_skip(line: str, path: Path) -> bool:
    if SKIP_RE.search(line):
        return True
    # Skip this script's own docstring
    if "find_chinese_in_code" in str(path):
        return True
    # Skip lines that are only a quoted string (no # or //)
    stripped = line.strip()
    if (stripped.startswith('"') or stripped.startswith("'")) and ("#" not in line and "//" not in line):
        return True
    return False


def has_chinese(line: str) -> bool:
    return bool(CJK_RANGE.search(line))


def main():
    parser = argparse.ArgumentParser(description="Find Chinese in code comments (for i18n).")
    parser.add_argument("--all", action="store_true", help="Include 3rdparty")
    parser.add_argument("--dir", nargs="+", default=None, help="Only these dirs (e.g. python afd)")
    parser.add_argument("-n", "--max-lines", type=int, default=None,
                        help="Max lines to print (default: 80 to terminal; no limit when -o file)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Write to file (outputs all lines by default)")
    parser.add_argument("--summary-only", action="store_true", help="Only print per-file counts")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.all:
        search_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    elif args.dir:
        search_dirs = [root / d for d in args.dir if (root / d).is_dir()]
    else:
        search_dirs = [root / d for d in DEFAULT_DIRS if (root / d).is_dir()]

    by_file = {}
    for d in search_dirs:
        for path in d.rglob("*"):
            if not path.is_file() or path.suffix not in CODE_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = path.relative_to(root)
            for i, line in enumerate(text.splitlines(), 1):
                if not has_chinese(line) or should_skip(line, path):
                    continue
                key = str(rel)
                by_file.setdefault(key, []).append((i, line.strip()[:100]))

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    # Summary
    total_lines = sum(len(v) for v in by_file.values())
    print(f"# Files with Chinese in comments: {len(by_file)}, total lines: {total_lines}", file=out)
    if args.summary_only:
        for f in sorted(by_file.keys()):
            print(f"  {f}: {len(by_file[f])}", file=out)
        if args.output:
            out.close()
        return 0 if total_lines == 0 else 1

    # When writing to file, no limit unless -n given; when to terminal, default 80
    max_lines = args.max_lines
    if max_lines is None and not args.output:
        max_lines = 80

    printed = 0
    for f in sorted(by_file.keys()):
        entries = by_file[f]
        print(f"\n## {f} ({len(entries)} lines)", file=out)
        for line_no, content in entries[: (max_lines - printed) if max_lines else None]:
            print(f"  {line_no}: {content}", file=out)
            printed += 1
            if max_lines and printed >= max_lines:
                remaining = total_lines - printed
                if remaining > 0:
                    print(f"\n# ... and {remaining} more lines. Use -n N or -o file.", file=out)
                break
        if max_lines and printed >= max_lines:
            break

    if args.output:
        out.close()
        print(f"Written to {args.output}", file=sys.stderr)
    return 0 if total_lines == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
