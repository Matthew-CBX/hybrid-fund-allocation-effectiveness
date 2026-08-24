"""Reject files that do not belong in the public portfolio repository."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BANNED_PARTS = {
    "source_data",
    "intermediate",
    "node_modules",
}
GENERATED_PARTS = {"__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".csv",
    ".json",
    ".yml",
    ".yaml",
}
MAC_USER_PREFIX = "/" + "Users/"
PRIVATE_KEY_MARKER = "-----BEGIN " + "PRIVATE KEY-----"
SENSITIVE_PATTERNS = {
    "absolute path": re.compile(re.escape(MAC_USER_PREFIX) + r"[^/\s]+/"),
    "private key": re.compile(re.escape(PRIVATE_KEY_MARKER)),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
}
MAX_FILE_SIZE = 25 * 1024 * 1024


def collect_boundary_violations(repo_root: Path) -> list[str]:
    """Return deterministic descriptions of public-boundary violations."""
    repo_root = Path(repo_root)
    violations: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(repo_root)
        if GENERATED_PARTS.intersection(relative.parts):
            continue
        if BANNED_PARTS.intersection(relative.parts):
            violations.append(f"banned path: {relative}")
        if path.stat().st_size > MAX_FILE_SIZE:
            violations.append(f"file exceeds 25 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{label}: {relative}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", nargs="?", default=".", type=Path)
    args = parser.parse_args()
    violations = collect_boundary_violations(args.repo_root.resolve())
    if violations:
        print("\n".join(violations))
        return 1
    print("Public boundary check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
