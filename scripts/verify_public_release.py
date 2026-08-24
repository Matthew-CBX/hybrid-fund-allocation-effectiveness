"""Aggregate checks that must pass before the public repository is pushed."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import BadZipFile, ZipFile

try:
    from scripts.verify_public_boundary import MAX_FILE_SIZE, collect_boundary_violations
except ModuleNotFoundError:
    from verify_public_boundary import MAX_FILE_SIZE, collect_boundary_violations


RESULT_SHAPES = {
    "enhanced_company_evaluation.csv": (101, 30),
    "enhanced_company_annualized_return.csv": (101, 15),
    "enhanced_fund_annualized_return.csv": (376, 24),
    "enhanced_threshold_robustness.csv": (303, 11),
    "enhanced_extreme_sensitivity.csv": (100, 7),
    "enhanced_direction_environment.csv": (425, 8),
    "enhanced_product_concentration.csv": (101, 9),
    "enhanced_audit.csv": (17, 5),
}
REPORT_PATH = Path("reports/hybrid_fund_allocation_effectiveness_report.xlsx")
REQUIRED_PATHS = [
    Path("README.md"),
    Path("NOTICE.md"),
    Path("requirements.txt"),
    Path("pyproject.toml"),
    Path("docs/methodology.md"),
    Path("docs/data-dictionary.md"),
    Path("docs/findings.md"),
    Path("data/sample/fund_quarter_demo.csv"),
    REPORT_PATH,
] + [Path("data/results") / name for name in RESULT_SHAPES]
EXPECTED_SHEETS = [
    "改进版总览",
    "可信度评价",
    "年化收益比较",
    "主动调仓明细",
    "阈值稳健性",
    "方向与环境",
    "三资产基准",
    "产品集中度",
    "极端值敏感性",
    "口径与限制",
    "审计",
]
MAC_USER_PREFIX = "/" + "Users/"
LOCAL_OR_PERSONAL = (MAC_USER_PREFIX, "file://", "Desktop/", "chen" + "boxi")
FORMULA_ERRORS = ("#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A")
BANNED_HISTORY_PARTS = (
    "source_data/",
    "intermediate/",
    "data/raw/",
    ".inspect.ndjson",
    "混合型开放式基金.xlsx",
    "混合基金季度资产配置基础数据",
    "Book2.xlsx",
)


def check_required_paths(repo_root: Path) -> list[str]:
    return [
        f"missing required path: {relative}"
        for relative in REQUIRED_PATHS
        if not (Path(repo_root) / relative).is_file()
    ]


def check_markdown_links(repo_root: Path) -> list[str]:
    repo_root = Path(repo_root)
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    violations: list[str] = []
    for markdown in sorted(repo_root.rglob("*.md")):
        relative = markdown.relative_to(repo_root)
        if {".git", ".pytest_cache", "__pycache__"}.intersection(relative.parts):
            continue
        text = markdown.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = target.split("#", 1)[0]
            if path_part and not (markdown.parent / path_part).resolve().exists():
                violations.append(f"broken Markdown link: {relative} -> {target}")
    return violations


def _read_csv_shape(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    return max(len(rows) - 1, 0), len(rows[0]) if rows else 0


def check_result_tables(repo_root: Path) -> list[str]:
    repo_root = Path(repo_root)
    violations: list[str] = []
    for name, expected in RESULT_SHAPES.items():
        path = repo_root / "data" / "results" / name
        if not path.is_file():
            continue
        actual = _read_csv_shape(path)
        if actual != expected:
            violations.append(f"result shape mismatch: {name} {actual} != {expected}")
    return violations


def check_audit_results(repo_root: Path) -> list[str]:
    path = Path(repo_root) / "data" / "results" / "enhanced_audit.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    failed = sum(row.get("检查结论") != "通过" for row in rows)
    return [f"published audit contains {failed} failed row(s)"] if failed else []


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def check_workbook(workbook_path: Path) -> list[str]:
    workbook_path = Path(workbook_path)
    violations: list[str] = []
    if not workbook_path.is_file():
        return [f"missing workbook: {workbook_path}"]
    try:
        with ZipFile(workbook_path) as archive:
            corrupt = archive.testzip()
            if corrupt:
                violations.append(f"corrupt workbook member: {corrupt}")
            names = archive.namelist()
            if any(name.startswith("xl/externalLinks/") for name in names):
                violations.append("workbook contains external links")
            xml_text: list[str] = []
            for name in names:
                if name.endswith((".xml", ".rels")):
                    xml_text.append(archive.read(name).decode("utf-8", errors="replace"))
            combined = "\n".join(xml_text)
            if any(fragment in combined for fragment in LOCAL_OR_PERSONAL):
                violations.append("workbook contains local or personal metadata")
            if any(error in combined for error in FORMULA_ERRORS):
                violations.append("workbook contains a formula error token")
            if "xl/workbook.xml" not in names:
                violations.append("workbook is missing xl/workbook.xml")
            else:
                root = ET.fromstring(archive.read("xl/workbook.xml"))
                sheets = [
                    element.attrib["name"]
                    for element in root.iter()
                    if _local_name(element.tag) == "sheet" and "name" in element.attrib
                ]
                if sheets != EXPECTED_SHEETS:
                    violations.append(f"workbook sheet order mismatch: {sheets}")
            chart_count = sum(
                re.fullmatch(r"xl/(?:[^/]+/)*charts/chart\d+\.xml", name) is not None
                for name in names
            )
            if chart_count != 3:
                violations.append(f"workbook chart count mismatch: {chart_count} != 3")
    except (BadZipFile, ET.ParseError) as error:
        violations.append(f"invalid workbook: {error}")
    return violations


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def check_git_repository(repo_root: Path) -> list[str]:
    repo_root = Path(repo_root)
    if not (repo_root / ".git").exists():
        return ["missing Git repository"]
    violations: list[str] = []
    tracked = _git_output(repo_root, "ls-files", "-z").split("\0")
    for relative in filter(None, tracked):
        path = repo_root / relative
        if path.is_file() and path.stat().st_size > MAX_FILE_SIZE:
            violations.append(f"tracked file exceeds 25 MiB: {relative}")
    history = _git_output(repo_root, "rev-list", "--objects", "--all").splitlines()
    for record in history:
        _, separator, relative = record.partition(" ")
        if separator and any(part in relative for part in BANNED_HISTORY_PARTS):
            violations.append(f"banned Git history path: {relative}")
    return sorted(set(violations))


def verify_release(repo_root: Path) -> list[str]:
    repo_root = Path(repo_root).resolve()
    violations = collect_boundary_violations(repo_root)
    violations.extend(check_required_paths(repo_root))
    violations.extend(check_markdown_links(repo_root))
    violations.extend(check_result_tables(repo_root))
    violations.extend(check_audit_results(repo_root))
    violations.extend(check_workbook(repo_root / REPORT_PATH))
    violations.extend(check_git_repository(repo_root))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", nargs="?", default=".", type=Path)
    args = parser.parse_args()
    violations = verify_release(args.repo_root)
    if violations:
        print("\n".join(violations))
        return 1
    print(
        "Public release verification: PASS "
        f"({len(RESULT_SHAPES)} result files, {len(EXPECTED_SHEETS)} sheets, 3 charts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
