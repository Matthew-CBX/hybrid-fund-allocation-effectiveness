import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_git(repo: Path) -> None:
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release-test@example.invalid")


def test_release_check_reports_missing_required_result(tmp_path: Path):
    from scripts.verify_public_release import check_required_paths

    violations = check_required_paths(tmp_path)

    assert any("enhanced_company_evaluation.csv" in item for item in violations)


def test_release_check_reports_broken_markdown_link(tmp_path: Path):
    from scripts.verify_public_release import check_markdown_links

    (tmp_path / "README.md").write_text("[missing](docs/missing.md)")

    assert check_markdown_links(tmp_path) == [
        "broken Markdown link: README.md -> docs/missing.md"
    ]


def test_release_check_reports_failed_audit(tmp_path: Path):
    from scripts.verify_public_release import check_audit_results

    result_dir = tmp_path / "data" / "results"
    result_dir.mkdir(parents=True)
    (result_dir / "enhanced_audit.csv").write_text(
        "检查项目,实际结果,预期或判断标准,检查结论,说明\n测试,1,0,失败,故意失败\n",
        encoding="utf-8-sig",
    )

    assert check_audit_results(tmp_path) == ["published audit contains 1 failed row(s)"]


def test_release_check_reports_workbook_local_path_metadata(tmp_path: Path):
    from scripts.verify_public_release import check_workbook

    workbook = tmp_path / "report.xlsx"
    local_path = "/" + "Users/example/Desktop/input.xlsx"
    with ZipFile(workbook, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", f"<workbook>{local_path}</workbook>")

    violations = check_workbook(workbook)

    assert any("local or personal metadata" in item for item in violations)


def test_release_check_accepts_artifact_tool_chart_member_paths(tmp_path: Path):
    from scripts.verify_public_release import check_workbook

    workbook = tmp_path / "report.xlsx"
    sheet_names = [
        "改进版总览", "可信度评价", "年化收益比较", "主动调仓明细", "阈值稳健性",
        "方向与环境", "三资产基准", "产品集中度", "极端值敏感性", "口径与限制", "审计",
    ]
    sheets = "".join(f'<sheet name="{name}" />' for name in sheet_names)
    with ZipFile(workbook, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", f"<workbook><sheets>{sheets}</sheets></workbook>")
        for number in range(1, 4):
            archive.writestr(
                f"xl/drawings/charts/chart{number}.xml",
                f"<chart id=\"{number}\" />",
            )

    assert check_workbook(workbook) == []


def test_release_check_reports_large_tracked_file(tmp_path: Path):
    from scripts.verify_public_release import check_git_repository

    _init_git(tmp_path)
    large = tmp_path / "large.bin"
    with large.open("wb") as stream:
        stream.seek(25 * 1024 * 1024)
        stream.write(b"0")
    _git(tmp_path, "add", "large.bin")

    violations = check_git_repository(tmp_path)

    assert any("tracked file exceeds 25 MiB" in item for item in violations)


def test_release_check_reports_raw_path_retained_in_git_history(tmp_path: Path):
    from scripts.verify_public_release import check_git_repository

    _init_git(tmp_path)
    raw_dir = tmp_path / "source_data"
    raw_dir.mkdir()
    raw = raw_dir / "raw.xlsx"
    raw.write_bytes(b"raw")
    _git(tmp_path, "add", "source_data/raw.xlsx")
    _git(tmp_path, "commit", "-m", "add raw fixture")
    raw.unlink()
    _git(tmp_path, "add", "-u")
    _git(tmp_path, "commit", "-m", "remove raw fixture")

    violations = check_git_repository(tmp_path)

    assert any("banned Git history path: source_data/raw.xlsx" in item for item in violations)
