from pathlib import Path


def test_boundary_rejects_raw_directory_local_path_secret_and_large_file(
    tmp_path: Path,
):
    from scripts.verify_public_boundary import collect_boundary_violations

    raw_dir = tmp_path / "source_data"
    raw_dir.mkdir()
    (raw_dir / "raw.xlsx").write_bytes(b"raw")
    local_path = "/" + "Users/example/Desktop/input.xlsx"
    (tmp_path / "notes.md").write_text(local_path)
    private_key_marker = "-----BEGIN " + "PRIVATE KEY-----"
    (tmp_path / "key.txt").write_text(private_key_marker)
    large = tmp_path / "large.bin"
    with large.open("wb") as stream:
        stream.seek(25 * 1024 * 1024)
        stream.write(b"0")

    violations = collect_boundary_violations(tmp_path)

    assert any("source_data" in item for item in violations)
    assert any("absolute path" in item for item in violations)
    assert any("private key" in item for item in violations)
    assert any("25 MiB" in item for item in violations)


def test_boundary_allows_derived_results_and_source_attribution(tmp_path: Path):
    from scripts.verify_public_boundary import collect_boundary_violations

    result_dir = tmp_path / "data" / "results"
    result_dir.mkdir(parents=True)
    (result_dir / "company.csv").write_text("company,metric\nA,0.1\n")
    (tmp_path / "README.md").write_text(
        "原始数据来自 Choice，未随仓库发布。",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".pytest_cache"
    cache_dir.mkdir()
    (cache_dir / "README.md").write_text("pytest cache")

    assert collect_boundary_violations(tmp_path) == []
