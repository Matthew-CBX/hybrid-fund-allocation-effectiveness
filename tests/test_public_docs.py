import re
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(".")
RESULT_DIR = ROOT / "data" / "results"


def test_published_project_facts_recompute_from_result_tables():
    funds = pd.read_csv(
        RESULT_DIR / "enhanced_fund_annualized_return.csv",
        dtype={"基金主代码": "string"},
    )
    companies = pd.read_csv(RESULT_DIR / "enhanced_company_evaluation.csv")
    formal_return = funds["annualized_evaluation_pool"].eq("正式年化")
    formal_contribution = funds["contribution_evaluation_pool"].eq("正式贡献")

    assert len(funds) == 376
    assert formal_return.sum() == 269
    assert formal_contribution.sum() == 332
    assert len(companies) == 101
    assert companies["evaluation_pool"].eq("正式评价").sum() == 16
    assert companies["credibility_grade"].value_counts().to_dict() == {
        "观察池": 85,
        "B级": 8,
        "C级": 8,
    }
    assert funds.loc[formal_return, "annualized_fund_return"].median() == pytest.approx(
        0.04566878051424439
    )
    assert funds.loc[
        formal_return, "annualized_three_asset_benchmark_return"
    ].median() == pytest.approx(0.04787931423040548)
    assert funds.loc[
        formal_return, "annualized_excess_vs_three_asset"
    ].median() == pytest.approx(-0.00525291032371733)
    assert funds.loc[
        formal_contribution, "annualized_active_allocation_contribution"
    ].median() == pytest.approx(-0.0001485412460169866)
    assert funds.loc[formal_return, "annualized_excess_vs_three_asset"].gt(0).sum() == 125
    assert funds.loc[
        formal_contribution, "annualized_active_allocation_contribution"
    ].gt(0).sum() == 160


def test_public_markdown_links_resolve_and_no_local_path_is_exposed():
    required = [
        ROOT / "README.md",
        ROOT / "NOTICE.md",
        ROOT / "docs" / "methodology.md",
        ROOT / "docs" / "findings.md",
    ]
    for path in required:
        assert path.is_file()

    mac_user_prefix = "/" + "Users/"
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for markdown in ROOT.rglob("*.md"):
        if ".git" in markdown.parts:
            continue
        text = markdown.read_text(encoding="utf-8")
        assert mac_user_prefix not in text
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = (markdown.parent / target.split("#", 1)[0]).resolve()
            assert resolved.exists(), f"broken link in {markdown}: {target}"


def test_public_repository_has_no_license_file():
    assert not any(path.name.lower().startswith("license") for path in ROOT.iterdir())
