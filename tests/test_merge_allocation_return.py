import pandas as pd
import pytest

from hybrid_fund_allocation.merge_allocation_return import (
    calculate_forward_quarter_return,
    build_allocation_style_classification,
    build_allocation_return_panel,
    build_merge_log,
    choose_representative_share,
    merge_allocation_with_returns,
    normalize_nav_export,
    classify_dynamic_allocation,
)


def test_normalize_nav_export_removes_footer_and_maps_reporting_quarter():
    raw = pd.DataFrame(
        {
            "证券代码": ["000001.OF", "数据来源：妙想Choice"],
            "证券名称": ["样本基金", None],
            "复权单位净值\n[交易日期]2023-09-29\n[单位]元": [1.2, None],
        }
    )

    result = normalize_nav_export(raw)

    assert result.to_dict("records") == [
        {"security_code": "000001.OF", "quarter": "2023Q3", "adjusted_nav": 1.2}
    ]


def test_normalize_nav_export_keeps_exchange_listed_fund_codes():
    raw = pd.DataFrame(
        {
            "证券代码": ["160105.SZ", "501064.SH", "数据来源：妙想Choice"],
            "复权单位净值\n[交易日期]2023-09-29\n[单位]元": [1.1, 1.2, None],
        }
    )

    result = normalize_nav_export(raw)

    assert result["security_code"].tolist() == ["160105.SZ", "501064.SH"]


def test_forward_return_uses_following_quarter_nav():
    nav_long = pd.DataFrame(
        {
            "security_code": ["000001.OF", "000001.OF"],
            "quarter": ["2023Q1", "2023Q2"],
            "adjusted_nav": [1.0, 1.1],
        }
    )

    result = calculate_forward_quarter_return(nav_long)

    assert result[["security_code", "quarter"]].to_dict("records") == [
        {"security_code": "000001.OF", "quarter": "2023Q1"}
    ]
    assert result.loc[0, "next_quarter_return"] == pytest.approx(0.1)


def test_representative_share_prefers_main_code_then_smallest_share_code():
    base = pd.DataFrame(
        {
            "证券代码": ["000002.OF", "000001.OF", "000004.OF", "000003.OF"],
            "基金主代码": ["000001.OF", "000001.OF", "000003.OF", "000003.OF"],
        }
    )

    result = choose_representative_share(base)

    assert result.to_dict("records") == [
        {"fund_main_code": "000001.OF", "security_code": "000001.OF"},
        {"fund_main_code": "000003.OF", "security_code": "000003.OF"},
    ]


def test_merge_allocation_with_returns_marks_unmatched_funds():
    allocation = pd.DataFrame(
        {
            "报告期": ["2023Q1", "2023Q1"],
            "基金主代码": ["000001.OF", "000002.OF"],
            "异常标记": ["可用", "可用"],
        }
    )
    representatives = pd.DataFrame(
        {
            "fund_main_code": ["000001.OF", "000002.OF"],
            "security_code": ["000001.OF", "000002.OF"],
        }
    )
    forward_returns = pd.DataFrame(
        {
            "security_code": ["000001.OF"],
            "quarter": ["2023Q1"],
            "next_quarter_return": [0.05],
        }
    )

    result = merge_allocation_with_returns(allocation, representatives, forward_returns)

    assert result["return_match_status"].tolist() == ["matched", "missing_nav_return"]
    assert result.loc[0, "next_quarter_return"] == pytest.approx(0.05)


def test_build_panel_keeps_only_quarters_with_forward_return(tmp_path):
    pd.DataFrame(
        {"证券代码": ["000001.OF"], "基金主代码": ["000001.OF"]}
    ).to_excel(tmp_path / "Book2.xlsx", index=False)
    pd.DataFrame(
        {
            "报告期": ["2023Q1", "2023Q2"],
            "基金主代码": ["000001.OF", "000001.OF"],
            "异常标记": ["可用", "可用"],
        }
    ).to_csv(tmp_path / "clean_fund_quarter.csv", index=False)
    nav_path = tmp_path / "nav.xlsx"
    pd.DataFrame(
        {
            "证券代码": ["000001.OF"],
            "复权单位净值\n[交易日期]2023-03-31\n[单位]元": [1.0],
            "复权单位净值\n[交易日期]2023-06-30\n[单位]元": [1.1],
        }
    ).to_excel(nav_path, index=False)

    result = build_allocation_return_panel(tmp_path, nav_path)

    assert result[["报告期", "return_match_status"]].to_dict("records") == [
        {"报告期": "2023Q1", "return_match_status": "matched"}
    ]
    assert result.loc[0, "next_quarter_return"] == pytest.approx(0.1)


def test_build_merge_log_reports_total_and_match_statuses():
    panel = pd.DataFrame({"return_match_status": ["matched", "missing_nav_return"]})

    result = build_merge_log(panel)

    assert set(result["stage"]) == {"panel_total", "matched", "missing_nav_return"}


def test_classify_dynamic_allocation_requires_large_stock_and_bond_changes():
    quarters = [f"2023Q{i}" for i in range(1, 5)] + [f"2024Q{i}" for i in range(1, 5)]
    dynamic = pd.DataFrame(
        {
            "基金主代码": ["dynamic.OF"] * 8,
            "报告期": quarters,
            "异常标记": ["可用"] * 8,
            "stock_weight": [0.70, 0.40, 0.65, 0.45, 0.68, 0.42, 0.66, 0.43],
            "bond_weight": [0.10, 0.40, 0.15, 0.38, 0.12, 0.39, 0.14, 0.37],
        }
    )
    static = pd.DataFrame(
        {
            "基金主代码": ["static.OF"] * 8,
            "报告期": quarters,
            "异常标记": ["可用"] * 8,
            "stock_weight": [0.80, 0.81, 0.82, 0.80, 0.81, 0.82, 0.80, 0.81],
            "bond_weight": [0.04, 0.05, 0.04, 0.05, 0.04, 0.05, 0.04, 0.05],
        }
    )

    result = classify_dynamic_allocation(pd.concat([dynamic, static], ignore_index=True))

    groups = result.set_index("基金主代码")["allocation_style_group"].to_dict()
    assert groups == {"dynamic.OF": "动态股债配置型", "static.OF": "静态配置型"}


def test_allocation_style_classification_uses_full_clean_history(tmp_path):
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    quarters = [f"2023Q{i}" for i in range(1, 5)] + [f"2024Q{i}" for i in range(1, 5)]
    pd.DataFrame(
        {
            "基金主代码": ["dynamic.OF"] * 8,
            "报告期": quarters,
            "异常标记": ["可用"] * 8,
            "stock_weight": [0.70, 0.40, 0.65, 0.45, 0.68, 0.42, 0.66, 0.43],
            "bond_weight": [0.10, 0.40, 0.15, 0.38, 0.12, 0.39, 0.14, 0.37],
        }
    ).to_csv(intermediate / "clean_fund_quarter.csv", index=False)

    result = build_allocation_style_classification(tmp_path)

    assert result.loc[0, "valid_quarter_count"] == 8
    assert result.loc[0, "allocation_style_group"] == "动态股债配置型"
