import pandas as pd
import pytest

from hybrid_fund_allocation.analyze_allocation_effectiveness import (
    add_allocation_change_labels,
    build_allocation_action_effectiveness,
    build_company_effectiveness,
    build_company_effectiveness_stability,
    build_ranked_company_effectiveness,
    calculate_peer_relative_return,
    write_effectiveness_outputs,
)


def test_peer_relative_return_uses_same_quarter_and_style_median():
    panel = pd.DataFrame(
        {
            "基金主代码": ["A", "B", "C"],
            "报告期": ["2024Q1", "2024Q1", "2024Q1"],
            "投资风格": ["成长型", "成长型", "成长型"],
            "allocation_style_group": ["动态股债配置型"] * 3,
            "return_match_status": ["matched"] * 3,
            "next_quarter_return": [0.02, 0.04, 0.10],
        }
    )

    result = calculate_peer_relative_return(panel)

    assert result["peer_median_return"].tolist() == pytest.approx([0.04, 0.04, 0.04])
    assert result["peer_relative_return"].tolist() == pytest.approx([-0.02, 0.00, 0.06])


def test_allocation_change_labels_use_five_percentage_point_threshold():
    panel = pd.DataFrame(
        {
            "基金主代码": ["A", "A"],
            "报告期": ["2024Q1", "2024Q2"],
            "stock_weight": [0.50, 0.56],
            "bond_weight": [0.30, 0.24],
        }
    )

    result = add_allocation_change_labels(panel)

    assert result["stock_allocation_action"].tolist() == ["首次观察", "加股票"]
    assert result["bond_allocation_action"].tolist() == ["首次观察", "减债券"]


def test_company_effectiveness_uses_start_of_quarter_assets_as_weights():
    fund_panel = pd.DataFrame(
        {
            "基金管理人简称": ["甲公司", "甲公司"],
            "报告期": ["2024Q1", "2024Q1"],
            "基金主代码": ["A", "B"],
            "fund_nav_yuan": [100.0, 300.0],
            "peer_relative_return": [0.10, -0.10],
        }
    )

    company_quarter, company_total = build_company_effectiveness(fund_panel)

    assert company_quarter.loc[0, "weighted_peer_relative_return"] == pytest.approx(-0.05)
    assert company_quarter.loc[0, "positive_relative_return_ratio"] == pytest.approx(0.5)
    assert company_total.loc[0, "unique_fund_count"] == 2


def test_write_effectiveness_outputs_saves_fund_and_company_files(tmp_path):
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    panel = pd.DataFrame(
        {
            "基金主代码": ["A", "B", "A", "B"],
            "报告期": ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
            "基金管理人简称": ["甲公司", "乙公司", "甲公司", "乙公司"],
            "投资风格": ["成长型"] * 4,
            "allocation_style_group": ["动态股债配置型"] * 4,
            "return_match_status": ["matched"] * 4,
            "next_quarter_return": [0.02, 0.04, 0.03, 0.01],
            "stock_weight": [0.6, 0.4, 0.5, 0.5],
            "bond_weight": [0.2, 0.4, 0.3, 0.3],
            "fund_nav_yuan": [100.0, 200.0, 110.0, 190.0],
        }
    )
    panel.to_csv(intermediate / "fund_quarter_allocation_return.csv", index=False)

    fund_output, company_output = write_effectiveness_outputs(tmp_path)

    assert len(fund_output) == 4
    assert set(company_output["aggregation_level"]) == {"company_quarter", "company_total"}
    assert (intermediate / "fund_quarter_effectiveness.csv").exists()
    assert (intermediate / "fund_company_effectiveness.csv").exists()


def test_ranked_company_effectiveness_excludes_insufficient_sample_companies():
    company = pd.DataFrame(
        {
            "基金管理人简称": ["合格公司", "样本不足公司"],
            "aggregation_level": ["company_total", "company_total"],
            "weighted_peer_relative_return": [0.02, 0.03],
            "matched_observation_count": [20, 26],
            "unique_fund_count": [3, 2],
            "total_start_of_quarter_nav_yuan": [1000.0, 2000.0],
            "positive_relative_return_ratio": [0.6, 0.7],
        }
    )

    result = build_ranked_company_effectiveness(company)

    assert result[["基金管理人简称", "company_rank"]].to_dict("records") == [
        {"基金管理人简称": "合格公司", "company_rank": 1}
    ]


def test_allocation_action_effectiveness_reports_stock_action_results():
    fund = pd.DataFrame(
        {
            "基金主代码": ["A", "B", "C"],
            "stock_allocation_action": ["加股票", "加股票", "减股票"],
            "bond_allocation_action": ["基本不变", "基本不变", "基本不变"],
            "peer_relative_return": [0.02, 0.04, -0.01],
        }
    )

    result = build_allocation_action_effectiveness(fund)
    stock = result.query("asset_class == '股票'").set_index("allocation_action")

    assert stock.loc["加股票", "observation_count"] == 2
    assert stock.loc["加股票", "mean_peer_relative_return"] == pytest.approx(0.03)
    assert stock.loc["减股票", "mean_peer_relative_return"] == pytest.approx(-0.01)


def test_company_stability_separates_strong_and_weak_environments():
    fund = pd.DataFrame(
        {
            "报告期": ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
            "next_quarter_return": [0.10, 0.20, -0.10, -0.20],
        }
    )
    company = pd.DataFrame(
        {
            "基金管理人简称": ["甲公司", "甲公司"],
            "报告期": ["2024Q1", "2024Q2"],
            "aggregation_level": ["company_quarter", "company_quarter"],
            "weighted_peer_relative_return": [0.02, -0.01],
        }
    )
    ranking = pd.DataFrame({"基金管理人简称": ["甲公司"]})

    result = build_company_effectiveness_stability(fund, company, ranking)

    assert result.loc[0, "positive_company_quarter_ratio"] == pytest.approx(0.5)
    assert result.loc[0, "strong_environment_mean_relative_return"] == pytest.approx(0.02)
    assert result.loc[0, "weak_environment_mean_relative_return"] == pytest.approx(-0.01)
