import pandas as pd
import pytest
import json

from hybrid_fund_allocation.analyze_csi300_benchmark import (
    add_csi300_effectiveness_metrics,
    build_company_csi300_effectiveness,
    build_company_csi300_timing_ranking,
    build_fund_csi300_effectiveness,
    build_quarterly_csi300_benchmark,
    write_csi300_outputs,
)


def test_quarterly_benchmark_uses_last_trading_day_and_forward_return():
    daily = pd.DataFrame(
        {
            "tradeDate": ["20240328", "20240329", "20240628"],
            "close": [100.0, 110.0, 121.0],
        }
    )

    result = build_quarterly_csi300_benchmark(daily)

    assert result[["报告期", "trade_date", "csi300_close"]].to_dict("records") == [
        {"报告期": "2024Q1", "trade_date": "2024-03-29", "csi300_close": 110.0},
        {"报告期": "2024Q2", "trade_date": "2024-06-28", "csi300_close": 121.0},
    ]
    assert result.loc[0, "csi300_next_quarter_return"] == pytest.approx(0.10)
    assert pd.isna(result.loc[1, "csi300_next_quarter_return"])


def test_csi300_metrics_measure_excess_return_and_stock_timing_direction():
    panel = pd.DataFrame(
        {
            "基金主代码": ["A", "A", "B", "B"],
            "报告期": ["2024Q1", "2024Q2", "2024Q1", "2024Q2"],
            "allocation_style_group": ["动态股债配置型"] * 4,
            "return_match_status": ["matched"] * 4,
            "异常标记": ["可用"] * 4,
            "stock_weight": [0.40, 0.50, 0.40, 0.30],
            "bond_weight": [0.40, 0.30, 0.40, 0.50],
            "next_quarter_return": [0.02, 0.15, 0.02, 0.05],
        }
    )
    benchmark = pd.DataFrame(
        {
            "报告期": ["2024Q1", "2024Q2"],
            "csi300_next_quarter_return": [-0.02, 0.10],
        }
    )

    result = add_csi300_effectiveness_metrics(panel, benchmark).set_index(["基金主代码", "报告期"])

    assert result.loc[("A", "2024Q2"), "excess_return_vs_csi300"] == pytest.approx(0.05)
    assert result.loc[("A", "2024Q2"), "outperformed_csi300"] == 1
    assert result.loc[("A", "2024Q2"), "stock_timing_hit"] == 1
    assert result.loc[("A", "2024Q2"), "simplified_stock_timing_contribution"] == pytest.approx(0.01)
    assert result.loc[("B", "2024Q2"), "stock_timing_hit"] == 0
    assert result.loc[("B", "2024Q2"), "simplified_stock_timing_contribution"] == pytest.approx(-0.01)


def test_stock_timing_requires_current_and_previous_usable_allocations():
    panel = pd.DataFrame(
        {
            "基金主代码": ["A", "A"],
            "报告期": ["2024Q1", "2024Q2"],
            "allocation_style_group": ["动态股债配置型"] * 2,
            "return_match_status": ["matched"] * 2,
            "异常标记": ["资产配置披露不完整", "可用"],
            "stock_weight": [0.40, 0.50],
            "bond_weight": [0.40, 0.30],
            "next_quarter_return": [0.02, 0.15],
        }
    )
    benchmark = pd.DataFrame(
        {"报告期": ["2024Q1", "2024Q2"], "csi300_next_quarter_return": [-0.02, 0.10]}
    )

    result = add_csi300_effectiveness_metrics(panel, benchmark)

    assert result.loc[result["报告期"].eq("2024Q2"), "stock_timing_assessment"].item() == "配置数据不可比较"
    assert pd.isna(result.loc[result["报告期"].eq("2024Q2"), "stock_timing_hit"].item())


def test_company_effectiveness_uses_nav_weighted_benchmark_excess_return():
    fund = pd.DataFrame(
        {
            "基金管理人简称": ["甲公司", "甲公司"],
            "报告期": ["2024Q2", "2024Q2"],
            "基金主代码": ["A", "B"],
            "fund_nav_yuan": [100.0, 300.0],
            "excess_return_vs_csi300": [0.10, -0.10],
            "outperformed_csi300": [1, 0],
            "csi300_market_environment": ["上涨", "上涨"],
            "stock_timing_hit": [1.0, 0.0],
            "simplified_stock_timing_contribution": [0.02, -0.01],
        }
    )

    company_quarter, company_total = build_company_csi300_effectiveness(fund)

    assert company_quarter.loc[0, "weighted_excess_return_vs_csi300"] == pytest.approx(-0.05)
    assert company_total.loc[0, "csi300_outperformance_ratio"] == pytest.approx(0.5)
    assert company_total.loc[0, "stock_timing_hit_ratio"] == pytest.approx(0.5)


def test_fund_effectiveness_summarizes_up_and_down_market_results():
    fund_quarter = pd.DataFrame(
        {
            "基金主代码": ["A", "A"],
            "基金名称": ["样本基金A", "样本基金A"],
            "基金管理人简称": ["甲公司", "甲公司"],
            "投资风格": ["平衡型", "平衡型"],
            "excess_return_vs_csi300": [0.10, -0.04],
            "outperformed_csi300": [1, 0],
            "csi300_market_environment": ["上涨", "下跌"],
            "stock_timing_hit": [1.0, 0.0],
            "simplified_stock_timing_contribution": [0.02, -0.01],
        }
    )

    result = build_fund_csi300_effectiveness(fund_quarter)

    assert result.loc[0, "mean_excess_return_vs_csi300"] == pytest.approx(0.03)
    assert result.loc[0, "annualized_information_ratio_vs_csi300"] == pytest.approx(0.6060915)
    assert result.loc[0, "up_market_mean_excess_return"] == pytest.approx(0.10)
    assert result.loc[0, "down_market_mean_excess_return"] == pytest.approx(-0.04)
    assert result.loc[0, "stock_timing_hit_ratio"] == pytest.approx(0.5)


def test_timing_ranking_enforces_minimum_fund_and_observation_counts():
    company = pd.DataFrame(
        {
            "基金管理人简称": ["甲公司", "样本不足公司", "乙公司"],
            "stock_timing_hit_ratio": [0.60, 0.80, 0.50],
            "mean_simplified_stock_timing_contribution": [0.01, 0.03, 0.02],
            "stock_timing_observation_count": [12, 20, 10],
            "stock_timing_unique_fund_count": [3, 2, 3],
            "weighted_excess_return_vs_csi300": [0.02, 0.05, 0.01],
        }
    )

    result = build_company_csi300_timing_ranking(company)

    assert result[["stock_timing_rank", "基金管理人简称"]].to_dict("records") == [
        {"stock_timing_rank": 1, "基金管理人简称": "甲公司"},
        {"stock_timing_rank": 2, "基金管理人简称": "乙公司"},
    ]


def test_write_csi300_outputs_creates_auditable_project_files(tmp_path):
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    panel = pd.DataFrame(
        {
            "基金主代码": ["A", "A"],
            "基金名称": ["样本基金A", "样本基金A"],
            "基金管理人简称": ["甲公司", "甲公司"],
            "投资风格": ["平衡型", "平衡型"],
            "报告期": ["2024Q1", "2024Q2"],
            "allocation_style_group": ["动态股债配置型"] * 2,
            "return_match_status": ["matched"] * 2,
            "异常标记": ["可用"] * 2,
            "stock_weight": [0.40, 0.50],
            "bond_weight": [0.40, 0.30],
            "next_quarter_return": [0.03, 0.15],
            "fund_nav_yuan": [100.0, 110.0],
        }
    )
    panel.to_csv(intermediate / "fund_quarter_allocation_return.csv", index=False)
    source = tmp_path / "csi300.json"
    source.write_text(
        json.dumps(
            {
                "code": "200",
                "msg": "Success",
                "data": [
                    {"tradeDate": "20240329", "close": 100.0},
                    {"tradeDate": "20240628", "close": 110.0},
                    {"tradeDate": "20240930", "close": 121.0},
                ],
            }
        )
    )

    write_csi300_outputs(tmp_path, source)

    expected = {
        "csi300_quarterly_benchmark.csv",
        "fund_quarter_csi300_effectiveness.csv",
        "fund_csi300_effectiveness.csv",
        "fund_company_csi300_effectiveness.csv",
        "fund_company_csi300_timing_ranking.csv",
        "csi300_benchmark_audit.csv",
    }
    assert expected.issubset({path.name for path in intermediate.glob("*.csv")})
    detail = pd.read_csv(intermediate / "fund_quarter_csi300_effectiveness.csv")
    assert detail.loc[detail["报告期"].eq("2024Q2"), "excess_return_vs_csi300"].item() == pytest.approx(0.05)
