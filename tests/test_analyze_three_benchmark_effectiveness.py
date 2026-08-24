import json

import pandas as pd
import pytest

from hybrid_fund_allocation.analyze_three_benchmark_effectiveness import (
    add_three_benchmark_metrics,
    build_company_three_benchmark_effectiveness,
    build_fund_three_benchmark_effectiveness,
    build_quarterly_bond_benchmark,
    build_stock_bond_switching_ranking,
    write_three_benchmark_outputs,
)


def test_quarterly_bond_benchmark_uses_last_observation_and_forward_return():
    raw = {
        "CFZS_00": {
            "1711555200000": 100.0,
            "1711641600000": 101.0,
            "1719504000000": 103.02,
        }
    }

    result = build_quarterly_bond_benchmark(raw)

    assert result[["报告期", "trade_date", "bond_index_close"]].to_dict("records") == [
        {"报告期": "2024Q1", "trade_date": "2024-03-29", "bond_index_close": 101.0},
        {"报告期": "2024Q2", "trade_date": "2024-06-28", "bond_index_close": 103.02},
    ]
    assert result.loc[0, "bond_next_quarter_return"] == pytest.approx(0.02)
    assert pd.isna(result.loc[1, "bond_next_quarter_return"])


def _fund_quarter_inputs():
    csi = pd.DataFrame(
        {
            "基金主代码": ["A", "A"],
            "基金名称": ["样本基金A", "样本基金A"],
            "基金管理人简称": ["甲公司", "甲公司"],
            "投资风格": ["平衡型", "平衡型"],
            "报告期": ["2024Q1", "2024Q2"],
            "异常标记": ["可用", "可用"],
            "fund_nav_yuan": [100.0, 120.0],
            "stock_weight": [0.40, 0.70],
            "bond_weight": [0.60, 0.30],
            "next_quarter_return": [0.07, -0.02],
            "csi300_next_quarter_return": [0.10, -0.05],
            "excess_return_vs_csi300": [-0.03, 0.03],
            "outperformed_csi300": [0, 1],
            "stock_timing_hit": [pd.NA, 0.0],
            "simplified_stock_timing_contribution": [pd.NA, -0.015],
        }
    )
    peer = pd.DataFrame(
        {
            "基金主代码": ["A", "A"],
            "报告期": ["2024Q1", "2024Q2"],
            "peer_median_return": [0.05, -0.03],
            "peer_group_size": [20, 21],
            "peer_relative_return": [0.02, 0.01],
        }
    )
    bond = pd.DataFrame(
        {
            "报告期": ["2024Q1", "2024Q2"],
            "bond_next_quarter_return": [0.02, 0.01],
        }
    )
    return csi, peer, bond


def test_three_benchmark_metrics_build_normalized_dynamic_benchmark_and_switching():
    csi, peer, bond = _fund_quarter_inputs()

    result = add_three_benchmark_metrics(csi, peer, bond).set_index("报告期")

    assert result.loc["2024Q1", "normalized_stock_weight"] == pytest.approx(0.40)
    assert result.loc["2024Q1", "normalized_bond_weight"] == pytest.approx(0.60)
    assert result.loc["2024Q1", "dynamic_stock_bond_benchmark_return"] == pytest.approx(0.052)
    assert result.loc["2024Q1", "excess_return_vs_dynamic_stock_bond"] == pytest.approx(0.018)
    assert result.loc["2024Q1", "excess_return_vs_bond"] == pytest.approx(0.05)
    assert result.loc["2024Q1", "peer_relative_return"] == pytest.approx(0.02)
    assert result.loc["2024Q2", "stock_bond_switching_contribution"] == pytest.approx(-0.018)
    assert result.loc["2024Q2", "stock_bond_switching_hit"] == 0
    assert result.loc["2024Q2", "stock_bond_switching_assessment"] == "方向错误"


def test_dynamic_benchmark_requires_usable_allocation_and_positive_stock_bond_total():
    csi, peer, bond = _fund_quarter_inputs()
    csi.loc[0, "异常标记"] = "资产配置披露不完整"
    csi.loc[1, ["stock_weight", "bond_weight"]] = 0.0

    result = add_three_benchmark_metrics(csi, peer, bond)

    assert result["dynamic_stock_bond_benchmark_return"].isna().all()
    assert result["stock_bond_switching_hit"].isna().all()


def test_company_summary_uses_nav_weighted_excess_and_reports_switching():
    fund = pd.DataFrame(
        {
            "基金管理人简称": ["甲公司", "甲公司"],
            "报告期": ["2024Q1", "2024Q1"],
            "基金主代码": ["A", "B"],
            "fund_nav_yuan": [100.0, 300.0],
            "peer_relative_return": [0.10, -0.10],
            "excess_return_vs_csi300": [0.08, -0.04],
            "excess_return_vs_bond": [0.07, -0.03],
            "excess_return_vs_dynamic_stock_bond": [0.06, -0.02],
            "outperformed_csi300": [1, 0],
            "outperformed_bond": [1, 0],
            "outperformed_dynamic_stock_bond": [1, 0],
            "stock_timing_hit": [1.0, 0.0],
            "stock_bond_switching_hit": [1.0, 0.0],
            "stock_bond_switching_contribution": [0.02, -0.01],
        }
    )

    _, total = build_company_three_benchmark_effectiveness(fund)

    assert total.loc[0, "weighted_peer_relative_return"] == pytest.approx(-0.05)
    assert total.loc[0, "weighted_excess_return_vs_dynamic_stock_bond"] == pytest.approx(0.0)
    assert total.loc[0, "stock_bond_switching_hit_ratio"] == pytest.approx(0.5)
    assert total.loc[0, "mean_stock_bond_switching_contribution"] == pytest.approx(0.005)


def test_switching_ranking_enforces_fund_return_and_switching_sample_thresholds():
    company = pd.DataFrame(
        {
            "基金管理人简称": ["甲公司", "切换样本不足", "基金样本不足", "乙公司"],
            "matched_observation_count": [30, 30, 30, 20],
            "unique_fund_count": [4, 4, 2, 3],
            "stock_bond_switching_observation_count": [12, 9, 20, 10],
            "stock_bond_switching_unique_fund_count": [4, 4, 2, 3],
            "stock_bond_switching_hit_ratio": [0.60, 0.90, 0.80, 0.50],
            "mean_stock_bond_switching_contribution": [0.01, 0.03, 0.02, 0.02],
            "weighted_excess_return_vs_dynamic_stock_bond": [0.02, 0.04, 0.03, 0.01],
        }
    )

    result = build_stock_bond_switching_ranking(company)

    assert result[["stock_bond_switching_rank", "基金管理人简称"]].to_dict("records") == [
        {"stock_bond_switching_rank": 1, "基金管理人简称": "甲公司"},
        {"stock_bond_switching_rank": 2, "基金管理人简称": "乙公司"},
    ]


def test_fund_summary_reports_all_benchmarks_and_switching_metrics():
    csi, peer, bond = _fund_quarter_inputs()
    detail = add_three_benchmark_metrics(csi, peer, bond)

    result = build_fund_three_benchmark_effectiveness(detail)

    assert result.loc[0, "mean_peer_relative_return"] == pytest.approx(0.015)
    assert result.loc[0, "mean_excess_return_vs_csi300"] == pytest.approx(0.0)
    assert result.loc[0, "mean_excess_return_vs_bond"] == pytest.approx(0.01)
    assert result.loc[0, "mean_excess_return_vs_dynamic_stock_bond"] == pytest.approx(0.015)
    assert result.loc[0, "stock_bond_switching_observation_count"] == 1
    assert result.loc[0, "stock_bond_switching_hit_ratio"] == pytest.approx(0.0)


def test_write_outputs_creates_integrated_csvs_and_passing_audit(tmp_path):
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    csi, peer, _ = _fund_quarter_inputs()
    csi.to_csv(intermediate / "fund_quarter_csi300_effectiveness.csv", index=False)
    peer.to_csv(intermediate / "fund_quarter_effectiveness.csv", index=False)
    source = tmp_path / "bond.json"
    source.write_text(
        json.dumps(
            {
                "CFZS_00": {
                    "1711670400000": 100.0,
                    "1719532800000": 102.0,
                    "1727654400000": 103.02,
                }
            }
        )
    )

    outputs = write_three_benchmark_outputs(tmp_path, source)

    expected = {
        "chinabond_cba00201_quarterly_benchmark.csv",
        "fund_quarter_three_benchmark_effectiveness.csv",
        "fund_three_benchmark_effectiveness.csv",
        "fund_company_three_benchmark_effectiveness.csv",
        "fund_company_stock_bond_switching_ranking.csv",
        "three_benchmark_audit.csv",
    }
    assert expected == set(outputs)
    assert expected.issubset({path.name for path in intermediate.glob("*.csv")})
    audit = pd.read_csv(intermediate / "three_benchmark_audit.csv")
    assert set(audit["检查结论"]) == {"通过"}
