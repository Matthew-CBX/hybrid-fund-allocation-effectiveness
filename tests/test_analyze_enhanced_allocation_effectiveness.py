import math

import pandas as pd
import pytest

from hybrid_fund_allocation.analyze_enhanced_allocation_effectiveness import (
    _continuous_active_contribution,
    add_drift_adjusted_metrics,
    annualize_quarterly_returns,
    build_company_annualized_metrics,
    assign_credibility_grade,
    build_company_evaluation,
    build_direction_environment,
    build_fund_annualized_metrics,
    build_threshold_robustness,
    calculate_leave_one_out_peer,
    two_way_cluster_bootstrap_mean,
    wilson_interval,
    winsorized_mean,
    write_enhanced_outputs,
)


def test_annualize_quarterly_returns_uses_geometric_compounding():
    returns = pd.Series([0.10, 0.00, -0.05, 0.02])
    expected = (1.10 * 1.00 * 0.95 * 1.02) - 1
    assert annualize_quarterly_returns(returns) == pytest.approx(expected)


def test_annualize_quarterly_returns_scales_partial_year():
    returns = pd.Series([0.10, 0.10])
    expected = ((1.10 * 1.10) ** 2) - 1
    assert annualize_quarterly_returns(returns) == pytest.approx(expected)


def test_annualize_quarterly_returns_rejects_total_loss_and_empty_series():
    assert math.isnan(annualize_quarterly_returns(pd.Series(dtype=float)))
    assert math.isnan(annualize_quarterly_returns(pd.Series([0.02, -1.00])))


def _annualized_detail_fixture(quarters=8):
    rows = []
    labels = [f"2024Q{1 + i % 4}" if i < 4 else f"2025Q{1 + i % 4}" for i in range(quarters)]
    for index, quarter in enumerate(labels):
        rows.append(
            {
                "报告期": quarter,
                "基金主代码": "F1",
                "基金名称": "测试基金",
                "基金管理人简称": "测试公司",
                "投资风格": "成长型",
                "fund_nav_yuan": 100 + index,
                "next_quarter_return": 0.02,
                "csi300_next_quarter_return": 0.01,
                "bond_next_quarter_return": 0.005,
                "three_asset_benchmark_return": 0.01,
                "three_asset_coverage_status": "可用",
                "loo_peer_median_return": 0.015,
                "drift_adjusted_comparable": True,
                "drift_adjusted_stock_rebalance": 0.04 if index == 0 else 0.06,
                "stock_bond_relative_return_spread": 0.05,
            }
        )
    return pd.DataFrame(rows)


def test_build_fund_annualized_metrics_uses_one_common_quarter_set():
    detail = _annualized_detail_fixture()
    result = build_fund_annualized_metrics(detail)
    row = result.iloc[0]
    assert row["return_quarter_count"] == 8
    assert row["annualized_evaluation_pool"] == "正式年化"
    assert row["annualized_fund_return"] == pytest.approx((1.02**8) ** 0.5 - 1)
    assert row["annualized_csi300_return"] == pytest.approx((1.01**8) ** 0.5 - 1)
    assert row["annualized_three_asset_benchmark_return"] == pytest.approx(
        (1.01**8) ** 0.5 - 1
    )
    assert row["annualized_excess_vs_three_asset"] == pytest.approx(
        row["annualized_fund_return"] - row["annualized_three_asset_benchmark_return"]
    )
    assert row["significant_switch_count"] == 7
    assert row["significant_switch_coverage"] == pytest.approx(7 / 8)


def test_build_fund_annualized_metrics_drops_the_same_quarter_from_all_returns():
    detail = _annualized_detail_fixture()
    detail.loc[0, "three_asset_benchmark_return"] = float("nan")
    result = build_fund_annualized_metrics(detail)
    assert result.iloc[0]["return_quarter_count"] == 7
    assert result.iloc[0]["annualized_evaluation_pool"] == "年化观察池"
    assert math.isnan(result.iloc[0]["annualized_fund_return"])


def test_build_fund_annualized_metrics_compounds_zero_for_small_switch_quarters():
    detail = _annualized_detail_fixture()
    result = build_fund_annualized_metrics(detail)
    expected_quarterly = [0.0] + [0.06 * 0.05] * 7
    expected = annualize_quarterly_returns(pd.Series(expected_quarterly))
    assert result.iloc[0]["annualized_active_allocation_contribution"] == pytest.approx(expected)


def test_build_company_annualized_metrics_uses_fund_level_median_and_latest_nav():
    funds = pd.DataFrame(
        {
            "基金主代码": ["F1", "F2", "F3"],
            "基金管理人简称": ["甲公司"] * 3,
            "annualized_evaluation_pool": ["正式年化", "正式年化", "年化观察池"],
            "contribution_evaluation_pool": ["正式贡献", "正式贡献", "贡献观察池"],
            "latest_valid_nav_yuan": [100.0, 300.0, 500.0],
            "annualized_fund_return": [0.10, 0.20, float("nan")],
            "annualized_three_asset_benchmark_return": [0.08, 0.15, float("nan")],
            "annualized_excess_vs_three_asset": [0.02, 0.05, float("nan")],
            "annualized_excess_vs_loo_peer": [0.01, -0.01, float("nan")],
            "annualized_active_allocation_contribution": [0.01, -0.02, float("nan")],
        }
    )
    company = pd.DataFrame(
        {
            "基金管理人简称": ["甲公司"],
            "evaluation_pool": ["正式评价"],
            "credibility_grade": ["B级"],
            "formal_rank": [1],
        }
    )

    result = build_company_annualized_metrics(funds, company).iloc[0]

    assert result["annualized_valid_fund_count"] == 2
    assert result["median_annualized_fund_return"] == pytest.approx(0.15)
    assert result["nav_weighted_annualized_fund_return"] == pytest.approx(0.175)
    assert result["median_annualized_excess_vs_three_asset"] == pytest.approx(0.035)
    assert result["annualized_excess_positive_fund_ratio"] == 1.0
    assert result["active_contribution_positive_fund_ratio"] == 0.5


def test_continuous_active_contribution_zeros_small_rebalances_only():
    frame = pd.DataFrame(
        {
            "drift_adjusted_comparable": [True, True, False],
            "drift_adjusted_stock_rebalance": [0.04, -0.06, 0.10],
            "stock_bond_relative_return_spread": [0.10, -0.05, 0.20],
        }
    )
    result = _continuous_active_contribution(frame, threshold=0.05)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == pytest.approx(0.003)
    assert math.isnan(result.iloc[2])


def test_continuous_active_contribution_keeps_missing_inputs_missing():
    frame = pd.DataFrame(
        {
            "drift_adjusted_comparable": [True, True],
            "drift_adjusted_stock_rebalance": [None, 0.06],
            "stock_bond_relative_return_spread": [0.10, None],
        }
    )
    assert _continuous_active_contribution(frame, 0.05).isna().all()


def _two_quarter_panel(
    previous_weight=0.60,
    current_weight=0.65,
    current_stock_return=0.10,
    current_bond_return=0.00,
    next_stock_return=0.08,
    next_bond_return=0.02,
    stock_weight=0.60,
    bond_weight=0.30,
    cash_weight=0.10,
):
    return pd.DataFrame(
        {
            "基金主代码": ["F1", "F1"],
            "报告期": ["2024Q1", "2024Q2"],
            "异常标记": ["可用", "可用"],
            "normalized_stock_weight": [previous_weight, current_weight],
            "normalized_bond_weight": [1 - previous_weight, 1 - current_weight],
            "csi300_next_quarter_return": [current_stock_return, next_stock_return],
            "bond_next_quarter_return": [current_bond_return, next_bond_return],
            "next_quarter_return": [0.04, 0.05],
            "stock_weight": [0.60, stock_weight],
            "bond_weight": [0.30, bond_weight],
            "bank_deposit_weight": [0.10, cash_weight],
            "fund_nav_yuan": [100.0, 110.0],
        }
    )


def test_leave_one_out_peer_excludes_focal_fund_and_enforces_minimum_group():
    panel = pd.DataFrame(
        {
            "基金主代码": [f"F{i}" for i in range(11)],
            "报告期": ["2024Q1"] * 11,
            "投资风格": ["成长型"] * 11,
            "next_quarter_return": [
                0.00,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                1.00,
            ],
        }
    )

    result = calculate_leave_one_out_peer(panel, minimum_other_funds=10)

    focal = result.loc[result["基金主代码"].eq("F10")].iloc[0]
    assert focal["loo_peer_group_size"] == 10
    assert focal["loo_peer_median_return"] == pytest.approx(0.045)
    assert focal["loo_peer_relative_return"] == pytest.approx(0.955)


def test_leave_one_out_peer_returns_missing_when_other_group_is_too_small():
    panel = pd.DataFrame(
        {
            "基金主代码": ["A", "B"],
            "报告期": ["2024Q1", "2024Q1"],
            "投资风格": ["收入型", "收入型"],
            "next_quarter_return": [0.01, 0.02],
        }
    )

    result = calculate_leave_one_out_peer(panel, minimum_other_funds=2)

    assert result["loo_peer_group_size"].tolist() == [1, 1]
    assert result["loo_peer_median_return"].isna().all()


def test_drift_adjustment_separates_passive_weight_change_from_active_rebalance():
    panel = _two_quarter_panel()

    result = add_drift_adjusted_metrics(panel, threshold=0.05)

    expected_passive = 0.60 * 1.10 / (0.60 * 1.10 + 0.40)
    assert result.loc[1, "passive_normalized_stock_weight"] == pytest.approx(
        expected_passive
    )
    assert result.loc[1, "drift_adjusted_stock_rebalance"] == pytest.approx(
        0.65 - expected_passive
    )
    assert result.loc[1, "drift_adjusted_comparable"]
    assert math.isnan(result.loc[1, "drift_adjusted_switch_hit"])


def test_drift_adjustment_marks_large_positive_active_rebalance_as_hit():
    panel = _two_quarter_panel(current_weight=0.75)

    result = add_drift_adjusted_metrics(panel, threshold=0.05)

    contribution = result.loc[1, "drift_adjusted_stock_rebalance"] * 0.06
    assert result.loc[1, "drift_adjusted_switch_direction"] == "加股票"
    assert result.loc[1, "drift_adjusted_switch_contribution"] == pytest.approx(
        contribution
    )
    assert result.loc[1, "drift_adjusted_switch_hit"] == 1


def test_three_asset_benchmark_uses_actual_weights_and_zero_cash_proxy():
    panel = _two_quarter_panel()

    result = add_drift_adjusted_metrics(panel)

    assert result.loc[1, "three_asset_coverage"] == pytest.approx(1.0)
    assert result.loc[1, "three_asset_benchmark_return"] == pytest.approx(
        0.60 * 0.08 + 0.30 * 0.02
    )
    assert result.loc[1, "excess_return_vs_three_asset"] == pytest.approx(
        0.05 - (0.60 * 0.08 + 0.30 * 0.02)
    )
    assert result.loc[1, "three_asset_coverage_status"] == "可用"


def test_three_asset_benchmark_flags_out_of_range_coverage():
    panel = _two_quarter_panel(stock_weight=0.30, bond_weight=0.20, cash_weight=0.10)

    result = add_drift_adjusted_metrics(panel)

    assert result.loc[1, "three_asset_coverage_status"] == "覆盖/杠杆异常"
    assert math.isnan(result.loc[1, "excess_return_vs_three_asset"])


def test_wilson_interval_matches_known_nine_of_twelve_case():
    lower, upper = wilson_interval(9, 12)

    assert lower == pytest.approx(0.4677, abs=0.001)
    assert upper == pytest.approx(0.9111, abs=0.001)


def test_wilson_interval_returns_missing_for_zero_observations():
    lower, upper = wilson_interval(0, 0)

    assert math.isnan(lower)
    assert math.isnan(upper)


def test_winsorized_mean_caps_both_tails():
    values = pd.Series([-100.0, 0.0, 1.0, 2.0, 100.0])

    assert winsorized_mean(values, 0.25) == pytest.approx(1.0)


def test_two_way_bootstrap_is_deterministic_for_constant_values():
    frame = pd.DataFrame(
        {
            "基金主代码": ["A", "A", "B", "B"],
            "报告期": ["2024Q1", "2024Q2", "2024Q1", "2024Q2"],
            "value": [0.01, 0.01, 0.01, 0.01],
        }
    )

    first = two_way_cluster_bootstrap_mean(frame, "value", draws=100, seed=7)
    second = two_way_cluster_bootstrap_mean(frame, "value", draws=100, seed=7)

    assert first == pytest.approx((0.01, 0.01))
    assert second == pytest.approx(first)


def _company_rows(company, observations, funds, successes, rebalance=0.06):
    rows = []
    for index in range(observations):
        hit = int(index < successes)
        contribution = 0.001 if hit else -0.001
        rows.append(
            {
                "基金管理人简称": company,
                "基金主代码": f"{company}-F{index % funds}",
                "报告期": f"202{3 + (index % 3)}Q{1 + index % 4}",
                "fund_nav_yuan": 100.0 + index,
                "drift_adjusted_comparable": True,
                "drift_adjusted_stock_rebalance": rebalance,
                "stock_bond_relative_return_spread": contribution / rebalance,
                "drift_adjusted_switch_hit": hit,
                "drift_adjusted_switch_contribution": contribution,
                "excess_return_vs_dynamic_stock_bond": 0.002,
                "excess_return_vs_three_asset": 0.001,
                "three_asset_coverage_status": "可用",
                "loo_peer_relative_return": 0.001,
            }
        )
    return rows


def test_company_evaluation_separates_formal_companies_from_observation_pool():
    detail = pd.DataFrame(
        _company_rows("正式公司", 30, 5, 18)
        + _company_rows("小样本公司", 12, 3, 9)
    )

    result = build_company_evaluation(detail, bootstrap_draws=100)
    by_company = result.set_index("基金管理人简称")

    assert by_company.loc["正式公司", "evaluation_pool"] == "正式评价"
    assert by_company.loc["小样本公司", "evaluation_pool"] == "观察池"
    assert by_company.loc["正式公司", "formal_rank"] == 1
    assert math.isnan(by_company.loc["小样本公司", "formal_rank"])


def test_formal_ranking_uses_wilson_lower_bound_before_raw_hit_rate():
    detail = pd.DataFrame(
        _company_rows("大样本公司", 100, 10, 65)
        + _company_rows("小样本高命中", 30, 5, 21)
    )

    result = build_company_evaluation(detail, bootstrap_draws=100)
    formal = result.query("evaluation_pool == '正式评价'").sort_values("formal_rank")

    assert formal.iloc[0]["基金管理人简称"] == "大样本公司"
    assert formal.iloc[0]["switch_hit_ratio"] < formal.iloc[1]["switch_hit_ratio"]
    assert formal.iloc[0]["wilson_lower_95"] > formal.iloc[1]["wilson_lower_95"]


def _complete_grade_row():
    return {
        "evaluation_pool": "正式评价",
        "wilson_lower_95": 0.49,
        "bootstrap_lower_95": -0.001,
        "mean_active_contribution": 0.001,
        "median_active_contribution": 0.001,
        "equal_weight_dynamic_excess": 0.001,
        "nav_weighted_dynamic_excess": 0.001,
        "fund_positive_ratio": 0.60,
        "switch_hit_ratio": 0.60,
        "positive_threshold_count": 2,
        "positive_signal_count": 6,
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"evaluation_pool": "观察池"}, "观察池"),
        (
            {
                "wilson_lower_95": 0.51,
                "bootstrap_lower_95": 0.0001,
                "positive_threshold_count": 3,
            },
            "A级",
        ),
        ({"positive_signal_count": 4}, "B级"),
        ({"positive_signal_count": 2}, "C级"),
    ],
)
def test_credibility_grade_rules(overrides, expected):
    row = _complete_grade_row() | overrides

    assert assign_credibility_grade(pd.Series(row)) == expected


def test_threshold_robustness_has_nested_observation_counts():
    detail = pd.DataFrame(_company_rows("甲公司", 30, 5, 18, rebalance=0.06))

    result = build_threshold_robustness(detail)
    counts = result.set_index("threshold")["switch_observation_count"]

    assert counts.loc[0.03] == 30
    assert counts.loc[0.05] == 30
    assert counts.loc[0.10] == 0


def test_direction_environment_splits_action_and_relative_market_strength():
    detail = pd.DataFrame(_company_rows("甲公司", 3, 3, 2))
    detail["drift_adjusted_stock_rebalance"] = [0.06, -0.06, 0.06]
    detail["stock_bond_relative_return_spread"] = [0.06, -0.06, 0.02]
    detail["drift_adjusted_switch_contribution"] = [0.0036, 0.0036, 0.0012]
    detail["drift_adjusted_switch_hit"] = [1, 1, 1]

    result = build_direction_environment(detail)

    assert set(result["switch_direction"]) == {"加股票", "减股票"}
    assert set(result["relative_market_environment"]) == {
        "股显著强",
        "债显著强",
        "差异较小",
    }


def _write_enhanced_input_fixture(project_dir):
    intermediate = project_dir / "intermediate"
    intermediate.mkdir()
    quarters = ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    rows = []
    for fund_index in range(5):
        for quarter_index, quarter in enumerate(quarters):
            normalized_stock = 0.40 if quarter_index % 2 == 0 else 0.70
            rows.append(
                {
                    "报告期": quarter,
                    "基金主代码": f"F{fund_index}",
                    "证券代码": f"F{fund_index}.OF",
                    "基金名称": f"测试基金{fund_index}",
                    "基金管理人简称": "甲公司",
                    "投资风格": "成长型",
                    "异常标记": "可用",
                    "fund_nav_yuan": 100.0 + fund_index * 10 + quarter_index,
                    "stock_weight": normalized_stock * 0.90,
                    "bond_weight": (1 - normalized_stock) * 0.90,
                    "bank_deposit_weight": 0.10,
                    "normalized_stock_weight": normalized_stock,
                    "normalized_bond_weight": 1 - normalized_stock,
                    "csi300_next_quarter_return": 0.08 if quarter_index % 2 == 0 else -0.02,
                    "bond_next_quarter_return": 0.01,
                    "next_quarter_return": 0.03,
                    "excess_return_vs_dynamic_stock_bond": 0.002,
                }
            )
    pd.DataFrame(rows).to_csv(
        intermediate / "fund_quarter_three_benchmark_effectiveness.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "报告期": "2024Q1",
                "基金主代码": "F0",
                "security_code": "F0.OF",
                "基金名称": "测试基金0",
                "基金管理人简称": "甲公司",
                "投资风格": "成长型",
                "start_adjusted_nav": 1.0,
                "next_adjusted_nav": 1.6,
                "next_quarter_return": 0.6,
                "复核结论": "原始净值可还原；未自动剔除",
            }
        ]
    ).to_csv(intermediate / "extreme_return_review.csv", index=False)


def test_write_enhanced_outputs_writes_all_tables_and_passing_audit(tmp_path):
    _write_enhanced_input_fixture(tmp_path)

    outputs = write_enhanced_outputs(tmp_path, bootstrap_draws=100)

    expected = {
        "enhanced_fund_quarter_detail.csv",
        "enhanced_company_evaluation.csv",
        "enhanced_threshold_robustness.csv",
        "enhanced_direction_environment.csv",
        "enhanced_three_asset_benchmark.csv",
        "enhanced_product_concentration.csv",
        "enhanced_extreme_sensitivity.csv",
        "enhanced_event_ledger.csv",
        "enhanced_audit.csv",
        "enhanced_fund_annualized_return.csv",
        "enhanced_company_annualized_return.csv",
    }
    assert set(outputs) == expected
    assert expected.issubset(
        {path.name for path in (tmp_path / "intermediate").glob("enhanced_*.csv")}
    )
    assert set(outputs["enhanced_audit.csv"]["检查结论"]) == {"通过"}
    assert outputs["enhanced_fund_annualized_return.csv"]["基金主代码"].nunique() == 5
    assert outputs["enhanced_company_annualized_return.csv"].iloc[0]["基金管理人简称"] == "甲公司"
    annualized_audit = outputs["enhanced_audit.csv"].set_index("检查项目")["实际结果"]
    for item in [
        "年化收益季度数超过13的基金数",
        "正式年化基金少于8季度的数量",
        "正式贡献基金少于8漂移季度的数量",
        "统一收益可比季度中收益小于等于-100%的基金数",
        "漂移可比季度中贡献小于等于-100%的基金数",
        "小于5%可比调仓未归零的数量",
        "基金与基准统一季度集合违反数",
        "公司有效基金数汇总差异",
    ]:
        assert annualized_audit[item] == 0
    assert outputs["enhanced_event_ledger.csv"][
        ["人工事件类型", "是否剔除", "备注"]
    ].isna().all().all()
