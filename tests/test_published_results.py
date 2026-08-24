from pathlib import Path

import pandas as pd
import pytest


RESULT_DIR = Path("data/results")
EXPECTED = {
    "enhanced_company_evaluation.csv": (
        (101, 30),
        [
            "基金管理人简称", "evaluation_pool", "return_observation_count",
            "switch_observation_count", "switch_unique_fund_count",
            "switch_success_count", "switch_hit_ratio", "wilson_lower_95",
            "wilson_upper_95", "mean_active_contribution",
            "median_active_contribution", "winsor_1_mean_contribution",
            "winsor_5_mean_contribution", "bootstrap_lower_95",
            "bootstrap_upper_95", "equal_weight_dynamic_excess",
            "nav_weighted_dynamic_excess", "equal_weight_three_asset_excess",
            "nav_weighted_three_asset_excess", "fund_positive_ratio",
            "company_quarter_positive_ratio", "worst_company_quarter_contribution",
            "max_single_fund_observation_share", "top_two_fund_observation_share",
            "max_single_fund_absolute_contribution_share", "formal_gate_reason",
            "positive_threshold_count", "positive_signal_count",
            "credibility_grade", "formal_rank",
        ],
        ["基金管理人简称"],
    ),
    "enhanced_company_annualized_return.csv": (
        (101, 15),
        [
            "基金管理人简称", "evaluation_pool", "credibility_grade", "formal_rank",
            "annualized_valid_fund_count", "active_contribution_valid_fund_count",
            "median_annualized_fund_return", "equal_weight_annualized_fund_return",
            "nav_weighted_annualized_fund_return",
            "median_annualized_three_asset_benchmark_return",
            "median_annualized_excess_vs_three_asset",
            "median_annualized_excess_vs_loo_peer",
            "median_annualized_active_allocation_contribution",
            "annualized_excess_positive_fund_ratio",
            "active_contribution_positive_fund_ratio",
        ],
        ["基金管理人简称"],
    ),
    "enhanced_fund_annualized_return.csv": (
        (376, 24),
        [
            "基金主代码", "基金名称", "基金管理人简称", "投资风格",
            "annualized_evaluation_pool", "return_quarter_count",
            "return_period_start", "return_period_end", "contribution_evaluation_pool",
            "contribution_quarter_count", "latest_valid_nav_yuan",
            "annualized_fund_return", "annualized_csi300_return",
            "annualized_bond_return", "annualized_three_asset_benchmark_return",
            "annualized_loo_peer_return", "annualized_excess_vs_three_asset",
            "annualized_excess_vs_loo_peer", "significant_switch_count",
            "significant_switch_coverage", "cumulative_active_allocation_contribution",
            "annualized_active_allocation_contribution", "annualized_excess_positive",
            "annualized_allocation_contribution_positive",
        ],
        ["基金主代码"],
    ),
    "enhanced_threshold_robustness.csv": (
        (303, 11),
        [
            "threshold", "基金管理人简称", "switch_observation_count",
            "switch_unique_fund_count", "switch_success_count", "switch_hit_ratio",
            "wilson_lower_95", "wilson_upper_95", "mean_active_contribution",
            "median_active_contribution", "threshold_rank",
        ],
        ["threshold", "基金管理人简称"],
    ),
    "enhanced_extreme_sensitivity.csv": (
        (100, 7),
        [
            "基金管理人简称", "observation_count", "raw_mean_contribution",
            "winsor_1_mean_contribution", "winsor_5_mean_contribution",
            "median_contribution", "all_four_same_sign",
        ],
        ["基金管理人简称"],
    ),
    "enhanced_direction_environment.csv": (
        (425, 8),
        [
            "基金管理人简称", "switch_direction", "relative_market_environment",
            "observation_count", "unique_fund_count", "hit_ratio",
            "mean_active_contribution", "median_active_contribution",
        ],
        ["基金管理人简称", "switch_direction", "relative_market_environment"],
    ),
    "enhanced_product_concentration.csv": (
        (101, 9),
        [
            "基金管理人简称", "evaluation_pool", "credibility_grade",
            "switch_observation_count", "switch_unique_fund_count",
            "max_single_fund_observation_share", "top_two_fund_observation_share",
            "max_single_fund_absolute_contribution_share", "formal_gate_reason",
        ],
        ["基金管理人简称"],
    ),
    "enhanced_audit.csv": (
        (17, 5),
        ["检查项目", "实际结果", "预期或判断标准", "检查结论", "说明"],
        ["检查项目"],
    ),
}


@pytest.mark.parametrize("filename", EXPECTED)
def test_published_result_has_exact_shape_schema_and_primary_key(filename: str):
    expected_shape, expected_columns, key = EXPECTED[filename]

    result = pd.read_csv(RESULT_DIR / filename, dtype={"基金主代码": "string"})

    assert result.shape == expected_shape
    assert result.columns.tolist() == expected_columns
    assert not result.duplicated(key).any()


def test_all_published_audit_checks_pass():
    audit = pd.read_csv(RESULT_DIR / "enhanced_audit.csv")

    assert audit["检查结论"].eq("通过").all()
