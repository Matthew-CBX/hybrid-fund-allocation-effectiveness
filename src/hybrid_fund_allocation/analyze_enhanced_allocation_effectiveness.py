"""Build statistically stricter mixed-fund allocation evidence tables."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def annualize_quarterly_returns(values: pd.Series) -> float:
    """Compound valid quarterly returns and convert them to a yearly rate."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty or numeric.le(-1).any():
        return float("nan")
    growth = float((1 + numeric).prod())
    return growth ** (4 / len(numeric)) - 1


def _continuous_active_contribution(
    group: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    comparable = (
        group["drift_adjusted_comparable"].fillna(False)
        & pd.to_numeric(group["drift_adjusted_stock_rebalance"], errors="coerce").notna()
        & pd.to_numeric(group["stock_bond_relative_return_spread"], errors="coerce").notna()
    )
    rebalance = pd.to_numeric(group["drift_adjusted_stock_rebalance"], errors="coerce")
    spread = pd.to_numeric(group["stock_bond_relative_return_spread"], errors="coerce")
    result = pd.Series(np.nan, index=group.index, dtype=float)
    result.loc[comparable] = np.where(
        rebalance.loc[comparable].abs().ge(threshold),
        rebalance.loc[comparable] * spread.loc[comparable],
        0.0,
    )
    return result


def build_fund_annualized_metrics(
    detail: pd.DataFrame,
    min_return_quarters: int = 8,
    min_contribution_quarters: int = 8,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Aggregate comparable quarterly evidence into fund-level annualized metrics."""
    return_columns = [
        "next_quarter_return",
        "csi300_next_quarter_return",
        "bond_next_quarter_return",
        "three_asset_benchmark_return",
        "loo_peer_median_return",
    ]
    rows = []
    for fund_code, group in detail.groupby("基金主代码", sort=True):
        group = group.sort_values("报告期")
        common = group[return_columns].notna().all(axis=1) & group[
            "three_asset_coverage_status"
        ].eq("可用")
        common_returns = group.loc[common]
        return_count = len(common_returns)
        geometric_return_valid = (
            common_returns[return_columns].gt(-1).all().all()
            if return_count
            else False
        )
        formal_return = return_count >= min_return_quarters and geometric_return_valid
        contribution = _continuous_active_contribution(group, threshold)
        valid_contribution = contribution.dropna()
        contribution_count = len(valid_contribution)
        geometric_contribution_valid = (
            valid_contribution.gt(-1).all() if contribution_count else False
        )
        formal_contribution = (
            contribution_count >= min_contribution_quarters
            and geometric_contribution_valid
        )
        latest_nav = pd.to_numeric(group["fund_nav_yuan"], errors="coerce")
        latest_nav = (
            latest_nav.loc[latest_nav.gt(0)].iloc[-1]
            if latest_nav.gt(0).any()
            else np.nan
        )
        annualized = {
            column: annualize_quarterly_returns(common_returns[column])
            if formal_return
            else np.nan
            for column in return_columns
        }
        active_annualized = (
            annualize_quarterly_returns(valid_contribution)
            if formal_contribution
            else np.nan
        )
        identity = group.iloc[-1]
        annualized_fund = annualized["next_quarter_return"] if formal_return else np.nan
        annualized_three_asset = (
            annualized["three_asset_benchmark_return"] if formal_return else np.nan
        )
        annualized_peer = (
            annualized["loo_peer_median_return"] if formal_return else np.nan
        )
        significant_count = int(
            (
                pd.to_numeric(
                    group.loc[
                        valid_contribution.index,
                        "drift_adjusted_stock_rebalance",
                    ],
                    errors="coerce",
                ).abs()
                >= threshold
            ).sum()
        )
        compounded_contribution = (
            float((1 + valid_contribution).prod() - 1)
            if not valid_contribution.empty and valid_contribution.gt(-1).all()
            else np.nan
        )
        rows.append(
            {
                "基金主代码": fund_code,
                "基金名称": identity.get("基金名称"),
                "基金管理人简称": identity.get("基金管理人简称"),
                "投资风格": identity.get("投资风格"),
                "annualized_evaluation_pool": (
                    "正式年化"
                    if formal_return
                    else (
                        "年化无效"
                        if return_count >= min_return_quarters
                        and not geometric_return_valid
                        else "年化观察池"
                    )
                ),
                "return_quarter_count": return_count,
                "return_period_start": (
                    common_returns["报告期"].min() if return_count else pd.NA
                ),
                "return_period_end": (
                    common_returns["报告期"].max() if return_count else pd.NA
                ),
                "contribution_evaluation_pool": (
                    "正式贡献"
                    if formal_contribution
                    else (
                        "贡献无效"
                        if contribution_count >= min_contribution_quarters
                        and not geometric_contribution_valid
                        else "贡献观察池"
                    )
                ),
                "contribution_quarter_count": contribution_count,
                "latest_valid_nav_yuan": latest_nav,
                "annualized_fund_return": annualized_fund,
                "annualized_csi300_return": (
                    annualized["csi300_next_quarter_return"]
                    if formal_return
                    else np.nan
                ),
                "annualized_bond_return": (
                    annualized["bond_next_quarter_return"]
                    if formal_return
                    else np.nan
                ),
                "annualized_three_asset_benchmark_return": annualized_three_asset,
                "annualized_loo_peer_return": annualized_peer,
                "annualized_excess_vs_three_asset": (
                    annualized_fund - annualized_three_asset
                ),
                "annualized_excess_vs_loo_peer": annualized_fund - annualized_peer,
                "significant_switch_count": significant_count,
                "significant_switch_coverage": (
                    significant_count / contribution_count
                    if contribution_count
                    else np.nan
                ),
                "cumulative_active_allocation_contribution": compounded_contribution,
                "annualized_active_allocation_contribution": active_annualized,
                "annualized_excess_positive": (
                    bool(annualized_fund - annualized_three_asset > 0)
                    if formal_return
                    and pd.notna(annualized_fund)
                    and pd.notna(annualized_three_asset)
                    else pd.NA
                ),
                "annualized_allocation_contribution_positive": (
                    bool(active_annualized > 0)
                    if formal_contribution and pd.notna(active_annualized)
                    else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def build_company_annualized_metrics(
    fund_metrics: pd.DataFrame,
    company_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fund-level annualized evidence to the company level."""
    rows = []
    evaluation = company_evaluation.set_index("基金管理人简称")
    for company, group in fund_metrics.groupby("基金管理人简称", sort=True):
        formal = group.loc[group["annualized_evaluation_pool"].eq("正式年化")]
        contribution = group.loc[
            group["contribution_evaluation_pool"].eq("正式贡献")
        ]
        nav = pd.to_numeric(formal["latest_valid_nav_yuan"], errors="coerce")
        valid_nav = nav.gt(0) & formal["annualized_fund_return"].notna()
        excess = formal["annualized_excess_vs_three_asset"].dropna()
        active = contribution["annualized_active_allocation_contribution"].dropna()
        metadata = (
            evaluation.loc[company]
            if company in evaluation.index
            else pd.Series(dtype=object)
        )
        rows.append(
            {
                "基金管理人简称": company,
                "evaluation_pool": metadata.get("evaluation_pool", "观察池"),
                "credibility_grade": metadata.get("credibility_grade", "观察池"),
                "formal_rank": metadata.get("formal_rank", np.nan),
                "annualized_valid_fund_count": len(formal),
                "active_contribution_valid_fund_count": len(contribution),
                "median_annualized_fund_return": formal[
                    "annualized_fund_return"
                ].median(),
                "equal_weight_annualized_fund_return": formal[
                    "annualized_fund_return"
                ].mean(),
                "nav_weighted_annualized_fund_return": (
                    float(
                        np.average(
                            formal.loc[valid_nav, "annualized_fund_return"],
                            weights=nav.loc[valid_nav],
                        )
                    )
                    if valid_nav.any()
                    else np.nan
                ),
                "median_annualized_three_asset_benchmark_return": formal[
                    "annualized_three_asset_benchmark_return"
                ].median(),
                "median_annualized_excess_vs_three_asset": formal[
                    "annualized_excess_vs_three_asset"
                ].median(),
                "median_annualized_excess_vs_loo_peer": formal[
                    "annualized_excess_vs_loo_peer"
                ].median(),
                "median_annualized_active_allocation_contribution": contribution[
                    "annualized_active_allocation_contribution"
                ].median(),
                "annualized_excess_positive_fund_ratio": (
                    excess.gt(0).mean() if len(excess) else np.nan
                ),
                "active_contribution_positive_fund_ratio": (
                    active.gt(0).mean() if len(active) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "evaluation_pool",
            "formal_rank",
            "median_annualized_excess_vs_three_asset",
            "基金管理人简称",
        ],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def wilson_interval(
    successes: int,
    observations: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial success rate."""
    if observations <= 0:
        return float("nan"), float("nan")
    proportion = successes / observations
    denominator = 1 + z * z / observations
    center = (proportion + z * z / (2 * observations)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / observations
            + z * z / (4 * observations**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def winsorized_mean(values: pd.Series, tail: float) -> float:
    """Return a two-sided winsorized mean using empirical quantiles."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    if not 0 <= tail < 0.5:
        raise ValueError("tail must be between 0 and 0.5")
    if tail == 0:
        return float(numeric.mean())
    lower, upper = numeric.quantile([tail, 1 - tail])
    return float(numeric.clip(lower=lower, upper=upper).mean())


def two_way_cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value_col: str,
    fund_col: str = "基金主代码",
    quarter_col: str = "报告期",
    draws: int = 2000,
    seed: int = 20260819,
) -> tuple[float, float]:
    """Estimate a mean interval by resampling fund and quarter clusters."""
    data = frame[[fund_col, quarter_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        return float("nan"), float("nan")
    funds = data[fund_col].unique()
    quarters = data[quarter_col].unique()
    rng = np.random.default_rng(seed)
    means: list[float] = []
    for _ in range(draws):
        fund_counts = pd.Series(
            rng.choice(funds, size=len(funds), replace=True)
        ).value_counts()
        quarter_counts = pd.Series(
            rng.choice(quarters, size=len(quarters), replace=True)
        ).value_counts()
        weights = (
            data[fund_col].map(fund_counts).fillna(0)
            * data[quarter_col].map(quarter_counts).fillna(0)
        )
        valid = weights.gt(0)
        if valid.any():
            means.append(
                float(
                    np.average(
                        data.loc[valid, value_col],
                        weights=weights.loc[valid],
                    )
                )
            )
    if not means:
        return float("nan"), float("nan")
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def calculate_leave_one_out_peer(
    panel: pd.DataFrame,
    minimum_other_funds: int = 10,
) -> pd.DataFrame:
    """Compare each fund with the median return of other same-style funds."""
    result = panel.copy()
    peer_sizes: dict[int, int] = {}
    peer_medians: dict[int, float] = {}
    for _, group in result.groupby(["报告期", "投资风格"], sort=False):
        values = pd.to_numeric(group["next_quarter_return"], errors="coerce")
        for index in group.index:
            peers = values.drop(index).dropna()
            peer_sizes[index] = len(peers)
            peer_medians[index] = (
                float(peers.median())
                if len(peers) >= minimum_other_funds
                else float("nan")
            )
    result["loo_peer_group_size"] = pd.Series(peer_sizes)
    result["loo_peer_median_return"] = pd.Series(peer_medians)
    result["loo_peer_relative_return"] = (
        pd.to_numeric(result["next_quarter_return"], errors="coerce")
        - result["loo_peer_median_return"]
    )
    return result


def add_drift_adjusted_metrics(
    panel: pd.DataFrame,
    threshold: float = 0.05,
    cash_return: float = 0.0,
) -> pd.DataFrame:
    """Add passive-drift, active-rebalance and actual-weight benchmark metrics."""
    result = panel.sort_values(["基金主代码", "报告期"]).reset_index(drop=True).copy()
    grouped = result.groupby("基金主代码", sort=False)
    previous_stock = grouped["normalized_stock_weight"].shift(1)
    previous_bond = grouped["normalized_bond_weight"].shift(1)
    current_period_stock_return = grouped["csi300_next_quarter_return"].shift(1)
    current_period_bond_return = grouped["bond_next_quarter_return"].shift(1)
    passive_denominator = (
        previous_stock * (1 + current_period_stock_return)
        + previous_bond * (1 + current_period_bond_return)
    )

    result["previous_normalized_stock_weight"] = previous_stock
    result["current_period_stock_return"] = current_period_stock_return
    result["current_period_bond_return"] = current_period_bond_return
    result["passive_normalized_stock_weight"] = (
        previous_stock * (1 + current_period_stock_return) / passive_denominator
    )
    result["drift_adjusted_stock_rebalance"] = (
        pd.to_numeric(result["normalized_stock_weight"], errors="coerce")
        - result["passive_normalized_stock_weight"]
    )
    result["stock_bond_relative_return_spread"] = (
        pd.to_numeric(result["csi300_next_quarter_return"], errors="coerce")
        - pd.to_numeric(result["bond_next_quarter_return"], errors="coerce")
    )

    previous_quality = grouped["异常标记"].shift(1)
    comparable = (
        result["异常标记"].eq("可用")
        & previous_quality.eq("可用")
        & result["drift_adjusted_stock_rebalance"].notna()
    )
    result["previous_allocation_quality_for_drift"] = previous_quality
    result["drift_adjusted_comparable"] = comparable
    eligible = (
        comparable
        & result["drift_adjusted_stock_rebalance"].abs().ge(threshold)
        & result["stock_bond_relative_return_spread"].ne(0)
    )
    positive = result["drift_adjusted_stock_rebalance"].gt(0)
    result["drift_adjusted_switch_direction"] = "基本不变"
    result.loc[eligible & positive, "drift_adjusted_switch_direction"] = "加股票"
    result.loc[eligible & ~positive, "drift_adjusted_switch_direction"] = "减股票"
    raw_contribution = (
        result["drift_adjusted_stock_rebalance"]
        * result["stock_bond_relative_return_spread"]
    )
    result["drift_adjusted_switch_contribution"] = raw_contribution.where(eligible)
    result["drift_adjusted_switch_hit"] = np.where(
        eligible,
        raw_contribution.gt(0).astype(float),
        np.nan,
    )

    stock = pd.to_numeric(result["stock_weight"], errors="coerce")
    bond = pd.to_numeric(result["bond_weight"], errors="coerce")
    cash = pd.to_numeric(result["bank_deposit_weight"], errors="coerce")
    result["three_asset_coverage"] = stock + bond + cash
    coverage_usable = result["three_asset_coverage"].between(0.80, 1.20, inclusive="both")
    result["three_asset_coverage_status"] = np.where(
        coverage_usable,
        "可用",
        "覆盖/杠杆异常",
    )
    result["three_asset_benchmark_return"] = (
        stock * pd.to_numeric(result["csi300_next_quarter_return"], errors="coerce")
        + bond * pd.to_numeric(result["bond_next_quarter_return"], errors="coerce")
        + cash * cash_return
    )
    result["excess_return_vs_three_asset"] = (
        pd.to_numeric(result["next_quarter_return"], errors="coerce")
        - result["three_asset_benchmark_return"]
    ).where(coverage_usable)
    return result


def _weighted_mean(group: pd.DataFrame, value_column: str) -> float:
    values = pd.to_numeric(group[value_column], errors="coerce")
    weights = pd.to_numeric(group["fund_nav_yuan"], errors="coerce")
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _switching_at_threshold(detail: pd.DataFrame, threshold: float) -> pd.DataFrame:
    comparable = detail.get(
        "drift_adjusted_comparable",
        detail["drift_adjusted_stock_rebalance"].notna(),
    )
    eligible = (
        comparable.fillna(False)
        & detail["drift_adjusted_stock_rebalance"].abs().ge(threshold)
        & detail["stock_bond_relative_return_spread"].ne(0)
    )
    switching = detail.loc[eligible].copy()
    switching["threshold_switch_contribution"] = (
        switching["drift_adjusted_stock_rebalance"]
        * switching["stock_bond_relative_return_spread"]
    )
    switching["threshold_switch_hit"] = switching[
        "threshold_switch_contribution"
    ].gt(0).astype(int)
    return switching


def build_threshold_robustness(
    detail: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.03, 0.05, 0.10),
) -> pd.DataFrame:
    """Summarize company switching evidence under alternate active thresholds."""
    companies = sorted(detail["基金管理人简称"].dropna().unique())
    rows = []
    for threshold in thresholds:
        switching = _switching_at_threshold(detail, threshold)
        for company in companies:
            group = switching.loc[switching["基金管理人简称"].eq(company)]
            observations = len(group)
            successes = int(group["threshold_switch_hit"].sum())
            lower, upper = wilson_interval(successes, observations)
            rows.append(
                {
                    "threshold": threshold,
                    "基金管理人简称": company,
                    "switch_observation_count": observations,
                    "switch_unique_fund_count": group["基金主代码"].nunique(),
                    "switch_success_count": successes,
                    "switch_hit_ratio": (
                        float(group["threshold_switch_hit"].mean())
                        if observations
                        else float("nan")
                    ),
                    "wilson_lower_95": lower,
                    "wilson_upper_95": upper,
                    "mean_active_contribution": group[
                        "threshold_switch_contribution"
                    ].mean(),
                    "median_active_contribution": group[
                        "threshold_switch_contribution"
                    ].median(),
                }
            )
    result = pd.DataFrame(rows)
    result["threshold_rank"] = float("nan")
    for threshold, index in result.groupby("threshold").groups.items():
        ranked = result.loc[index].loc[
            result.loc[index, "switch_observation_count"].gt(0)
        ].sort_values(
            ["wilson_lower_95", "mean_active_contribution", "switch_observation_count"],
            ascending=[False, False, False],
        )
        result.loc[ranked.index, "threshold_rank"] = range(1, len(ranked) + 1)
    return result.sort_values(["threshold", "threshold_rank", "基金管理人简称"], na_position="last").reset_index(drop=True)


def build_direction_environment(detail: pd.DataFrame) -> pd.DataFrame:
    """Split main-threshold evidence by action direction and market spread."""
    switching = detail.loc[detail["drift_adjusted_switch_hit"].notna()].copy()
    switching["switch_direction"] = np.where(
        switching["drift_adjusted_stock_rebalance"].gt(0),
        "加股票",
        "减股票",
    )
    spread = switching["stock_bond_relative_return_spread"]
    switching["relative_market_environment"] = np.select(
        [spread.ge(0.05), spread.le(-0.05)],
        ["股显著强", "债显著强"],
        default="差异较小",
    )
    if switching.empty:
        return pd.DataFrame(
            columns=[
                "基金管理人简称",
                "switch_direction",
                "relative_market_environment",
                "observation_count",
                "unique_fund_count",
                "hit_ratio",
                "mean_active_contribution",
                "median_active_contribution",
            ]
        )
    return (
        switching.groupby(
            ["基金管理人简称", "switch_direction", "relative_market_environment"],
            as_index=False,
        )
        .agg(
            observation_count=("drift_adjusted_switch_hit", "size"),
            unique_fund_count=("基金主代码", "nunique"),
            hit_ratio=("drift_adjusted_switch_hit", "mean"),
            mean_active_contribution=("drift_adjusted_switch_contribution", "mean"),
            median_active_contribution=("drift_adjusted_switch_contribution", "median"),
        )
        .sort_values(["基金管理人简称", "switch_direction", "relative_market_environment"])
        .reset_index(drop=True)
    )


def assign_credibility_grade(row: pd.Series) -> str:
    """Assign the approved A/B/C or observation-pool credibility label."""
    if row["evaluation_pool"] != "正式评价":
        return "观察池"
    a_grade = (
        row["wilson_lower_95"] > 0.50
        and row["mean_active_contribution"] > 0
        and row["median_active_contribution"] > 0
        and row["bootstrap_lower_95"] > 0
        and row["equal_weight_dynamic_excess"] > 0
        and row["nav_weighted_dynamic_excess"] > 0
        and row["positive_threshold_count"] == 3
    )
    if a_grade:
        return "A级"
    return "B级" if row["positive_signal_count"] >= 4 else "C级"


def build_company_evaluation(
    detail: pd.DataFrame,
    threshold: float = 0.05,
    bootstrap_draws: int = 2000,
) -> pd.DataFrame:
    """Build formal-company evidence and an explicitly separate observation pool."""
    threshold_table = build_threshold_robustness(detail)
    positive_thresholds = (
        threshold_table.assign(
            positive=lambda frame: frame["mean_active_contribution"].gt(0)
        )
        .groupby("基金管理人简称")["positive"]
        .sum()
    )
    main_switching = _switching_at_threshold(detail, threshold)
    rows = []
    for company, group in detail.groupby("基金管理人简称", sort=True):
        switching = main_switching.loc[
            main_switching["基金管理人简称"].eq(company)
        ]
        observations = len(switching)
        successes = int(switching["threshold_switch_hit"].sum())
        switch_funds = switching["基金主代码"].nunique()
        fund_counts = switching.groupby("基金主代码").size().sort_values(ascending=False)
        max_share = float(fund_counts.iloc[0] / observations) if observations else float("nan")
        top_two_share = float(fund_counts.head(2).sum() / observations) if observations else float("nan")
        absolute_by_fund = switching.assign(
            _absolute=switching["threshold_switch_contribution"].abs()
        ).groupby("基金主代码")["_absolute"].sum()
        absolute_total = absolute_by_fund.sum()
        maximum_absolute_share = (
            float(absolute_by_fund.max() / absolute_total)
            if absolute_total > 0
            else float("nan")
        )
        return_observations = int(
            pd.to_numeric(
                group["excess_return_vs_dynamic_stock_bond"], errors="coerce"
            ).notna().sum()
        )
        failed_gates = []
        if observations < 30:
            failed_gates.append("切换观察<30")
        if switch_funds < 5:
            failed_gates.append("切换基金<5")
        if return_observations < 30:
            failed_gates.append("收益观察<30")
        if pd.notna(max_share) and max_share > 0.50:
            failed_gates.append("单基金观察占比>50%")
        evaluation_pool = "正式评价" if not failed_gates else "观察池"
        lower, upper = wilson_interval(successes, observations)
        if evaluation_pool == "正式评价":
            bootstrap_lower, bootstrap_upper = two_way_cluster_bootstrap_mean(
                switching,
                "threshold_switch_contribution",
                draws=bootstrap_draws,
            )
        else:
            bootstrap_lower, bootstrap_upper = float("nan"), float("nan")
        fund_positive_ratio = (
            switching.groupby("基金主代码")["threshold_switch_contribution"]
            .mean()
            .gt(0)
            .mean()
            if observations
            else float("nan")
        )
        quarter_means = switching.groupby("报告期")[
            "threshold_switch_contribution"
        ].mean()
        dynamic_equal = pd.to_numeric(
            group["excess_return_vs_dynamic_stock_bond"], errors="coerce"
        ).mean()
        dynamic_weighted = _weighted_mean(group, "excess_return_vs_dynamic_stock_bond")
        valid_three_asset = group.loc[
            group["three_asset_coverage_status"].eq("可用")
            & pd.to_numeric(group["excess_return_vs_three_asset"], errors="coerce").notna()
        ]
        mean_contribution = switching["threshold_switch_contribution"].mean()
        median_contribution = switching["threshold_switch_contribution"].median()
        hit_ratio = switching["threshold_switch_hit"].mean()
        positive_signals = sum(
            bool(value)
            for value in [
                pd.notna(hit_ratio) and hit_ratio > 0.50,
                pd.notna(mean_contribution) and mean_contribution > 0,
                pd.notna(median_contribution) and median_contribution > 0,
                pd.notna(dynamic_equal) and dynamic_equal > 0,
                pd.notna(dynamic_weighted) and dynamic_weighted > 0,
                pd.notna(fund_positive_ratio) and fund_positive_ratio > 0.50,
            ]
        )
        rows.append(
            {
                "基金管理人简称": company,
                "evaluation_pool": evaluation_pool,
                "return_observation_count": return_observations,
                "switch_observation_count": observations,
                "switch_unique_fund_count": switch_funds,
                "switch_success_count": successes,
                "switch_hit_ratio": hit_ratio,
                "wilson_lower_95": lower,
                "wilson_upper_95": upper,
                "mean_active_contribution": mean_contribution,
                "median_active_contribution": median_contribution,
                "winsor_1_mean_contribution": winsorized_mean(
                    switching["threshold_switch_contribution"], 0.01
                ),
                "winsor_5_mean_contribution": winsorized_mean(
                    switching["threshold_switch_contribution"], 0.05
                ),
                "bootstrap_lower_95": bootstrap_lower,
                "bootstrap_upper_95": bootstrap_upper,
                "equal_weight_dynamic_excess": dynamic_equal,
                "nav_weighted_dynamic_excess": dynamic_weighted,
                "equal_weight_three_asset_excess": pd.to_numeric(
                    valid_three_asset["excess_return_vs_three_asset"], errors="coerce"
                ).mean(),
                "nav_weighted_three_asset_excess": _weighted_mean(
                    valid_three_asset, "excess_return_vs_three_asset"
                ),
                "fund_positive_ratio": fund_positive_ratio,
                "company_quarter_positive_ratio": quarter_means.gt(0).mean(),
                "worst_company_quarter_contribution": quarter_means.min(),
                "max_single_fund_observation_share": max_share,
                "top_two_fund_observation_share": top_two_share,
                "max_single_fund_absolute_contribution_share": maximum_absolute_share,
                "formal_gate_reason": "通过" if not failed_gates else "；".join(failed_gates),
                "positive_threshold_count": int(positive_thresholds.get(company, 0)),
                "positive_signal_count": positive_signals,
            }
        )
    result = pd.DataFrame(rows)
    result["credibility_grade"] = result.apply(assign_credibility_grade, axis=1)
    result["formal_rank"] = float("nan")
    formal = result.loc[result["evaluation_pool"].eq("正式评价")].sort_values(
        ["wilson_lower_95", "mean_active_contribution", "nav_weighted_dynamic_excess", "switch_observation_count"],
        ascending=[False, False, False, False],
    )
    result.loc[formal.index, "formal_rank"] = range(1, len(formal) + 1)
    pool_order = pd.Categorical(
        result["evaluation_pool"],
        categories=["正式评价", "观察池"],
        ordered=True,
    )
    result = result.assign(_pool_order=pool_order).sort_values(
        ["_pool_order", "formal_rank", "switch_hit_ratio"],
        ascending=[True, True, False],
        na_position="last",
    )
    return result.drop(columns="_pool_order").reset_index(drop=True)


def _build_extreme_sensitivity(detail: pd.DataFrame) -> pd.DataFrame:
    switching = detail.loc[detail["drift_adjusted_switch_contribution"].notna()]
    rows = []
    for company, group in switching.groupby("基金管理人简称", sort=True):
        values = group["drift_adjusted_switch_contribution"]
        rows.append(
            {
                "基金管理人简称": company,
                "observation_count": len(values),
                "raw_mean_contribution": values.mean(),
                "winsor_1_mean_contribution": winsorized_mean(values, 0.01),
                "winsor_5_mean_contribution": winsorized_mean(values, 0.05),
                "median_contribution": values.median(),
                "all_four_same_sign": len(
                    {
                        int(np.sign(value))
                        for value in [
                            values.mean(),
                            winsorized_mean(values, 0.01),
                            winsorized_mean(values, 0.05),
                            values.median(),
                        ]
                        if pd.notna(value)
                    }
                )
                <= 1,
            }
        )
    return pd.DataFrame(rows)


def _audit_row(item, result, expected, passed, note):
    return {
        "检查项目": item,
        "实际结果": result,
        "预期或判断标准": expected,
        "检查结论": "通过" if passed else "失败",
        "说明": note,
    }


def write_enhanced_outputs(
    project_dir: Path,
    bootstrap_draws: int = 2000,
) -> dict[str, pd.DataFrame]:
    """Run the enhanced analysis and write auditable CSV outputs."""
    project_dir = Path(project_dir)
    intermediate = project_dir / "intermediate"
    detail = pd.read_csv(
        intermediate / "fund_quarter_three_benchmark_effectiveness.csv",
        dtype={"基金主代码": "string", "证券代码": "string", "security_code": "string"},
    )
    detail = calculate_leave_one_out_peer(detail)
    detail = add_drift_adjusted_metrics(detail)
    company = build_company_evaluation(
        detail,
        bootstrap_draws=bootstrap_draws,
    )
    fund_annualized = build_fund_annualized_metrics(detail)
    company_annualized = build_company_annualized_metrics(fund_annualized, company)
    threshold = build_threshold_robustness(detail)
    direction = build_direction_environment(detail)
    three_asset_columns = [
        column
        for column in [
            "报告期",
            "基金主代码",
            "基金名称",
            "基金管理人简称",
            "stock_weight",
            "bond_weight",
            "bank_deposit_weight",
            "three_asset_coverage",
            "three_asset_coverage_status",
            "csi300_next_quarter_return",
            "bond_next_quarter_return",
            "three_asset_benchmark_return",
            "next_quarter_return",
            "excess_return_vs_three_asset",
        ]
        if column in detail.columns
    ]
    three_asset = detail[three_asset_columns].copy()
    concentration = company[
        [
            "基金管理人简称",
            "evaluation_pool",
            "credibility_grade",
            "switch_observation_count",
            "switch_unique_fund_count",
            "max_single_fund_observation_share",
            "top_two_fund_observation_share",
            "max_single_fund_absolute_contribution_share",
            "formal_gate_reason",
        ]
    ].copy()
    extreme = _build_extreme_sensitivity(detail)
    event_ledger = pd.read_csv(
        intermediate / "extreme_return_review.csv",
        dtype={"基金主代码": "string", "security_code": "string"},
    )
    event_ledger["人工事件类型"] = pd.NA
    event_ledger["是否剔除"] = pd.NA
    event_ledger["备注"] = pd.NA

    duplicate_count = int(detail.duplicated(["基金主代码", "报告期"]).sum())
    drift_rows = detail.loc[detail["passive_normalized_stock_weight"].notna()]
    passive_recomputed = (
        drift_rows["previous_normalized_stock_weight"]
        * (1 + drift_rows["current_period_stock_return"])
        / (
            drift_rows["previous_normalized_stock_weight"]
            * (1 + drift_rows["current_period_stock_return"])
            + (1 - drift_rows["previous_normalized_stock_weight"])
            * (1 + drift_rows["current_period_bond_return"])
        )
    )
    drift_error = (
        passive_recomputed - drift_rows["passive_normalized_stock_weight"]
    ).abs()
    switching = detail.loc[detail["drift_adjusted_switch_contribution"].notna()]
    contribution_error = (
        switching["drift_adjusted_stock_rebalance"]
        * switching["stock_bond_relative_return_spread"]
        - switching["drift_adjusted_switch_contribution"]
    ).abs()
    threshold_counts = threshold.groupby("threshold")["switch_observation_count"].sum()
    nested = (
        threshold_counts.get(0.03, 0)
        >= threshold_counts.get(0.05, 0)
        >= threshold_counts.get(0.10, 0)
    )
    formal = company.loc[company["evaluation_pool"].eq("正式评价")]
    formal_violations = int(
        (
            formal["switch_observation_count"].lt(30)
            | formal["switch_unique_fund_count"].lt(5)
            | formal["return_observation_count"].lt(30)
            | formal["max_single_fund_observation_share"].gt(0.50)
        ).sum()
    )
    grade_violations = int(
        (company.apply(assign_credibility_grade, axis=1) != company["credibility_grade"]).sum()
    )
    coverage_violations = int(
        (
            detail["three_asset_coverage_status"].ne("可用")
            & detail["excess_return_vs_three_asset"].notna()
        ).sum()
    )
    wilson_violations = int(
        (
            company["wilson_lower_95"].notna()
            & (
                company["wilson_lower_95"].lt(0)
                | company["wilson_upper_95"].gt(1)
                | company["wilson_lower_95"].gt(company["wilson_upper_95"])
            )
        ).sum()
    )
    bootstrap_difference = 0.0
    if not formal.empty:
        sample_company = formal.iloc[0]["基金管理人简称"]
        sample_switching = switching.loc[
            switching["基金管理人简称"].eq(sample_company)
        ]
        first_interval = two_way_cluster_bootstrap_mean(
            sample_switching,
            "drift_adjusted_switch_contribution",
            draws=bootstrap_draws,
        )
        second_interval = two_way_cluster_bootstrap_mean(
            sample_switching,
            "drift_adjusted_switch_contribution",
            draws=bootstrap_draws,
        )
        bootstrap_difference = max(
            abs(first_interval[0] - second_interval[0]),
            abs(first_interval[1] - second_interval[1]),
        )
    return_columns = [
        "next_quarter_return",
        "csi300_next_quarter_return",
        "bond_next_quarter_return",
        "three_asset_benchmark_return",
        "loo_peer_median_return",
    ]
    return_quarters_over_13 = int(fund_annualized["return_quarter_count"].gt(13).sum())
    formal_return_under_8 = int(
        (
            fund_annualized["annualized_evaluation_pool"].eq("正式年化")
            & fund_annualized["return_quarter_count"].lt(8)
        ).sum()
    )
    formal_contribution_under_8 = int(
        (
            fund_annualized["contribution_evaluation_pool"].eq("正式贡献")
            & fund_annualized["contribution_quarter_count"].lt(8)
        ).sum()
    )
    return_loss_funds = 0
    contribution_loss_funds = 0
    small_rebalance_not_zero = 0
    unified_quarter_violations = 0
    annualized_by_fund = fund_annualized.set_index("基金主代码")
    annualized_value_columns = [
        "annualized_fund_return",
        "annualized_csi300_return",
        "annualized_bond_return",
        "annualized_three_asset_benchmark_return",
        "annualized_loo_peer_return",
    ]
    for fund_code, group in detail.groupby("基金主代码", sort=True):
        common = group[return_columns].notna().all(axis=1) & group[
            "three_asset_coverage_status"
        ].eq("可用")
        common_returns = group.loc[common, return_columns]
        return_loss_funds += int(
            not common_returns.empty and common_returns.le(-1).any().any()
        )
        contribution = _continuous_active_contribution(group, threshold=0.05)
        valid_contribution = contribution.dropna()
        contribution_loss_funds += int(
            not valid_contribution.empty and valid_contribution.le(-1).any()
        )
        rebalance = pd.to_numeric(
            group["drift_adjusted_stock_rebalance"], errors="coerce"
        )
        comparable = (
            group["drift_adjusted_comparable"].fillna(False)
            & rebalance.notna()
            & pd.to_numeric(
                group["stock_bond_relative_return_spread"], errors="coerce"
            ).notna()
        )
        small_rebalance_not_zero += int(
            contribution.loc[comparable & rebalance.abs().lt(0.05)].ne(0).sum()
        )
        annualized_row = annualized_by_fund.loc[fund_code]
        unified_quarter_violations += int(
            annualized_row["return_quarter_count"] != len(common_returns)
            or (
                annualized_row["annualized_evaluation_pool"] == "正式年化"
                and annualized_row[annualized_value_columns].isna().any()
            )
        )
    expected_company_counts = fund_annualized.loc[
        fund_annualized["annualized_evaluation_pool"].eq("正式年化")
    ].groupby("基金管理人简称").size()
    observed_company_counts = company_annualized.set_index("基金管理人简称")[
        "annualized_valid_fund_count"
    ]
    company_count_difference = sum(
        int(
            observed_company_counts.get(company, 0)
            != expected_company_counts.get(company, 0)
        )
        for company in set(observed_company_counts.index) | set(expected_company_counts.index)
    )
    audit = pd.DataFrame(
        [
            _audit_row("基金-季度键重复数", duplicate_count, 0, duplicate_count == 0, "每只基金每季度唯一"),
            _audit_row("被动漂移公式最大误差", float(drift_error.max()) if not drift_error.empty else 0.0, "<=1e-12", drift_error.empty or drift_error.max() <= 1e-12, "上期权重按本期股债收益漂移"),
            _audit_row("主动切换贡献公式最大误差", float(contribution_error.max()) if not contribution_error.empty else 0.0, "<=1e-12", contribution_error.empty or contribution_error.max() <= 1e-12, "主动调仓×下期股债收益差"),
            _audit_row("阈值样本包含关系", f"{threshold_counts.get(0.03, 0)}/{threshold_counts.get(0.05, 0)}/{threshold_counts.get(0.10, 0)}", "3%>=5%>=10%", nested, "阈值越高样本不得增加"),
            _audit_row("正式公司门槛违反数", formal_violations, 0, formal_violations == 0, "30次、5只、30条收益和50%集中度"),
            _audit_row("分级规则违反数", grade_violations, 0, grade_violations == 0, "A/B/C/观察池规则复算"),
            _audit_row("三资产覆盖异常超额数", coverage_violations, 0, coverage_violations == 0, "80%-120%之外不计超额"),
            _audit_row("Wilson区间越界数", wilson_violations, 0, wilson_violations == 0, "区间必须在0到1之间"),
            _audit_row("bootstrap重复运行最大差异", bootstrap_difference, 0, bootstrap_difference == 0, "固定随机种子需可复现"),
            _audit_row("年化收益季度数超过13的基金数", return_quarters_over_13, 0, return_quarters_over_13 == 0, "统一收益可比季度最多13个"),
            _audit_row("正式年化基金少于8季度的数量", formal_return_under_8, 0, formal_return_under_8 == 0, "正式年化至少8个统一收益可比季度"),
            _audit_row("正式贡献基金少于8漂移季度的数量", formal_contribution_under_8, 0, formal_contribution_under_8 == 0, "正式贡献至少8个漂移可比较季度"),
            _audit_row("统一收益可比季度中收益小于等于-100%的基金数", return_loss_funds, 0, return_loss_funds == 0, "几何年化输入必须大于-100%"),
            _audit_row("漂移可比季度中贡献小于等于-100%的基金数", contribution_loss_funds, 0, contribution_loss_funds == 0, "几何贡献输入必须大于-100%"),
            _audit_row("小于5%可比调仓未归零的数量", small_rebalance_not_zero, 0, small_rebalance_not_zero == 0, "5%以内可比调仓的连续贡献为0"),
            _audit_row("基金与基准统一季度集合违反数", unified_quarter_violations, 0, unified_quarter_violations == 0, "基金和四类基准使用同一季度集合"),
            _audit_row("公司有效基金数汇总差异", company_count_difference, 0, company_count_difference == 0, "公司正式年化基金数与基金明细一致"),
        ]
    )

    outputs = {
        "enhanced_fund_quarter_detail.csv": detail,
        "enhanced_company_evaluation.csv": company,
        "enhanced_threshold_robustness.csv": threshold,
        "enhanced_direction_environment.csv": direction,
        "enhanced_three_asset_benchmark.csv": three_asset,
        "enhanced_product_concentration.csv": concentration,
        "enhanced_extreme_sensitivity.csv": extreme,
        "enhanced_event_ledger.csv": event_ledger,
        "enhanced_fund_annualized_return.csv": fund_annualized,
        "enhanced_company_annualized_return.csv": company_annualized,
        "enhanced_audit.csv": audit,
    }
    intermediate.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(intermediate / filename, index=False, encoding="utf-8-sig")
    return outputs


if __name__ == "__main__":
    write_enhanced_outputs(Path(__file__).parent)
