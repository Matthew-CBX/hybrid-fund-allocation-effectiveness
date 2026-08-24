"""Measure allocation effectiveness for dynamically allocated mixed funds."""

from pathlib import Path

import pandas as pd


def calculate_peer_relative_return(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate return relative to same-quarter, same-style dynamic stock-bond peers."""
    eligible = panel.loc[
        panel["allocation_style_group"].eq("动态股债配置型")
        & panel["return_match_status"].eq("matched")
    ].copy()
    peer_groups = eligible.groupby(["报告期", "投资风格"])["next_quarter_return"]
    eligible["peer_median_return"] = peer_groups.transform("median")
    eligible["peer_group_size"] = peer_groups.transform("size")
    eligible["peer_relative_return"] = (
        eligible["next_quarter_return"] - eligible["peer_median_return"]
    )
    return eligible.reset_index(drop=True)


def _allocation_action(change: float, increase_label: str, decrease_label: str) -> str:
    if pd.isna(change):
        return "首次观察"
    if change >= 0.05:
        return increase_label
    if change <= -0.05:
        return decrease_label
    return "基本不变"


def add_allocation_change_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Add quarter-over-quarter stock/bond changes and direction labels."""
    result = panel.sort_values(["基金主代码", "报告期"]).copy()
    result["stock_weight_change"] = result.groupby("基金主代码")["stock_weight"].diff()
    result["bond_weight_change"] = result.groupby("基金主代码")["bond_weight"].diff()
    result["stock_allocation_action"] = result["stock_weight_change"].map(
        lambda value: _allocation_action(value, "加股票", "减股票")
    )
    result["bond_allocation_action"] = result["bond_weight_change"].map(
        lambda value: _allocation_action(value, "加债券", "减债券")
    )
    return result.reset_index(drop=True)


def _summarize_company(group: pd.DataFrame) -> pd.Series:
    valid_weight = group["fund_nav_yuan"].gt(0) & group["fund_nav_yuan"].notna()
    weighted_group = group.loc[valid_weight]
    weighted_return = (
        (weighted_group["peer_relative_return"] * weighted_group["fund_nav_yuan"]).sum()
        / weighted_group["fund_nav_yuan"].sum()
        if not weighted_group.empty
        else float("nan")
    )
    return pd.Series(
        {
            "weighted_peer_relative_return": weighted_return,
            "matched_observation_count": len(group),
            "unique_fund_count": group["基金主代码"].nunique(),
            "total_start_of_quarter_nav_yuan": weighted_group["fund_nav_yuan"].sum(),
            "positive_relative_return_ratio": group["peer_relative_return"].gt(0).mean(),
        }
    )


def build_company_effectiveness(fund_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate peer-relative performance to company-quarter and company-total levels."""
    company_quarter = (
        fund_panel.groupby(["基金管理人简称", "报告期"], as_index=False)
        .apply(_summarize_company)
        .reset_index(drop=True)
    )
    company_total = (
        fund_panel.groupby("基金管理人简称", as_index=False)
        .apply(_summarize_company)
        .reset_index(drop=True)
    )
    return company_quarter, company_total


def build_ranked_company_effectiveness(company_effectiveness: pd.DataFrame) -> pd.DataFrame:
    """Return only companies with enough dynamic-fund observations for formal ranking."""
    ranked = company_effectiveness.loc[
        company_effectiveness["aggregation_level"].eq("company_total")
        & company_effectiveness["unique_fund_count"].ge(3)
        & company_effectiveness["matched_observation_count"].ge(20)
    ].copy()
    ranked = ranked.sort_values(
        ["weighted_peer_relative_return", "matched_observation_count"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "company_rank", range(1, len(ranked) + 1))
    return ranked


def build_allocation_action_effectiveness(fund_effectiveness: pd.DataFrame) -> pd.DataFrame:
    """Summarize relative performance after stock and bond allocation actions."""
    tables = []
    for asset_class, action_column in [
        ("股票", "stock_allocation_action"),
        ("债券", "bond_allocation_action"),
    ]:
        eligible = fund_effectiveness.loc[
            fund_effectiveness[action_column].ne("首次观察"),
            ["基金主代码", action_column, "peer_relative_return"],
        ].copy()
        summary = (
            eligible.groupby(action_column)
            .agg(
                observation_count=("peer_relative_return", "size"),
                unique_fund_count=("基金主代码", "nunique"),
                mean_peer_relative_return=("peer_relative_return", "mean"),
                median_peer_relative_return=("peer_relative_return", "median"),
                positive_relative_return_ratio=(
                    "peer_relative_return", lambda series: series.gt(0).mean()
                ),
            )
            .reset_index()
            .rename(columns={action_column: "allocation_action"})
        )
        summary.insert(0, "asset_class", asset_class)
        tables.append(summary)
    return pd.concat(tables, ignore_index=True)


def build_company_effectiveness_stability(
    fund_effectiveness: pd.DataFrame,
    company_effectiveness: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Describe ranked companies' consistency in strong and weak peer environments."""
    environment = (
        fund_effectiveness.groupby("报告期")["next_quarter_return"]
        .median()
        .rename("principal_sample_median_return")
        .reset_index()
    )
    environment["market_environment"] = environment["principal_sample_median_return"].map(
        lambda value: "偏强环境" if value >= 0 else "偏弱环境"
    )
    eligible_companies = ranking["基金管理人简称"].drop_duplicates()
    company_quarter = company_effectiveness.loc[
        company_effectiveness["aggregation_level"].eq("company_quarter")
        & company_effectiveness["基金管理人简称"].isin(eligible_companies)
    ].merge(environment[["报告期", "market_environment"]], on="报告期", how="left")

    rows = []
    for company, group in company_quarter.groupby("基金管理人简称", sort=False):
        strong = group.loc[group["market_environment"].eq("偏强环境"), "weighted_peer_relative_return"]
        weak = group.loc[group["market_environment"].eq("偏弱环境"), "weighted_peer_relative_return"]
        rows.append(
            {
                "基金管理人简称": company,
                "covered_quarter_count": len(group),
                "mean_company_quarter_relative_return": group["weighted_peer_relative_return"].mean(),
                "median_company_quarter_relative_return": group["weighted_peer_relative_return"].median(),
                "positive_company_quarter_ratio": group["weighted_peer_relative_return"].gt(0).mean(),
                "strong_environment_quarter_count": len(strong),
                "strong_environment_mean_relative_return": strong.mean(),
                "weak_environment_quarter_count": len(weak),
                "weak_environment_mean_relative_return": weak.mean(),
            }
        )
    columns = [
        "基金管理人简称",
        "covered_quarter_count",
        "mean_company_quarter_relative_return",
        "median_company_quarter_relative_return",
        "positive_company_quarter_ratio",
        "strong_environment_quarter_count",
        "strong_environment_mean_relative_return",
        "weak_environment_quarter_count",
        "weak_environment_mean_relative_return",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values("mean_company_quarter_relative_return", ascending=False).reset_index(drop=True)


def write_effectiveness_outputs(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fund-level and company-level effectiveness outputs from the analysis panel."""
    intermediate = project_dir / "intermediate"
    panel = pd.read_csv(intermediate / "fund_quarter_allocation_return.csv")
    labelled_panel = add_allocation_change_labels(panel)
    fund_effectiveness = calculate_peer_relative_return(labelled_panel)
    company_quarter, company_total = build_company_effectiveness(fund_effectiveness)
    company_quarter["aggregation_level"] = "company_quarter"
    company_total["aggregation_level"] = "company_total"
    company_total["报告期"] = "总体"
    company_effectiveness = pd.concat([company_quarter, company_total], ignore_index=True, sort=False)
    company_ranking = build_ranked_company_effectiveness(company_effectiveness)
    allocation_action_effectiveness = build_allocation_action_effectiveness(fund_effectiveness)
    company_stability = build_company_effectiveness_stability(
        fund_effectiveness, company_effectiveness, company_ranking
    )
    fund_effectiveness.to_csv(
        intermediate / "fund_quarter_effectiveness.csv", index=False, encoding="utf-8-sig"
    )
    company_effectiveness.to_csv(
        intermediate / "fund_company_effectiveness.csv", index=False, encoding="utf-8-sig"
    )
    company_ranking.to_csv(
        intermediate / "fund_company_effectiveness_ranking.csv", index=False, encoding="utf-8-sig"
    )
    allocation_action_effectiveness.to_csv(
        intermediate / "allocation_action_effectiveness.csv", index=False, encoding="utf-8-sig"
    )
    company_stability.to_csv(
        intermediate / "fund_company_effectiveness_stability.csv", index=False, encoding="utf-8-sig"
    )
    return fund_effectiveness, company_effectiveness


if __name__ == "__main__":
    write_effectiveness_outputs(Path(__file__).parent)
