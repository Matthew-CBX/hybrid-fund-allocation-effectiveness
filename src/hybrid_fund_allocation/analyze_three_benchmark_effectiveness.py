"""Integrate peer, equity, bond and dynamic stock-bond benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CHINABOND_INDEX_CODE = "CBA00201"
CHINABOND_INDEX_NAME = "中债-综合财富（总值）指数"
CHINABOND_SOURCE_URL = (
    "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQuery?"
    "indexid=2c90818811afed8d0111c0c672b31578&qxlxt=00&zslxt=CFZS&lx=1&locale="
)


def build_quarterly_bond_benchmark(raw: dict) -> pd.DataFrame:
    """Convert official ChinaBond wealth-index JSON to quarter-end forward returns."""
    values = raw["CFZS_00"]
    result = pd.DataFrame(
        {
            "date": (
                pd.to_datetime(
                    pd.Index(values.keys()).astype("int64"),
                    unit="ms",
                    utc=True,
                )
                .tz_convert("Asia/Shanghai")
                .tz_localize(None)
            ),
            "bond_index_close": pd.to_numeric(pd.Series(values.values()), errors="coerce"),
        }
    ).dropna()
    result = result.sort_values("date")
    result["报告期"] = result["date"].dt.to_period("Q").astype(str)
    result = result.groupby("报告期", sort=True).tail(1).copy()
    result["trade_date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["next_quarter"] = result["报告期"].shift(-1)
    result["bond_next_quarter_return"] = (
        result["bond_index_close"].shift(-1) / result["bond_index_close"] - 1
    )
    return result[
        [
            "报告期",
            "trade_date",
            "bond_index_close",
            "next_quarter",
            "bond_next_quarter_return",
        ]
    ].reset_index(drop=True)


def add_three_benchmark_metrics(
    csi_panel: pd.DataFrame,
    peer_panel: pd.DataFrame,
    bond_benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Add bond and normalized dynamic stock-bond metrics to the current panel."""
    result = csi_panel.copy().sort_values(["基金主代码", "报告期"]).reset_index(drop=True)
    peer_columns = [
        "基金主代码",
        "报告期",
        "peer_median_return",
        "peer_group_size",
        "peer_relative_return",
    ]
    result = result.merge(
        peer_panel[peer_columns].drop_duplicates(["基金主代码", "报告期"]),
        on=["基金主代码", "报告期"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        bond_benchmark[["报告期", "bond_next_quarter_return"]],
        on="报告期",
        how="left",
        validate="many_to_one",
    )

    stock = pd.to_numeric(result["stock_weight"], errors="coerce")
    bond = pd.to_numeric(result["bond_weight"], errors="coerce")
    stock_bond_total = stock + bond
    usable = result["异常标记"].eq("可用") & stock.notna() & bond.notna() & stock_bond_total.gt(0)
    result["stock_bond_weight_total"] = stock_bond_total
    result["normalized_stock_weight"] = float("nan")
    result["normalized_bond_weight"] = float("nan")
    result.loc[usable, "normalized_stock_weight"] = stock.loc[usable] / stock_bond_total.loc[usable]
    result.loc[usable, "normalized_bond_weight"] = bond.loc[usable] / stock_bond_total.loc[usable]
    result["normalized_stock_weight"] = pd.to_numeric(
        result["normalized_stock_weight"], errors="coerce"
    )
    result["normalized_bond_weight"] = pd.to_numeric(
        result["normalized_bond_weight"], errors="coerce"
    )

    result["dynamic_stock_bond_benchmark_return"] = (
        result["normalized_stock_weight"] * result["csi300_next_quarter_return"]
        + result["normalized_bond_weight"] * result["bond_next_quarter_return"]
    )
    result["excess_return_vs_bond"] = (
        result["next_quarter_return"] - result["bond_next_quarter_return"]
    )
    result["excess_return_vs_dynamic_stock_bond"] = (
        result["next_quarter_return"] - result["dynamic_stock_bond_benchmark_return"]
    )
    result["outperformed_bond"] = result["excess_return_vs_bond"].gt(0).astype("Int64")
    result["outperformed_dynamic_stock_bond"] = (
        result["excess_return_vs_dynamic_stock_bond"].gt(0).astype("Int64")
    )
    result["bond_market_environment"] = result["bond_next_quarter_return"].map(
        lambda value: "上涨" if value > 0 else ("下跌" if value < 0 else "持平")
    )

    result["previous_allocation_quality_for_switching"] = result.groupby("基金主代码")[
        "异常标记"
    ].shift()
    result["normalized_stock_weight_change"] = result.groupby("基金主代码")[
        "normalized_stock_weight"
    ].diff()
    result["stock_bond_relative_return_spread"] = (
        result["csi300_next_quarter_return"] - result["bond_next_quarter_return"]
    )
    result["stock_bond_switching_contribution"] = (
        result["normalized_stock_weight_change"]
        * result["stock_bond_relative_return_spread"]
    )

    first = result["previous_allocation_quality_for_switching"].isna()
    comparable = (
        result["异常标记"].eq("可用")
        & result["previous_allocation_quality_for_switching"].eq("可用")
        & result["normalized_stock_weight_change"].notna()
    )
    active = result["normalized_stock_weight_change"].abs().ge(0.05)
    nonzero_spread = result["stock_bond_relative_return_spread"].ne(0)
    eligible = comparable & active & nonzero_spread
    positive = result["stock_bond_switching_contribution"].gt(0)

    result["stock_bond_switching_assessment"] = "未明显切换"
    result.loc[first, "stock_bond_switching_assessment"] = "首次观察"
    result.loc[~first & ~comparable, "stock_bond_switching_assessment"] = "配置数据不可比较"
    result.loc[eligible & positive, "stock_bond_switching_assessment"] = "方向正确"
    result.loc[eligible & ~positive, "stock_bond_switching_assessment"] = "方向错误"
    result["stock_bond_switching_hit"] = pd.NA
    result.loc[eligible, "stock_bond_switching_hit"] = positive.loc[eligible].astype(int)
    result["stock_bond_switching_hit"] = pd.to_numeric(
        result["stock_bond_switching_hit"], errors="coerce"
    )
    result.loc[~eligible, "stock_bond_switching_contribution"] = pd.NA
    result["stock_bond_switching_contribution"] = pd.to_numeric(
        result["stock_bond_switching_contribution"], errors="coerce"
    )
    return result


def _nav_weighted(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce")
    weights = pd.to_numeric(group["fund_nav_yuan"], errors="coerce")
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float((values.loc[valid] * weights.loc[valid]).sum() / weights.loc[valid].sum())


def _summarize_company(group: pd.DataFrame) -> pd.Series:
    switching = group.loc[group["stock_bond_switching_hit"].notna()]
    stock_timing = group.loc[group["stock_timing_hit"].notna()]
    return pd.Series(
        {
            "weighted_peer_relative_return": _nav_weighted(group, "peer_relative_return"),
            "weighted_excess_return_vs_csi300": _nav_weighted(group, "excess_return_vs_csi300"),
            "weighted_excess_return_vs_bond": _nav_weighted(group, "excess_return_vs_bond"),
            "weighted_excess_return_vs_dynamic_stock_bond": _nav_weighted(
                group, "excess_return_vs_dynamic_stock_bond"
            ),
            "peer_outperformance_ratio": group["peer_relative_return"].gt(0).mean(),
            "csi300_outperformance_ratio": group["outperformed_csi300"].mean(),
            "bond_outperformance_ratio": group["outperformed_bond"].mean(),
            "dynamic_stock_bond_outperformance_ratio": group[
                "outperformed_dynamic_stock_bond"
            ].mean(),
            "matched_observation_count": len(group),
            "unique_fund_count": group["基金主代码"].nunique(),
            "covered_quarter_count": group["报告期"].nunique(),
            "total_start_of_quarter_nav_yuan": pd.to_numeric(
                group["fund_nav_yuan"], errors="coerce"
            ).sum(),
            "stock_timing_observation_count": len(stock_timing),
            "stock_timing_hit_ratio": stock_timing["stock_timing_hit"].mean(),
            "stock_bond_switching_observation_count": len(switching),
            "stock_bond_switching_unique_fund_count": switching["基金主代码"].nunique(),
            "stock_bond_switching_hit_ratio": switching["stock_bond_switching_hit"].mean(),
            "mean_stock_bond_switching_contribution": switching[
                "stock_bond_switching_contribution"
            ].mean(),
        }
    )


def build_company_three_benchmark_effectiveness(
    fund_quarter: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate integrated metrics to company-quarter and company-total levels."""
    company_quarter = (
        fund_quarter.groupby(["基金管理人简称", "报告期"], as_index=False)
        .apply(_summarize_company)
        .reset_index(drop=True)
    )
    company_total = (
        fund_quarter.groupby("基金管理人简称", as_index=False)
        .apply(_summarize_company)
        .reset_index(drop=True)
    )
    return company_quarter, company_total


def build_stock_bond_switching_ranking(company_total: pd.DataFrame) -> pd.DataFrame:
    """Rank companies meeting both fund-return and switching sample thresholds."""
    eligible = company_total.loc[
        company_total["unique_fund_count"].ge(3)
        & company_total["matched_observation_count"].ge(20)
        & company_total["stock_bond_switching_unique_fund_count"].ge(3)
        & company_total["stock_bond_switching_observation_count"].ge(10)
    ].copy()
    eligible = eligible.sort_values(
        [
            "stock_bond_switching_hit_ratio",
            "mean_stock_bond_switching_contribution",
            "weighted_excess_return_vs_dynamic_stock_bond",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    eligible.insert(0, "stock_bond_switching_rank", range(1, len(eligible) + 1))
    return eligible


def build_fund_three_benchmark_effectiveness(fund_quarter: pd.DataFrame) -> pd.DataFrame:
    """Summarize all benchmark-relative and switching metrics for each fund."""
    rows = []
    identity_columns = ["基金主代码", "基金名称", "基金管理人简称", "投资风格"]
    for _, group in fund_quarter.groupby("基金主代码", sort=True):
        row = {column: group.iloc[0][column] for column in identity_columns}
        dynamic_excess = group["excess_return_vs_dynamic_stock_bond"]
        dynamic_volatility = dynamic_excess.std(ddof=1)
        switching = group.loc[group["stock_bond_switching_hit"].notna()]
        stock_timing = group.loc[group["stock_timing_hit"].notna()]
        row.update(
            {
                "matched_observation_count": len(group),
                "mean_peer_relative_return": group["peer_relative_return"].mean(),
                "peer_outperformance_ratio": group["peer_relative_return"].gt(0).mean(),
                "mean_excess_return_vs_csi300": group["excess_return_vs_csi300"].mean(),
                "csi300_outperformance_ratio": group["outperformed_csi300"].mean(),
                "mean_excess_return_vs_bond": group["excess_return_vs_bond"].mean(),
                "bond_outperformance_ratio": group["outperformed_bond"].mean(),
                "mean_excess_return_vs_dynamic_stock_bond": dynamic_excess.mean(),
                "median_excess_return_vs_dynamic_stock_bond": dynamic_excess.median(),
                "dynamic_stock_bond_outperformance_ratio": group[
                    "outperformed_dynamic_stock_bond"
                ].mean(),
                "annualized_information_ratio_vs_dynamic_stock_bond": (
                    dynamic_excess.mean() / dynamic_volatility * 2
                    if pd.notna(dynamic_volatility) and dynamic_volatility > 0
                    else float("nan")
                ),
                "stock_timing_observation_count": len(stock_timing),
                "stock_timing_hit_ratio": stock_timing["stock_timing_hit"].mean(),
                "stock_bond_switching_observation_count": len(switching),
                "stock_bond_switching_hit_ratio": switching[
                    "stock_bond_switching_hit"
                ].mean(),
                "mean_stock_bond_switching_contribution": switching[
                    "stock_bond_switching_contribution"
                ].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _audit_row(item, result, expected, passed, note):
    return {
        "检查项目": item,
        "实际结果": result,
        "预期或判断标准": expected,
        "检查结论": "通过" if passed else "失败",
        "说明": note,
    }


def write_three_benchmark_outputs(
    project_dir: Path,
    source_json_path: Path,
) -> dict[str, pd.DataFrame]:
    """Run the integrated analysis and save auditable CSV outputs."""
    project_dir = Path(project_dir)
    source_json_path = Path(source_json_path)
    intermediate = project_dir / "intermediate"
    raw = json.loads(source_json_path.read_text())
    full_benchmark = build_quarterly_bond_benchmark(raw)

    csi = pd.read_csv(
        intermediate / "fund_quarter_csi300_effectiveness.csv",
        dtype={"基金主代码": "string", "证券代码": "string", "security_code": "string"},
    )
    peer = pd.read_csv(
        intermediate / "fund_quarter_effectiveness.csv",
        dtype={"基金主代码": "string", "证券代码": "string", "security_code": "string"},
    )
    panel_periods = pd.PeriodIndex(csi["报告期"].unique(), freq="Q")
    first_period = panel_periods.min()
    last_point_period = panel_periods.max() + 1
    benchmark_periods = pd.PeriodIndex(full_benchmark["报告期"], freq="Q")
    benchmark = full_benchmark.loc[
        (benchmark_periods >= first_period) & (benchmark_periods <= last_point_period)
    ].copy()
    benchmark.insert(0, "index_code", CHINABOND_INDEX_CODE)
    benchmark.insert(1, "index_name", CHINABOND_INDEX_NAME)
    benchmark["benchmark_type"] = "财富指数"
    benchmark["source_url"] = CHINABOND_SOURCE_URL

    detail = add_three_benchmark_metrics(csi, peer, benchmark)
    fund = build_fund_three_benchmark_effectiveness(detail)
    company_quarter, company_total = build_company_three_benchmark_effectiveness(detail)
    company_quarter["aggregation_level"] = "company_quarter"
    company_total["aggregation_level"] = "company_total"
    company_total["报告期"] = "总体"
    company = pd.concat([company_quarter, company_total], ignore_index=True, sort=False)
    ranking = build_stock_bond_switching_ranking(company_total)

    raw_series = pd.Series(raw["CFZS_00"], dtype="float64")
    expected_quarter_count = len(panel_periods) + 1
    normalized = detail.loc[
        detail["normalized_stock_weight"].notna()
        & detail["normalized_bond_weight"].notna()
    ]
    normalized_error = (
        normalized["normalized_stock_weight"] + normalized["normalized_bond_weight"] - 1
    ).abs()
    dynamic_error = (
        detail["normalized_stock_weight"] * detail["csi300_next_quarter_return"]
        + detail["normalized_bond_weight"] * detail["bond_next_quarter_return"]
        - detail["dynamic_stock_bond_benchmark_return"]
    ).abs()
    bond_excess_error = (
        detail["next_quarter_return"]
        - detail["bond_next_quarter_return"]
        - detail["excess_return_vs_bond"]
    ).abs()
    switching = detail.loc[detail["stock_bond_switching_hit"].notna()]
    switching_error = (
        switching["normalized_stock_weight_change"]
        * switching["stock_bond_relative_return_spread"]
        - switching["stock_bond_switching_contribution"]
    ).abs()
    ranking_violation_count = int(
        (
            ranking["unique_fund_count"].lt(3)
            | ranking["matched_observation_count"].lt(20)
            | ranking["stock_bond_switching_unique_fund_count"].lt(3)
            | ranking["stock_bond_switching_observation_count"].lt(10)
        ).sum()
    ) if not ranking.empty else 0

    audit = pd.DataFrame(
        [
            _audit_row("债券指数日度记录数", len(raw_series), ">0", len(raw_series) > 0, "CBA00201财富指数"),
            _audit_row("债券指数日期重复", int(raw_series.index.duplicated().sum()), 0, not raw_series.index.duplicated().any(), "时间戳必须唯一"),
            _audit_row("债券指数非正点位", int(raw_series.le(0).sum()), 0, not raw_series.le(0).any(), "指数点位必须为正"),
            _audit_row("债券指数季度末点位数量", len(benchmark), expected_quarter_count, len(benchmark) == expected_quarter_count, "基金收益期间数加一个终点季度"),
            _audit_row("基金季度债券基准缺失", int(detail["bond_next_quarter_return"].isna().sum()), 0, detail["bond_next_quarter_return"].notna().all(), "每条基金收益记录均需债券基准"),
            _audit_row("基金季度同类基准缺失", int(detail["peer_relative_return"].isna().sum()), 0, detail["peer_relative_return"].notna().all(), "每条基金收益记录均需同类基准"),
            _audit_row("归一化股债权重和最大误差", float(normalized_error.max()) if not normalized_error.empty else 0.0, "<=1e-12", normalized_error.empty or normalized_error.max() <= 1e-12, "归一化股票权重+债券权重=1"),
            _audit_row("动态股债基准公式最大误差", float(dynamic_error.max()) if dynamic_error.notna().any() else 0.0, "<=1e-12", not dynamic_error.notna().any() or dynamic_error.max() <= 1e-12, "股债权重乘以对应市场收益"),
            _audit_row("相对债券收益公式最大误差", float(bond_excess_error.max()), "<=1e-12", bond_excess_error.max() <= 1e-12, "基金收益-债券指数收益"),
            _audit_row("股债切换贡献公式最大误差", float(switching_error.max()) if not switching_error.empty else 0.0, "<=1e-12", switching_error.empty or switching_error.max() <= 1e-12, "归一化股票权重变化×股债相对收益差"),
            _audit_row("有效股债切换观察", len(switching), "信息项", True, "权重变化至少5个百分点且配置可比较"),
            _audit_row("正式股债切换排名公司数", len(ranking), "信息项", True, "同时满足收益和切换样本门槛"),
            _audit_row("排名门槛误入公司", ranking_violation_count, 0, ranking_violation_count == 0, "复核3只基金、20条收益、10条切换观察"),
        ]
    )

    outputs = {
        "chinabond_cba00201_quarterly_benchmark.csv": benchmark,
        "fund_quarter_three_benchmark_effectiveness.csv": detail,
        "fund_three_benchmark_effectiveness.csv": fund,
        "fund_company_three_benchmark_effectiveness.csv": company,
        "fund_company_stock_bond_switching_ranking.csv": ranking,
        "three_benchmark_audit.csv": audit,
    }
    intermediate.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(intermediate / filename, index=False, encoding="utf-8-sig")
    return outputs


if __name__ == "__main__":
    root = Path(__file__).parent
    write_three_benchmark_outputs(
        root,
        root / "source_data" / "chinabond_cba00201_wealth.json",
    )
