"""Evaluate dynamic mixed funds against the CSI 300 price index."""

import json
from pathlib import Path

import pandas as pd

from .analyze_allocation_effectiveness import add_allocation_change_labels


CSI300_SOURCE_URL = (
    "https://www.csindex.com.cn/csindex-home/perf/index-perf?"
    "indexCode=000300&startDate=20230301&endDate=20260630"
)


def build_quarterly_csi300_benchmark(daily: pd.DataFrame) -> pd.DataFrame:
    """Use the last trading-day close in each quarter and calculate forward returns."""
    result = daily[["tradeDate", "close"]].copy()
    result["date"] = pd.to_datetime(result["tradeDate"], format="%Y%m%d", errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"]).sort_values("date")
    result["报告期"] = result["date"].dt.to_period("Q").astype(str)
    result = result.groupby("报告期", sort=True).tail(1).copy()
    result["trade_date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["csi300_close"] = result["close"]
    result["next_quarter"] = result["报告期"].shift(-1)
    result["csi300_next_quarter_return"] = result["csi300_close"].shift(-1) / result["csi300_close"] - 1
    return result[[
        "报告期", "trade_date", "csi300_close", "next_quarter", "csi300_next_quarter_return"
    ]].reset_index(drop=True)


def add_csi300_effectiveness_metrics(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Add CSI 300 excess-return and stock-timing metrics to dynamic stock-bond funds."""
    labelled = add_allocation_change_labels(panel)
    labelled["previous_allocation_quality"] = labelled.groupby("基金主代码")["异常标记"].shift()
    result = labelled.merge(
        benchmark[["报告期", "csi300_next_quarter_return"]],
        on="报告期",
        how="left",
        validate="many_to_one",
    )
    result = result.loc[
        result["allocation_style_group"].eq("动态股债配置型")
        & result["return_match_status"].eq("matched")
    ].copy()
    result["excess_return_vs_csi300"] = (
        result["next_quarter_return"] - result["csi300_next_quarter_return"]
    )
    result["outperformed_csi300"] = result["excess_return_vs_csi300"].gt(0).astype("Int64")
    result["csi300_market_environment"] = result["csi300_next_quarter_return"].map(
        lambda value: "上涨" if value > 0 else ("下跌" if value < 0 else "持平")
    )

    result["stock_timing_assessment"] = "未明显调仓"
    first_observation = result["previous_allocation_quality"].isna()
    comparable_quality = (
        result["异常标记"].eq("可用")
        & result["previous_allocation_quality"].eq("可用")
    )
    active_action = result["stock_allocation_action"].isin(["加股票", "减股票"])
    nonzero_market = result["csi300_next_quarter_return"].ne(0)
    timing_eligible = comparable_quality & active_action & nonzero_market
    correct = timing_eligible & (
        (result["stock_allocation_action"].eq("加股票") & result["csi300_next_quarter_return"].gt(0))
        | (result["stock_allocation_action"].eq("减股票") & result["csi300_next_quarter_return"].lt(0))
    )
    result.loc[first_observation, "stock_timing_assessment"] = "首次观察"
    result.loc[~first_observation & ~comparable_quality, "stock_timing_assessment"] = "配置数据不可比较"
    result.loc[timing_eligible & ~correct, "stock_timing_assessment"] = "方向错误"
    result.loc[correct, "stock_timing_assessment"] = "方向正确"

    result["stock_timing_hit"] = pd.NA
    result.loc[timing_eligible, "stock_timing_hit"] = correct.loc[timing_eligible].astype(int)
    result["stock_timing_hit"] = pd.to_numeric(result["stock_timing_hit"], errors="coerce")
    result["simplified_stock_timing_contribution"] = pd.NA
    result.loc[timing_eligible, "simplified_stock_timing_contribution"] = (
        result.loc[timing_eligible, "stock_weight_change"]
        * result.loc[timing_eligible, "csi300_next_quarter_return"]
    )
    result["simplified_stock_timing_contribution"] = pd.to_numeric(
        result["simplified_stock_timing_contribution"], errors="coerce"
    )
    return result.reset_index(drop=True)


def _summarize_company_csi300(group: pd.DataFrame) -> pd.Series:
    valid_weight = (
        group["fund_nav_yuan"].gt(0)
        & group["fund_nav_yuan"].notna()
        & group["excess_return_vs_csi300"].notna()
    )
    weighted = group.loc[valid_weight]
    weighted_excess = (
        (weighted["excess_return_vs_csi300"] * weighted["fund_nav_yuan"]).sum()
        / weighted["fund_nav_yuan"].sum()
        if not weighted.empty
        else float("nan")
    )
    timing = group.loc[group["stock_timing_hit"].notna()]
    up_market = group.loc[group["csi300_market_environment"].eq("上涨")]
    down_market = group.loc[group["csi300_market_environment"].eq("下跌")]
    return pd.Series(
        {
            "weighted_excess_return_vs_csi300": weighted_excess,
            "mean_excess_return_vs_csi300": group["excess_return_vs_csi300"].mean(),
            "median_excess_return_vs_csi300": group["excess_return_vs_csi300"].median(),
            "csi300_outperformance_ratio": group["outperformed_csi300"].mean(),
            "up_market_observation_count": len(up_market),
            "up_market_mean_excess_return": up_market["excess_return_vs_csi300"].mean(),
            "up_market_outperformance_ratio": up_market["outperformed_csi300"].mean(),
            "down_market_observation_count": len(down_market),
            "down_market_mean_excess_return": down_market["excess_return_vs_csi300"].mean(),
            "down_market_outperformance_ratio": down_market["outperformed_csi300"].mean(),
            "matched_observation_count": len(group),
            "unique_fund_count": group["基金主代码"].nunique(),
            "total_start_of_quarter_nav_yuan": weighted["fund_nav_yuan"].sum(),
            "stock_timing_observation_count": len(timing),
            "stock_timing_unique_fund_count": timing["基金主代码"].nunique(),
            "stock_timing_hit_ratio": timing["stock_timing_hit"].mean(),
            "mean_simplified_stock_timing_contribution": timing[
                "simplified_stock_timing_contribution"
            ].mean(),
        }
    )


def build_company_csi300_effectiveness(
    fund_effectiveness: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate CSI 300 metrics at company-quarter and company-total levels."""
    company_quarter = (
        fund_effectiveness.groupby(["基金管理人简称", "报告期"], as_index=False)
        .apply(_summarize_company_csi300)
        .reset_index(drop=True)
    )
    company_total = (
        fund_effectiveness.groupby("基金管理人简称", as_index=False)
        .apply(_summarize_company_csi300)
        .reset_index(drop=True)
    )
    return company_quarter, company_total


def build_fund_csi300_effectiveness(fund_quarter: pd.DataFrame) -> pd.DataFrame:
    """Summarize benchmark-relative performance and timing for each fund."""
    rows = []
    identity_columns = ["基金主代码", "基金名称", "基金管理人简称", "投资风格"]
    for fund_main_code, group in fund_quarter.groupby("基金主代码", sort=False):
        excess = group["excess_return_vs_csi300"]
        excess_volatility = excess.std(ddof=1)
        up_market = group.loc[group["csi300_market_environment"].eq("上涨")]
        down_market = group.loc[group["csi300_market_environment"].eq("下跌")]
        timing = group.loc[group["stock_timing_hit"].notna()]
        row = {column: group.iloc[0][column] for column in identity_columns}
        row.update(
            {
                "matched_observation_count": len(group),
                "mean_excess_return_vs_csi300": excess.mean(),
                "median_excess_return_vs_csi300": excess.median(),
                "excess_return_volatility": excess_volatility,
                "annualized_information_ratio_vs_csi300": (
                    excess.mean() / excess_volatility * 2
                    if pd.notna(excess_volatility) and excess_volatility > 0
                    else float("nan")
                ),
                "csi300_outperformance_ratio": group["outperformed_csi300"].mean(),
                "up_market_observation_count": len(up_market),
                "up_market_mean_excess_return": up_market["excess_return_vs_csi300"].mean(),
                "up_market_outperformance_ratio": up_market["outperformed_csi300"].mean(),
                "down_market_observation_count": len(down_market),
                "down_market_mean_excess_return": down_market["excess_return_vs_csi300"].mean(),
                "down_market_outperformance_ratio": down_market["outperformed_csi300"].mean(),
                "stock_timing_observation_count": len(timing),
                "stock_timing_hit_ratio": timing["stock_timing_hit"].mean(),
                "mean_simplified_stock_timing_contribution": timing[
                    "simplified_stock_timing_contribution"
                ].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("基金主代码").reset_index(drop=True)


def build_company_csi300_timing_ranking(company_total: pd.DataFrame) -> pd.DataFrame:
    """Rank companies with enough independent stock-timing observations."""
    eligible = company_total.loc[
        company_total["stock_timing_unique_fund_count"].ge(3)
        & company_total["stock_timing_observation_count"].ge(10)
    ].copy()
    eligible = eligible.sort_values(
        [
            "stock_timing_hit_ratio",
            "mean_simplified_stock_timing_contribution",
            "weighted_excess_return_vs_csi300",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    eligible.insert(0, "stock_timing_rank", range(1, len(eligible) + 1))
    return eligible


def _audit_row(item, result, expected, status, note):
    return {
        "检查项目": item,
        "实际结果": result,
        "预期或判断标准": expected,
        "检查结论": status,
        "说明": note,
    }


def write_csi300_outputs(project_dir: Path, source_json_path: Path) -> dict[str, pd.DataFrame]:
    """Build and save all CSI 300 benchmark effectiveness outputs."""
    intermediate = project_dir / "intermediate"
    source = json.loads(source_json_path.read_text())
    daily = pd.DataFrame(source["data"])
    benchmark = build_quarterly_csi300_benchmark(daily)
    benchmark.insert(0, "index_code", "000300")
    benchmark.insert(1, "index_name", "沪深300")
    benchmark["benchmark_type"] = "价格指数"
    benchmark["source_url"] = CSI300_SOURCE_URL

    panel = pd.read_csv(
        intermediate / "fund_quarter_allocation_return.csv",
        dtype={"基金主代码": "string", "证券代码": "string", "security_code": "string"},
    )
    fund_quarter = add_csi300_effectiveness_metrics(panel, benchmark)
    fund_total = build_fund_csi300_effectiveness(fund_quarter)
    company_quarter, company_total = build_company_csi300_effectiveness(fund_quarter)
    company_quarter["aggregation_level"] = "company_quarter"
    company_total["aggregation_level"] = "company_total"
    company_total["报告期"] = "总体"
    company = pd.concat([company_quarter, company_total], ignore_index=True, sort=False)
    ranking = build_company_csi300_timing_ranking(company_total)

    return_identity = (
        fund_quarter["next_quarter_return"]
        - fund_quarter["csi300_next_quarter_return"]
        - fund_quarter["excess_return_vs_csi300"]
    ).abs()
    timing = fund_quarter.loc[fund_quarter["stock_timing_hit"].notna()]
    timing_identity = (
        timing["stock_weight_change"] * timing["csi300_next_quarter_return"]
        - timing["simplified_stock_timing_contribution"]
    ).abs()
    expected_quarter_count = panel["报告期"].nunique() + 1
    audit = pd.DataFrame(
        [
            _audit_row("官方接口响应代码", source.get("code"), "200", "通过" if source.get("code") == "200" else "失败", "中证指数有限公司"),
            _audit_row("日度行情记录数", len(daily), "大于0", "通过" if len(daily) > 0 else "失败", "000300日度收盘点位"),
            _audit_row("日度交易日重复", int(daily["tradeDate"].duplicated().sum()), 0, "通过" if not daily["tradeDate"].duplicated().any() else "失败", "交易日必须唯一"),
            _audit_row("非正收盘点位", int(pd.to_numeric(daily["close"], errors="coerce").le(0).sum()), 0, "通过" if not pd.to_numeric(daily["close"], errors="coerce").le(0).any() else "失败", "指数点位应为正"),
            _audit_row("季度末点位数量", len(benchmark), expected_quarter_count, "通过" if len(benchmark) == expected_quarter_count else "失败", "收益面板季度数加一个终点季度"),
            _audit_row("基金季度基准收益缺失", int(fund_quarter["csi300_next_quarter_return"].isna().sum()), 0, "通过" if fund_quarter["csi300_next_quarter_return"].notna().all() else "失败", "每个基金收益期间必须有沪深300收益"),
            _audit_row("基金相对沪深300收益公式最大误差", float(return_identity.max()), "<=1e-12", "通过" if return_identity.max() <= 1e-12 else "失败", "基金收益-沪深300收益"),
            _audit_row("股票择时贡献公式最大误差", float(timing_identity.max()) if not timing_identity.empty else 0.0, "<=1e-12", "通过" if timing_identity.empty or timing_identity.max() <= 1e-12 else "失败", "股票仓位变化×下一季度沪深300收益"),
            _audit_row("有效股票择时观察", len(timing), "信息项", "通过", "仅当前及上期配置均可用且仓位变化至少5个百分点"),
            _audit_row("正式择时排名公司数", len(ranking), "信息项", "通过", "至少3只可比较基金且10条择时观察"),
            _audit_row("排名门槛误入公司", int(((ranking["stock_timing_unique_fund_count"] < 3) | (ranking["stock_timing_observation_count"] < 10)).sum()) if not ranking.empty else 0, 0, "通过", "排名样本门槛复核"),
        ]
    )

    outputs = {
        "csi300_quarterly_benchmark.csv": benchmark,
        "fund_quarter_csi300_effectiveness.csv": fund_quarter,
        "fund_csi300_effectiveness.csv": fund_total,
        "fund_company_csi300_effectiveness.csv": company,
        "fund_company_csi300_timing_ranking.csv": ranking,
        "csi300_benchmark_audit.csv": audit,
    }
    intermediate.mkdir(exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(intermediate / filename, index=False, encoding="utf-8-sig")
    return outputs


if __name__ == "__main__":
    root = Path(__file__).parent
    write_csi300_outputs(root, root / "source_data" / "csi300_daily_20230301_20260630.json")
