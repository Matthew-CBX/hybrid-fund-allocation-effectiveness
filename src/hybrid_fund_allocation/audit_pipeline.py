"""Create reproducible data-quality and recalculation audit outputs."""

from pathlib import Path

import pandas as pd

from .merge_allocation_return import (
    _quarter_from_choice_header,
    calculate_forward_quarter_return,
    choose_representative_share,
    normalize_nav_export,
)


def _audit_row(category, item, result, expected, status, note):
    return {
        "检查类别": category,
        "检查项目": item,
        "实际结果": result,
        "预期或判断标准": expected,
        "检查结论": status,
        "说明": note,
    }


def write_audit_outputs(project_dir: Path, nav_export_path: Path) -> pd.DataFrame:
    intermediate = project_dir / "intermediate"
    base = pd.read_excel(project_dir / "Book2.xlsx", dtype={"证券代码": "string", "基金主代码": "string"})
    nav_raw = pd.read_excel(nav_export_path, dtype={"证券代码": "string"})
    clean = pd.read_csv(intermediate / "clean_fund_quarter.csv", dtype={"基金主代码": "string"})
    panel = pd.read_csv(
        intermediate / "fund_quarter_allocation_return.csv",
        dtype={"基金主代码": "string", "security_code": "string"},
    )
    classification = pd.read_csv(
        intermediate / "fund_dynamic_allocation_classification.csv", dtype={"基金主代码": "string"}
    )
    effectiveness = pd.read_csv(
        intermediate / "fund_quarter_effectiveness.csv", dtype={"基金主代码": "string"}
    )
    company = pd.read_csv(intermediate / "fund_company_effectiveness.csv")
    ranking = pd.read_csv(intermediate / "fund_company_effectiveness_ranking.csv")

    valid_nav_code = nav_raw["证券代码"].astype("string").str.match(
        r"^\d{6}\.(?:OF|SZ|SH)$", na=False
    )
    nav_long = normalize_nav_export(nav_raw)
    direct_returns = calculate_forward_quarter_return(nav_long).rename(
        columns={"quarter": "报告期", "next_quarter_return": "direct_return"}
    )
    reconciled = panel.merge(
        direct_returns,
        on=["security_code", "报告期"],
        how="left",
        validate="many_to_one",
    )
    matched = reconciled["return_match_status"].eq("matched")
    return_diff = (
        reconciled.loc[matched, "next_quarter_return"] - reconciled.loc[matched, "direct_return"]
    ).abs()

    company_total = company.loc[company["aggregation_level"].eq("company_total")]
    manual_company = (
        effectiveness.groupby("基金管理人简称")
        .apply(
            lambda group: (group["peer_relative_return"] * group["fund_nav_yuan"]).sum()
            / group["fund_nav_yuan"].sum()
        )
        .rename("manual_weighted_return")
        .reset_index()
    )
    company_check = company_total.merge(manual_company, on="基金管理人简称", how="inner")
    company_diff = (
        company_check["weighted_peer_relative_return"] - company_check["manual_weighted_return"]
    ).abs()

    dynamic = classification["allocation_style_group"].eq("动态股债配置型")
    invalid_dynamic = dynamic & ~(
        classification["valid_quarter_count"].ge(8)
        & classification["stock_weight_range"].ge(0.20)
        & classification["bond_weight_range"].ge(0.15)
        & classification["mean_adjacent_weight_change"].ge(0.05)
    )
    peer_identity = (
        effectiveness["next_quarter_return"]
        - effectiveness["peer_median_return"]
        - effectiveness["peer_relative_return"]
    ).abs()

    rows = [
        _audit_row("原始数据", "配置源文件份额记录", len(base), "信息项", "通过", "Book2.xlsx"),
        _audit_row("原始数据", "唯一基金主代码", base["基金主代码"].nunique(), "信息项", "通过", "全部份额合并后的基金数量"),
        _audit_row("原始数据", "净值源文件有效证券代码", int(valid_nav_code.sum()), "信息项", "通过", "包括.OF、.SZ、.SH"),
        _audit_row("清洗数据", "基金-季度记录数", len(clean), clean["基金主代码"].nunique() * clean["报告期"].nunique(), "通过" if len(clean) == clean["基金主代码"].nunique() * clean["报告期"].nunique() else "失败", "应为基金数×14个季度"),
        _audit_row("清洗数据", "基金-季度重复键", int(clean.duplicated(["基金主代码", "报告期"]).sum()), 0, "通过" if not clean.duplicated(["基金主代码", "报告期"]).any() else "失败", "主代码和季度必须唯一"),
        _audit_row("清洗数据", "可用配置记录", int(clean["异常标记"].eq("可用").sum()), "信息项", "通过", "参与动态配置筛选"),
        _audit_row("清洗数据", "资产配置披露不完整", int(clean["异常标记"].eq("资产配置披露不完整").sum()), "需保留审计、不用于动态筛选", "提示", "不擅自补零"),
        _audit_row("清洗数据", "净资产缺失或非正", int(clean["异常标记"].eq("净资产缺失或非正").sum()), "需保留审计、不用于动态筛选", "提示", "多数为尚未成立或尚无历史披露"),
        _audit_row("清洗数据", "仓位异常记录", int(clean["异常标记"].eq("仓位异常，需复核").sum()), "人工复核", "提示", "不进入动态配置筛选"),
        _audit_row("收益率", "基金-季度收益面板记录", len(panel), panel["基金主代码"].nunique() * panel["报告期"].nunique(), "通过" if len(panel) == panel["基金主代码"].nunique() * panel["报告期"].nunique() else "失败", "最后一个配置季度没有下一季度收益"),
        _audit_row("收益率", "收益面板重复键", int(panel.duplicated(["基金主代码", "报告期"]).sum()), 0, "通过" if not panel.duplicated(["基金主代码", "报告期"]).any() else "失败", "主代码和季度必须唯一"),
        _audit_row("收益率", "已匹配下一季度收益", int(panel["return_match_status"].eq("matched").sum()), "信息项", "通过", "使用代表份额季度末复权单位净值"),
        _audit_row("收益率", "缺失下一季度收益", int(panel["return_match_status"].eq("missing_nav_return").sum()), "需保留审计", "提示", "不参与有效性收益计算"),
        _audit_row("收益率", "收益公式最大绝对误差", float(return_diff.max()), "<=1e-12", "通过" if return_diff.max() <= 1e-12 else "失败", "复权净值(t+1)/复权净值(t)-1"),
        _audit_row("收益率", "单季度收益绝对值超过50%的记录", int((panel["next_quarter_return"].abs() > 0.50).sum()), "人工复核、暂不删除", "提示", "可从原始复权净值逐笔还原"),
        _audit_row("动态筛选", "动态股债配置型基金", int(dynamic.sum()), "信息项", "通过", "使用完整14季度配置历史"),
        _audit_row("动态筛选", "不满足阈值却被选中的基金", int(invalid_dynamic.sum()), 0, "通过" if not invalid_dynamic.any() else "失败", "检查8季、股票20个百分点、债券15个百分点、相邻变化5个百分点"),
        _audit_row("相对收益", "基金-季度有效样本", len(effectiveness), "信息项", "通过", "动态股债型且收益已匹配"),
        _audit_row("相对收益", "相对收益恒等式最大绝对误差", float(peer_identity.max()), "<=1e-12", "通过" if peer_identity.max() <= 1e-12 else "失败", "基金收益-同季同风格中位数"),
        _audit_row("公司汇总", "公司加权收益最大绝对误差", float(company_diff.max()), "<=1e-12", "通过" if company_diff.max() <= 1e-12 else "失败", "按期初整只基金净资产加权"),
        _audit_row("公司汇总", "正式排名公司数", len(ranking), "信息项", "通过", "至少3只基金且20条有效观察"),
        _audit_row("公司汇总", "不符合正式排名门槛的公司", int(((ranking["unique_fund_count"] < 3) | (ranking["matched_observation_count"] < 20)).sum()), 0, "通过" if not ((ranking["unique_fund_count"] < 3) | (ranking["matched_observation_count"] < 20)).any() else "失败", "排名门槛复核"),
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(intermediate / "pipeline_recalculation_audit.csv", index=False, encoding="utf-8-sig")

    nav_wide = nav_long.pivot(index="security_code", columns="quarter", values="adjusted_nav")
    quarter_order = sorted(nav_long["quarter"].unique())
    next_quarter = dict(zip(quarter_order[:-1], quarter_order[1:]))
    extreme = panel.loc[
        panel["return_match_status"].eq("matched") & panel["next_quarter_return"].abs().gt(0.50)
    ].copy()
    extreme["start_adjusted_nav"] = [
        nav_wide.at[code, quarter] for code, quarter in zip(extreme["security_code"], extreme["报告期"])
    ]
    extreme["next_adjusted_nav"] = [
        nav_wide.at[code, next_quarter[quarter]]
        for code, quarter in zip(extreme["security_code"], extreme["报告期"])
    ]
    extreme["复核结论"] = "原始净值可还原；未自动剔除"
    extreme[[
        "报告期", "基金主代码", "security_code", "基金名称", "基金管理人简称", "投资风格",
        "start_adjusted_nav", "next_adjusted_nav", "next_quarter_return", "复核结论",
    ]].sort_values(["报告期", "next_quarter_return"]).to_csv(
        intermediate / "extreme_return_review.csv", index=False, encoding="utf-8-sig"
    )

    nav_columns = [column for column in nav_raw.columns if "复权单位净值" in str(column)]
    full_nav = nav_raw.loc[valid_nav_code, ["证券代码", *nav_columns]].melt(
        id_vars="证券代码", var_name="source_column", value_name="adjusted_nav"
    )
    full_nav["quarter"] = full_nav["source_column"].map(_quarter_from_choice_header)
    full_nav["adjusted_nav"] = pd.to_numeric(full_nav["adjusted_nav"], errors="coerce")
    full_nav_wide = full_nav.pivot(index="证券代码", columns="quarter", values="adjusted_nav")
    available_codes = set(full_nav_wide.index)
    missing = panel.loc[panel["return_match_status"].eq("missing_nav_return")].copy()

    def missing_reason(row):
        code = row["security_code"]
        if code not in available_codes:
            return "代表份额代码不在净值源文件"
        start_nav = full_nav_wide.at[code, row["报告期"]]
        end_nav = full_nav_wide.at[code, next_quarter[row["报告期"]]]
        if pd.isna(start_nav) and pd.isna(end_nav):
            return "期初和期末净值均缺失"
        if pd.isna(start_nav):
            return "期初净值缺失"
        if pd.isna(end_nav):
            return "期末净值缺失"
        return "其他"

    missing["缺失原因"] = missing.apply(missing_reason, axis=1)
    missing[[
        "报告期", "基金主代码", "security_code", "基金名称", "基金管理人简称", "投资风格", "缺失原因",
    ]].sort_values(["报告期", "基金主代码"]).to_csv(
        intermediate / "missing_return_review.csv", index=False, encoding="utf-8-sig"
    )
    return audit


if __name__ == "__main__":
    raise SystemExit("Use the documented pipeline with explicit input paths.")
