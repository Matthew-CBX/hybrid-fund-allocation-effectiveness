"""Build an auditable allocation-to-forward-return panel for mixed funds."""

import re
from pathlib import Path

import pandas as pd


def _quarter_from_choice_header(header: str) -> str:
    """Return natural reporting quarter from a Choice adjusted-NAV header."""
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", header)
    if not match:
        raise ValueError(f"Cannot find a transaction date in header: {header}")
    year, month, _ = map(int, match.groups())
    return f"{year}Q{((month - 1) // 3) + 1}"


def normalize_nav_export(nav_raw: pd.DataFrame) -> pd.DataFrame:
    """Remove Choice footer rows and reshape quarterly adjusted NAV values."""
    nav_columns = [column for column in nav_raw.columns if "复权单位净值" in str(column)]
    valid_fund_code = nav_raw["证券代码"].astype("string").str.match(
        r"^\d{6}\.(?:OF|SZ|SH)$", na=False
    )
    clean = nav_raw.loc[
        valid_fund_code,
        ["证券代码", *nav_columns],
    ].copy()
    long = clean.melt(
        id_vars="证券代码",
        value_vars=nav_columns,
        var_name="source_column",
        value_name="adjusted_nav",
    )
    long["quarter"] = long["source_column"].map(_quarter_from_choice_header)
    long["adjusted_nav"] = pd.to_numeric(long["adjusted_nav"], errors="coerce")
    long = long.dropna(subset=["adjusted_nav"])
    return (
        long.rename(columns={"证券代码": "security_code"})[
            ["security_code", "quarter", "adjusted_nav"]
        ]
        .sort_values(["security_code", "quarter"])
        .reset_index(drop=True)
    )


def calculate_forward_quarter_return(nav_long: pd.DataFrame) -> pd.DataFrame:
    """Calculate the return from each quarter-end to the following quarter-end."""
    ordered = nav_long.sort_values(["security_code", "quarter"]).copy()
    ordered["next_nav"] = ordered.groupby("security_code")["adjusted_nav"].shift(-1)
    ordered["next_quarter_return"] = ordered["next_nav"] / ordered["adjusted_nav"] - 1
    return (
        ordered.dropna(subset=["next_quarter_return"])[
            ["security_code", "quarter", "next_quarter_return"]
        ]
        .reset_index(drop=True)
    )


def choose_representative_share(base: pd.DataFrame) -> pd.DataFrame:
    """Choose one share class per master fund using the project-wide rule."""
    candidates = base[["证券代码", "基金主代码"]].dropna().copy()
    candidates["证券代码"] = candidates["证券代码"].astype("string").str.strip()
    candidates["基金主代码"] = candidates["基金主代码"].astype("string").str.strip()
    candidates["is_master_share"] = candidates["证券代码"].eq(candidates["基金主代码"])
    candidates = candidates.sort_values(
        ["基金主代码", "is_master_share", "证券代码"],
        ascending=[True, False, True],
    )
    return (
        candidates.drop_duplicates("基金主代码", keep="first")
        .rename(columns={"基金主代码": "fund_main_code", "证券代码": "security_code"})[
            ["fund_main_code", "security_code"]
        ]
        .reset_index(drop=True)
    )


def merge_allocation_with_returns(
    allocation: pd.DataFrame,
    representatives: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Attach representative-share forward returns to each fund-quarter allocation."""
    panel = allocation.merge(
        representatives,
        how="left",
        left_on="基金主代码",
        right_on="fund_main_code",
        validate="many_to_one",
        indicator="representative_match",
    )
    panel = panel.merge(
        forward_returns,
        how="left",
        left_on=["security_code", "报告期"],
        right_on=["security_code", "quarter"],
        validate="many_to_one",
    )
    panel["return_match_status"] = "matched"
    panel.loc[panel["representative_match"].eq("left_only"), "return_match_status"] = (
        "missing_representative_share"
    )
    panel.loc[
        panel["representative_match"].eq("both") & panel["next_quarter_return"].isna(),
        "return_match_status",
    ] = "missing_nav_return"
    return panel.drop(columns=["representative_match", "fund_main_code", "quarter"])


def build_allocation_return_panel(project_dir: Path, nav_export_path: Path) -> pd.DataFrame:
    """Load project sources and return allocation rows that have a following-quarter return."""
    allocation = load_clean_allocation(project_dir)
    base = pd.read_excel(project_dir / "Book2.xlsx", dtype={"证券代码": "string", "基金主代码": "string"})
    nav_raw = pd.read_excel(nav_export_path, dtype={"证券代码": "string"})
    forward_returns = calculate_forward_quarter_return(normalize_nav_export(nav_raw))
    panel = merge_allocation_with_returns(
        allocation,
        choose_representative_share(base),
        forward_returns,
    )
    analysis_quarters = set(forward_returns["quarter"])
    return (
        panel.loc[panel["报告期"].isin(analysis_quarters)]
        .sort_values(["报告期", "基金主代码"])
        .reset_index(drop=True)
    )


def load_clean_allocation(project_dir: Path) -> pd.DataFrame:
    """Load the complete cleaned allocation history used for classification."""
    allocation_path = project_dir / "intermediate" / "clean_fund_quarter.csv"
    if not allocation_path.exists():
        allocation_path = project_dir / "clean_fund_quarter.csv"
    return pd.read_csv(
        allocation_path,
        dtype={"基金主代码": "string", "证券代码": "string"},
    )


def build_merge_log(panel: pd.DataFrame) -> pd.DataFrame:
    """Return concise record-count checks for the merged panel."""
    records = [{"stage": "panel_total", "record_count": len(panel), "note": "全部分析期基金—季度记录"}]
    for status, count in panel["return_match_status"].value_counts(dropna=False).items():
        records.append(
            {
                "stage": str(status),
                "record_count": int(count),
                "note": "按净值收益匹配状态汇总",
            }
        )
    if "报告期" in panel.columns:
        for quarter, count in (
            panel.loc[panel["return_match_status"].eq("matched")]
            .groupby("报告期")
            .size()
            .items()
        ):
            records.append(
                {
                    "stage": f"matched_{quarter}",
                    "record_count": int(count),
                    "note": f"{quarter} 已匹配下一季度收益",
                }
            )
    return pd.DataFrame(records)


def classify_dynamic_allocation(panel: pd.DataFrame) -> pd.DataFrame:
    """Classify funds by whether stock and bond allocations materially change."""
    all_funds = panel[["基金主代码"]].drop_duplicates().copy()
    valid = panel.loc[
        panel["异常标记"].eq("可用"),
        ["基金主代码", "报告期", "stock_weight", "bond_weight"],
    ].copy()
    valid["stock_weight"] = pd.to_numeric(valid["stock_weight"], errors="coerce")
    valid["bond_weight"] = pd.to_numeric(valid["bond_weight"], errors="coerce")
    valid = valid.dropna(subset=["stock_weight", "bond_weight"])

    rows = []
    for fund_main_code, group in valid.groupby("基金主代码", sort=False):
        group = group.sort_values("报告期")
        adjacent_change = group[["stock_weight", "bond_weight"]].diff().abs().max(axis=1).dropna()
        rows.append(
            {
                "基金主代码": fund_main_code,
                "valid_quarter_count": len(group),
                "stock_weight_range": group["stock_weight"].max() - group["stock_weight"].min(),
                "bond_weight_range": group["bond_weight"].max() - group["bond_weight"].min(),
                "mean_adjacent_weight_change": adjacent_change.mean(),
            }
        )
    metrics = pd.DataFrame(rows)
    result = all_funds.merge(metrics, on="基金主代码", how="left")
    result["allocation_style_group"] = "静态配置型"
    dynamic_stock_bond = (
        result["valid_quarter_count"].ge(8)
        & result["stock_weight_range"].ge(0.20)
        & result["bond_weight_range"].ge(0.15)
        & result["mean_adjacent_weight_change"].ge(0.05)
    )
    dynamic_stock_cash = (
        result["valid_quarter_count"].ge(8)
        & result["stock_weight_range"].ge(0.20)
        & ~dynamic_stock_bond
    )
    result.loc[dynamic_stock_cash, "allocation_style_group"] = "动态股现金配置型"
    result.loc[dynamic_stock_bond, "allocation_style_group"] = "动态股债配置型"
    return result.sort_values("基金主代码").reset_index(drop=True)


def build_allocation_style_classification(project_dir: Path) -> pd.DataFrame:
    """Classify funds from every cleaned allocation quarter, including the last one."""
    return classify_dynamic_allocation(load_clean_allocation(project_dir))


def write_project_outputs(project_dir: Path, nav_export_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the panel, classify allocation styles, and save auditable CSV outputs."""
    panel = build_allocation_return_panel(project_dir, nav_export_path)
    classification = build_allocation_style_classification(project_dir)
    panel = panel.merge(classification, on="基金主代码", how="left", validate="many_to_one")
    merge_log = build_merge_log(panel)
    style_log = (
        classification["allocation_style_group"]
        .value_counts()
        .rename_axis("stage")
        .reset_index(name="record_count")
        .assign(
            stage=lambda frame: "allocation_style_" + frame["stage"],
            note="按基金主代码统计的动态配置分类",
        )
    )
    merge_log = pd.concat([merge_log, style_log], ignore_index=True)
    output_dir = project_dir / "intermediate"
    output_dir.mkdir(exist_ok=True)
    panel.to_csv(output_dir / "fund_quarter_allocation_return.csv", index=False, encoding="utf-8-sig")
    classification.to_csv(
        output_dir / "fund_dynamic_allocation_classification.csv", index=False, encoding="utf-8-sig"
    )
    merge_log.to_csv(output_dir / "allocation_return_merge_log.csv", index=False, encoding="utf-8-sig")
    return panel, classification, merge_log


if __name__ == "__main__":
    raise SystemExit("Use the documented pipeline with explicit input paths.")
