"""Clean Choice mixed-fund allocation data to one row per master fund and quarter."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
SOURCE = ROOT / "Book2.xlsx"
OUT = ROOT / "intermediate"

PERIODS = [
    ("2023Q1", "2023年一季"), ("2023Q2", "2023年二季/中报"),
    ("2023Q3", "2023年三季"), ("2023Q4", "2023年年报"),
    ("2024Q1", "2024年一季"), ("2024Q2", "2024年二季/中报"),
    ("2024Q3", "2024年三季"), ("2024Q4", "2024年年报"),
    ("2025Q1", "2025年一季"), ("2025Q2", "2025年二季/中报"),
    ("2025Q3", "2025年三季"), ("2025Q4", "2025年年报"),
    ("2026Q1", "2026年一季"), ("2026Q2", "2026年二季/中报"),
]

IDENTITY_COLUMNS = ["证券代码", "证券名称", "基金管理人简称", "基金主代码", "投资风格"]


def _normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.copy().replace({"--": pd.NA, "—": pd.NA, "": pd.NA})
    result["证券代码"] = result["证券代码"].astype("string").str.strip()
    result["基金主代码"] = result["基金主代码"].astype("string").str.strip()
    result["is_master_share"] = result["证券代码"].eq(result["基金主代码"]).astype(int)
    return result.sort_values(
        ["基金主代码", "is_master_share", "证券代码"],
        ascending=[True, False, True],
    )


def build_clean_fund_quarter(
    raw: pd.DataFrame,
    periods: list[tuple[str, str]] = PERIODS,
) -> pd.DataFrame:
    """Return fund-quarter allocations using total NAV across all share classes."""
    normalized = _normalize_raw(raw)
    canonical = normalized.drop_duplicates("基金主代码", keep="first").copy()
    canonical["份额处理"] = canonical["is_master_share"].map(
        {1: "保留主份额作为名称和资产披露代表；净资产汇总全部份额", 0: "未找到主份额，保留最小代码作为代表；净资产汇总全部份额"}
    )

    records = []
    for quarter, marker in periods:
        source_columns = {
            "fund_nav_billion": f"基金资产净值[报告期]{marker}[单位]亿元",
            "stock_value_yuan": f"股票投资市值[报告期]{marker}[单位]元",
            "bond_value_yuan": f"债券投资市值[报告期]{marker}[单位]元",
            "bank_deposit_yuan": f"银行存款[报告期]{marker}[单位]元",
        }
        numeric = normalized[["基金主代码", *source_columns.values()]].copy()
        for source in source_columns.values():
            numeric[source] = pd.to_numeric(numeric[source], errors="coerce")

        frame = canonical[[*IDENTITY_COLUMNS, "份额处理"]].copy()
        frame = frame.rename(columns={"证券名称": "基金名称"})
        frame.insert(0, "报告期", quarter)

        total_nav = (
            numeric.groupby("基金主代码")[source_columns["fund_nav_billion"]]
            .sum(min_count=1)
            .mul(100_000_000)
            .rename("fund_nav_yuan")
        )
        frame = frame.merge(total_nav, on="基金主代码", how="left", validate="one_to_one")

        difference_count = pd.Series(0, index=total_nav.index, dtype="int64")
        for output, source in source_columns.items():
            if output == "fund_nav_billion":
                continue
            # Holdings are fund-level values repeated on each share class. Keep
            # one disclosure, but audit disagreements across share rows.
            disclosed = numeric.groupby("基金主代码")[source].first().rename(output)
            frame = frame.merge(disclosed, on="基金主代码", how="left", validate="one_to_one")
            difference_count = difference_count.add(
                numeric.groupby("基金主代码")[source].nunique(dropna=True).gt(1).astype(int),
                fill_value=0,
            )
        frame = frame.merge(
            difference_count.astype(int).rename("份额间资产披露差异项数"),
            on="基金主代码",
            how="left",
            validate="one_to_one",
        )
        records.append(frame)

    clean = pd.concat(records, ignore_index=True)
    weight_map = {
        "stock_value_yuan": "stock_weight",
        "bond_value_yuan": "bond_weight",
        "bank_deposit_yuan": "bank_deposit_weight",
    }
    for asset, weight in weight_map.items():
        clean[weight] = clean[asset] / clean["fund_nav_yuan"]

    asset_columns = list(weight_map)
    weight_columns = list(weight_map.values())
    clean["资产配置已披露项数"] = clean[asset_columns].notna().sum(axis=1)
    clean["异常标记"] = ""
    invalid_nav = clean["fund_nav_yuan"].isna() | clean["fund_nav_yuan"].le(0)
    incomplete = ~invalid_nav & clean["资产配置已披露项数"].lt(3)
    inconsistent = ~invalid_nav & clean["份额间资产披露差异项数"].gt(0)
    weight_sum = clean[weight_columns].sum(axis=1, min_count=1)
    extreme = ~invalid_nav & (clean[weight_columns].lt(0).any(axis=1) | weight_sum.gt(1.5))

    clean.loc[invalid_nav, "异常标记"] = "净资产缺失或非正"
    clean.loc[incomplete & ~invalid_nav, "异常标记"] = "资产配置披露不完整"
    clean.loc[inconsistent & ~invalid_nav & ~incomplete, "异常标记"] = "份额间资产披露不一致，需复核"
    clean.loc[extreme & ~inconsistent, "异常标记"] = "仓位异常，需复核"
    clean.loc[clean["异常标记"].eq(""), "异常标记"] = "可用"

    ordered = [
        "报告期", "基金主代码", "证券代码", "基金名称", "基金管理人简称", "投资风格", "份额处理",
        "fund_nav_yuan", "stock_value_yuan", "bond_value_yuan", "bank_deposit_yuan",
        "stock_weight", "bond_weight", "bank_deposit_weight", "资产配置已披露项数",
        "份额间资产披露差异项数", "异常标记",
    ]
    return clean[ordered].sort_values(["报告期", "基金主代码"]).reset_index(drop=True)


def build_cleaning_log(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Summarize cleaning checks and the corrected share-class aggregation rule."""
    rows = [
        ["原始份额级记录数", len(raw), "Choice：混合型开放式基金"],
        ["唯一基金主代码数", raw["基金主代码"].nunique(), "以基金主代码去重；净资产汇总全部份额"],
        ["清洗后基金-季度记录数", len(clean), "仅保留2023Q1至2026Q2"],
    ]
    descriptions = {
        "可用": "整只基金净资产为正、三类资产完整、份额间披露一致",
        "资产配置披露不完整": "保留但不参与完整配置比较",
        "净资产缺失或非正": "保留供审计，不参与权重计算",
        "份额间资产披露不一致，需复核": "同一基金不同份额的基金级资产披露值不一致",
        "仓位异常，需复核": "三项权重之和大于150%或有负值",
    }
    for status, description in descriptions.items():
        rows.append([status, int(clean["异常标记"].eq(status).sum()), description])
    return pd.DataFrame(rows, columns=["检查项目", "数量", "口径说明"])


def write_clean_outputs(source: Path = SOURCE, output_dir: Path = OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Choice source, calculate corrected allocations, and save CSV outputs."""
    raw = pd.read_excel(source, dtype={"证券代码": "string", "基金主代码": "string"})
    clean = build_clean_fund_quarter(raw)
    log = build_cleaning_log(raw, clean)
    output_dir.mkdir(exist_ok=True)
    clean.to_csv(output_dir / "clean_fund_quarter.csv", index=False)
    log.to_csv(output_dir / "cleaning_log.csv", index=False)
    return clean, log


if __name__ == "__main__":
    write_clean_outputs()
