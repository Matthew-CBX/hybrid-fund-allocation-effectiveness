import pandas as pd
import pytest

from hybrid_fund_allocation.clean_fund_allocation import build_clean_fund_quarter


def test_cleaning_sums_share_class_nav_but_keeps_fund_level_holdings_once():
    raw = pd.DataFrame(
        {
            "证券代码": ["000001.OF", "000002.OF"],
            "证券名称": ["样本基金A", "样本基金C"],
            "基金管理人简称": ["样本公司", "样本公司"],
            "基金主代码": ["000001.OF", "000001.OF"],
            "投资风格": ["平衡型", "平衡型"],
            "基金资产净值[报告期]2024年一季[单位]亿元": [1.0, 2.0],
            "股票投资市值[报告期]2024年一季[单位]元": [150_000_000, 150_000_000],
            "债券投资市值[报告期]2024年一季[单位]元": [60_000_000, 60_000_000],
            "银行存款[报告期]2024年一季[单位]元": [30_000_000, 30_000_000],
        }
    )

    result = build_clean_fund_quarter(raw, periods=[("2024Q1", "2024年一季")])

    assert len(result) == 1
    assert result.loc[0, "证券代码"] == "000001.OF"
    assert result.loc[0, "fund_nav_yuan"] == pytest.approx(300_000_000)
    assert result.loc[0, "stock_value_yuan"] == pytest.approx(150_000_000)
    assert result.loc[0, "stock_weight"] == pytest.approx(0.5)
    assert result.loc[0, "bond_weight"] == pytest.approx(0.2)
    assert result.loc[0, "bank_deposit_weight"] == pytest.approx(0.1)
    assert result.loc[0, "份额间资产披露差异项数"] == 0
    assert result.loc[0, "异常标记"] == "可用"


def test_cleaning_flags_different_fund_level_holdings_across_share_classes():
    raw = pd.DataFrame(
        {
            "证券代码": ["000001.OF", "000002.OF"],
            "证券名称": ["样本基金A", "样本基金C"],
            "基金管理人简称": ["样本公司", "样本公司"],
            "基金主代码": ["000001.OF", "000001.OF"],
            "投资风格": ["平衡型", "平衡型"],
            "基金资产净值[报告期]2024年一季[单位]亿元": [1.0, 2.0],
            "股票投资市值[报告期]2024年一季[单位]元": [150_000_000, 151_000_000],
            "债券投资市值[报告期]2024年一季[单位]元": [60_000_000, 60_000_000],
            "银行存款[报告期]2024年一季[单位]元": [30_000_000, 30_000_000],
        }
    )

    result = build_clean_fund_quarter(raw, periods=[("2024Q1", "2024年一季")])

    assert result.loc[0, "份额间资产披露差异项数"] == 1
    assert result.loc[0, "异常标记"] == "份额间资产披露不一致，需复核"
