from pathlib import Path

import pandas as pd


def test_demo_writes_formal_annualized_result(tmp_path: Path):
    from hybrid_fund_allocation.demo import run_demo

    source = Path("data/sample/fund_quarter_demo.csv")
    output = tmp_path / "demo_result.csv"

    result = run_demo(source, output)

    assert output.exists()
    assert len(result) == 2
    assert result["annualized_evaluation_pool"].eq("正式年化").all()
    assert result["return_quarter_count"].eq(8).all()
    saved = pd.read_csv(output, dtype={"基金主代码": "string"})
    assert saved["基金主代码"].nunique() == 2
