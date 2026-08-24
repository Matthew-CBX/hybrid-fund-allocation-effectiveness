"""Run a small annualized-return demonstration on synthetic fund data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analyze_enhanced_allocation_effectiveness import build_fund_annualized_metrics


def run_demo(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Calculate and save fund-level annualized metrics for the demo panel."""
    detail = pd.read_csv(input_path, dtype={"基金主代码": "string"})
    result = build_fund_annualized_metrics(detail)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sample/fund_quarter_demo.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample/demo_fund_annualized_result.csv"),
    )
    args = parser.parse_args()
    result = run_demo(args.input, args.output)
    print(f"Synthetic demo complete: {len(result)} fund rows written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
