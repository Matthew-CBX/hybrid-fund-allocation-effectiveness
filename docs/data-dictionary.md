# Data dictionary

## Input boundary

The full production pipeline expects authorized quarterly mixed-fund allocation data, adjusted fund NAV observations, CSI 300 index levels, and ChinaBond Aggregate Wealth index levels. Raw inputs are not included in this public repository.

The essential raw fields are fund/share code, master fund code, fund name, manager, investment style, report quarter, fund NAV, stock market value, bond market value, bank-deposit market value, and adjusted unit NAV. Share-class NAV is aggregated to the master-fund level before allocation ratios are calculated.

Percentages and returns in CSV files use decimals: `0.05` means 5%. Monetary NAV fields use yuan. Missing values mean the observation was unavailable or did not meet the comparability gate; they must not be silently converted to zero.

## Sample input

### `data/sample/fund_quarter_demo.csv`

- Grain: one fictional fund × quarter.
- Primary key: `基金主代码` + `报告期`.
- Purpose: exercise fund-level annualization with eight comparable quarters.
- Status: entirely synthetic; no row represents a real product.

## Published result tables

### `enhanced_company_evaluation.csv`

- Grain: one fund management company.
- Primary key: `基金管理人简称`.
- Rows: 101; columns: 30.
- Main measures: switching observations, unique funds, hit ratio, Wilson 95% interval, mean/median contribution, winsorized means, two-way clustered bootstrap interval, excess-return summaries, concentration measures, credibility grade, and formal rank.
- Formal gate: at least 30 switching observations, at least 5 unique funds, at least 30 return observations, and maximum single-fund observation share no greater than 50%.
- `credibility_grade`: `A`, `B`, `C`, or `观察池`; the grade is an evidence-strength assessment, not a recommendation.

### `enhanced_company_annualized_return.csv`

- Grain: one fund management company.
- Primary key: `基金管理人简称`.
- Rows: 101; columns: 15.
- Main measures: valid fund counts, median/equal-weight/NAV-weight annualized fund return, annualized three-asset benchmark, annualized excess versus three-asset and leave-one-out peer benchmarks, annualized active allocation contribution, and positive-fund ratios.
- Annualized company values aggregate fund-level annualized observations; they do not annualize a company-level quarterly average.

### `enhanced_fund_annualized_return.csv`

- Grain: one master fund.
- Primary key: `基金主代码`.
- Rows: 376; columns: 24.
- Formal annualization: at least 8 quarters for which fund, CSI 300, bond, three-asset, and leave-one-out peer returns are all simultaneously available and greater than -100%.
- Formal contribution: at least 8 drift-comparable quarters with valid contribution input greater than -100%.
- Return fields use geometric annualization. `annualized_excess_vs_three_asset` is the difference between two annualized return series calculated on the same quarter set.

### `enhanced_threshold_robustness.csv`

- Grain: switching threshold × fund management company.
- Primary key: `threshold` + `基金管理人简称`.
- Rows: 303; columns: 11.
- Threshold values are 3%, 5%, and 10% absolute drift-adjusted stock rebalancing. The table tests whether company conclusions remain directionally stable as the event threshold changes.

### `enhanced_extreme_sensitivity.csv`

- Grain: one company with usable contribution observations.
- Primary key: `基金管理人简称`.
- Rows: 100; columns: 7.
- Compares raw mean, 1% winsorized mean, 5% winsorized mean, and median contribution. `all_four_same_sign` identifies directional stability, not statistical significance.

### `enhanced_direction_environment.csv`

- Grain: company × switching direction × relative stock/bond market environment.
- Primary key: `基金管理人简称` + `switch_direction` + `relative_market_environment`.
- Rows: 425; columns: 8.
- Measures: observations, unique funds, hit ratio, mean contribution, and median contribution.

### `enhanced_product_concentration.csv`

- Grain: one fund management company.
- Primary key: `基金管理人简称`.
- Rows: 101; columns: 9.
- Measures whether a company conclusion is dominated by one or two products, using observation shares and absolute-contribution shares.

### `enhanced_audit.csv`

- Grain: one audit rule.
- Primary key: `检查项目`.
- Rows: 17; columns: 5.
- `检查结论` must equal `通过` for every published run.
- Audits cover unique keys, passive-drift and contribution formulas, nested event thresholds, formal gates, grade rules, benchmark coverage, Wilson bounds, bootstrap reproducibility, annualization sample sets, total-loss exclusions, 5% event treatment, and company/fund count reconciliation.

## Interpretation rules

- `evaluation_pool = 正式评价` means the company met the predefined evidence-volume and concentration gates.
- `annualized_evaluation_pool = 正式年化` means the fund met the common-quarter and geometric-return gates.
- `contribution_evaluation_pool = 正式贡献` means the fund met the drift-comparable contribution gate.
- A positive metric is descriptive evidence for the observation period. It is not proof of persistent manager skill.
- Company tables should be read together with sample size, interval estimates, concentration, and robustness outputs.
