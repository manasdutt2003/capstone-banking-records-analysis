# Banking Records Analysis

An end-to-end **Pandas** capstone covering **179 tasks** across data cleaning, filtering,
aggregation, joins, performance optimization, data governance, business analytics and
visualization on banking transaction data.

> Week 2 individual capstone · Perficient Global AI First Academy · Author: **Manas Dutt**

## Scale

| | |
|---|---|
| Tasks answered | **179** (Q1–Q179) |
| Bank records | **13,002 rows × 41 columns** |
| Loan records | 3,000 rows × 12 columns |
| Charts produced | **25** figures and dashboards |
| Verified output | 1,609 lines of captured console results |

## The data problem

This is deliberately messy, real-shaped data:

- **Currency columns** (`account_balance`, `Fixed_Deposite`, `Credit_Amount`) arrive as text like
  `€1,234.5`, `INR 200`, `$10k` — parsed by a `to_num()` helper.
- **Dates** (`txn_date`) arrive in mixed formats (`February4-February-2025`, `28122026`, ISO
  strings) — normalized to `YYYY-MM-DD` by a `to_date()` helper.
- **Categoricals** (`region`, `payment_method`, `payment_status`, `txn_mode`, `txn_type`) are
  stripped, case-standardized and mapped to canonical values (`CC` → `Credit Card`,
  `south east` → `South-East`).
- The long column `customer_lifetime_value_amount (CLV)` is renamed to `clv`.

Raw and aggregated data are kept in separate variables, all mutations happen on copies, and
row counts are validated after every cleaning step.

## Sprints

| Sprint | Tasks | Topics |
|---|---|---|
| 1 | Q1–Q60 | Missing values, duplicates, type conversion, standardization, filtering, group analysis, aggregations |
| 2 | Q61–Q120 | `loc`/`iloc`, pagination, inner/left/right/outer joins, multi-key merges, ranking, cumulative calculations, optimization |
| 3 | Q121–Q179 | Data-quality scoring, audit tables, constraint validation, CLV analysis, 25 matplotlib figures and dashboards |

## Quick start

Requires `pandas`, `numpy`, `matplotlib`, `openpyxl`.

```bash
# Q1–Q154: data cleaning and analysis — prints results for every task
python "Capstone_2/Task_Files/Data_Cleaning_And_Analysis/dbg.py"

# Q155–Q179: visualization — saves all PNGs next to the script
python "Capstone_2/Task_Files/Visualization & Pandas-Specific Tasks/plots.py"
```

Both scripts begin with `os.chdir(os.path.dirname(os.path.abspath(__file__)))`, so they resolve
their input workbooks relative to the script's own folder and run correctly from **any** working
directory — no path edits needed.

## Outputs

- `output_q1_q154.txt` — console results for every task
- `cleaned_bank_records.xlsx` (Q130) and `final_clean.xlsx` (Q132) — cleaned datasets
- **25 figures**: CLV distribution, correlation and clustered heatmaps, scatter with regression,
  monthly trends with rolling average, stacked and segment bars, pairplot, pie/donut, top-10
  customers, box/KDE/CDF/bubble/lag-1/area charts, pivot heatmap, gateway error bars, CLV-tier
  metrics, and a 2×3 investment dashboard

## Layout

```
Capstone_2/
├── Docs/sprint-notes/            sprint-01.md, sprint-02.md, sprint-03.md
└── Task_Files/
    ├── Data_Cleaning_And_Analysis/           dbg.py  (Q1–Q154) + data + outputs
    └── Visualization & Pandas-Specific Tasks/ plots.py (Q155–Q179) + figures
```

> **Note:** the datasets here are course-provided sample data for a fictional banking scenario.
