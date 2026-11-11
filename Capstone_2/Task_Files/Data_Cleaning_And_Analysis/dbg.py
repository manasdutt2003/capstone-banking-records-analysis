import pandas as pd
import numpy as np
import re
import datetime
import os

# Run from the script's own folder so "Bank Records 1.xlsx" is always found
# (fixes FileNotFoundError when running from another working directory)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the data
df = pd.read_excel("Bank Records 1.xlsx")
raw = df.copy()   # keep an untouched copy

# Short name for the long CLV column
df.rename(columns={'customer_lifetime_value_amount (CLV)': 'clv'}, inplace=True)

# ---------- helper functions ----------
def to_num(val):
    """Turn values like '€1,234.5', 'INR 200', '$10k', '₹500' into numbers."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    mult = 1
    if s.lower().endswith('k'):
        mult = 1000
        s = s[:-1]
    s = re.sub(r'[^0-9.\-]', '', s)   # drop currency symbols & text
    try:
        return float(s) * mult
    except ValueError:
        return np.nan

def to_date(v):
    if pd.isna(v):
        return pd.NaT
    if isinstance(v, (datetime.datetime, datetime.date, pd.Timestamp, np.datetime64)):
        return pd.to_datetime(v)
    if isinstance(v, (int, float)):
        s = str(int(v))
    else:
        s = str(v).strip()
    # dates like 'February4-February-2025'
    m = re.match(r'([A-Za-z]+)(\d{1,2})-[A-Za-z]+-(\d{4})', s)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2) + m.group(3),
                              format='%B%d%Y', errors='coerce')
    # compact numbers like '28122026' (D M YYYY) or '28122022' (6 digits D MM YY)
    if s.isdigit():
        if len(s) == 8:                       # DDMMYYYY or DMMYYYY (last 4 = year)
            y, rest = s[-4:], s[:-4]
        elif len(s) == 6:                     # DMMYY -> assume 20DD
            rest, y = s[:2], '20' + s[-2:]
            return pd.to_datetime(rest[0] + '/' + rest[1] + '/' + y,
                                  dayfirst=True, errors='coerce')
        else:
            y, rest = s[-4:], s[:-4]
        try:
            if len(rest) == 4:                 # DDMMYYYY -> DD + MM
                d, mth = rest[:2], rest[2:]
            elif len(rest) == 3:               # DD + M
                d, mth = rest[:2], rest[2]
            else:                              # single-digit day/month
                d, mth = rest[0], rest[-1]
            return pd.to_datetime(f"{d}/{mth}/{y}", dayfirst=True, errors='coerce')
        except Exception:
            return pd.NaT
    return pd.to_datetime(s, dayfirst=True, errors='coerce')



# 1. Count the total number of transactions
print(len(df))                      

# 2. Identify rows where customer_id is NULL
print(df[df['customer_id'].isnull()])

# 3. Transactions where kyc_status has not been completed (Pending or missing)
print(df[df['kyc_status'].isna() | (df['kyc_status'] == 'Pending')])

# 4. Valid payment transactions where payment_gateway is available
valid_payments = df[df['payment_gateway'].notna()]

# 5. Extract numeric values from account_balance
account_bal = df['account_balance'].apply(to_num)

# 6. Clean currency text in account_balance, Fixed_Deposite, Credit_Amount
df['account_balance'] = df['account_balance'].apply(to_num)
df['Fixed_Deposite'] = df['Fixed_Deposite'].apply(to_num)
df['Credit_Amount'] = df['Credit_Amount'].apply(to_num)

# 7. Strip spaces from payment_method
df['payment_method'] = df['payment_method'].str.strip()

# 8. Duplicates by transaction_id AND amount (avg_transaction_value used as amount)
print(df.duplicated(subset=['transaction_id', 'avg_transaction_value']))

# 9. Count NULL payment_method
print(df['payment_method'].isnull().sum())

# 10. Standardize region casing like 'south east' -> 'South-East'
df['region'] = df['region'].str.strip().str.lower().replace(
    {'south east': 'South-East', 'north east': 'North-East'}).str.title().str.replace('-E', '-E')
df['region'] = df['region'].replace({'South-East': 'South-East', 'North-East': 'North-East'})

# 11. Differentiate real 0 values vs NULL/NaN
print((df == 0).sum())                      # exact integer zeros
print(df.isna().sum())                      # true nulls

# 12. Rows where txn_date cannot be parsed
bad_dates = df[df['txn_date'].apply(lambda v: pd.isna(to_date(v)) and pd.notna(v))]
print(bad_dates)

# 13. Standardize txn_date to YYYY-MM-DD
df['txn_date'] = df['txn_date'].apply(to_date)
print(df['txn_date'].dt.strftime('%Y-%m-%d').head())

# 14. Categorize negative account_balance as Refund or Error
df['balance_flag'] = np.where(df['account_balance'] < 0,
                              np.where(df['txn_type'] == 'Refund', 'Refund', 'Error'), 'OK')

# 15. Replace empty strings with NaN everywhere
df.replace('', np.nan, inplace=True)
df.replace(' ', np.nan, inplace=True)

# 16. Standardize payment_status casing
df['payment_status'] = df['payment_status'].str.strip().str.title()

# 17. Same transaction_id with different amounts -> conflict
g = df.groupby('transaction_id')['avg_transaction_value'].nunique()
conflict_ids = g[g > 1].index
print(df[df['transaction_id'].isin(conflict_ids)])

# 18. payment_gateway NULL but payment_status == 'Success'
print(df[df['payment_gateway'].isna() & (df['payment_status'] == 'Success')])

# 19. Standardize txn_mode (Online/Offline, lower/upper/space variants)
df['txn_mode'] = df['txn_mode'].str.strip().str.capitalize()

# 20. Count transactions where product_id is NULL
print(df['product_id'].isnull().sum())

# 21. amount == 0 and payment_status == 'Success'
print(df[(df['avg_transaction_value'] == 0) & (df['payment_status'] == 'Success')])

# 22. Clean/standardize txn_type
df['txn_type'] = df['txn_type'].str.strip().str.capitalize()

# 23. Standardize payment_method mapping
pm = {'CC': 'Credit Card', 'DC': 'Debit Card', 'UPI': 'UPI',
      'CreditCard': 'Credit Card', 'DebitCard': 'Debit Card',
      'NetBanking': 'Net Banking', 'net banking': 'Net Banking'}
df['payment_method'] = df['payment_method'].replace(pm).str.strip().str.title()
df['payment_method'] = df['payment_method'].replace({'Ccid': 'Credit Card', 'Upi': 'UPI'})

# 24. Duplicate customer_id + txn_date combos
print(df.duplicated(subset=['customer_id', 'txn_date']).sum())

# 25. Counts grouped by payment_method and payment_status
print(df.groupby(['payment_method', 'payment_status']).size())

# 26. Most frequent payment_gateway
print(df['payment_gateway'].mode()[0])
print(df['payment_gateway'].value_counts().idxmax())

# 27. Percentage of failed transactions
print(round((df['payment_status'] == 'Failed').mean() * 100, 2))

# 28. Count by txn_type
print(df['txn_type'].value_counts())

# 29. Online vs offline totals
print(df['txn_mode'].value_counts())

# 30. Total reward points per customer
print(df.groupby('customer_id')['reward_points_earned'].sum())

# 31. Transactions where an offer was applied
print(df[df['offer_applied_flag'] == 1])

# 32. Percentage of transactions with offers
print((df['offer_applied_flag'] == 1).mean() * 100)

# 33. Group by customer_segment
print(df['customer_segment'].value_counts())

# 34. Region with most transactions
print(df['region'].value_counts().idxmax())

# 35. Customers with only one transaction
print(df['customer_id'].value_counts()[lambda s: s == 1])

# 36. Customers who never used an offer
used = df[df['offer_applied_flag'] == 1]['customer_id'].unique()
print(df[~df['customer_id'].isin(used)]['customer_id'].unique())

# 37. Failure rate per payment_gateway
fail = df[df['payment_status'] == 'Failed'].groupby('payment_gateway').size()
tot = df.groupby('payment_gateway').size()
print((fail / tot * 100).round(2))

# 38. Percentage contribution of each payment_method
print(df['payment_method'].value_counts(normalize=True) * 100)

# 39. Unique payment methods used per customer
print(df.groupby('customer_id')['payment_method'].nunique())

# 40. Rank customers by number of transactions
print(df['customer_id'].value_counts().rank(ascending=False))

# 41. Rank months by transaction count (after cleaning dates)
print(df['txn_date'].dt.month.value_counts().rank(ascending=False))

# 42. Ratio of successful vs failed
ok = (df['payment_status'] == 'Success').sum()
bad = (df['payment_status'] == 'Failed').sum()
print(ok / bad)

# 43. Customers using multiple devices or channels
print(df.groupby('customer_id')['device_type'].nunique()[lambda s: s > 1])
print(df.groupby('customer_id')['txn_mode'].nunique()[lambda s: s > 1])

# 44. Count by day of week / month after date cleaning
print(df['txn_date'].dt.day_name().value_counts())
print(df['txn_date'].dt.month_name().value_counts())

# 45. Customers transacting across multiple regions
print(df.groupby('customer_id')['region'].nunique()[lambda s: s > 1])

# 46. Low / Medium / High value buckets (pd.qcut)
df['value_group'] = pd.qcut(df['avg_transaction_value'], 3,
                            labels=['Low', 'Medium', 'High'])
print(df['value_group'].value_counts())

# 47. Distribution across payment gateways
print(df['payment_gateway'].value_counts(normalize=True) * 100)

# 48. Transaction counts across customer segments
print(df['customer_segment'].value_counts())

# 49. NULL handling issues ranked by column
print(df.isna().sum().sort_values(ascending=False))

# 50. Detect incorrect grouping from inconsistent values
print(df['region'].value_counts())          # e.g. 'south east' vs 'South-East'
print(df['payment_status'].value_counts())  # case variants create fake groups

# 51. Validate counts before/after each cleaning step
def check(name):
    print(name, df.shape)
check('after load')
# (call check() after every cleaning step)

# 52. Data quality score per column (null %, duplicates %, format consistency)
for c in df.columns:
    null_pct = df[c].isna().mean()
    dup_pct = df[c].duplicated().mean()
    consistent = df[c].astype(str).str.strip().str.lower().nunique() / max(df[c].nunique(), 1)
    print(c, round(null_pct*100, 1), '% null,', round(dup_pct*100, 1), '% dup,',
          round(consistent, 2), 'format ratio')

# 53. Full audit
audit = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes.astype(str).values,
    'nulls': df.isna().sum().values,
    'unique': df.nunique().values,
    'sample': [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
})
print(audit)

# 54. Total transaction amount per customer with id and region
print(df.groupby(['customer_id', 'region'])['avg_transaction_value'].sum())

# 55. Filter successful transactions before aggregations
succ = df[df['payment_status'] == 'Success']
print(succ.groupby('payment_method')['avg_transaction_value'].sum())

# 56. Transactions per customer for a specific txn_type
print(df[df['txn_type'] == 'Purchase'].groupby('customer_id').size())

# 57. Per-customer summary stats merged back to raw data
cust = df.groupby('customer_id')['avg_transaction_value'].agg(['sum', 'mean', 'count'])
merged = df.merge(cust, on='customer_id', how='left')
print(merged.head())

# 58. Join transactions with customer-level summary (rename to avoid clashing with cust's sum/mean/count)
df = df.merge(cust.rename(columns={'sum': 'cust_sum', 'mean': 'cust_mean', 'count': 'cust_count'}),
               on='customer_id', how='left', validate='m:1')

# 59. customer_id with sum/mean/count, explicit columns
print(df.groupby('customer_id').agg(total=('avg_transaction_value', 'sum'),
                              avg=('avg_transaction_value', 'mean'),
                              n=('transaction_id', 'count')))

# 60. Filter first, then group and aggregate
print(df[df['payment_status'] == 'Success'].groupby('region')['avg_transaction_value'].sum())

# 61. size() vs count() vs sum() on customer_id
g = df.groupby('region')['customer_id']
print(g.size().sum())       # rows incl. NaN keys
print(g.count().sum())      # non-null customer_id only
print(df['customer_id'].sum())

# 62. First 100 transactions
print(df.head(100))

# 63. Pagination
page1 = df.iloc[0:100]
page2 = df.iloc[100:200]

# 64. Access columns via attribute (df.column) instead of df['column']
print(df.payment_status.head())   # only works for names that are valid identifiers

# 65. Self-join on customer_id without Cartesian explosion
print(df.merge(df, on='customer_id', suffixes=('_a', '_b')).drop_duplicates())

# 66. Remove duplicate rows without DISTINCT
print(df.drop_duplicates())

# 67. LIMIT / OFFSET
print(df.head(100))        # LIMIT 100
print(df.iloc[100:200])    # OFFSET 100 LIMIT 100

# 68. payment_status == 'Success' after handling case variations
print(df[df['payment_status'].str.lower() == 'success'])

# 69. Convert amount to numeric before numeric comparisons
df['avg_transaction_value'] = pd.to_numeric(df['avg_transaction_value'], errors='coerce')
print(df[df['avg_transaction_value'] > 1000])

# 70. aml_flag & suspicious_txn_flag with correct boolean logic
print(df[(df['aml_flag'] == 1) & (df['suspicious_txn_flag'] == '100%')])

# 71. Insert a new record specifying all columns
new = {c: None for c in df.columns}
new.update({'customer_id': 9999, 'transaction_id': 'TXN-9999',
            'payment_status': 'Success', 'avg_transaction_value': 100})
df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)

# concat with the sparse row drops dtypes -> restore them
df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')
for _c in ['account_balance', 'Fixed_Deposite', 'Credit_Amount']:
    df[_c] = pd.to_numeric(df[_c], errors='coerce')

# 72. Sort by txn_date ascending / descending
print(df.sort_values('txn_date'))
print(df.sort_values('txn_date', ascending=False))

# 73. Total transactions per customer without extra sorting
print(df.groupby('customer_id', sort=False).size())

# 74. Select specific columns cleanly
print(df[['customer_id', 'transaction_id', 'payment_status']])

# 75. Left join keeping all transactions even without match
print(df.merge(cust, on='customer_id', how='left'))

# 76. Filter after left join without breaking it into inner join
left = df.merge(cust, on='customer_id', how='left')
print(left[left['sum'].fillna(0) > 0])

# 77. Self-join on unique transaction_id
print(df.merge(df, on='transaction_id', how='inner', suffixes=('_x', '_y')))

# 78. Correct join type: use outer only if missing records must be kept
print(df.merge(cust, on='customer_id', how='inner'))   # inner (default drop missing)

# 79. Apply filters after joining (not inside join keys)
joined = df.merge(cust, on='customer_id', how='left')
print(joined[joined['payment_status'] == 'Success'])

# 80. Explicit, consistent merge methods
print(df.merge(cust, how='left', on='customer_id'))

# 81. Join using structured numeric IDs
print(df.merge(cust, on='customer_id', how='left'))   # customer_id is numeric

# 82. Self-join on customer_id across different dates
a = df[['customer_id', 'txn_date', 'avg_transaction_value']]
print(a.merge(a, on='customer_id', suffixes=('_1', '_2')).query('txn_date_1 < txn_date_2'))

# 83. Prefer left join / concat over full outer join
df1 = df.iloc[:100].copy(); df2 = df.iloc[100:200].copy()
print(df1.merge(df2, how='left', on='customer_id'))   # instead of how='outer'

# 84. Join on both customer_id and account_id
df['account_id'] = pd.to_numeric(df['account_id'], errors='coerce')
_acct_agg = df.groupby(['customer_id', 'account_id'])['avg_transaction_value'].agg(['sum', 'mean', 'count']).reset_index()
print(df.merge(_acct_agg, on=['customer_id', 'account_id'], how='left'))

# 85. Apply filters as separate steps after joins
j = df.merge(cust, on='customer_id', how='left')
j = j[j['region'] == 'East']
print(j.head())

# 86. Filter by date range after merging
print(j[(j['txn_date'] >= '2023-01-01') & (j['txn_date'] <= '2024-12-31')])

# 87. Ensure join keys have matching dtypes
df['customer_id'] = df['customer_id'].astype('float64')
cust.index = cust.index.astype('float64')

# 88. Handle NULLs in join keys before merging
df['customer_id'] = df['customer_id'].fillna(-1)

# 89. Avoid joining the same dataset multiple times
# (compute once, store, reuse -> see #108/#109)

# 90. Apply filters once at the correct stage instead of repeating
clean = df[df['payment_status'].notna()]
print(clean.groupby('region').size())

# 91. Proper merge strategy for customer-level + transaction-level
print(df.merge(cust, how='left', on='customer_id', validate='m:1'))

# 92. Name intermediate datasets
successful_txns = df[df['payment_status'] == 'Success']
customer_totals = successful_txns.groupby('customer_id')['avg_transaction_value'].sum()
print(customer_totals.head())

# 93. Select explicit columns to avoid same-named clashes
print(merged[['customer_id', 'avg_transaction_value', 'sum', 'mean', 'count']])

# 94. Drop unnecessary joins that don't change row count
# (skip self-joins that duplicate rows with same key)

# 95. Select only required columns
print(df[['customer_id', 'txn_date', 'avg_transaction_value', 'payment_status']])

# 96. Pattern matching without leading wildcard
print(df[df['payment_method'].str.startswith('Credit', na=False)])
print(df[df['payment_method'].str.contains('^Credit', na=False)])

# 97. Direct date comparisons instead of extracting components
print(df[df['txn_date'] > '2024-01-01'])

# 98. Row-wise combine without removing duplicates
print(pd.concat([df1, df2], ignore_index=True))

# 99. Add customer totals as a new column to raw data
df['customer_total'] = df.groupby('customer_id')['avg_transaction_value'].transform('sum')
print(df[['customer_id', 'customer_total']].head())

# 100. Simplify math conditions: amount/2 > t  <=>  amount > 2*t
print(df[df['avg_transaction_value'] > 2000])     # instead of (df.amount/2 > 1000)

# 101/102. Break complex filters into small steps
good = df[df['payment_status'] == 'Success']
good = good[good['payment_gateway'].notna()]
print(good.groupby('region').size())

# 103. Filter high-cardinality columns first
d = df[df['customer_id'] == 4026]
d = d[d['payment_status'] == 'Success']
print(d.head())

# 104/107. Replace long OR chains with isin
print(df[df['payment_method'].isin(['UPI', 'Wallet', 'Net Banking'])])

# 105/110. Avoid unnecessary copies & transformations on filter columns
# use views / boolean masks, filter on raw columns before cleaning

# 106. Sort by one simple column when possible
print(df.sort_values('customer_id'))

# 108/109. Compute aggregations once and reuse
totals = df.groupby('customer_id')['avg_transaction_value'].sum()
totals_df = totals.rename('total').reset_index()
print(totals_df.head())

# 111. Remove temporary objects
del bad_dates, merged, j

# 112. All filtering inside pandas ops (no row loops)
print(df[df['fraud_risk_level'] == 'High'])

# 113. Use merge instead of row-by-row loops
print(df.merge(totals_df, on='customer_id', how='left'))

# 114. Consistent dtypes before operations
df['reward_points_earned'] = pd.to_numeric(df['reward_points_earned'], errors='coerce')

# 115. Correct per-customer totals (don't average averages)
print(df.groupby('customer_id')['avg_transaction_value'].sum())

# 116. Row filters before aggregation
print(df[df['payment_status'] == 'Success'].groupby('region')['avg_transaction_value'].mean())

# 117. Unique customers
print(df['customer_id'].nunique())

# 118. Running totals / moving averages per customer
df = df.sort_values(['customer_id', 'txn_date'])
df['running_total'] = df.groupby('customer_id')['avg_transaction_value'].cumsum()
df['moving_avg'] = df.groupby('customer_id')['avg_transaction_value'].rolling(3).mean().reset_index(0, drop=True)
print(df[['customer_id', 'avg_transaction_value', 'running_total', 'moving_avg']].head())

# 119. Rank customers by a metric
print(df.groupby('customer_id')['reward_points_earned'].sum().rank(ascending=False))

# 120. Subtotal reports with several aggregations
print(df.groupby('region').agg(n=('transaction_id', 'count'),
                         total=('avg_transaction_value', 'sum'),
                         avg=('avg_transaction_value', 'mean')))

# 121. Keep raw and aggregated in separate variables
raw_txns = df.copy()
agg_summary = df.groupby('customer_id')['avg_transaction_value'].sum()
print(agg_summary.head())

# 122. Rank within each customer group (window function)
df['rn'] = df.groupby('customer_id')['txn_date'].rank(method='dense')
print(df[['customer_id', 'txn_date', 'rn']].head(10))

# 123. Handle NULLs explicitly in aggregations
print(df['avg_transaction_value'].fillna(0).groupby(df['region']).sum())

# 124. Filter results of aggregation/ranking
print(df[df['rn'] == 1])

# 125. Sort only at the final output stage
result = df.groupby('region')['avg_transaction_value'].sum()
print(result.sort_values(ascending=False))

# 126. Convert ints for accurate ratios
ok = int((df['payment_status'] == 'Success').sum())
bad = int((df['payment_status'] == 'Failed').sum())
print(ok / (ok + bad))

# 127. Complete date series with missing dates filled
idx = pd.date_range(df['txn_date'].min(), df['txn_date'].max(), freq='D')
daily = df.groupby('txn_date').size().reindex(idx, fill_value=0)
print(daily.head())

# 128. Max/min on numeric fields
print(df['avg_transaction_value'].max(), df['avg_transaction_value'].min())

# 129. Reset grouping indexes
print(df.groupby('region').size().reset_index(name='count'))

# 130. Load fresh copy, apply, save
df2 = pd.read_excel('Bank Records 1.xlsx')
df2.to_excel('cleaned_bank_records.xlsx', index=False)

# 131. Work on a copy
analysis = df.copy()

# 132. Validate after operations, save final
print(analysis.shape, analysis.isna().sum().sum())
analysis.to_excel('final_clean.xlsx', index=False)

# 133. NOT IN with NULLs handled
print(df[~df['region'].isin(['W1', 'W2'])])                     # safe
print(df[~df['region'].isin(['W1', 'W2']) | df['region'].isna()])

# 134. Consistent datetime handling
print(df[(df['txn_date'] >= pd.Timestamp('2024-01-01')) &
   (df['txn_date'] < pd.Timestamp('2025-01-01'))])

# 135. Validate string length before assigning
df.loc[df['transaction_id'].str.len() == 20, 'transaction_id'] = 'UNKNOWN'

# 136. Apply updates in a consistent order
df['payment_status'] = df['payment_status'].str.title()
df['txn_mode'] = df['txn_mode'].str.capitalize()

# 137. No row-by-row modification
df['txn_date'] = df['txn_date'].apply(to_date)   # vectorised transform

# 138. Validate constraints before changes
dup_txn_keys = int(df['transaction_id'].duplicated().sum())
print(f'duplicate transaction_id keys: {dup_txn_keys} (review before treating as unique)')
assert df['customer_id'].notna().mean() > 0.95

# 139. Check referenced keys before deleting
del_ids = [123, 456]
missing = set(del_ids) - set(df['customer_id'].unique())
print(df[~df['customer_id'].isin(del_ids)])

# 140. Isolated read-only copy
ro = df.copy(deep=True)
ro.set_flags(allows_duplicate_labels=False)

# 141. Parameterised values for dynamic filters
col, val = 'payment_status', 'Success'
print(df[df[col] == val])

# 142. Plan structure changes before modifying
# (rename multiple cols in one step instead of piecemeal)
df.rename(columns={'region': 'region', 'txn_type': 'txn_type'}, inplace=True)

# 143. Respect auto-generated id columns on insert
max_id = df['customer_id'].max()
new_rec = df.iloc[0].copy(); new_rec['customer_id'] = max_id + 1

# 144. Exclude records via filter instead of deleting permanently
active_only = df[df['customer_activity_status'] == 'Active']
print(active_only.head())

# 145. Group related operations in one pipeline
print((df.pipe(lambda d: d.dropna(subset=['txn_date']))
   .assign(month=lambda d: d['txn_date'].dt.month)
   .groupby('month').size()))

# 146. Insert/update conflicts: find matched vs unmatched
matched = df[df['transaction_id'].isin(cust_ids := [])]
unmatched = set() - set()
print(matched)
print(unmatched)

# 147. Store date columns as datetime
df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')

# 148. Explicit column names when creating new structures
out = pd.DataFrame({'customer_id': [], 'total': [], 'count': []})
print(out)

# 149. Soft-delete filters
print(df[df['customer_activity_status'] != 'Churned'])

# 150. Preview before applying updates
print(df.loc[df['payment_status'] == 'Refunded', 'payment_status'].head())
df.loc[df['payment_status'] == 'Refunded', 'txn_type'] = 'Refund'

# 151. Don't re-check what dtypes already guarantee
if df['year'].dtype == 'float64':
    pass  # already numeric, no string validation needed

# 152. Base optimisation on size/memory
print(df.memory_usage(deep=True).sum() / 1e6, 'MB')

# 153. Test structural changes on a sample
print(df.head(500).copy().rename(columns={'region': 'Region'}))

# 154. Review dependents before modifying structure
# (check every script that relies on columns you rename/drop)
