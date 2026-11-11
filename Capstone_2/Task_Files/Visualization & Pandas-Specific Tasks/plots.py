import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless: save figures without opening windows
import matplotlib.pyplot as plt
import re, datetime, os

# Run from the script's own folder so the Excel file is always found
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_excel("Bank Records 1.xlsx")
df.rename(columns={'customer_lifetime_value_amount (CLV)': 'clv'}, inplace=True)
pd.set_option('display.width', 200)

# date cleaning (same logic as solution.py)
def to_date(v):
    if pd.isna(v): return pd.NaT
    if isinstance(v, (datetime.datetime, datetime.date)): return pd.to_datetime(v)
    s = str(int(v)) if isinstance(v, (int, float)) else str(v).strip()
    m = re.match(r'([A-Za-z]+)(\d{1,2})-[A-Za-z]+-(\d{4})', s)
    if m: return pd.to_datetime(m.group(1)+m.group(2)+m.group(3), format='%B%d%Y', errors='coerce')
    if s.isdigit():
        y, rest = s[-4:], s[:-4]
        try:
            d, mth = (rest[0], rest[1]) if len(rest) == 2 else (rest[:-1], rest[-1])
            return pd.to_datetime(f"{d}/{mth}/{y}", dayfirst=True, errors='coerce')
        except Exception: return pd.NaT
    return pd.to_datetime(s, dayfirst=True, errors='coerce')

df['txn_date'] = df['txn_date'].apply(to_date)
df['account_balance'] = pd.to_numeric(
    df['account_balance'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

NUM = ['clv', 'avg_transaction_value', 'txn_frequency_30d', 'txn_velocity_1hr',
       'failed_txn_count_24hr', 'reward_points_earned']

# ---------- SECTION B ----------

# 155. Histogram of CLV with 30 bins, mean line
plt.figure(figsize=(9, 5))
plt.hist(df['clv'].dropna(), bins=30, color='steelblue', edgecolor='black')
plt.axvline(df['clv'].mean(), color='red', linestyle='--', label='Mean CLV')
plt.title('Customer Lifetime Value Distribution')
plt.xlabel('CLV'); plt.ylabel('Frequency'); plt.legend()
plt.savefig('clv_distribution.png', dpi=150, bbox_inches='tight'); plt.close()

# 156. 2x2 subplot grid
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
df.boxplot(column='clv', by='region', ax=ax[0, 0]); ax[0, 0].set_title('CLV by Region')
for s, sub in df.groupby('customer_activity_status'):
    ax[0, 1].hist(sub['txn_frequency_30d'].dropna(), bins=30, alpha=0.4, label=s, density=True)
ax[0, 1].set_title('Txn Frequency by Activity Status'); ax[0, 1].legend()
df.groupby('customer_segment')['clv'].mean().plot.bar(ax=ax[1, 0], color='orange')
ax[1, 0].set_title('Avg CLV by Segment')
df['most_common_txn_time_of_day'].value_counts().plot.bar(ax=ax[1, 1], color='green')
ax[1, 1].set_title('Txn Time of Day')
plt.suptitle(''); plt.tight_layout(); plt.savefig('grid_156.png', dpi=150); plt.close()

# 157. Correlation heatmap with annotations
corr = df[NUM].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(NUM))); ax.set_yticks(range(len(NUM)))
ax.set_xticklabels(NUM, rotation=45, ha='right'); ax.set_yticklabels(NUM)
for i in range(len(NUM)):
    for j in range(len(NUM)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)
plt.colorbar(im); plt.title('Correlation Heatmap'); plt.tight_layout()
plt.savefig('corr_heatmap.png', dpi=150); plt.close()

# 158. Scatter CLV vs avg_transaction_value colored by status, regression per group
plt.figure(figsize=(9, 6))
for s, sub in df.groupby('customer_activity_status'):
    sub2 = sub.dropna(subset=['clv', 'avg_transaction_value'])
    x, y = sub2['clv'], sub2['avg_transaction_value']
    plt.scatter(x, y, s=8, alpha=0.4, label=s)
    b, a = np.polyfit(x, y, 1)
    xs = np.sort(x)
    plt.plot(xs, b*xs + a, linewidth=2)
plt.title('CLV vs Avg Transaction Value'); plt.xlabel('CLV'); plt.ylabel('Avg Txn Value'); plt.legend()
plt.savefig('scatter_158.png', dpi=150); plt.close()

# 159. Monthly txn counts with 3-month rolling average
monthly = df.set_index('txn_date').resample('ME').size()
plt.figure(figsize=(11, 5))
plt.plot(monthly.index, monthly.values, marker='o', label='Monthly count')
plt.plot(monthly.index, monthly.rolling(3).mean(), '--', label='3-month rolling avg')
plt.legend(); plt.title('Monthly Transactions'); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig('monthly_159.png', dpi=150); plt.close()

# 160. Stacked bar: payment methods by segment with % annotations
pt = pd.crosstab(df['customer_segment'], df['payment_method'], normalize='index') * 100
ax = pt.plot.bar(stacked=True, figsize=(10, 6), colormap='tab20')
for c in ax.containers:
    ax.bar_label(c, fmt='%.0f%%', fontsize=7)
plt.title('Payment Method by Segment'); plt.ylabel('Percent'); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig('stacked_160.png', dpi=150); plt.close()

# 161. Pairplot (no seaborn -> manual scatter grid)
cols = NUM
fig, axes = plt.subplots(len(cols), len(cols), figsize=(14, 14))
for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        ax = axes[i, j]
        for r, sub in df.groupby('region'):
            ax.scatter(sub[b], sub[a], s=3, alpha=0.2, label=r)
        if j == 0: ax.set_ylabel(a)
        if i == len(cols)-1: ax.set_xlabel(b)
fig.suptitle('Pairwise Scatter by Region'); plt.tight_layout()
plt.savefig('pairplot_161.png', dpi=150); plt.close()

# 162. Pie chart of activity status with 'Active' exploded
counts = df['customer_activity_status'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%',
        explode=[0.1 if i == 'Active' else 0 for i in counts.index])
plt.title('Customer Activity Status'); plt.savefig('pie_162.png', dpi=150); plt.close()

# 163. Top 10 customers by CLV (horizontal bar + labels)
top = df.groupby('customer_id')['clv'].max().nlargest(10).sort_values()
plt.figure(figsize=(10, 6))
bars = plt.barh(top.index.astype(str), top.values, color='teal')
plt.bar_label(bars, fmt='%.0f')
plt.title('Top 10 Customers by CLV'); plt.xlabel('CLV')
plt.tight_layout(); plt.savefig('top10_clv_customers.png', dpi=150); plt.close()

# 164. Box plot txn_velocity vs fraud_risk_level + strip points
groups = [g['txn_velocity_1hr'].dropna() for _, g in df.groupby('fraud_risk_level')]
labels = [k for k, _ in df.groupby('fraud_risk_level')]
plt.figure(figsize=(8, 6))
plt.boxplot(groups, tick_labels=labels, showfliers=False)
for i, g in enumerate(groups, start=1):
    plt.scatter(np.random.normal(i, 0.06, len(g)), g, alpha=0.15, s=6)
plt.title('Txn Velocity by Fraud Risk'); plt.ylabel('Velocity'); plt.savefig('box_164.png', dpi=150); plt.close()

# 165. Faceted bar charts of monthly counts per region
regions = df['region'].dropna().unique()[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for ax, r in zip(axes.ravel(), regions):
    df[df['region'] == r].set_index('txn_date').resample('ME').size().plot.bar(ax=ax)
    ax.set_title(r); ax.tick_params(axis='x', rotation=45)
for ax in axes.ravel()[len(regions):]: ax.set_visible(False)
plt.suptitle('Monthly Transactions by Region'); plt.tight_layout()
plt.savefig('facet_165.png', dpi=150); plt.close()

# 166. KDE CLV: offer applied vs not (filled + medians)
used = df[df['offer_applied_flag'] == 1]['clv'].dropna()
notused = df[df['offer_applied_flag'] == 0]['clv'].dropna()
for d, label, color in [(used, 'Offer applied', 'blue'), (notused, 'No offer', 'orange')]:
    d.plot.kde(alpha=0.3, label=label, color=color)
    plt.axvline(d.median(), color=color, linestyle='--')
plt.title('CLV Distribution by Offer Usage'); plt.legend(); plt.savefig('kde_166.png', dpi=150); plt.close()

# 167. Bubble chart
plt.figure(figsize=(10, 7))
for r, sub in df.groupby('region'):
    plt.scatter(sub['clv'], sub['avg_transaction_value'],
                s=sub['txn_frequency_30d']/2, alpha=0.4, label=r)
plt.xlabel('CLV'); plt.ylabel('Avg Txn Value'); plt.title('Bubble Chart of Transactions')
plt.legend(fontsize=7); plt.savefig('bubble_167.png', dpi=150); plt.close()

# 168. Pivot heatmap: region x payment_method counts
pv = df.pivot_table(index='region', columns='payment_method',
                    values='transaction_id', aggfunc='count', fill_value=0)
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(pv, cmap='YlGnBu', aspect='auto')
ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=45, ha='right')
ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        ax.text(j, i, pv.iloc[i, j], ha='center', va='center', fontsize=8)
plt.colorbar(im); plt.title('Region vs Payment Method'); plt.tight_layout()
plt.savefig('pivot_heatmap_168.png', dpi=150); plt.close()

# 169. CDF of CLV with reference lines
s = df['clv'].dropna().sort_values()
y = np.arange(1, len(s)+1) / len(s)
plt.figure(figsize=(9, 5))
plt.plot(s, y)
for p in [0.25, 0.5, 0.75, 0.9]:
    plt.axhline(p, color='gray', linestyle='--', alpha=0.6)
    plt.text(s.min(), p, f' {int(p*100)}%')
plt.title('CDF of CLV'); plt.xlabel('CLV'); plt.ylabel('Cumulative %')
plt.savefig('cdf_169.png', dpi=150); plt.close()

# 170. Monthly CLV trends for top 4 regions with markers + legend
top4 = df.groupby('region')['clv'].sum().nlargest(4).index
monthly_clv = df[df['region'].isin(top4)].groupby([df['txn_date'].dt.to_period('M'), 'region'])['clv'].sum().unstack()
plt.figure(figsize=(11, 6))
for r in top4:
    plt.plot(monthly_clv.index.astype(str), monthly_clv[r], marker='o', label=r)
plt.xticks(rotation=45); plt.grid(True); plt.legend(); plt.title('Monthly CLV by Region')
plt.tight_layout(); plt.savefig('clv_trend_170.png', dpi=150); plt.close()

# 171. Donut of weekend vs weekday high-value txns
wknd = df[df['averag_high_txn_weekend_flag'] == 1]
counts = wknd['averag_high_txn_day_of_week'].isin(['Sat', 'Sun']).value_counts()
plt.figure(figsize=(7, 7))
plt.pie(counts, labels=['Weekend', 'Weekday'], autopct='%1.1f%%', pctdistance=0.75)
plt.gca().add_artist(plt.Circle((0, 0), 0.5, fc='white'))
plt.text(0, 0, f"{counts.iloc[0]/counts.sum()*100:.1f}% weekend", ha='center')
plt.title('Weekend vs Weekday High-Value Txns'); plt.savefig('donut_171.png', dpi=150); plt.close()

# 172. 1x3: balance histogram / box by segment / KDE by activity
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].hist(df['account_balance'].dropna(), bins=30, color='purple', edgecolor='black')
ax[0].set_title('Account Balance')
df.boxplot(column='account_balance', by='customer_segment', ax=ax[1]); ax[1].set_title('Balance by Segment')
for s, sub in df.groupby('customer_activity_status'):
    if s in ('Active', 'Churned'):
        sub['account_balance'].dropna().plot.kde(ax=ax[2], label=s)
ax[2].legend(); ax[2].set_title('Balance KDE')
plt.suptitle(''); plt.tight_layout(); plt.savefig('balance_172.png', dpi=150); plt.close()

# 173. Lag-1 scatter plot
plt.figure(figsize=(7, 7))
for cust, sub in df.sort_values('txn_date').groupby('customer_id'):
    prev = sub['avg_transaction_value'].iloc[:-1].values
    nxt = sub['avg_transaction_value'].iloc[1:].values
    plt.scatter(prev, nxt, alpha=0.05, s=4, color='navy')
mx = df['avg_transaction_value'].max()
plt.plot([0, mx], [0, mx], 'r--')
plt.xlabel('Current amount'); plt.ylabel('Next amount'); plt.title('Lag-1 Scatter')
plt.savefig('lag1_173.png', dpi=150); plt.close()

# 174. Area chart of cumulative CLV with 80% reference line
s = df['clv'].dropna().sort_values(ascending=False)
cum = s.cumsum() / s.sum()
plt.figure(figsize=(10, 5))
plt.fill_between(range(len(cum)), cum, alpha=0.5)
plt.axhline(0.8, color='red', linestyle='--', label='80%')
plt.legend(); plt.title('Cumulative CLV Contribution'); plt.xlabel('Customers (ranked)')
plt.savefig('cumclv_174.png', dpi=150); plt.close()

# 175. Clustered heatmap (clustermap) without seaborn: hierarchical clustering via corr sort
corr = df[NUM].corr()
# simple ordering: sort rows/cols by their correlation with CLV
order = corr['clv'].sort_values().index
corr2 = corr.loc[order, order]
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr2, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr2))); ax.set_xticklabels(corr2.columns, rotation=90)
ax.set_yticks(range(len(corr2))); ax.set_yticklabels(corr2.index)
plt.colorbar(im); plt.title('Clustered Correlation Heatmap'); plt.tight_layout()
plt.savefig('clustermap_175.png', dpi=150); plt.close()

# 176. Count plot of time-of-day split by activity status (grouped bars + %)
ct = pd.crosstab(df['most_common_txn_time_of_day'], df['customer_activity_status'], normalize='columns') * 100
ax = ct.plot.bar(figsize=(10, 6))
for c in ax.containers: ax.bar_label(c, fmt='%.0f%%', fontsize=7)
plt.title('Txn Time by Activity Status'); plt.ylabel('Percent'); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig('countplot_176.png', dpi=150); plt.close()

# 177. Gateways by avg amount with std error bars + color gradient
g = df.groupby('payment_gateway')['avg_transaction_value'].agg(['mean', 'std']).sort_values('mean')
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(g)))
plt.figure(figsize=(9, 6))
plt.bar(g.index, g['mean'], yerr=g['std'], color=colors, capsize=5)
plt.title('Avg Transaction Amount by Gateway'); plt.ylabel('Avg Amount')
plt.tight_layout(); plt.savefig('gateway_177.png', dpi=150); plt.close()

# 178. 2x3 dashboard
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
ax[0, 0].hist(df['clv'].dropna(), bins=30, color='steelblue'); ax[0, 0].set_title('CLV Histogram')
df['payment_method'].value_counts().plot.bar(ax=ax[0, 1], color='darkorange'); ax[0, 1].set_title('Payment Methods')
df['region'].value_counts().head(8).plot.bar(ax=ax[0, 2], color='forestgreen'); ax[0, 2].set_title('Top Regions')
df['customer_activity_status'].value_counts().plot.pie(ax=ax[1, 0], autopct='%1.1f%%'); ax[1, 0].set_ylabel(''); ax[1, 0].set_title('Activity Status')
monthly.plot(ax=ax[1, 1]); ax[1, 1].set_title('Monthly Txns')
ax[1, 2].imshow(df[NUM].corr(), cmap='coolwarm', vmin=-1, vmax=1); ax[1, 2].set_title('Correlation')
fig.suptitle('Investment Dashboard', fontsize=16)
plt.tight_layout(); plt.savefig('investment_dashboard.png', dpi=150); plt.close()

# 179. 4 CLV tiers: grouped bars of avg freq, avg value, failed count
df['clv_tier'] = pd.qcut(df['clv'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
g = df.groupby('clv_tier', observed=True).agg(
    avg_freq=('txn_frequency_30d', 'mean'),
    avg_value=('avg_transaction_value', 'mean'),
    avg_failed=('failed_txn_count_24hr', 'mean'))
g.plot.bar(figsize=(10, 6))
plt.title('Metrics by CLV Tier'); plt.ylabel('Average'); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig('clv_tier_179.png', dpi=150); plt.close()

print('Done - all figures generated.')
