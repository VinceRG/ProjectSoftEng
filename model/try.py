# Random Forest Monthly Case Prediction Script
# Fixed version — includes missing 'annual' DataFrame creation and minor cleanup

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# Show versions
print('pandas', pd.__version__)
print('numpy', np.__version__)

# === Load Files ===
DATA_CSV = 'master_dataset_cleaned.csv'
CASE_DICT = 'case_dictionary.json'
AGE_MAP = 'age_group_mapping.txt'

df = pd.read_csv(DATA_CSV, parse_dates=True, low_memory=False)
with open(CASE_DICT, 'r', encoding='utf-8') as f:
    case_dict = json.load(f)
with open(AGE_MAP, 'r', encoding='utf-8') as f:
    age_map_txt = f.read()

print('Data shape:', df.shape)
df.head()

# === Identify Date Column ===
date_col = None
for col in df.columns:
    low = col.lower()
    if 'date' in low or 'admit' in low or 'visit' in low or 'consult' in low:
        date_col = col
        break
date_col = date_col or df.columns[0]
print('Using date column:', date_col)

df['__date'] = pd.to_datetime(df[date_col], errors='coerce')
missing_dates = df['__date'].isna().sum()
print('Missing or unparsable dates:', missing_dates)

df = df.dropna(subset=['__date']).copy()
df['year'] = df['__date'].dt.year
df['month'] = df['__date'].dt.month
df['year_month'] = df['__date'].dt.to_period('M').dt.to_timestamp()

df.columns = [c.strip() for c in df.columns]

print('After processing:', df.shape)
df[['__date', 'year', 'month']].head()

# === Map / Clean Case Column ===
case_col = None
for col in df.columns:
    if 'case' in col.lower() or 'diagnos' in col.lower() or 'chief' in col.lower() or 'top' in col.lower():
        case_col = col
        break
case_col = case_col or 'CASE' if 'CASE' in df.columns else df.columns[0]
print('Using case column:', case_col)

df['case_raw'] = df[case_col].astype(str).str.upper().str.strip()
df['case_raw'] = df['case_raw'].str.replace('\s+', ' ', regex=True)

known_cases = set(case_dict.keys())
df['case_mapped'] = df['case_raw'].where(df['case_raw'].isin(known_cases), other=df['case_raw'])

print('Unique mapped cases sample:', list(df['case_mapped'].unique())[:20])

# === Identify Consultation Type Column ===
consult_col = None
for col in df.columns:
    if 'consult' in col.lower() or 'type' in col.lower() and 'consult' in col.lower():
        consult_col = col
        break

if consult_col is None:
    for c in ['consultation_type', 'consultation', 'type', 'SERVICE_TYPE', 'SERVICE']:
        if c in df.columns:
            consult_col = c
            break

if consult_col is None:
    df['consultation_type'] = 'UNKNOWN'
    consult_col = 'consultation_type'
else:
    df['consultation_type'] = df[consult_col].astype(str).str.upper().str.strip()

print('Using consultation_type column:', consult_col)

# === Group by Year/Month ===
grouped = df.groupby(['year_month', 'consultation_type', 'case_mapped']).size().reset_index(name='count')
print('Grouped shape:', grouped.shape)
grouped.head()

# === FIX: Create annual totals DataFrame ===
annual = df.groupby(['year', 'consultation_type']).size().reset_index(name='count')

# === Create pivot function ===
def make_pivot_for_consult(ct):
    g = grouped[grouped['consultation_type'] == ct].copy()
    if g.empty:
        return None
    pivot = g.pivot_table(index='year_month', columns='case_mapped', values='count', aggfunc='sum').fillna(0).sort_index()
    return pivot

consult_types = grouped['consultation_type'].unique().tolist()[:10]
consult_types[:10]

# === Helper Functions for Modeling ===
from sklearn.preprocessing import StandardScaler

def prepare_features(pivot_df, n_lags=1, top_n=10):
    dfp = pivot_df.copy().asfreq('MS').fillna(0).sort_index()
    df_feat = dfp.copy()
    df_feat['total_admissions'] = dfp.sum(axis=1)
    df_feat['is_christmas_month'] = (df_feat.index.month == 12).astype(int)
    df_feat['is_undas_month'] = (df_feat.index.month == 11).astype(int)
    df_feat['is_newyear_month'] = (df_feat.index.month == 1).astype(int)

    mean_counts = dfp.mean().sort_values(ascending=False)
    top_cases = mean_counts.head(top_n).index.tolist()
    df_cases = dfp[top_cases].copy()

    for lag in range(1, n_lags + 1):
        lagged = df_cases.shift(lag).add_prefix(f'lag{lag}_')
        df_feat = pd.concat([df_feat, lagged], axis=1)

    features = df_feat.drop(columns=top_cases)
    target = df_cases.shift(-1)
    valid_idx = target.dropna().index.intersection(features.dropna().index)
    X = features.loc[valid_idx].copy()
    y = target.loc[valid_idx].copy()
    return X, y, top_cases


def train_and_evaluate(pivot_df, test_size_months=6, n_lags=1, top_n=10, random_state=42):
    X, y, top_cases = prepare_features(pivot_df, n_lags=n_lags, top_n=top_n)
    if X.shape[0] < 12:
        print('Not enough data for reliable training. Got', X.shape[0])
        return None
    train_X = X.iloc[:-test_size_months]
    train_y = y.iloc[:-test_size_months]
    test_X = X.iloc[-test_size_months:]
    test_y = y.iloc[-test_size_months:]
    scaler = StandardScaler()
    num_cols = train_X.select_dtypes(include=[np.number]).columns.tolist()
    train_X[num_cols] = scaler.fit_transform(train_X[num_cols])
    test_X[num_cols] = scaler.transform(test_X[num_cols])
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1))
    model.fit(train_X, train_y)
    preds = pd.DataFrame(model.predict(test_X), index=test_X.index, columns=train_y.columns)
    r2s = {col: r2_score(test_y[col], preds[col]) for col in test_y.columns}
    maes = {col: mean_absolute_error(test_y[col], preds[col]) for col in test_y.columns}
    metrics = {
        'r2_per_case': r2s,
        'mae_per_case': maes,
        'r2_mean': np.mean(list(r2s.values())),
        'mae_mean': np.mean(list(maes.values()))
    }
    return {'model': model, 'scaler': scaler, 'y_test': test_y, 'preds': preds, 'metrics': metrics, 'top_cases': top_cases}

# === Train per consultation type ===
results = {}
for ct in grouped['consultation_type'].unique():
    pivot = make_pivot_for_consult(ct)
    if pivot is None:
        continue
    res = train_and_evaluate(pivot)
    if res:
        results[ct] = res
        print(f"Trained {ct}: mean R²={res['metrics']['r2_mean']:.3f}, MAE={res['metrics']['mae_mean']:.3f}")

# === Annual pivot (now works) ===
annual_pivot = annual.pivot(index='year', columns='consultation_type', values='count').fillna(0).astype(int)
print("\nAnnual totals preview:")
print(annual_pivot.head())

# === Save metrics summary ===
metrics_summary = []
for ct, info in results.items():
    ms = info['metrics']
    metrics_summary.append({
        'consultation_type': ct,
        'r2_mean': ms['r2_mean'],
        'mae_mean': ms['mae_mean'],
        'top_cases': ','.join(info['top_cases'])
    })
metrics_df = pd.DataFrame(metrics_summary).sort_values('r2_mean', ascending=False)
metrics_df.to_csv('/mnt/data/metrics_summary.csv', index=False)
print("\nMetrics summary saved to /mnt/data/metrics_summary.csv")
print(metrics_df.head())
