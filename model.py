import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

print('All libraries imported successfully!')

# ── Load data ──────────────────────────────────────────────────────────────────
ds1 = pd.read_csv('DS1_material_properties_5500.csv')
ds4 = pd.read_csv('DS4_ mqi_weights.csv')

print('DS1 shape:', ds1.shape)
print('\nMQI Weights (DS4):')
print(ds4)

df = ds1.copy()

# ── Categorical encoding ───────────────────────────────────────────────────────
le_crystal  = LabelEncoder()
le_category = LabelEncoder()

df['crystal_system_enc'] = le_crystal.fit_transform(df['crystal_system'])
df['category_enc']       = le_category.fit_transform(df['category'])

print('Encoding done!')

# ── D4: Compute MQI weights from DS4 ──────────────────────────────────────────
# Map DS4 property names to DS1 column names
_property_map = {
    'Bulk Modulus (K)':  'bulk_modulus_GPa',
    'Shear Modulus (G)': 'shear_modulus_GPa',
    'Formation Energy':  'formation_energy_per_atom_eV',
    'Density':           'density_g_cm3',
    'Melting Point':     'melting_point_K',
    'Band Gap':          'band_gap_eV',
}
weights = {
    _property_map[prop]: w
    for prop, w in ds4.set_index('Property')['Weights'].items()
    if prop in _property_map
}

# Copy only MQI-related columns
mqi_df = df[list(weights.keys())].copy()

# Invert formation energy (more negative = more stable = better)
mqi_df['formation_energy_per_atom_eV'] = -mqi_df['formation_energy_per_atom_eV']

# Normalize each column to [0, 1]
scaler_mqi = MinMaxScaler()
mqi_norm   = pd.DataFrame(
    scaler_mqi.fit_transform(mqi_df),
    columns=mqi_df.columns
)

# Weighted sum = MQI
df['MQI'] = sum(mqi_norm[col] * w for col, w in weights.items())

print('MQI computed successfully!')
print(df['MQI'].describe())

plt.figure(figsize=(8, 4))
plt.hist(df['MQI'], bins=60, color='mediumseagreen', edgecolor='white')
plt.title('Distribution of MQI Scores', fontsize=13, fontweight='bold')
plt.xlabel('MQI Score (0 = worst, 1 = best)')
plt.ylabel('Number of Materials')
plt.tight_layout()
plt.show()

print('\nTop 5 materials by MQI:')
print(df[['material_id', 'formula', 'category', 'MQI']].sort_values(
    'MQI', ascending=False).head(5))

# ── Feature set & train/test split ────────────────────────────────────────────
FEATURES = [
    # Structural
    'n_elements',
    'spacegroup_number',
    'crystal_system_enc',
    'category_enc',
    'nsites',
    'volume_A3',
    # Electronic
    'band_gap_eV',
    'is_metal',
    # Mechanical
    'bulk_modulus_GPa',
    'shear_modulus_GPa',
    'poisson_ratio',
    # Thermal
    'melting_point_K',
    # Stability
    'formation_energy_per_atom_eV',
    'energy_above_hull_eV',
    # Physical
    'density_g_cm3',
]

X = df[FEATURES]   # Input features
y = df['MQI']      # Target

print('Input shape  (X):', X.shape)
print('Target shape (y):', y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f'Training samples : {X_train.shape[0]}')
print(f'Testing  samples : {X_test.shape[0]}')

# ── Build and train XGBoost model ──────────────────────────────────────────────
model = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(
        n_estimators     = 300,   # number of trees
        max_depth        = 6,     # depth of each tree
        learning_rate    = 0.05,  # how fast it learns
        subsample        = 0.8,   # 80% of data per tree
        colsample_bytree = 0.8,   # 80% of features per tree
        tree_method      = 'hist',
        random_state     = 42,
        n_jobs           = -1     # use all CPU cores
    ))
])

print('Training XGBoost model...')
model.fit(X_train, y_train)
print('Training complete!')

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print('===== Model Evaluation =====')
print(f'MAE  (Mean Absolute Error) : {mae:.4f}')
print(f'RMSE (Root Mean Sq. Error) : {rmse:.4f}')
print(f'R²   (R-squared Score)     : {r2:.4f}')
print()
print(f'Model explains {r2*100:.2f}% of the variation in MQI')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.3, color='royalblue', s=10)
axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual MQI')
axes[0].set_ylabel('Predicted MQI')
axes[0].set_title(f'Actual vs Predicted MQI\nR² = {r2:.4f}')
axes[0].legend()

# Plot 2: Residuals
residuals = y_test.values - y_pred
axes[1].hist(residuals, bins=50, color='salmon', edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--', lw=2)
axes[1].set_xlabel('Error (Actual - Predicted)')
axes[1].set_ylabel('Count')
axes[1].set_title('Prediction Error Distribution')

plt.suptitle('XGBoost — MQI Prediction Results', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

importances = model.named_steps['xgb'].feature_importances_
feat_imp    = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
feat_imp.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — XGBoost', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print('Most important feature:', feat_imp.idxmax())

# ── Save output ────────────────────────────────────────────────────────────────
df['MQI_predicted'] = model.predict(X)

output = df[['material_id', 'formula', 'category', 'crystal_system', 'MQI', 'MQI_predicted']]
output.to_csv('DS1_with_MQI.csv', index=False)

print('Saved to DS1_with_MQI.csv')
print(output.head(10))

# ══════════════════════════════════════════════════════════════════════════════
# D2 + D3 — Commodity Price Prediction with Cross-Domain Material Features
# ══════════════════════════════════════════════════════════════════════════════

# ── Load DS2 (commodity prices) and DS3 (cross-domain features) ───────────────
ds2 = pd.read_csv('DS2_commodity_prices_10yr.csv', parse_dates=['date'])
ds3 = pd.read_csv('DS3_crossdomain_features_daily.csv', parse_dates=['date'])

print('\nDS2 shape:', ds2.shape)
print('DS3 shape:', ds3.shape)

# ── Merge DS2 + DS3 on date and commodity (DS3 enriches DS2) ──────────────────
merged = pd.merge(ds2, ds3, on=['date', 'commodity'], how='inner')
print('Merged shape (DS2 + DS3):', merged.shape)

# ── Target: next-day close price ──────────────────────────────────────────────
merged = merged.sort_values(['commodity', 'date']).reset_index(drop=True)
merged['next_close'] = merged.groupby('commodity')['close'].shift(-1)
merged = merged.dropna(subset=['next_close'])

# ── Encode commodity as a numeric feature ─────────────────────────────────────
le_commodity = LabelEncoder()
merged['commodity_enc'] = le_commodity.fit_transform(merged['commodity'])

# ── Feature sets ──────────────────────────────────────────────────────────────
# DS2 financial indicators
DS2_FEATURES = [
    'commodity_enc',
    'open', 'high', 'low', 'close', 'volume',
    'daily_return', 'return_5d', 'return_21d',
    'volatility_5d_ann', 'volatility_21d_ann', 'volatility_63d_ann',
    'sma_21', 'sma_63',
    'bollinger_upper', 'bollinger_lower', 'bollinger_z',
    'rsi_14', 'macd', 'macd_signal',
    'momentum_10d', 'momentum_21d',
    'term_spread',
]

# DS3 cross-domain material-science signals
DS3_FEATURES = [
    'mqi',
    'supply_disruption_prob',
    'substitution_elasticity',
    'green_premium_per_kg',
    'carbon_intensity_virgin',
    'carbon_intensity_recycled',
    'herfindahl_index',
    'mqi_5d_trend',
    'mqi_21d_trend',
    'mqi_63d_trend',
]

ALL_FEATURES = DS2_FEATURES + DS3_FEATURES

# Drop rows where any feature is NaN
merged_clean = merged[ALL_FEATURES + ['next_close']].dropna()
print(f'Samples after dropping NaNs: {len(merged_clean)}')

X_comm = merged_clean[ALL_FEATURES]
y_comm = merged_clean['next_close']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_comm, y_comm, test_size=0.2, random_state=42
)
print(f'Commodity training samples : {X_train_c.shape[0]}')
print(f'Commodity testing  samples : {X_test_c.shape[0]}')

# ── Train Model A: DS2 features only (baseline) ───────────────────────────────
model_ds2 = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        tree_method      = 'hist',
        random_state     = 42,
        n_jobs           = -1,
    ))
])
print('\nTraining commodity model (DS2 only)...')
model_ds2.fit(X_train_c[DS2_FEATURES], y_train_c)

y_pred_ds2 = model_ds2.predict(X_test_c[DS2_FEATURES])
mae_ds2  = mean_absolute_error(y_test_c, y_pred_ds2)
rmse_ds2 = np.sqrt(mean_squared_error(y_test_c, y_pred_ds2))
r2_ds2   = r2_score(y_test_c, y_pred_ds2)
print('=== DS2-only Model ===')
print(f'MAE  : {mae_ds2:.4f}')
print(f'RMSE : {rmse_ds2:.4f}')
print(f'R²   : {r2_ds2:.4f}')

# ── Train Model B: DS2 + DS3 features (cross-domain enriched) ─────────────────
model_ds2_ds3 = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        tree_method      = 'hist',
        random_state     = 42,
        n_jobs           = -1,
    ))
])
print('\nTraining commodity model (DS2 + DS3)...')
model_ds2_ds3.fit(X_train_c[ALL_FEATURES], y_train_c)

y_pred_all = model_ds2_ds3.predict(X_test_c[ALL_FEATURES])
mae_all  = mean_absolute_error(y_test_c, y_pred_all)
rmse_all = np.sqrt(mean_squared_error(y_test_c, y_pred_all))
r2_all   = r2_score(y_test_c, y_pred_all)
print('=== DS2 + DS3 (Cross-Domain) Model ===')
print(f'MAE  : {mae_all:.4f}')
print(f'RMSE : {rmse_all:.4f}')
print(f'R²   : {r2_all:.4f}')

# ── Compare the two models ─────────────────────────────────────────────────────
print('\n=== Impact of Adding DS3 Cross-Domain Features ===')
print(f'R² improvement : {(r2_all - r2_ds2)*100:.2f} percentage points')
print(f'MAE  reduction : {mae_ds2 - mae_all:.4f}')
print(f'RMSE reduction : {rmse_ds2 - rmse_all:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test_c, y_pred_all, alpha=0.3, color='royalblue', s=5)
axes[0].plot(
    [y_test_c.min(), y_test_c.max()],
    [y_test_c.min(), y_test_c.max()],
    'r--', lw=2, label='Perfect Prediction'
)
axes[0].set_xlabel('Actual Next-Day Close')
axes[0].set_ylabel('Predicted Next-Day Close')
axes[0].set_title(f'Actual vs Predicted (DS2 + DS3)\nR² = {r2_all:.4f}')
axes[0].legend()

metrics = ['MAE', 'RMSE', 'R²']
ds2_scores   = [mae_ds2,  rmse_ds2,  r2_ds2]
ds2ds3_scores = [mae_all, rmse_all,  r2_all]
x = np.arange(len(metrics))
width = 0.35
axes[1].bar(x - width/2, ds2_scores,   width, label='DS2 only',    color='steelblue')
axes[1].bar(x + width/2, ds2ds3_scores, width, label='DS2 + DS3',  color='mediumseagreen')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics)
axes[1].set_title('Model Comparison: DS2 vs DS2+DS3')
axes[1].legend()

plt.suptitle('Commodity Price Prediction — XGBoost', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Feature importance for cross-domain model ─────────────────────────────────
importances_comm = model_ds2_ds3.named_steps['xgb'].feature_importances_
feat_imp_comm    = pd.Series(importances_comm, index=ALL_FEATURES).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
colors = ['mediumseagreen' if f in DS3_FEATURES else 'steelblue' for f in feat_imp_comm.index]
feat_imp_comm.plot(kind='barh', color=colors)
plt.title('Feature Importance — Commodity Price Model (DS2 + DS3)', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')

blue_patch  = mpatches.Patch(color='steelblue',      label='DS2 (Financial)')
green_patch = mpatches.Patch(color='mediumseagreen', label='DS3 (Material Science)')
plt.legend(handles=[blue_patch, green_patch])
plt.tight_layout()
plt.show()

print('Most important feature (commodity model):', feat_imp_comm.idxmax())

# ── Save enriched commodity predictions ───────────────────────────────────────
merged_clean = merged_clean.copy()
merged_clean['next_close_predicted'] = model_ds2_ds3.predict(X_comm)
merged_clean.to_csv('DS2_DS3_with_predictions.csv', index=False)
print('Saved to DS2_DS3_with_predictions.csv')