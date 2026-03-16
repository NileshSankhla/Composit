import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ── Compute MQI (Material Quality Index) ──────────────────────────────────────
# Weights from DS4
weights = {
    'bulk_modulus_GPa':             0.20,
    'shear_modulus_GPa':            0.20,
    'formation_energy_per_atom_eV': 0.20,
    'density_g_cm3':                0.10,
    'melting_point_K':              0.15,
    'band_gap_eV':                  0.15,
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