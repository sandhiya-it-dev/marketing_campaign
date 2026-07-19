import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score

# ==========================================================
# CONFIGURATION: Set your brand here nykaa
# ==========================================================
brand_name = "nykaa"  
csv_path = f'/Users/sandhiyachandrasekar/marketing_campaign/cleaned_dataset/nykaa_campaign_data_cleaned.csv'

print(f"🚀 Starting Standardized training pipeline for: {brand_name.upper()}")
# ==========================================================
# STEP 1: LOAD THE DATA
# ==========================================================
print("📦 Loading dataset...")
df = pd.read_csv(csv_path)

# ==========================================================
# STEP 2: FEATURE ENGINEERING
# ==========================================================

print("🧪 Creating marketing features...")
df["ctr"] = df["clicks"] / df["impressions"]
df["lead_conversion_rate"] = (
    df["conversions"] / df["leads"]
)

df["cost_per_click"] = (
    df["acquisition_cost"] / df["clicks"]
)
df["cost_per_conversion"] = (
    df["acquisition_cost"] / df["conversions"]
)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ==========================================================
# STEP 3: PREPROCESS THE FEATURES (Matching Tira Exactly)
# ==========================================================
print("🛠️ Preprocessing data with pandas...")

# Drop the targets and metadata
drop_cols = ['revenue', 'profit_loss_flag', 'roi', 'channel_used', 'campaign_id', 'id']
X_raw = df.drop(columns=drop_cols, errors='ignore')

# One-hot encode text columns (handles audience_size, device_type, etc.)
X = pd.get_dummies(
    X_raw,
    drop_first=True,
    dtype=int
)
# Define targets
y_reg = np.log1p(df['revenue']) 
y_clf = df['profit_loss_flag']

# Keep track of training column order
trained_columns = list(X.columns)

# ==========================================================
# STEP 4: TRAIN THE REGRESSOR
# ==========================================================
print("🎯 Training the Regressor...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
param_grid = {

    "learning_rate":[0.01,0.03,0.05,0.08,0.1],

    "max_iter":[200,300,500,700],

    "max_depth":[5,8,10,None],

    "min_samples_leaf":[10,20,30]

}

base_model = HistGradientBoostingRegressor(
    random_state=42
)

search = RandomizedSearchCV(

    estimator=base_model,

    param_distributions=param_grid,

    n_iter=5,

    cv=3,

    scoring="r2",

    random_state=42,

    verbose=2,

    n_jobs=-1

)

search.fit(X_train_r, y_train_r)

reg_model = search.best_estimator_

reg_model.fit(X_train_r, y_train_r)

reg_preds = reg_model.predict(X_test_r)
r2_result = r2_score(y_test_r, reg_preds) * 100
actual = np.expm1(y_test_r)

predicted = np.expm1(reg_preds)

mae = mean_absolute_error(actual, predicted)

# ==========================================================
# STEP 5: TRAIN THE CLASSIFIER
# ==========================================================
print("🎯 Training the Classifier...")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

clf_model = HistGradientBoostingClassifier(random_state=42)
clf_model.fit(X_train_c, y_train_c)

clf_preds = clf_model.predict(X_test_c)
accuracy_result = accuracy_score(y_test_c, clf_preds) * 100

print(f"📈 Regressor R² Score: {r2_result:.2f}%")
print(f"💵 Regressor MAE: ${mae:,.2f}")
print(f"🎯 Classifier Accuracy: {accuracy_result:.2f}%")
# ==========================================================
# STEP 6: SAVE THE MODEL FILES WITH CLEAN NAMES
# ==========================================================
print("\n💾 Saving standardized models and configuration...")
model_dir = "/Users/sandhiyachandrasekar/marketing_campaign/saved_models/"
os.makedirs(model_dir, exist_ok=True)

# 1. Save Regressor
reg_path = os.path.join(model_dir, f"nykaa_cosmetics_regressor.pkl")
with open(reg_path, "wb") as f:
    pickle.dump(reg_model, f)

# 2. Save Classifier
clf_path = os.path.join(model_dir, f"nykaa_cosmetics_classifier.pkl")
with open(clf_path, "wb") as f:
    pickle.dump(clf_model, f)

# 3. Save Features list (Replacing the old _scaler.pkl file with a clean list of columns)
features_path = os.path.join(model_dir, f"nykaa_cosmetics_model.pkl")
with open(features_path, "wb") as f:
    pickle.dump(trained_columns, f)

print(f"✅ All assets successfully saved to: {model_dir}")
print(f"📈 Regressor R² Score: {r2_result:.2f}%")
print(f"🎯 Classifier Accuracy: {accuracy_result:.2f}%")
print("="*50)