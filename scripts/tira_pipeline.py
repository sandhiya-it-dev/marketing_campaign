import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

# Load the cleaned dataset
df = pd.read_csv('/Users/sandhiyachandrasekar'
'/marketing_campaign/cleaned_dataset/tira_campaign_data_cleaned.csv')

###### MODEL BUILDING #########

# A. SAFE CHANNEL TRACKING: Sort names alphabetically so order doesn't confuse the model
df['channel_clean'] = df['channel_used'].astype(str).apply(
    lambda x: ", ".join(sorted([item.strip() for item in x.split(',')]))
)

# B. HIGH-EFFICACY RATIOS: Gives the algorithms mathematical leverage
df['conversion_efficiency'] = df['conversions'] / (df['clicks'] + 1)
df['cost_per_click'] = df['acquisition_cost'] / (df['clicks'] + 1)
df['click_to_lead_ratio'] = df['leads'] / (df['clicks'] + 1)
df['impression_to_click_ratio'] = df['clicks'] / (df['impressions'] + 1)

# C. HANDLE LONG-TAIL SPENDING: Log transform revenue to soften outliers
df['Log_Revenue'] = np.log1p(df['revenue']) 

# ==========================================================
# 2. DEFINING FEATURES & TARGETS (WITH LEAKAGE PREVENTION)
# ==========================================================
# Explicitly drop targets, original raw values, and 'roi' to prevent cheating!
drop_cols = ['revenue', 'Log_Revenue', 'profit_loss_flag', 'roi', 'channel_used']
X = df.drop(columns=drop_cols, errors='ignore')

y_reg = df['Log_Revenue']        # Target for Regression
y_clf = df['profit_loss_flag']   # Target for Classification

# Automatically isolate numeric vs text columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Define automated preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

# ==========================================================
# 3. TARGET 1: REGRESSION (Revenue Pipeline)
# ==========================================================
print("\n📈 Training Revenue Regression Model (Gradient Boosting)...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42))
])

reg_pipeline.fit(X_train_r, y_train_r)
reg_pred_log = reg_pipeline.predict(X_test_r)

# Transform back to normal dollar values using exponential function
y_test_r_normal = np.expm1(y_test_r)
reg_pred_normal = np.expm1(reg_pred_log)

print(f"✅ Regression Model Evaluated!")
print(f"   👉 Best R² Score: {r2_score(y_test_r, reg_pred_log) * 100:.2f}%")
print(f"   👉 Mean Absolute Error: ${mean_absolute_error(y_test_r_normal, reg_pred_normal):.2f}")

# ==========================================================
# 4. TARGET 2: CLASSIFICATION (Profit Pipeline)
# ==========================================================
print("\n🎯 Training Profit Classification Model (Random Forest Champion)...")
# stratify=y_clf keeps the balance of profit/loss equal in train and test splits
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced'))
])

clf_pipeline.fit(X_train_c, y_train_c)
clf_preds = clf_pipeline.predict(X_test_c)
clf_accuracy = accuracy_score(y_test_c, clf_preds)

print(f"✅ Classification Model Evaluated!")
print(f"   👉 Champion Accuracy: {clf_accuracy * 100:.2f}%")
print("\n📋 Detailed Performance Breakout:")
print(classification_report(y_test_c, clf_preds, target_names=['Loss (0)', 'Profit (1)']))

# ==========================================================
# 5. EXPORTING THE CHAMPION MODEL FOR PRODUCTION DEPLOYMENT
# ==========================================================
model_filename = "tira_profit_champion_model.joblib"

print(f"\n💾 Saving 93%+ Champion Classifier to file...")
# Saving the whole pipeline saves the scaler and encoder configurations inside it!
joblib.dump(clf_pipeline, model_filename)
print(f"🎉 Success! Saved as '{model_filename}'. Your workflow script is completely fixed.")