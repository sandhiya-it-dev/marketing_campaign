import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score
)
# ==========================================================
# STEP 1 : LOAD DATA
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(
    "/Users/sandhiyachandrasekar/marketing_campaign/cleaned_dataset/purplle_campaign_data_cleaned.csv"
)

# ==========================================================
# STEP 2 : FEATURE ENGINEERING (NO DATA LEAKAGE)
# ==========================================================

print("Creating marketing features...")

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
# STEP 3 : PREPROCESS FEATURES
# ==========================================================

drop_cols = [
    "revenue",
    "profit_loss_flag",
    "roi",
    "campaign_id",
    "channel_used",
    "id"
]

X_raw = df.drop(columns=drop_cols, errors="ignore")

X = pd.get_dummies(
    X_raw,
    drop_first=True,
    dtype=int
)

y_reg = np.log1p(df["revenue"])
y_clf = df["profit_loss_flag"]

trained_columns = list(X.columns)

# ==========================================================
# STEP 4 : TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)
# ==========================================================
# STEP 5 : HYPERPARAMETER TUNING
# ==========================================================

print("Searching best parameters...")
param_grid = {

    "learning_rate":[0.01,0.03,0.05,0.08,0.1],

    "max_iter":[200,300,500,700],

    "max_depth":[5,8,10,None],

    "min_samples_leaf":[10,20,30,40]

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
    n_jobs=-1,
    verbose=3
)
search.fit(X_train,y_train)

reg_model = search.best_estimator_

print("\nBest Parameters")
print(search.best_params_)

# ==========================================================
# STEP 6 : EVALUATE REGRESSOR
# ==========================================================
pred_log = reg_model.predict(X_test)

r2 = r2_score(y_test,pred_log)

actual = np.expm1(y_test)

predicted = np.expm1(pred_log)

mae = mean_absolute_error(actual,predicted)

print("\nRegression Results")
print("------------------")
print("R2 Score :",round(r2*100,2),"%")
print("MAE :",round(mae,2))
# ==========================================================
# STEP 7 : CLASSIFIER
# ==========================================================
X_train_c,X_test_c,y_train_c,y_test_c = train_test_split(

    X,

    y_clf,

    test_size=0.20,

    random_state=42

)
clf_model = HistGradientBoostingClassifier(

    random_state=42

)
clf_model.fit(

    X_train_c,

    y_train_c

)
clf_pred = clf_model.predict(X_test_c)

accuracy = accuracy_score(

    y_test_c,

    clf_pred
)

print("\nClassification Accuracy :",round(accuracy*100,2),"%")
# ==========================================================
# STEP 8 : SAVE MODELS
# ==========================================================

model_dir="/Users/sandhiyachandrasekar/marketing_campaign/saved_models"

os.makedirs(model_dir,exist_ok=True)

pickle.dump(
    reg_model,
    open(os.path.join(model_dir,
    "purplle_cosmetics_regressor.pkl"),"wb")
)

pickle.dump(
    clf_model,
    open(os.path.join(model_dir,
    "purplle_cosmetics_classifier.pkl"),"wb")
)

pickle.dump(
    trained_columns,
    open(os.path.join(model_dir,
    "purplle_cosmetics_model.pkl"),"wb")
)

print("\nModels Saved Successfully")