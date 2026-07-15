import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_score, recall_score, f1_score
)

df = pd.read_csv('/Users/sandhiyachandrasekar/marketing_campaign/raw_dataset'
'/purplle_campaign_data_with_nulls.csv')

df.info()
print(df.info())

# 1. ₹Clean the column names by stripping whitespace and converting to lowercase
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# 2. Check for missing values in the dataset
df.isnull().sum()
print(df.isnull().sum())

# 3. Check for duplicates in the dataset
df.duplicated().sum()
print(df.duplicated().sum())# no duplicates found

# 4.Handle categorical missing values by filling them with the mode (most frequent value)
if "customer_segment" in df.columns and "target_audience" in df.columns:
    df["customer_segment"] = df.groupby("target_audience")["customer_segment"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
    )
categorical_cols = df.select_dtypes(include=["object", "string"]).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# 5. Handle numerical missing values by filling them with the median
numeric_cols = ["duration", "impressions", "clicks", "leads", "conversions", "revenue", "acquisition_cost", "engagement_score"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

# --- FEATURE ENGINEERING & DATA CLEANING ---
# 5. Clean, Validate, and Correct ROI & Flag Values
df["roi"] = (((df["revenue"] - df["acquisition_cost"]) / df["acquisition_cost"]) * 100).round(2)
median_roi = df["roi"].median()
df["profit_loss_flag"] = (df["roi"] > median_roi).astype(int)

# 6. Multi-Label Encoding for channel_used
def clean_and_untangle_channels(x):
    if pd.isnull(x):
        return []
    text = str(x).lower()
    valid_channels = ['whatsapp', 'youtube', 'facebook', 'instagram', 'google', 'email']
    found_channels = []
    for channel in valid_channels:
        if channel in text:
            found_channels.append(channel)
    return found_channels

df["channel_list"] = df["channel_used"].apply(clean_and_untangle_channels)

#  Create the binary tracking columns in STRICT lowercase
expected_channels = ['whatsapp', 'youtube', 'facebook', 'instagram', 'google', 'email']
channel_cols = []
for channel in expected_channels:
    # This guarantees names like 'channel_whatsapp' in the final dataset file
    clean_col_name = f"channel_{channel}"
    df[clean_col_name] = df["channel_list"].apply(lambda x: 1 if channel in x else 0)
    channel_cols.append(clean_col_name)

# Drop the temporary list column
df = df.drop(columns=["channel_list"])

# 7. Add Conversion Rate metric interaction column
df["conversion_rate"] = (df["conversions"] / df["clicks"]) * 100
df["conversion_rate"] = df["conversion_rate"].fillna(0) # Safeguard divide by zero exceptions   

# checking the null values after cleaning 
print(df.isnull().sum())

# 8. Save the cleanly processed dataframe asset
output_file_path = '/Users/sandhiyachandrasekar/marketing_campaign/cleaned_dataset/purplle_campaign_data_cleaned.csv'
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
df.to_csv(output_file_path, index=False)
print(f"💾 Cleaned Purplle dataset saved to: {output_file_path}")

# ---  EXPLORATORY DATA ANALYSIS (EDA) ---

# --- 1. BASIC STRUCTURAL SUMMARIES ---
print("--- 1. Data Type & Schema Overview ---")
print(df.info())
print("\n--- 2. Key Numeric Distributions ---")
print(df[["duration", "impressions", "clicks", "leads", "conversions", "revenue", "acquisition_cost", "roi", "conversion_rate"]].describe().round(2).to_string())

# --- 2. MULTI-LABEL LOWERCASE CHANNEL BREAKDOWN ---
print("\n--- 3. Channel Performance Breakdown (Lowercase Columns) ---")
lowercase_channels = ['channel_whatsapp', 'channel_youtube', 'channel_facebook', 'channel_instagram', 'channel_google', 'channel_email']

channel_summary_list = []
for ch_col in lowercase_channels:
    if ch_col in df.columns:
        # Filter campaigns where this specific channel was utilized
        active_campaigns = df[df[ch_col] == 1]
        
        channel_summary_list.append({
            "Channel Column": ch_col,
            "Total Volume": len(active_campaigns),
            "Avg Spend": active_campaigns["acquisition_cost"].mean(),
            "Avg Revenue": active_campaigns["revenue"].mean(),
            "Avg ROI (%)": active_campaigns["roi"].mean()
        })

df_ch_metrics = pd.DataFrame(channel_summary_list).sort_values(by="Avg ROI (%)", ascending=False)
print(df_ch_metrics.to_string(index=False, formatters={
    'Total Volume': lambda x: f"{x:,}",
    'Avg Spend': lambda x: f"${x:,.2f}",
    'Avg Revenue': lambda x: f"${x:,.2f}",
    'Avg ROI (%)': lambda x: f"{x:,.2f}%"
}))

# --- 3. AUDIENCE & SEGMENT INTERACTIONS ---
print("\n--- 4. Target Audience Performance Matrix ---")
if "target_audience" in df.columns:
    audience_group = df.groupby("target_audience")[["clicks", "revenue", "roi"]].mean()
    print(audience_group.round(2).to_string())

print("\n--- 5. Customer Segment Performance Matrix ---")
if "customer_segment" in df.columns:
    segment_group = df.groupby("customer_segment")[["acquisition_cost", "revenue", "roi"]].mean()
    print(segment_group.round(2).to_string())

# --- 4. ATTRIBUTE CORRELATIONS ---
print("\n--- 6. Key Variable Correlation Matrix ---")
core_metrics = ["acquisition_cost", "clicks", "revenue", "roi", "conversion_rate", "engagement_score"]
existing_core = [m for m in core_metrics if m in df.columns]
print(df[existing_core].corr().round(3).to_string())



 