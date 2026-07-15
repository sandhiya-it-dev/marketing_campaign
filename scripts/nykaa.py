import numpy as np
import pandas as pd
import joblib
import os


df = pd.read_csv('/Users/sandhiyachandrasekar/marketing_campaign/raw_dataset'
'/nykaa_campaign_data_with_nulls.csv')
print(df.shape)
print(df.info())
print(df.dtypes)
# Check for null values
print(df.isnull().sum()) # as per output only campaign_id doesn't have null values, all other columns have null values  


# editing column names to remove spaces and make them more readable
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print(df.columns)

# checking for duplicate records
duplicate_records = df.duplicated().sum()
print(f"Number of duplicate records: {duplicate_records}") # no duplicate records found

# Handle missing values (Smarter Group-By/Mode Imputation)
# If Customer_Segment is missing, fill it based on the most common value for that row's Target_Audience
if "Customer_Segment" in df.columns and "Target_Audience" in df.columns:
    df["Customer_Segment"] = df.groupby("Target_Audience")[
        "Customer_Segment"
].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
    )

# Fill any remaining categorical missing values with their column mode
categorical_cols = df.select_dtypes(include=["object", "string"]).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
print(df.isnull().sum())  # Check for null values after imputation

# Perform data type conversion
numeric_cols = [
    "duration",
    "impressions",
    "clicks",
    "leads",
    "conversions",
    "revenue",
    "acquisition_cost",
    "engagement_score",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values in numeric columns by filling with median
for col in numeric_cols:
    if col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(
                f"   -> Filled {null_count} missing values in '{col}' with median ({median_value})"
            )
# convert date columns to datetime format
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

# clean ,validate and correct ROI values
df["roi"] = (((df["revenue"] - df["acquisition_cost"]) / df["acquisition_cost"]) * 100).round(2)
print( "-> Preprocessing completed: Missing values handled (Categorical & Numerical), types converted, and ROI corrected."
)

print(df.isnull().sum())  # validating if all null values and ROI have been handled

# FEATURE ENGINEERING
# 1. Calculate the median ROI across the entire dataset
median_roi = df["roi"].median()

# 2. Flag campaigns as 1 if they beat the median, 0 if they fell below it
df["profit_loss_flag"] = (df["roi"] > median_roi).astype(int)

print("\n==================================================")
print("FIXING CLASS IMBALANCE")
print("==================================================")
print(f"📊 Median ROI Threshold: {median_roi:.2f}%")
print("🔄 New Target Variable Distribution:")
print(df["profit_loss_flag"].value_counts())
print("==================================================")

# 2. Multi-Label Encoding for channel_used
print("\n--- Explicit Multi-Label Encoding for channel_used ---")

def clean_and_untangle_channels(x):
    if pd.isnull(x):
        return []
    
    # 1. Convert to completely lowercase to eliminate casing issues entirely
    text = str(x).lower()
    
    # 2. Define the true target marketing channels we expect
    valid_channels = ['whatsapp', 'youtube', 'facebook', 'instagram', 'google', 'email']
    
    found_channels = []
    # 3. Check if any valid channel exists inside the text string
    for channel in valid_channels:
        if channel in text:
            found_channels.append(channel)
            
    return found_channels

# Apply our precise untangling logic to build the array list
df["channel_list"] = df["channel_used"].apply(clean_and_untangle_channels)

# Extract every unique channel found dynamically
unique_channels = set([item for sublist in df["channel_list"] for item in sublist])

# Create individual binary tracking columns (1 or 0) for each unique channel
channel_cols = []
for channel in unique_channels:
    if channel:
        # Format name cleanly in lowercase with underscores
        clean_col_name = f"channel_{channel}"
        
        # Check if the channel exists in that row's array list
        df[clean_col_name] = df["channel_list"].apply(lambda x: 1 if channel in x else 0)
        channel_cols.append(clean_col_name)

# Drop the temporary parsing list column
df = df.drop(columns=["channel_list"])

print(f"-> Precise multi-label processing complete!")
print(f"-> Cleaned unique channels discovered: {list(unique_channels)}")
print(f"-> Created binary columns: {channel_cols}") 

# Check how the binary switches were mapped (First 10 rows)
print("\n--- Side-by-Side Channel Binary Check ---")
check_columns = ["channel_used"] + channel_cols
print(df[check_columns].head(10).to_string())

# 3. Feature selection and transformation for modeling
print("\n--- Feature Selection and Transformation ---")

# Calculate conversion rate as a new feature
df["conversion_rate"] = (df["conversions"] / df["clicks"]) * 100
df["conversion_rate"] = df["conversion_rate"].fillna(0)  # Handle 0-click rows safely

# Categorical one-hot encoding
categorical_cols = ["campaign_type", "target_audience", "language", "customer_segment"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save the cleaned DataFrame to a new CSV file
output_file_path = '/Users/sandhiyachandrasekar/marketing_campaign/cleaned_dataset/nykaa_campaign_data_cleaned.csv'
df.to_csv(output_file_path, index=False)
print(f"Cleaned data saved to {output_file_path}")

# Gathering the channel switches for final feature set
channel_cols = [col for col in df_encoded.columns if col.startswith("channel_")and col != "channel_used"]

# setting up the target predictors
base_modeling_features = [
    "duration", 
    "impressions", 
    "clicks", 
    "leads", 
    "acquisition_cost", 
    "engagement_score", 
    "conversion_rate"
] + channel_cols

print(df['channel_used'].value_counts(dropna=False).head(15))

# Finalising the independant features for modeling
X_pool = df_encoded[base_modeling_features]

# EDA 
print("\n==================================================")
print("PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("==================================================")

# 1. Analyze campaign performance across customer segments
print("--- 1. Average Performance by Customer Segment ---")
# Find the columns that were generated by our one-hot encoding step for customer segments
segment_cols = [col for col in df_encoded.columns if col.startswith("customer_segment_")]

# Group by the original segment column from the original df for a clean printout
segment_summary = df.groupby("customer_segment")[["acquisition_cost", "revenue", "roi"]].mean()
print(segment_summary.round(2).to_string())


# 2. Identify top-performing and low-performing campaigns by ROI
print("\n--- 2. Extreme Campaign Performers ---")
top_idx = df_encoded["roi"].idxmax()
low_idx = df_encoded["roi"].idxmin()

print(f"🏆 TOP CAMPAIGN -> ID: {df_encoded.loc[top_idx, 'campaign_id']}")
print(f"   Spend: ${df_encoded.loc[top_idx, 'acquisition_cost']:,.2f} | Revenue: ${df_encoded.loc[top_idx, 'revenue']:,.2f} | ROI: {df_encoded.loc[top_idx, 'roi']:.2f}%")

print(f"📉 LOWEST CAMPAIGN -> ID: {df_encoded.loc[low_idx, 'campaign_id']}")
print(f"   Spend: ${df_encoded.loc[low_idx, 'acquisition_cost']:,.2f} | Revenue: ${df_encoded.loc[low_idx, 'revenue']:,.2f} | ROI: {df_encoded.loc[low_idx, 'roi']:.2f}%")


# 3. Explore relationships between spend, clicks, revenue, and ROI
print("\n--- 3. Correlation Matrix (Metric Interactions) ---")
metrics_list = ["acquisition_cost", "clicks", "revenue", "roi", "conversion_rate"]
correlation_matrix = df_encoded[metrics_list].corr()
print(correlation_matrix.round(3).to_string())
print("\n💡 Note: Correlation scores near 1.0 indicate strong positive relationships.")


# 4. Analyze channel-wise effectiveness using your binary flags
print("\n--- 4. Channel Effectiveness Breakdown ---")
channel_summary = []

for ch in channel_cols:
    ch_df = df_encoded[df_encoded[ch] == 1]
    if len(ch_df) > 0:
        channel_summary.append({
            "Channel": ch.replace("channel_", "").capitalize(),
            "Total Campaigns": len(ch_df),
            "Avg Revenue": ch_df["revenue"].mean(),
            "Avg ROI (%)": ch_df["roi"].mean()
        })

df_channels = pd.DataFrame(channel_summary).sort_values(by="Avg ROI (%)", ascending=False)
print(df_channels.to_string(index=False, formatters={
    'Avg Revenue': lambda x: f"${x:,.2f}",
    'Avg ROI (%)': lambda x: f"{x:.2f}%"
}))

