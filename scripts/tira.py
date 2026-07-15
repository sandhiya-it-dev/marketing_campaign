import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/sandhiyachandrasekar/marketing_campaign/raw_dataset/tira_campaign_data_with_nulls.csv')
print(df.shape)
print(df.info())
print(df.isnull().sum()) 

# Editing column names to remove spaces and make them more readable
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print(df.columns)
print(df.channel_used.unique())  # Check unique values in channel_used column

# Check for duplicate records
duplicate_records = df.duplicated().sum()
print(f"Number of duplicate records: {duplicate_records}")  # No duplicate records found as per output

# Handle missing values for categorical columns using smarter group-by/mode imputation
if "customer_segment" in df.columns and "target_audience" in df.columns:
    df["customer_segment"] = df.groupby("target_audience")["customer_segment"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
    )
categorical_cols = df.select_dtypes(include=["object", "string"]).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Handle missing values for numeric columns by filling with median
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

# Check for null values after imputation, median filling and ROI correction
print(df.isnull().sum())

# FEATURE ENGINEERING 
# 1. Calculate the median ROI across the entire dataset
median_roi = df["roi"].median()
print(f"Median ROI across the dataset: {median_roi}")
# 2. Clean, Validate, and Correct ROI & Flag Values
df["roi"] = (((df["revenue"] - df["acquisition_cost"]) / df["acquisition_cost"]) * 100).round(2)
median_roi = df["roi"].median()
df["profit_loss_flag"] = (df["roi"] > median_roi).astype(int)

# 3. Multi label encoding for the channel used
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

# 4. Add Conversion Rate metric interaction column
df["conversion_rate"] = (df["conversions"] / df["clicks"]) * 100
df["conversion_rate"] = df["conversion_rate"].fillna(0) # Safeguard divide by zero exceptions   

# checking the null values after cleaning 
print(df.isnull().sum())

# 5. Save the cleaned dataset 
output_file_path = '/Users/sandhiyachandrasekar/marketing_campaign/cleaned_dataset/tira_campaign_data_cleaned.csv'
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
df.to_csv(output_file_path, index=False)
print(f"💾 Cleaned tira dataset saved to: {output_file_path}")


### EXPLORATORY DATA ANALYSIS
##1. TIRA campaign performance across brands
print("==========================================================")
print("📊 REQUIREMENT 1: TIRA TOTAL PERFORMANCE OVERVIEW")
print("==========================================================")
print(f"Total Tira Campaigns Run: {len(df):,}")
print(f"Total Combined Revenue:   ₹{df['revenue'].sum():,.2f}")
print(f"Total Combined Spend:     ₹{df['acquisition_cost'].sum():,.2f}")
print("----------------------------------------------------------")
print(f"Average Revenue per Campaign: ₹{df['revenue'].mean():,.2f}")
print(f"Average Spend per Campaign:   ₹{df['acquisition_cost'].mean():,.2f}")
print(f"Average Campaign ROI:         {df['roi'].mean():.4f}")
print("==========================================================")

print("==========================================================")
print("🏆 REQUIREMENT 2: TIRA CAMPAIGN EXTREMES REPORT")
print("==========================================================")


# 2. Top 5 Performing Campaigns by Revenue
top_5 = df.nlargest(5, 'revenue')
print("🔥 TOP 5 HIGH-PERFORMING CAMPAIGNS (BY REVENUE):")
print("----------------------------------------------------------")
for idx, row in top_5.iterrows():
    print(f"ID: {idx:<5} | Type: {row['campaign_type']:<12} | Audience: {row['target_audience']:<10} | Spend: ₹{row['acquisition_cost']:,.2f} | Revenue: ₹{row['revenue']:,.2f} | ROI: {row['roi']:.2f}")

print("\n")

# 3. Bottom 5 Performing Campaigns by Revenue
bottom_5 = df.nsmallest(5, 'revenue')
print("📉 BOTTOM 5 LOW-PERFORMING CAMPAIGNS (BY REVENUE):")
print("----------------------------------------------------------")
for idx, row in bottom_5.iterrows():
    print(f"ID: {idx:<5} | Type: {row['campaign_type']:<12} | Audience: {row['target_audience']:<10} | Spend: ₹{row['acquisition_cost']:,.2f} | Revenue: ₹{row['revenue']:,.2f} | ROI: {row['roi']:.2f}")
print("==========================================================")

print("==========================================================")
print("📈 REQUIREMENT 3: TIRA METRIC RELATIONSHIPS & CORRELATIONS")
print("==========================================================")

# 4. Calculate the Correlation Matrix for core financial indicators
# A value near 1.0 means strong positive correlation, near 0 means no correlation
correlation_matrix = df[['acquisition_cost', 'clicks', 'revenue', 'roi']].corr()

print("🔗 CORRELATION MATRIX OVERVIEW:")
print("----------------------------------------------------------")
print(correlation_matrix.to_string())
print("\n")

# 5. Extract specific descriptive relationship insights
avg_cost_per_click = df['acquisition_cost'].sum() / df['clicks'].sum()
avg_revenue_per_click = df['revenue'].sum() / df['clicks'].sum()

print("💡 STRATEGIC METRIC INTERACTION PAIRS:")
print("----------------------------------------------------------")
print(f"👉 Average Marketing Spend per Click (CPC):   ₹{avg_cost_per_click:.2f}")
print(f"👉 Average Revenue Generated per Click (RPC): ₹{avg_revenue_per_click:.2f}")
print("==========================================================")

print("==========================================================")
print("🎯 REQUIREMENT 4: TIRA CHANNEL EFFICACY SCOREBOARD")
print("==========================================================")

# 2. ISOLATE MIXED CHANNELS
df_isolated = df.copy()
# Using your exact column 'channel_used'
df_isolated["Individual Channel"] = (
    df_isolated["channel_used"].astype(str).str.split(",")
)
df_isolated = df_isolated.explode("Individual Channel")
df_isolated["Individual Channel"] = df_isolated["Individual Channel"].str.strip()

# 3. CRUNCH REAL METRICS FOR THE SCOREBOARD
# Grouping by your clean channel column and computing totals & averages
channel_scoreboard = (
    df_isolated.groupby("Individual Channel")
    .agg(
        Total_Campaigns=("campaign_id", "count"),
        Avg_Cost=("acquisition_cost", "mean"),
        Avg_Clicks=("clicks", "mean"),
        Avg_Leads=("leads", "mean"),
        Avg_Conversions=("conversions", "mean"),
        Avg_Revenue=("revenue", "mean"),
        Avg_ROI=("roi", "mean"),
    )
    .reset_index()
    .sort_values(by="Avg_ROI", ascending=False)
)

print("🎯 TIRA CHANNEL EFFICACY SCOREBOARD")
print("==========================================================================")
print(channel_scoreboard.round(2).to_string(index=False))
print("==========================================================================")

# 4. DATA VISUALIZATION
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Revenue vs. Acquisition Cost per Channel
melted_df = pd.melt(
    channel_scoreboard,
    id_vars=["Individual Channel"],
    value_vars=["Avg_Revenue", "Avg_Cost"],
    var_name="Metric",
    value_name="Amount",
)

sns.barplot(
    x="Individual Channel",
    y="Amount",
    hue="Metric",
    data=melted_df,
    ax=axes[0],
    palette="viridis",
)
axes[0].set_title("Average Revenue vs. Acquisition Cost", fontsize=14, pad=10)
axes[0].set_xlabel("Marketing Channel")
axes[0].set_ylabel("Amount")

# Chart 2: Full Conversion Funnel Analysis (Clicks -> Leads -> Conversions)
funnel_df = pd.melt(
    channel_scoreboard,
    id_vars=["Individual Channel"],
    value_vars=["Avg_Clicks", "Avg_Leads", "Avg_Conversions"],
    var_name="Funnel Stage",
    value_name="Count",
)

sns.barplot(
    x="Individual Channel",
    y="Count",
    hue="Funnel Stage",
    data=funnel_df,
    ax=axes[1],
    palette="magma",
)
axes[1].set_title("Funnel Volume: Clicks vs. Leads vs. Conversions", fontsize=14, pad=10)
axes[1].set_xlabel("Marketing Channel")
axes[1].set_ylabel("Average Count")

plt.tight_layout()
plt.show()