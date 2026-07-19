import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import altair as alt

# ==========================================================
# CONFIGURATION & ASSET LOADER
# ==========================================================
st.set_page_config(page_title="Multi-Brand Campaign Predictor", page_icon="🛍️", layout="wide")
MODEL_DIR = "/Users/sandhiyachandrasekar/marketing_campaign/saved_models/"

@st.cache_resource
def load_brand_assets(brand_prefix):
    paths = [os.path.join(MODEL_DIR, f"{brand_prefix}_cosmetics_{suffix}.pkl") 
             for suffix in ["classifier", "regressor", "model"]]
    return [pickle.load(open(p, "rb")) for p in paths]

# ==========================================================
# HEADER & BRAND SELECTION
# ==========================================================
st.title("🛍️ Multi-Brand Marketing Campaign Predictor")
st.write("Predict Revenue and Campaign Profitability using Machine Learning.")

brand = st.selectbox("Select Brand", ["Tira Cosmetics", "Purplle Cosmetics", "Nykaa Cosmetics"])
brand_prefix = brand.split()[0].lower()
clf_model, reg_model, trained_columns = load_brand_assets(brand_prefix)

st.success(f"Connected to {brand} models")
st.markdown("---")

left_col, right_col = st.columns([1, 1.2])

# ==========================================================
# INPUT FORM (LEFT)
# ==========================================================
with left_col:
    st.subheader("Campaign Configuration")
    with st.form("prediction_form"):
        campaign_type = st.selectbox("Campaign Type", ["Email", "SEO", "Influencer", "Paid Ads", "Social Media"])
        target_audience = st.selectbox("Target Audience", ["College Students", "Premium Shoppers", "Tier 2 City Customers", "Working Women", "Youth"])
        
        c1, c2 = st.columns(2)
        impressions = c1.number_input("Impressions", min_value=0, value=50000, step=1000)
        clicks = c2.number_input("Clicks", min_value=0, value=3000, step=100)
        leads = c1.number_input("Leads", min_value=0, value=1500, step=100)
        conversions = c2.number_input("Conversions", min_value=0, value=900, step=50)
        
        acquisition_cost = st.number_input("Acquisition Cost ($)", min_value=0.0, value=300.0, step=10.0)
        engagement_score = st.slider("Engagement Score", 0.0, 100.0, 25.0)
        
        language = st.selectbox("Language", ["English", "Hindi", "Tamil", "Bengali"])
        customer_segment = st.selectbox("Customer Segment", ["College Students", "Premium Shoppers", "Tier 2 City Customers", "Working Women", "Youth"])

        st.markdown("### Channels Used")
        ch_cols = st.columns(3)
        email = ch_cols[0].checkbox("Email")
        youtube = ch_cols[1].checkbox("YouTube")
        instagram = ch_cols[2].checkbox("Instagram")
        whatsapp = ch_cols[0].checkbox("WhatsApp")
        google = ch_cols[1].checkbox("Google")
        facebook = ch_cols[2].checkbox("Facebook")

        submitted = st.form_submit_button("🚀 Predict Campaign")

# ==========================================================
# PREDICTION DASHBOARD (RIGHT)
# ==========================================================
with right_col:
    st.subheader("Prediction Dashboard")

    if submitted:
        # Calculate rates & base features
        ctr = clicks / impressions if impressions > 0 else 0
        lcr = conversions / leads if leads > 0 else 0
        cpc = acquisition_cost / clicks if clicks > 0 else 0
        cpa = acquisition_cost / conversions if conversions > 0 else 0

        raw_input = pd.DataFrame([{
            "campaign_type": campaign_type, "target_audience": target_audience, "impressions": impressions,
            "clicks": clicks, "leads": leads, "conversions": conversions, "acquisition_cost": acquisition_cost,
            "language": language, "engagement_score": engagement_score, "customer_segment": customer_segment,
            "channel_email": int(email), "channel_youtube": int(youtube), "channel_instagram": int(instagram),
            "channel_whatsapp": int(whatsapp), "channel_google": int(google), "channel_facebook": int(facebook),
            "ctr": ctr, "lead_conversion_rate": lcr, "cost_per_click": cpc, "cost_per_conversion": cpa
        }])

        # One-Hot Encode the single input row
        encoded = pd.get_dummies(raw_input, dtype=int)
        
        # Space-safe vector alignment loop
        aligned_data = {}
        for col in trained_columns:
            if col in encoded.columns:
                aligned_data[col] = float(encoded[col].iloc[0])
            else:
                base_col = col.split('_')[0] if "_" in col else col
                if base_col in raw_input.columns:
                    val_str = str(raw_input[base_col].iloc[0])
                    expected_dummy_name = f"{base_col}_{val_str}"
                    aligned_data[col] = 1.0 if expected_dummy_name == col else 0.0
                else:
                    aligned_data[col] = 0.0

        processed_df = pd.DataFrame([aligned_data]).astype(np.float64)

        # Generate Engine Forecasts
        predicted_revenue = np.expm1(reg_model.predict(processed_df)[0])
        prediction = clf_model.predict(processed_df)[0]
        probability = clf_model.predict_proba(processed_df)[0][1] * 100
        roi = ((predicted_revenue - acquisition_cost) / acquisition_cost * 100) if acquisition_cost > 0 else 0.0

        # UI Layout Representation
        st.success("Prediction Completed Successfully")
        m1, m2 = st.columns(2)
        m1.metric("Predicted Revenue", f"${predicted_revenue:,.2f}", f"{roi:.2f}% ROI")
        
        # Dynamic Confidence Assessment
        if prediction == 1:
            status = "🟢 PROFITABLE"
            display_conf = probability
        else:
            status = "🔴 LOSS"
            display_conf = 100.0 - probability
            
        m2.metric("Campaign Status", status, f"{display_conf:.2f}% Confidence")
        st.markdown("---")

        # Compact Visual Performance Section
        chart_df = pd.DataFrame({"Category": ["Acquisition Cost", "Predicted Revenue"], "Amount": [acquisition_cost, predicted_revenue]})
        chart = alt.Chart(chart_df).mark_bar(size=50).encode(
            x=alt.X("Category:N", axis=alt.Axis(labelAngle=0)), y="Amount:Q",
            color=alt.Color("Category:N", scale=alt.Scale(range=['#ff4b4b', '#29b5e8']))
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### 📈 Core Campaign Performance")
        p1, p2 = st.columns(2)
        p1.metric("Click-Through Rate (CTR)", f"{ctr*100:.2f}%")
        p1.metric("Lead Conversion Rate", f"{lcr*100:.2f}%")
        p2.metric("Cost Per Click (CPC)", f"${cpc:.2f}")
        p2.metric("Cost Per Conversion", f"${cpa:.2f}")

        # ==========================================================
        # 📊 STRATEGIC INSIGHTS ENGINE
        # ==========================================================
        st.markdown("---")
        st.markdown("## 🧠 Predictive Strategy & Insights")
        
        with st.expander("📈 1. Generate Insights on Campaign Performance", expanded=True):
            st.write(f"""
            * **Volume & Value Balance:** Operating at an absolute volume of `{impressions:,}` impressions yielded a projected top-line performance of **${predicted_revenue:,.2f}**.
            * **Conversion Efficiency:** Your current framework demonstrates a Click-Through Rate (CTR) of **{ctr*100:.2f}%** and a Lead-to-Conversion efficiency of **{lcr*100:.2f}%**.
            * **Unit Economic Health:** The strategy maps out an average Cost-Per-Click (CPC) of **${cpc:.2f}** against a Cost-Per-Conversion load of **${cpa:.2f}**. 
            """)
            
        with st.expander("🔑 2. Identify Key Factors Affecting Profitability", expanded=False):
            efficiency_ratio = (predicted_revenue / acquisition_cost) if acquisition_cost > 0 else 0
            cost_pressure_status = "⚠️ High CAC Risk" if efficiency_ratio < 1.5 else "✅ Optimized Unit Margin"
            
            st.write(f"""
            * **Categorical Drivers:** The selected combinations of **{campaign_type}** targeting **{target_audience}** are primary anchors shifting the probability baseline.
            * **Cost Elasticity ({cost_pressure_status}):** An acquisition cost of **${acquisition_cost:.2f}** vs. an engagement slider of **{engagement_score}/100** creates the definitive ceiling for your net variance margins.
            * **Segment Vulnerability:** Operating within the **{customer_segment}** demographic heavily weights the classifier engine's boundary toward a `{status}` outcome.
            """)
            
        with st.expander("💡 3. Data-Driven Recommendations for Marketing Strategies", expanded=False):
            if prediction == 0:
                rec_action = "Pivot channel focus or scale down unit acquisition costs immediately."
                rec_detail = f"The model flags this structure as high-risk. Consider reallocating budget from channels with weak organic lift to highly focused {language}-language programmatic campaigns."
            else:
                rec_action = "Aggressively scale budget footprints across these specific attributes."
                rec_detail = f"Current configurations indicate highly stable yield structures. Maximize ad-spend penetration across the {target_audience} segment while tracking early fatigue trends."

            st.write(f"""
            * **Immediate Strategic Action:** *{rec_action}*
            * **Target Optimization:** {rec_detail}
            * **Channel Allocation Recommendation:** Focus activation points heavily across chosen high-intent digital loops to actively dilute the baseline conversion cost.
            """)
            
        with st.expander("🤖 4. Support Decision-Making Using Model Predictions", expanded=False):
            st.write(f"""
            * **Statistical Confidence:** The machine learning engines have processed your parameters and assigned a **{display_conf:.2f}% stability rating** to this exact operational output.
            * **Capital Deployment Guardrail:** With a projected revenue outlook of **${predicted_revenue:,.2f}** against a capital allocation of **${acquisition_cost:.2f}**, the calculated path maps to a net **{roi:.2f}% ROI**.
            * **Go/No-Go Directive:** The data indicates an operational **{status}** outcome. Use this projection as a risk-mitigation benchmark before launching the actual live campaign asset tracking.
            """)
    else:
        st.info("Configure the campaign settings on the left and hit forecast to view model results.")