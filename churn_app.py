import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { color: #64748B; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .risk-card { border-radius: 12px; padding: 22px; margin-top: 15px; border: 1px solid #E2E8F0; }
    .high-risk { background-color: #FEF2F2; border-left: 6px solid #DC2626; color: #991B1B; }
    .low-risk { background-color: #ECFDF5; border-left: 6px solid #059669; color: #065F46; }
    .metric-value { font-size: 2.2rem; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Load Pipeline Safely ---
@st.cache_resource
def load_churn_pipeline():
    # Resolve absolute path to saved_models relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "saved_models", "customer_churn_pipeline.pkl")
    
    if not os.path.exists(model_path):
        # Fallback to current working directory
        model_path = "saved_models/customer_churn_pipeline.pkl"
        if not os.path.exists(model_path):
            return None
    return joblib.load(model_path)

pipeline = load_churn_pipeline()

# --- 3. App Header ---
st.markdown('<div class="main-title"> Bank Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered decision support system to detect customer attrition risk and recommend retention actions.</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error(" Model file not found! Please ensure `customer_churn_pipeline.pkl` is inside `saved_models/`.")
    st.stop()

# --- 4. User Inputs (Form Layout) ---
st.sidebar.header("Customer Profile Attributes")

geography = st.sidebar.selectbox("Country / Geography", ["France", "Spain", "Germany"], index=0)
gender = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
age = st.sidebar.slider("Age", min_value=18, max_value=95, value=38, step=1)
tenure = st.sidebar.slider("Tenure with Bank (Years)", min_value=0, max_value=10, value=5, step=1)
credit_score = st.sidebar.slider("Credit Score", min_value=350, max_value=850, value=650, step=1)

balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
estimated_salary = st.sidebar.number_input("Estimated Annual Salary ($)", min_value=1000.0, max_value=300000.0, value=75000.0, step=1000.0)
num_of_products = st.sidebar.selectbox("Number of Bank Products Held", [1, 2, 3, 4], index=1)

has_cr_card = st.sidebar.selectbox("Has Active Credit Card?", ["Yes", "No"], index=0)
is_active_member = st.sidebar.selectbox("Is Active Banking Member?", ["Yes", "No"], index=0)

# --- 5. Data Assembly ---
input_df = pd.DataFrame([{
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1.0 if has_cr_card == "Yes" else 0.0,
    'IsActiveMember': 1.0 if is_active_member == "Yes" else 0.0,
    'EstimatedSalary': estimated_salary
}])

# --- 6. Prediction & Analytics ---
col_profile, col_results = st.columns([1, 1.2], gap="large")

with col_profile:
    st.subheader(" Customer Snapshot")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

with col_results:
    st.subheader(" Churn Risk Assessment")
    
    if st.button(" Assess Customer Churn Risk", type="primary", use_container_width=True):
        # Generate predictions
        churn_pred = pipeline.predict(input_df)[0]
        churn_proba = pipeline.predict_proba(input_df)[0][1] * 100
        
        # Display Gauge & Metrics
        st.metric(label="Predicted Churn Probability", value=f"{churn_proba:.1f}%")
        st.progress(min(int(churn_proba), 100))
        
        if churn_pred == 1 or churn_proba >= 50.0:
            st.markdown(f"""
                <div class="risk-card high-risk">
                    <h3> High Churn Risk Detected</h3>
                    <p>The model predicts this customer is <b>likely to close their account or leave the bank</b>.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("####  Recommended Retention Strategies:")
            recommendations = []
            if num_of_products >= 3:
                recommendations.append("• **Review Product Complexity/Fees:** Customer holds 3+ products; investigate fee dissatisfaction or account consolidation.")
            if is_active_member == "No":
                recommendations.append("• **Re-engagement Campaign:** Reach out with personalized advisory services or digital onboarding incentives.")
            if age >= 45:
                recommendations.append("• **Senior Banking Relationship:** Offer premium wealth or retirement deposit options suited for mature demographics.")
            if geography == "Germany":
                recommendations.append("• **Regional Review:** Review competitive fee pressures common in the German banking market.")
            if not recommendations:
                recommendations.append("• Offer a relationship manager consultation and preferential deposit/loan interest rates.")
            
            for rec in recommendations:
                st.write(rec)
                
        else:
            st.markdown(f"""
                <div class="risk-card low-risk">
                    <h3> Low Churn Risk (Loyal Customer)</h3>
                    <p>The model predicts this customer is <b>satisfied and likely to remain with the bank</b>.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("####  Growth Opportunities:")
            st.write("• Customer is in good standing. Suitable candidate for loyalty rewards, premium credit cards, or investment products.")
