import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config for a premium look
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .churn-card {
        padding: 2rem;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
        text-align: center;
    }
    .risk-high {
        color: #ff4b4b;
        font-size: 3rem;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc96;
        font-size: 3rem;
        font-weight: bold;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# Load model pipeline
@st.cache_resource
def load_churn_model():
    model_path = os.path.join('models', 'churn_model.pkl')
    if not os.path.exists(model_path):
        # Check alternative path if run from workspace root
        model_path = os.path.join('Customer-Churn-Prediction', 'models', 'churn_model.pkl')
    return joblib.load(model_path)

try:
    model = load_churn_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}. Please make sure you have run the model training step first.")

st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("Analyze customer demographics, service choices, and billing status to predict the risk of churn.")

if model_loaded:
    # Organize fields in tabs or columns
    st.markdown("### Customer Information Input")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        
        st.markdown("#### Account Charges")
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0, step=1.0)
        # Default TotalCharges to tenure * monthly_charges
        total_charges_est = tenure * monthly_charges
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(total_charges_est), step=10.0)

    with col2:
        st.markdown("#### Services Signed Up")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security Service", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup Service", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection Service", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support Service", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("#### Media & Contract")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        predict_btn = st.button("Analyze Customer Churn Risk", use_container_width=True)

    if predict_btn:
        # Create dictionary of inputs
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Convert to DataFrame (matching format used during training)
        input_df = pd.DataFrame([input_data])
        
        # Predict probability
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.markdown('<div class="churn-card">', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("#### Risk Status")
                st.markdown('<span class="risk-high">HIGH RISK</span>', unsafe_allow_html=True)
                st.markdown(f"**Churn Probability:** `{prob * 100:.1f}%`**")
            else:
                st.markdown("#### Risk Status")
                st.markdown('<span class="risk-low">LOW RISK</span>', unsafe_allow_html=True)
                st.markdown(f"**Churn Probability:** `{prob * 100:.1f}%`**")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_res2:
            st.markdown("#### Key Risk Drivers & Recommendations")
            if prediction == 1:
                st.markdown("🚨 **Warning Indicators:**")
                if contract == "Month-to-month":
                    st.markdown("- **Month-to-month Contract**: High association with short-term churn. Consider offering incentives to switch to a 1-year or 2-year contract.")
                if tech_support == "No":
                    st.markdown("- **No Tech Support**: Customers without support churn at higher rates. Offer a free trial of premium tech support.")
                if internet_service == "Fiber optic":
                    st.markdown("- **Fiber Optic Service**: Fiber optic customers experience high churn rates (potentially due to price or satisfaction issues). Review quality/satisfaction.")
                st.markdown("💡 **Action Plan:** Proactively contact this customer with a retention discount or contract upgrade offer.")
            else:
                st.markdown("✅ **Stability Indicators:**")
                if contract != "Month-to-month":
                    st.markdown("- **Long-term Contract**: Stable relationship with locked-in contract duration.")
                if tenure > 24:
                    st.markdown(f"- **High Loyalty ({tenure} months)**: Established tenure reduces churn probability.")
                st.markdown("💡 **Action Plan:** Maintain regular engagement, suggest loyalty programs, or upsell premium services where appropriate.")
                
else:
    st.info("Please train the model using `python Customer-Churn-Prediction/src/train_model.py` first to enable predictions.")
