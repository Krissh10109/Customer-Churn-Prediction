import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Enterprise Churn Analytics | SaiKet Systems",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding-left: 20px;
        padding-right: 20px;
        color: #ffffff;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
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
        font-size: 2.8rem;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc96;
        font-size: 2.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model pipeline
@st.cache_resource
def load_churn_model():
    paths = [
        os.path.join('models', 'churn_model.pkl'),
        os.path.join('Customer-Churn-Prediction', 'models', 'churn_model.pkl')
    ]
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError("Model file not found.")

@st.cache_data
def load_dataset():
    paths = [
        os.path.join('data', 'Telco_Customer_Churn_Dataset.csv'),
        os.path.join('Customer-Churn-Prediction', 'data', 'Telco_Customer_Churn_Dataset.csv')
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

try:
    model = load_churn_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

dataset = load_dataset()

# Title Header
st.title("⚡ Enterprise Customer Churn Platform")
st.caption("Powered by Machine Learning & Predictive Analytics — SaiKet Systems Task Solution")

if model_loaded:
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Single Customer Predictor",
        "📁 Batch CSV Predictor",
        "📊 Executive Cohort Analytics",
        "🎛️ What-If Simulator"
    ])

    # TAB 1: Single Customer Predictor
    with tab1:
        st.markdown("### Input Customer Profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            
            st.markdown("#### Charges & Tenure")
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)

        with c2:
            st.markdown("#### Phone & Internet Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        with c3:
            st.markdown("#### Streaming & Contract")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("Run Risk Analysis", use_container_width=True, type="primary")

        if predict_btn:
            input_df = pd.DataFrame([{
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
            }])

            prob = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]

            st.markdown("---")
            res_c1, res_c2 = st.columns([1, 2])
            
            with res_c1:
                st.markdown('<div class="churn-card">', unsafe_allow_html=True)
                if pred == 1:
                    st.markdown("#### Predicted Status")
                    st.markdown('<span class="risk-high">HIGH RISK</span>', unsafe_allow_html=True)
                    st.markdown(f"**Churn Probability:** `{prob * 100:.1f}%`")
                else:
                    st.markdown("#### Predicted Status")
                    st.markdown('<span class="risk-low">LOW RISK</span>', unsafe_allow_html=True)
                    st.markdown(f"**Churn Probability:** `{prob * 100:.1f}%`")
                st.markdown('</div>', unsafe_allow_html=True)

            with res_c2:
                st.markdown("#### Strategic Retention Insights")
                if pred == 1:
                    st.warning("⚠️ **Risk Factors Identified:**")
                    if contract == "Month-to-month":
                        st.markdown("- Short-term Month-to-Month contract.")
                    if internet_service == "Fiber optic":
                        st.markdown("- Fiber Optic plan without security add-ons.")
                    if tech_support == "No":
                        st.markdown("- Lack of Tech Support active on account.")
                    st.info("💡 **Recommended Action:** Offer 12-month contract lock-in with 15% discount or bundled tech support.")
                else:
                    st.success("✅ **Customer Profile Healthy:**")
                    st.markdown("- High retention probability. Suitable for premium upsell offers.")

    # TAB 2: Batch CSV Predictor
    with tab2:
        st.markdown("### Batch Inference Engine")
        st.markdown("Upload a customer CSV dataset to process predictions across thousands of customer records at once.")

        uploaded_file = st.file_uploader("Upload Telco CSV File", type=["csv"])
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"Loaded `{len(batch_data)}` customer records.")
            
            if st.button("Process Batch Predictions", type="primary"):
                # Clean TotalCharges for batch input
                infer_df = batch_data.copy()
                if 'customerID' in infer_df.columns:
                    cust_ids = infer_df['customerID']
                    infer_df = infer_df.drop(columns=['customerID'])
                else:
                    cust_ids = infer_df.index
                    
                if 'Churn' in infer_df.columns:
                    infer_df = infer_df.drop(columns=['Churn'])

                infer_df['TotalCharges'] = pd.to_numeric(infer_df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0.0)

                probs = model.predict_proba(infer_df)[:, 1]
                preds = model.predict(infer_df)

                results_df = batch_data.copy()
                results_df['Churn_Probability'] = np.round(probs * 100, 2)
                results_df['Predicted_Churn'] = np.where(preds == 1, 'Yes (High Risk)', 'No (Low Risk)')

                st.success("Batch predictions completed!")
                st.dataframe(results_df.head(20))

                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Predictions CSV",
                    data=csv_data,
                    file_name="churn_predictions_annotated.csv",
                    mime="text/csv"
                )

    # TAB 3: Executive Cohort Analytics
    with tab3:
        st.markdown("### Executive Cohort Insights & Churn Drivers")
        if dataset is not None:
            df_viz = dataset.copy()
            df_viz['TotalCharges'] = pd.to_numeric(df_viz['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0.0)
            
            m1, m2, m3, m4 = st.columns(4)
            churn_rate = (df_viz['Churn'].value_counts(normalize=True).get('Yes', 0)) * 100
            m1.metric("Total Customers", f"{len(df_viz):,}")
            m2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
            m3.metric("Avg Tenure", f"{df_viz['tenure'].mean():.1f} mos")
            m4.metric("Avg Monthly Spend", f"${df_viz['MonthlyCharges'].mean():.2f}")

            st.markdown("---")
            g1, g2 = st.columns(2)

            with g1:
                st.markdown("#### Contract Type vs Churn")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Contract', hue='Churn', data=df_viz, palette='Set2', ax=ax)
                plt.title("Churn Rate by Contract Type")
                st.pyplot(fig)

            with g2:
                st.markdown("#### Internet Service Type vs Churn")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='InternetService', hue='Churn', data=df_viz, palette='Set2', ax=ax)
                plt.title("Churn Rate by Internet Service Provider")
                st.pyplot(fig)

            st.markdown("#### Tenure Distribution by Churn Status")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.kdeplot(data=df_viz, x='tenure', hue='Churn', common_norm=False, fill=True, palette='Set2', ax=ax)
            plt.title("Customer Tenure Kernel Density Distribution")
            st.pyplot(fig)
        else:
            st.warning("Dataset not found to render executive charts.")

    # TAB 4: What-If Simulator
    with tab4:
        st.markdown("### 🎛️ What-If Retention Simulator")
        st.markdown("Select a baseline profile and toggle retention levers to instantly see the impact on churn probability.")
        
        sim_c1, sim_c2 = st.columns([1, 1])
        with sim_c1:
            st.markdown("#### Base Profile")
            base_tenure = st.slider("Simulated Tenure (Months)", min_value=1, max_value=72, value=3)
            base_monthly = st.number_input("Base Monthly Charges ($)", min_value=10.0, max_value=150.0, value=75.0, step=1.0)
            base_internet = st.selectbox("Base Internet", ["Fiber optic", "DSL", "No"])
            base_contract = st.selectbox("Base Contract", ["Month-to-month", "One year", "Two year"])
        
        with sim_c2:
            st.markdown("#### Retention Levers")
            lever_tech_support = st.checkbox("Add Tech Support?", value=False)
            lever_online_security = st.checkbox("Add Online Security?", value=False)
            lever_contract = st.selectbox("Upgrade Contract To:", ["Same as Base", "One year", "Two year"])
            
        st.markdown("---")
        
        def build_sim_df(tenure, monthly, internet, contract, has_tech, has_sec):
            return pd.DataFrame([{
                'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
                'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
                'InternetService': internet,
                'OnlineSecurity': 'Yes' if has_sec else 'No' if internet != 'No' else 'No internet service',
                'OnlineBackup': 'No', 'DeviceProtection': 'No',
                'TechSupport': 'Yes' if has_tech else 'No' if internet != 'No' else 'No internet service',
                'StreamingTV': 'No', 'StreamingMovies': 'No',
                'Contract': contract, 'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': monthly,
                'TotalCharges': tenure * monthly
            }])

        df_base = build_sim_df(base_tenure, base_monthly, base_internet, base_contract, False, False)
        base_prob = model.predict_proba(df_base)[0][1] * 100
        
        new_contract = lever_contract if lever_contract != "Same as Base" else base_contract
        # Estimate new monthly (add $5 for tech support, $5 for security, -$10 for long term contract if changed)
        new_monthly = base_monthly
        if lever_tech_support: new_monthly += 5
        if lever_online_security: new_monthly += 5
        if lever_contract != "Same as Base": new_monthly -= 10
        
        df_new = build_sim_df(base_tenure, max(10, new_monthly), base_internet, new_contract, lever_tech_support, lever_online_security)
        new_prob = model.predict_proba(df_new)[0][1] * 100
        
        delta = new_prob - base_prob
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Base Churn Probability", f"{base_prob:.1f}%")
        res_col2.metric("New Churn Probability", f"{new_prob:.1f}%", f"{delta:.1f}%", delta_color="inverse")
        
        if delta < -5:
            res_col3.success("✅ Highly effective retention strategy!")
        elif delta > 0:
            res_col3.error("⚠️ Changes increased churn risk!")
        else:
            res_col3.info("ℹ️ Strategy has minor impact.")
