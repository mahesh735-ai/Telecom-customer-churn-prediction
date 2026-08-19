"""
Telecom Customer Churn Prediction App
Author: Mahesh Thakare

Takes customer details as input, recreates the same cleaning + feature
engineering steps used in the training notebook, and predicts churn
probability + risk segment using the tuned XGBoost model.
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="centered")

# ----------------------------------------------------
# Theme - Dark Navy / Coral (same family as my Heart Disease app,
# just tuned a bit for a telecom/product feel)
# ----------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1C2F;
        color: #F5F5F5;
    }

    /* top banner */
    .app-header {
        background: linear-gradient(135deg, #13253F 0%, #0E1C2F 100%);
        border: 1px solid #1F3350;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 22px;
    }
    .app-header h1 {
        color: #FF6B5B;
        margin: 0;
        font-size: 28px;
    }
    .app-header p {
        color: #B8C4D9;
        margin: 6px 0 0 0;
        font-size: 15px;
    }

    h2, h3, .stSubheader {
        color: #FF6B5B;
    }

    /* section cards for the form */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #13253F;
        border-radius: 10px;
        border: 1px solid #1F3350;
    }

    div.stButton > button {
        background-color: #FF6B5B;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.6em 2em;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #e85a4a;
    }

    /* result card */
    .result-card {
        background-color: #13253F;
        border-radius: 12px;
        border: 1px solid #1F3350;
        padding: 20px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Trained Model + Column List
# (saved at the end of the ML notebook using joblib)
# ----------------------------------------------------
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.markdown("""
    <div class="app-header">
        <h1>📡 Telecom Customer Churn Predictor</h1>
        <p>Enter a customer's account and service details to estimate their risk of churning,
        and get a recommended risk category for the retention team.</p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Input Form
# Grouped the same way I grouped features during EDA - account/billing,
# demographics, then services - just easier to fill out this way
# ----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Account & Billing")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges (₹)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges (₹)", 0.0, 10000.0, 1000.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

with col2:
    st.subheader("Phone & Internet")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    st.subheader("Add-on Services")
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("---")
predict_clicked = st.button("Predict Churn Risk")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
if predict_clicked:

    # Step 1: put the raw inputs into a single-row dataframe, using the
    # same column names and category text as the original dataset
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
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
        'TotalCharges': total_charges,
    }
    input_df = pd.DataFrame([input_data])

    # Step 2: rebuild the same engineered features from the ML notebook
    # (tenure_group, total_services, has_streaming)
    def get_tenure_group(t):
        if t <= 6:
            return '0-6_Months'
        elif t <= 12:
            return '6-12_Months'
        elif t <= 24:
            return '1-2_Years'
        elif t <= 48:
            return '2-4_Years'
        else:
            return '4+_Years'

    input_df['tenure_group'] = get_tenure_group(tenure)

    addon_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_df['total_services'] = (input_df[addon_services] == 'Yes').sum(axis=1)

    input_df['has_streaming'] = int(
        (streaming_tv == 'Yes') or (streaming_movies == 'Yes')
    )

    # Step 3: one-hot encode this single row the same way training data was encoded
    input_encoded = pd.get_dummies(input_df, drop_first=True, dtype=int)

    # Step 4: rename a few columns to match what the notebook renamed them to
    input_encoded.rename(columns={
        'gender_Male': 'is_male',
        'SeniorCitizen_Yes': 'is_senior_citizen',
        'Partner_Yes': 'has_partner',
        'Dependents_Yes': 'has_dependents',
        'PaperlessBilling_Yes': 'has_paperless_billing',
        'PhoneService_Yes': 'has_phone_service'
    }, inplace=True)

    # Step 5: align columns with what the model actually expects - any
    # column the model saw in training but that's missing here (because that
    # category wasn't picked for this customer) just gets filled with 0
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Step 6: predict churn probability (no scaling needed, final model is XGBoost)
    churn_probability = model.predict_proba(input_encoded)[0][1]
    churn_pct = round(churn_probability * 100, 2)

    # Step 7: convert probability into a risk segment, same bins as the notebook
    if churn_probability < 0.30:
        risk_segment = "Low Risk"
        color = "#4CAF50"
    elif churn_probability < 0.50:
        risk_segment = "Medium Risk"
        color = "#FFC107"
    elif churn_probability < 0.70:
        risk_segment = "High Risk"
        color = "#FF6B5B"
    else:
        risk_segment = "Critical Risk"
        color = "#D32F2F"

    # ----------------------------------------------------
    # Result
    # ----------------------------------------------------
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### Prediction Result")

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Churn Probability", f"{churn_pct}%")
    with result_col2:
        st.markdown(
            f"<h3 style='color:{color};'>{risk_segment}</h3>",
            unsafe_allow_html=True
        )

    st.progress(min(int(churn_pct), 100))

    if risk_segment in ["High Risk", "Critical Risk"]:
        st.warning("This customer needs immediate retention action.")
    else:
        st.success("This customer is likely to stay.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built by Mahesh Thakare | Model: Tuned XGBoost (Recall-optimized) | Dataset: IBM Telco Customer Churn")