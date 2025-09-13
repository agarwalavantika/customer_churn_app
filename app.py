import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="",
    layout="wide"
)

# ----------------- CUSTOM STYLING -----------------
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2C3E50;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODEL -----------------
with open("churn_model_pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)

# ----------------- TITLE -----------------
st.title(" Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn and understand why.")

# ----------------- USER INPUT FORM -----------------
st.header(" Customer Information")

with st.expander(" Demographics"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with st.expander("Services Used"):
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streamtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streammovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with st.expander(" Contract & Billing"):
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ])

with st.expander(" Financials"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.35)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=845.5)

# ----------------- PREDICTION -----------------
input_data = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": onlinesec,
    "OnlineBackup": onlinebackup,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": streamtv,
    "StreamingMovies": streammovies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

if st.button("Predict Churn"):
    df = pd.DataFrame([input_data])
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    # ----------------- RESULT -----------------
    st.header(" Prediction Result")
    st.write("Churn: **Yes**" if prediction == 1 else "Churn: **No**")
    st.write("Probability:", round(probability, 2))

    # Risk message
    if probability < 0.3:
        risk_text = "Low Risk"
        color = "green"
        st.success("This customer is unlikely to churn ")
    elif probability < 0.6:
        risk_text = "Medium Risk"
        color = "orange"
        st.warning(" This customer may churn. Keep an eye on them ")
    else:
        risk_text = "High Risk"
        color = "red"
        st.error("High risk of churn! Immediate retention strategy needed.")

    # ----------------- RISK GAUGE -----------------
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            number={'valueformat': '.2f'},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.6], 'color': "lightyellow"},
                    {'range': [0.6, 1], 'color': "lightcoral"},
                ],
            },
            title={'text': f"Churn Risk: {risk_text}"}
        ))
        st.plotly_chart(fig, use_container_width=True)

    # ----------------- SHAP EXPLANATION -----------------
    with col2:
        st.subheader(" Why this prediction?")
        model = pipeline.named_steps["classifier"]    
        preprocessor = pipeline.named_steps["preprocessor"]

        # Transform input
        feature_names = preprocessor.get_feature_names_out()
        df_processed = pd.DataFrame(preprocessor.transform(df), columns=feature_names)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_processed)

        # Top 3 features explanation
        top_features = pd.Series(shap_values[0], index=feature_names).sort_values(
            key=abs, ascending=False
        )[:3]

        for feat, val in top_features.items():
            if val > 0:
                st.write(f"**{feat.replace('cat__','').replace('num__','')}** increases churn risk.")
            else:
                st.write(f" **{feat.replace('cat__','').replace('num__','')}** helps retain this customer.")

    # ----------------- ADVANCED SHAP PLOT -----------------
    with st.expander(" Advanced Explanation (SHAP Plot)"):
        plt.figure(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            feature_names=feature_names,
            max_display=10,
            show=False
        )
        st.pyplot(plt)
