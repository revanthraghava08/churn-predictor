import json
import joblib
import pandas as pd
import streamlit as st
from train import add_features

st.set_page_config(page_title="Churn Predictor", page_icon="📉")

@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

st.title("Customer Churn Predictor")
st.caption("Logistic regression on the Telco churn dataset. Enter a customer to get a churn probability.")

st.sidebar.header("Decision settings")
cost_missed = st.sidebar.number_input("Cost of a missed churner", min_value=1, value=500, step=50)
cost_alarm = st.sidebar.number_input("Cost of a wasted retention offer", min_value=1, value=50, step=10)
breakeven = cost_alarm / (cost_alarm + cost_missed)
st.sidebar.write("Break-even threshold from these costs:", round(breakeven, 3))
threshold = st.sidebar.slider("Flag as likely churner if probability is at least",
                              min_value=0.01, max_value=0.90,
                              value=float(meta["threshold"]), step=0.01)

def dependent_select(label, options, locked, locked_value):
    """Show a dropdown, or a disabled one with a fixed value when another answer forces it."""
    if locked:
        st.selectbox(label, [locked_value], disabled=True)
        return locked_value
    return st.selectbox(label, options)

yn = ["No", "Yes"]
c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
    partner = st.selectbox("Partner", yn)
    dependents = st.selectbox("Dependents", yn)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", yn, index=1)
    lines = dependent_select("Multiple lines", yn, phone == "No", "No phone service")
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with c2:
    security = dependent_select("Online security", yn, internet == "No", "No internet service")
    backup = dependent_select("Online backup", yn, internet == "No", "No internet service")
    protection = dependent_select("Device protection", yn, internet == "No", "No internet service")
    support = dependent_select("Tech support", yn, internet == "No", "No internet service")
    tv = dependent_select("Streaming TV", yn, internet == "No", "No internet service")
    movies = dependent_select("Streaming movies", yn, internet == "No", "No internet service")
    paperless = st.selectbox("Paperless billing", yn)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

total = st.number_input("Total charges", min_value=0.0, max_value=20000.0, value=float(tenure * monthly), step=10.0)
st.caption(
    "Suggested estimate: tenure × monthly charges. "
    "You can adjust this to the customer's actual total charges."
    + str(round(total, 2))
)

row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
    "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
    "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
}])
row = add_features(row).reindex(columns=meta["input_columns"])

proba = float(model.predict_proba(row)[0, 1])

st.divider()
st.metric("Churn probability", str(round(proba * 100, 1)) + "%")
st.progress(min(max(proba, 0.0), 1.0))
if proba >= threshold:
    st.error("Likely churner at this threshold: consider a retention offer.")
else:
    st.success("Not flagged at this threshold.")
st.caption("Threshold in use: " + str(round(threshold, 2)))