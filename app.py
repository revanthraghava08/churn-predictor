from __future__ import annotations

import json
import textwrap
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train import add_features


# ============================================================
# CHURN.AI — Futuristic customer churn intelligence dashboard
# Keeps the existing trained model, feature engineering,
# feature names, probability calculation, and metadata threshold.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
META_PATH = BASE_DIR / "model_meta.json"

st.set_page_config(
    page_title="CHURN.AI | Customer Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------- Theme -----------------------------

st.markdown(
    textwrap.dedent(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg: #0A0E1A;
            --surface: #0F1424;
            --surface-elevated: #141B2E;
            --surface-hover: #1A2340;
            --border-subtle: rgba(71, 85, 105, .18);
            --border-strong: rgba(100, 116, 139, .28);
            --primary: #0EA5E9;
            --primary-hover: #0284C7;
            --accent-purple: #A78BFA;
            --accent-cyan: #06B6D4;
            --accent-green: #10B981;
            --accent-red: #EF4444;
            --text-primary: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-muted: #94A3B8;
            --text-disabled: #64748B;
        }

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        .stApp {
            color: var(--text-primary);
            background:
                radial-gradient(ellipse at 8% 12%, rgba(14, 165, 233, .08), transparent 40%),
                radial-gradient(ellipse at 95% 85%, rgba(139, 92, 246, .06), transparent 45%),
                radial-gradient(ellipse at 50% 50%, rgba(6, 182, 212, .03), transparent 60%),
                linear-gradient(135deg, #0A0E1A 0%, #0F1424 40%, #0A0E1A 100%);
        }

        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        .block-container {
            max-width: 1480px;
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F1424 0%, #0A0E1A 100%);
            border-right: 1px solid var(--border-subtle);
        }
        section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -.5px;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border-strong);
            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 24px;
            background:
                radial-gradient(circle at 85% 20%, rgba(6, 182, 212, .12), transparent 28%),
                radial-gradient(circle at 15% 90%, rgba(139, 92, 246, .08), transparent 35%),
                linear-gradient(125deg, rgba(14, 165, 233, .08) 0%, var(--surface-elevated) 52%, var(--surface) 100%);
            box-shadow: 0 20px 60px rgba(0, 0, 0, .35), inset 0 1px 1px rgba(255, 255, 255, .04);
            backdrop-filter: blur(8px);
        }
        .hero:after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            right: 10%;
            top: -120px;
            border: 1px solid rgba(6, 182, 212, .15);
            border-radius: 50%;
            box-shadow: 0 0 0 20px rgba(6, 182, 212, .02),
                        0 0 0 40px rgba(139, 92, 246, .02);
            pointer-events: none;
        }
        .brand {
            font: 800 38px/1.0 'Space Grotesk', sans-serif;
            letter-spacing: -1.8px;
            color: var(--text-primary);
            background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .brand span {
            background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-sub {
            color: #CBD5E1;
            font-size: 12px;
            letter-spacing: 0.8px;
            margin-top: 10px;
            font-weight: 500;
        }
        .online-pill {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 20px;
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, .35);
            background: rgba(16, 185, 129, .08);
            font: 600 11px 'DM Mono', monospace;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .eyebrow {
            color: #0EA5E9;
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .section-heading {
            color: #F8FAFC;
            font: 700 22px 'Space Grotesk', sans-serif;
            margin: 20px 0 6px 0;
            letter-spacing: -0.3px;
        }
        .section-note {
            color: #CBD5E1;
            font-size: 13px;
            margin-bottom: 16px;
            line-height: 1.5;
        }

        .glass {
            background: linear-gradient(145deg, rgba(20, 27, 46, .92), rgba(15, 20, 36, .95));
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, .25), inset 0 1px 1px rgba(255, 255, 255, .02);
            backdrop-filter: blur(4px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }
        .glass:hover {
            background: linear-gradient(145deg, rgba(25, 36, 64, .98), rgba(18, 26, 48, 1));
            border-color: rgba(6, 182, 212, .7) !important;
            box-shadow: 0 16px 48px rgba(0, 0, 0, .4), 0 0 32px rgba(6, 182, 212, .4), inset 0 1px 2px rgba(6, 182, 212, .3) !important;
            transform: translateY(-3px);
        }
        .metric-label {
            color: #94A3B8;
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1.2px;
            text-transform: uppercase;
        }
        .metric-value {
            color: #F8FAFC;
            font: 700 28px 'Space Grotesk', sans-serif;
            margin-top: 8px;
            letter-spacing: -0.5px;
        }
        .metric-foot {
            color: #64748B;
            font-size: 11px;
            margin-top: 6px;
        }

        .field-card {
            border: 1px solid var(--border-subtle);
            border-radius: 13px;
            padding: 14px 16px;
            margin: 8px 0 13px 0;
            background: linear-gradient(120deg, rgba(14, 165, 233, .06), rgba(6, 182, 212, .03));
        }
        .field-card span {
            color: #0EA5E9;
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .result-panel {
            border: 1px solid var(--border-strong);
            border-radius: 20px;
            padding: 28px;
            background:
                radial-gradient(circle at 95% 5%, rgba(14, 165, 233, .12), transparent 40%),
                linear-gradient(145deg, var(--surface-elevated), var(--surface));
            box-shadow: 0 18px 50px rgba(0, 0, 0, .3), inset 0 1px 1px rgba(255, 255, 255, .03);
            backdrop-filter: blur(6px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .result-panel:hover {
            border-color: rgba(14, 165, 233, .4);
            box-shadow: 0 24px 60px rgba(0, 0, 0, .35), 0 0 30px rgba(14, 165, 233, .12);
        }
        .result-kicker {
            color: #94A3B8;
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
        .probability {
            color: var(--text-primary);
            font: 700 56px/.96 'Space Grotesk', sans-serif;
            letter-spacing: -1.5px;
            margin: 14px 0 8px 0;
        }
        .probability span { color: var(--accent-cyan); }
        .decision {
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 18px;
            letter-spacing: 0.3px;
        }
        .decision-yes {
            color: #FCA5A5;
            border: 1px solid rgba(239, 68, 68, .25);
            background: rgba(239, 68, 68, .06);
        }
        .decision-no {
            color: #86EFAC;
            border: 1px solid rgba(16, 185, 129, .28);
            background: rgba(16, 185, 129, .06);
        }
        .scale {
            display: flex;
            height: 12px;
            overflow: hidden;
            border-radius: 20px;
            margin-top: 20px;
            background: #1A2340;
            position: relative;
            border: 1px solid var(--border-subtle);
        }
        .scale-fill {
            height: 100%;
            border-radius: 20px;
            background: linear-gradient(90deg, #0EA5E9, #06B6D4, #10B981);
            box-shadow: 0 0 16px rgba(14, 165, 233, .4), inset 0 1px 1px rgba(255, 255, 255, .1);
        }
        .threshold-marker {
            position: absolute;
            top: -4px;
            bottom: -4px;
            width: 2.5px;
            background: #F8FAFC;
            border-radius: 1px;
            box-shadow: 0 0 10px rgba(248, 250, 252, .8);
            z-index: 2;
        }
        .scale-legend {
            display: flex;
            justify-content: space-between;
            color: #64748B;
            font: 10px 'DM Mono', monospace;
            margin-top: 8px;
            letter-spacing: 0.3px;
        }
        .detail-line {
            display: flex;
            justify-content: space-between;
            gap: 14px;
            padding: 11px 0;
            border-bottom: 1px solid var(--border-subtle);
            font-size: 13px;
        }
        .detail-line:last-child { border-bottom: 0; }
        .detail-line .key { color: #94A3B8; font-weight: 500; }
        .detail-line .value { color: #F8FAFC; font-weight: 600; text-align: right; }

        .footer {
            color: #475569;
            text-align: center;
            font: 10px 'DM Mono', monospace;
            letter-spacing: 0.5px;
            padding: 32px 0 8px 0;
            margin-top: 24px;
            border-top: 1px solid var(--border-subtle);
        }

        /* Streamlit widgets */
        .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #F8FAFC !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            letter-spacing: 0.2px !important;
        }
        div[data-baseweb="select"] > div {
            background: var(--surface-elevated) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 12px !important;
        }
        div[data-baseweb="select"] span { color: #F8FAFC !important; font-size: 13px !important; }
        div[data-testid="stNumberInput"] input {
            background: var(--surface-elevated) !important;
            color: #F8FAFC !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 12px !important;
            font-size: 13px !important;
        }
        div[data-testid="stNumberInput"] input:focus {
            border-color: #0EA5E9 !important;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, .1) !important;
        }
        .stSlider > div > div > div { background: var(--surface-elevated) !important; }
        .stButton > button {
            min-height: 48px;
            color: #fff;
            font-weight: 700;
            letter-spacing: 0.3px;
            border: 1px solid rgba(14, 165, 233, .4);
            border-radius: 12px;
            background: linear-gradient(100deg, #0EA5E9, #06B6D4 58%, #0284C7);
            box-shadow: 0 8px 28px rgba(14, 165, 233, .18), inset 0 1px 1px rgba(255, 255, 255, .1);
            transition: .2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .stButton > button:hover {
            color: #fff;
            border-color: #06B6D4;
            box-shadow: 0 12px 36px rgba(14, 165, 233, .28), inset 0 1px 1px rgba(255, 255, 255, .15);
            transform: translateY(-2px);
        }
        .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 4px 16px rgba(14, 165, 233, .18), inset 0 1px 1px rgba(255, 255, 255, .1);
        }
        [data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid var(--border-subtle);
            background: var(--surface-elevated) !important;
        }
        hr { border-color: var(--border-subtle); }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ----------------------------- Model loading -----------------------------

@st.cache_resource
def load_artifacts(model_path: str, meta_path: str):
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)
    return model, meta


if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        "Model files are missing. Place model.joblib and model_meta.json "
        "in the same project folder as app.py."
    )
    st.stop()

try:
    model, meta = load_artifacts(str(MODEL_PATH), str(META_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model artifacts: {exc}")
    st.stop()

if "threshold" not in meta or "input_columns" not in meta:
    st.error("model_meta.json must contain 'threshold' and 'input_columns'.")
    st.stop()

try:
    default_threshold = float(meta["threshold"])
except (TypeError, ValueError):
    st.error("The threshold in model_meta.json must be numeric.")
    st.stop()

if not 0.0 <= default_threshold <= 1.0:
    st.error("The saved threshold must be between 0 and 1.")
    st.stop()


# ----------------------------- Hero -----------------------------

st.markdown(
    textwrap.dedent(
        """
        <div class="hero">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;">
                <div>
                    <div class="brand">CHURN<span>.AI</span></div>
                    <div class="hero-sub">CUSTOMER INTELLIGENCE / PREDICTIVE RETENTION PLATFORM</div>
                </div>
                <div class="online-pill">● MODEL READY</div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# ----------------------------- Sidebar -----------------------------

with st.sidebar:
    st.markdown("## ◈ Decision Engine")
    st.caption("Tune the decision rule using your business cost assumptions.")

    st.markdown("### Business cost assumptions")
    cost_missed = st.number_input(
        "Cost of a missed churner",
        min_value=1,
        value=500,
        step=50,
        key="cost_missed",
    )
    cost_alarm = st.number_input(
        "Cost of a wasted retention offer",
        min_value=1,
        value=50,
        step=10,
        key="cost_alarm",
    )

    break_even = cost_alarm / (cost_alarm + cost_missed)

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="glass">
                <div class="metric-label">Break-even reference</div>
                <div class="metric-value" style="color:#22D3EE;">{break_even:.3f}</div>
                <div class="metric-foot">Cost-based reference only; it does not change the selected threshold.</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Classification threshold")
    threshold = st.slider(
        "Flag as likely churner at or above",
        min_value=0.01,
        max_value=0.90,
        value=default_threshold,
        step=0.01,
        key="decision_threshold",
    )

    st.caption(f"Active threshold: **{threshold:.2f}**")
    st.divider()

    st.markdown("### Model information")
    st.markdown(
        textwrap.dedent(
            """
            <div class="info">
                <b style="color:#C4B5FD;">Algorithm</b><br>Logistic Regression<br><br>
                <b style="color:#C4B5FD;">Dataset</b><br>IBM Telco Customer Churn<br><br>
                <b style="color:#C4B5FD;">Task</b><br>Binary classification<br><br>
                <b style="color:#C4B5FD;">Output</b><br>Predicted churn probability
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ----------------------------- Summary strip -----------------------------

summary_1, summary_2, summary_3 = st.columns(3)

for column, label, value, foot, accent in [
    (summary_1, "MODEL", "LOGISTIC", "Regression classifier", "#A78BFA"),
    (summary_2, "THRESHOLD", f"{threshold:.2f}", "Current decision cutoff", "#22D3EE"),
    (summary_3, "ENGINE", "READY", "Saved model loaded", "#34D399"),
]:
    with column:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="glass">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{accent};">{value}</div>
                    <div class="metric-foot">{foot}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )


# ----------------------------- Customer inputs -----------------------------

st.markdown(
    textwrap.dedent(
        """
        <div class="section-heading">◈ Customer intelligence</div>
        <div class="section-note">Enter the customer's profile, services, subscription and billing details.</div>
        """
    ),
    unsafe_allow_html=True,
)

yes_no = ["No", "Yes"]


def dependent_select(label: str, options: list[str], locked: bool, locked_value: str, key: str):
    if locked:
        st.selectbox(label, [locked_value], disabled=True, key=key)
        return locked_value
    return st.selectbox(label, options, key=key)


left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        '<div class="field-card"><span>01 / PERSONAL PROFILE</span></div>',
        unsafe_allow_html=True,
    )
    gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
    senior = st.selectbox(
        "Senior citizen",
        [0, 1],
        format_func=lambda value: "Yes" if value else "No",
        key="senior",
    )
    partner = st.selectbox("Partner", yes_no, key="partner")
    dependents = st.selectbox("Dependents", yes_no, key="dependents")

    st.markdown(
        '<div class="field-card"><span>02 / SUBSCRIPTION PROFILE</span></div>',
        unsafe_allow_html=True,
    )
    tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
        key="contract",
    )
    paperless = st.selectbox("Paperless billing", yes_no, key="paperless")
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        key="payment",
    )

with right:
    st.markdown(
        '<div class="field-card"><span>03 / CONNECTIVITY & SERVICES</span></div>',
        unsafe_allow_html=True,
    )
    phone = st.selectbox("Phone service", yes_no, index=1, key="phone")
    lines = dependent_select(
        "Multiple lines", yes_no, phone == "No", "No phone service", "lines"
    )
    internet = st.selectbox(
        "Internet service", ["DSL", "Fiber optic", "No"], key="internet"
    )
    security = dependent_select(
        "Online security", yes_no, internet == "No", "No internet service", "security"
    )
    backup = dependent_select(
        "Online backup", yes_no, internet == "No", "No internet service", "backup"
    )
    protection = dependent_select(
        "Device protection", yes_no, internet == "No", "No internet service", "protection"
    )
    support = dependent_select(
        "Tech support", yes_no, internet == "No", "No internet service", "support"
    )
    tv = dependent_select(
        "Streaming TV", yes_no, internet == "No", "No internet service", "tv"
    )
    movies = dependent_select(
        "Streaming movies", yes_no, internet == "No", "No internet service", "movies"
    )


# ----------------------------- Billing -----------------------------

st.markdown(
    textwrap.dedent(
        """
        <div class="section-heading">◈ Billing intelligence</div>
        <div class="section-note">Total charges are estimated using tenure × monthly charges by default. Enable the actual total option if you know the customer's recorded billing total.</div>
        """
    ),
    unsafe_allow_html=True,
)

billing_left, billing_right = st.columns(2, gap="large")

with billing_left:
    monthly = st.number_input(
        "Monthly charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
        key="monthly_charges",
    )

estimated_total = float(tenure * monthly)

with billing_right:
    use_actual_total = st.checkbox("I know the actual total charges", value=False, key="use_actual_total")
    if use_actual_total:
        total = st.number_input(
            "Total charges", min_value=0.0, max_value=20000.0,
            value=float(estimated_total), step=10.0, key="total_charges",
        )
    else:
        total = estimated_total

st.caption(
    f"Suggested estimate: {estimated_total:.2f} | "
    f"Current total charges used: {total:.2f}"
)


# ----------------------------- Build model row -----------------------------

input_row = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
    ]
)

try:
    input_row = add_features(input_row)
    expected_columns = meta["input_columns"]

    missing_columns = [name for name in expected_columns if name not in input_row.columns]
    if missing_columns:
        st.error(
            "The app's input row does not match the saved model metadata. "
            f"Missing columns: {missing_columns}. Check add_features() and train.py."
        )
        st.stop()

    # Match training feature order and exclude any unexpected extra columns.
    input_row = input_row.reindex(columns=expected_columns)

except Exception as exc:
    st.error(f"Could not prepare the customer input row: {exc}")
    st.stop()


# ----------------------------- Prediction action -----------------------------

st.markdown("---")
st.markdown(
    textwrap.dedent(
        """
        <div class="section-heading">◈ Prediction engine</div>
        <div class="section-note">Run the saved model against the customer profile above.</div>
        """
    ),
    unsafe_allow_html=True,
)

action_col, explanation_col = st.columns([1, 1.4], gap="large")

with action_col:
    run_prediction = st.button(
        "⚡  ANALYZE CUSTOMER",
        use_container_width=True,
        type="primary",
    )

with explanation_col:
    st.caption(
        "The app uses the saved preprocessing and Logistic Regression pipeline. "
        "The probability is a model estimate, not a guarantee of future behavior."
    )


if run_prediction:
    try:
        probability = float(model.predict_proba(input_row)[0, 1])

        if not 0.0 <= probability <= 1.0:
            raise ValueError("The model returned a probability outside the 0–1 range.")

        current_signature = {
            **input_row.iloc[0].to_dict(),
            "threshold": float(threshold),
        }

        st.session_state["last_prediction"] = {
            "probability": probability,
            "threshold": float(threshold),
            "flagged": bool(probability >= threshold),
            "tenure": int(tenure),
            "monthly": float(monthly),
            "total": float(total),
            "contract": str(contract),
            "internet": str(internet),
            "signature": current_signature,
        }

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# Show the last submitted prediction until the next analysis is run.
result = st.session_state.get("last_prediction")
current_signature = {
    **input_row.iloc[0].to_dict(),
    "threshold": float(threshold),
}

if result is not None:

    if result.get("signature") != current_signature:
        st.warning(
            "Inputs or threshold changed since this analysis. "
            "Click ANALYZE CUSTOMER to update."
        )

    probability = result["probability"]
    used_threshold = result["threshold"]
    flagged = result["flagged"]

    st.markdown("---")
    st.markdown(
        '<div class="section-heading">◈ Prediction result</div>',
        unsafe_allow_html=True,
    )

    result_col, detail_col = st.columns([1.15, 0.85], gap="large")

    with result_col:
        decision_class = "decision-yes" if flagged else "decision-no"
        decision_text = (
            "⚠ LIKELY CHURNER — consider reviewing a retention offer."
            if flagged
            else "✓ NOT FLAGGED — probability is below the selected threshold."
        )

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="result-panel">
                    <div class="result-kicker">MODEL-PREDICTED CHURN PROBABILITY</div>
                    <div class="probability">{probability * 100:.1f}<span>%</span></div>
                    <div class="scale">
                        <div class="scale-fill" style="width:{probability * 100:.2f}%;"></div>
                        <div class="threshold-marker" style="left:{used_threshold * 100:.2f}%;"></div>
                    </div>
                    <div class="scale-legend">
                        <span>0%</span>
                        <span>White marker = threshold</span>
                        <span>100%</span>
                    </div>
                    <div class="decision {decision_class}">{decision_text}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with detail_col:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="glass" style="height:100%;">
                    <div class="metric-label">DECISION SNAPSHOT</div>
                    <div class="detail-line">
                        <span class="key">Probability</span>
                        <span class="value">{probability * 100:.2f}%</span>
                    </div>
                    <div class="detail-line">
                        <span class="key">Threshold used</span>
                        <span class="value">{used_threshold:.2f}</span>
                    </div>
                    <div class="detail-line">
                        <span class="key">Tenure</span>
                        <span class="value">{result['tenure']} months</span>
                    </div>
                    <div class="detail-line">
                        <span class="key">Monthly charges</span>
                        <span class="value">{result['monthly']:.2f}</span>
                    </div>
                    <div class="detail-line">
                        <span class="key">Total charges</span>
                        <span class="value">{result['total']:.2f}</span>
                    </div>
                    <div class="detail-line">
                        <span class="key">Contract</span>
                        <span class="value">{result['contract']}</span>
                    </div>
                    <div class="detail-line">
                        <span class="key">Internet service</span>
                        <span class="value">{result['internet']}</span>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # Threshold-position visualization. This shows the decision boundary
    # and the customer's score without inventing risk bands.
    st.markdown(
        '<div class="section-heading">◈ Probability vs. decision threshold</div>',
        unsafe_allow_html=True,
    )

    score_col, threshold_col = st.columns(2)
    with score_col:
        st.metric("Customer probability", f"{probability:.1%}")
    with threshold_col:
        st.metric("Selected threshold", f"{used_threshold:.0%}")

    st.progress(min(max(probability, 0.0), 1.0), text="Predicted churn probability")
    st.caption(
        "The customer is flagged when the predicted probability is greater than "
        "or equal to the selected threshold."
    )

    if flagged:
        st.warning(
            "This customer meets the selected flagging rule. "
            "Consider the customer's context and the economics of a retention action."
        )
    else:
        st.success(
            "This customer is below the selected flagging threshold. "
            "This does not mean churn is impossible."
        )

    st.caption(
        "Interpretation: the probability is the model's estimated likelihood. "
        "It is not a measure of model confidence or a causal explanation."
    )
else:
    st.markdown(
        textwrap.dedent(
            """
            <div class="glass">
                <div class="eyebrow">AWAITING CUSTOMER ANALYSIS</div>
                <div style="font:600 17px 'Space Grotesk',sans-serif;color:#F5F7FF;">
                    Your prediction workspace is ready.
                </div>
                <div style="color:#929BB5;font-size:12px;margin-top:7px;">
                    Enter the customer details and select <b>ANALYZE CUSTOMER</b>
                    to display the model's churn probability and threshold decision.
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ----------------------------- Footer -----------------------------

st.markdown(
    textwrap.dedent(
        """
        <div class="footer">
            CHURN.AI &nbsp; / &nbsp; CUSTOMER INTELLIGENCE PLATFORM<br>
            LOGISTIC REGRESSION · TELCO CUSTOMER CHURN
        </div>
        """
    ),
    unsafe_allow_html=True,
)
