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
            --bg: #080A12;
            --panel: #111525;
            --panel-2: #171D30;
            --line: rgba(157, 140, 255, .20);
            --purple: #A78BFA;
            --violet: #8B5CF6;
            --cyan: #22D3EE;
            --green: #34D399;
            --red: #FB7185;
            --text: #F5F7FF;
            --muted: #929BB5;
        }

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(ellipse at 12% 0%, rgba(124,58,237,.16), transparent 34%),
                radial-gradient(ellipse at 92% 16%, rgba(34,211,238,.09), transparent 30%),
                linear-gradient(145deg, #080A12 0%, #0B0E19 55%, #080A12 100%);
        }

        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        .block-container {
            max-width: 1480px;
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111426 0%, #0B0E19 100%);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -.5px;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(167,139,250,.28);
            border-radius: 24px;
            padding: 27px 30px;
            margin-bottom: 22px;
            background:
                radial-gradient(circle at 86% 25%, rgba(34,211,238,.15), transparent 23%),
                radial-gradient(circle at 72% 100%, rgba(139,92,246,.23), transparent 34%),
                linear-gradient(115deg, rgba(124,58,237,.20), rgba(17,21,37,.97) 58%);
            box-shadow: 0 18px 65px rgba(0,0,0,.25);
        }
        .hero:after {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            right: 7%;
            top: -145px;
            border: 1px solid rgba(34,211,238,.22);
            border-radius: 50%;
            box-shadow: 0 0 0 22px rgba(34,211,238,.025),
                        0 0 0 48px rgba(167,139,250,.025);
            pointer-events: none;
        }
        .brand {
            font: 800 34px/1.05 'Space Grotesk', sans-serif;
            letter-spacing: -1.7px;
            color: #fff;
        }
        .brand span { color: var(--purple); }
        .hero-sub {
            color: #AAB3CC;
            font-size: 13px;
            margin-top: 9px;
        }
        .online-pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            color: #86F0C5;
            border: 1px solid rgba(52,211,153,.28);
            background: rgba(52,211,153,.08);
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1px;
        }

        .eyebrow {
            color: var(--purple);
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 7px;
        }
        .section-heading {
            color: #F5F7FF;
            font: 700 20px 'Space Grotesk', sans-serif;
            margin: 18px 0 4px 0;
        }
        .section-note {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 15px;
        }

        .glass {
            background: linear-gradient(145deg, rgba(23,29,48,.96), rgba(15,19,33,.98));
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: 18px 19px;
            box-shadow: 0 10px 32px rgba(0,0,0,.17);
        }
        .metric-label {
            color: #929BB5;
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1.15px;
            text-transform: uppercase;
        }
        .metric-value {
            color: #F7F8FF;
            font: 700 25px 'Space Grotesk', sans-serif;
            margin-top: 7px;
        }
        .metric-foot {
            color: #8791AC;
            font-size: 11px;
            margin-top: 4px;
        }

        .field-card {
            border: 1px solid rgba(157,140,255,.17);
            border-radius: 15px;
            padding: 13px 15px;
            margin: 7px 0 12px 0;
            background: linear-gradient(120deg, rgba(139,92,246,.09), rgba(34,211,238,.035));
        }
        .field-card span {
            color: #C4B5FD;
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1.1px;
        }

        .result-panel {
            border: 1px solid rgba(167,139,250,.31);
            border-radius: 22px;
            padding: 24px;
            background:
                radial-gradient(circle at 90% 0%, rgba(139,92,246,.18), transparent 35%),
                linear-gradient(145deg, #171D32, #101422);
            box-shadow: 0 15px 45px rgba(0,0,0,.22);
        }
        .result-kicker {
            color: #C4B5FD;
            font: 700 10px 'DM Mono', monospace;
            letter-spacing: 1.5px;
        }
        .probability {
            color: #FFFFFF;
            font: 700 54px/.98 'Space Grotesk', sans-serif;
            letter-spacing: -2px;
            margin: 12px 0 7px 0;
        }
        .probability span { color: var(--cyan); }
        .decision {
            border-radius: 12px;
            padding: 13px 15px;
            font-size: 13px;
            font-weight: 700;
            margin-top: 17px;
        }
        .decision-yes {
            color: #FDA4AF;
            border: 1px solid rgba(251,113,133,.28);
            background: rgba(251,113,133,.09);
        }
        .decision-no {
            color: #86EFC1;
            border: 1px solid rgba(52,211,153,.25);
            background: rgba(52,211,153,.08);
        }
        .scale {
            display: flex;
            height: 11px;
            overflow: hidden;
            border-radius: 99px;
            margin-top: 19px;
            background: #262D42;
            position: relative;
        }
        .scale-fill {
            height: 100%;
            border-radius: 99px;
            background: linear-gradient(90deg, #7C3AED, #A78BFA, #22D3EE);
            box-shadow: 0 0 14px rgba(167,139,250,.45);
        }
        .threshold-marker {
            position: absolute;
            top: -3px;
            bottom: -3px;
            width: 2px;
            background: #FFFFFF;
            box-shadow: 0 0 8px rgba(255,255,255,.9);
            z-index: 2;
        }
        .scale-legend {
            display: flex;
            justify-content: space-between;
            color: #8E98B2;
            font: 10px 'DM Mono', monospace;
            margin-top: 7px;
        }
        .detail-line {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(157,140,255,.12);
            font-size: 12px;
        }
        .detail-line:last-child { border-bottom: 0; }
        .detail-line .key { color: #929BB5; }
        .detail-line .value { color: #F3F5FF; font-weight: 600; text-align: right; }

        .footer {
            color: #66708A;
            text-align: center;
            font: 10px 'DM Mono', monospace;
            letter-spacing: .8px;
            padding: 25px 0 5px 0;
        }

        /* Streamlit widgets */
        .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #D7DCEF !important;
            font-size: 12px !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] > div {
            background: #171D30 !important;
            border: 1px solid #303951 !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="select"] span { color: #F5F7FF !important; }
        div[data-testid="stNumberInput"] input {
            background: #171D30 !important;
            color: #FFFFFF !important;
            border: 1px solid #303951 !important;
            border-radius: 10px !important;
        }
        .stButton > button {
            min-height: 46px;
            color: #fff;
            font-weight: 800;
            border: 1px solid rgba(196,181,253,.45);
            border-radius: 11px;
            background: linear-gradient(100deg, #6D3BEA, #9B5CF6 58%, #5268E8);
            box-shadow: 0 5px 22px rgba(124,58,237,.20);
            transition: .18s ease;
        }
        .stButton > button:hover {
            color: #fff;
            border-color: #22D3EE;
            box-shadow: 0 0 22px rgba(34,211,238,.22);
            transform: translateY(-1px);
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        hr { border-color: rgba(157,140,255,.16); }
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
        <div class="section-note">The total-charge field starts with a tenure × monthly-charge estimate and remains editable.</div>
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
    # Streamlit keeps widget state after first creation. This is intentional:
    # users can override the suggested estimate with the actual accumulated total.
    if "total_charges" not in st.session_state:
        st.session_state["total_charges"] = estimated_total

    total = st.number_input(
        "Total charges",
        min_value=0.0,
        max_value=20000.0,
        value=float(estimated_total),
        step=10.0,
        key="total_charges",
    )

st.caption(
    f"Suggested estimate: {estimated_total:.2f} | "
    f"Current total charges entered: {total:.2f}"
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

        st.session_state["last_prediction"] = {
            "probability": probability,
            "threshold": float(threshold),
            "flagged": bool(probability >= threshold),
            "tenure": int(tenure),
            "monthly": float(monthly),
            "total": float(total),
            "contract": str(contract),
            "internet": str(internet),
        }

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# Show the last submitted prediction until the next analysis is run.
result = st.session_state.get("last_prediction")

if result is not None:
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
