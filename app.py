# ============================================================
#  AI Crop Recommendation System — Main Application
#  app.py
#
#  This is the MAIN file that connects everything together:
#    - login.py       -> Login page
#    - register.py     -> Registration page
#    - dashboard.py    -> Dashboard page
#    - auth.py          -> Login/Registration backend logic
#
#  Prediction and History pages are kept in this file since they
#  depend directly on the ML model.
#
#  REQUIRED FILES IN THE SAME FOLDER:
#    app.py                    <- this file (run this one)
#    auth.py                   <- login/register backend logic
#    login.py                  <- login page
#    register.py               <- registration page
#    dashboard.py              <- dashboard page
#    crop_model.pkl            <- from Colab notebook
#    crop_label_encoder.pkl    <- from Colab notebook
#    crop_scaler.pkl           <- from Colab notebook
#    feature_columns.json      <- from Colab notebook
#
#  HOW TO RUN:
#    pip install streamlit pandas numpy scikit-learn joblib
#    python -m streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

import auth                              # login/register backend logic
from login import show_login_page        # Login page (separate file)
from register import show_register_page  # Registration page (separate file)
from dashboard import show_dashboard_page # Dashboard page (separate file)

# -- Page config ----------------------------------------------
st.set_page_config(
    page_title = "Crop Recommendation System",
    page_icon  = "🌱",
    layout     = "centered"
)

# -- Global CSS -------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f7f0; }

    .header-box {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-box h1 { color: white; font-size: 2rem; margin: 0; }
    .header-box p  { color: #d8f3dc; font-size: 1rem; margin: 0.5rem 0 0 0; }

    .section-label {
        font-size: 0.78rem; font-weight: 600; color: #40916c;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem;
    }

    .result-card {
        background: white; border-left: 6px solid #52b788; border-radius: 10px;
        padding: 1.5rem 2rem; margin-top: 1.5rem; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }
    .result-card h2 { margin: 0 0 0.3rem 0; color: #1b4332; font-size: 1.6rem; }
    .result-card .confidence { font-size: 1rem; font-weight: 500; }
    .result-card .tip {
        margin-top: 0.8rem; font-size: 0.9rem; color: #555;
        border-top: 1px solid #e8f5e9; padding-top: 0.8rem;
    }

    .warn-card {
        background: #fff8e1; border-left: 6px solid #f9a825; border-radius: 10px;
        padding: 1rem 1.5rem; margin-top: 1rem; font-size: 0.9rem; color: #555;
    }

    .slider-row {
        display: flex; justify-content: space-between; font-size: 0.8rem;
        color: #777; margin-top: -0.5rem; margin-bottom: 0.5rem;
    }

    div.stButton > button {
        background-color: #2d6a4f; color: white; font-size: 1.05rem; font-weight: 600;
        padding: 0.65rem 2.5rem; border-radius: 8px; border: none; width: 100%;
        transition: background 0.2s;
    }
    div.stButton > button:hover { background-color: #1b4332; color: white; }

    .info-pill {
        display: inline-block; background: #d8f3dc; color: #1b4332; font-size: 0.78rem;
        font-weight: 600; padding: 0.2rem 0.7rem; border-radius: 20px; margin-bottom: 1rem;
    }

    .welcome-card {
        background: white; border-radius: 12px; padding: 1.5rem 2rem;
        margin-bottom: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    .stat-box {
        background: white; border-radius: 10px; padding: 1.2rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stat-box .num { font-size: 1.8rem; font-weight: 700; color: #2d6a4f; }
    .stat-box .label { font-size: 0.8rem; color: #777; margin-top: 0.2rem; }

    .history-row {
        background: white; border-radius: 8px; padding: 0.8rem 1.2rem;
        margin-bottom: 0.6rem; border-left: 4px solid #52b788;
    }

    .footer { text-align: center; margin-top: 3rem; font-size: 0.8rem; color: #aaa; }
</style>
""", unsafe_allow_html=True)

# -- Crop info dictionary ----------------------------------------
crop_info = {
    "rice":        {"emoji": "🌾", "tip": "Grows best in flooded fields. Requires high rainfall and warm temperature."},
    "maize":       {"emoji": "🌽", "tip": "Needs well-drained soil. Sensitive to frost. Good for warm, moderate-rain regions."},
    "chickpea":    {"emoji": "🫘", "tip": "Drought-tolerant crop. Grows well in dry, cool conditions with low humidity."},
    "kidneybeans": {"emoji": "🫘", "tip": "Requires moderate rainfall and well-drained soil. Avoid waterlogging."},
    "pigeonpeas":  {"emoji": "🫘", "tip": "Drought-resistant and grows in semi-arid regions. Long growing season."},
    "mothbeans":   {"emoji": "🫘", "tip": "Extremely drought-tolerant. Ideal for arid and semi-arid sandy soils."},
    "mungbean":    {"emoji": "🫘", "tip": "Short-duration crop. Thrives in warm, humid conditions with moderate rain."},
    "blackgram":   {"emoji": "🫘", "tip": "Grows in warm, humid climates. Suitable for loamy and clay-loam soils."},
    "lentil":      {"emoji": "🫘", "tip": "Cool-season crop. Low water requirement. Good for winter growing."},
    "pomegranate": {"emoji": "🍎", "tip": "Drought-tolerant fruit. Prefers hot, dry summers and cool winters."},
    "banana":      {"emoji": "🍌", "tip": "Needs high humidity, high rainfall, and rich soil. Frost-sensitive."},
    "mango":       {"emoji": "🥭", "tip": "Tropical fruit. Needs dry spell before flowering. Grows in hot climates."},
    "grapes":      {"emoji": "🍇", "tip": "Needs well-drained soil, high K levels, and a dry summer climate."},
    "watermelon":  {"emoji": "🍉", "tip": "Needs sandy loam soil, high temperature, and moderate irrigation."},
    "muskmelon":   {"emoji": "🍈", "tip": "Warm-season crop. Needs dry, hot weather during ripening stage."},
    "apple":       {"emoji": "🍎", "tip": "Needs cold winters for dormancy. Grows best at higher altitudes."},
    "orange":      {"emoji": "🍊", "tip": "Subtropical fruit. Needs moderate rainfall and well-drained soil."},
    "papaya":      {"emoji": "🍑", "tip": "Fast-growing tropical fruit. Needs high humidity and warm temperature."},
    "coconut":     {"emoji": "🥥", "tip": "Coastal crop. Thrives in high humidity, sandy soil, and heavy rainfall."},
    "cotton":      {"emoji": "🌿", "tip": "Needs high temperature, moderate rainfall, and well-drained black soil."},
    "jute":        {"emoji": "🌿", "tip": "Grows in warm, humid climate with heavy rainfall. Needs alluvial soil."},
    "coffee":      {"emoji": "☕", "tip": "Grows in tropical highlands. Needs moderate temperature and rainfall."},
}

# -- Load ML model (cached so it loads only once) ------------------
@st.cache_resource
def load_model():
    model = joblib.load("crop_model.pkl")
    le    = joblib.load("crop_label_encoder.pkl")
    with open("feature_columns.json") as f:
        feature_cols = json.load(f)
    return model, le, feature_cols


def model_files_exist():
    import os
    required = ["crop_model.pkl", "crop_label_encoder.pkl", "feature_columns.json"]
    return all(os.path.exists(f) for f in required)


# -- Session state initialization ------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "full_name" not in st.session_state:
    st.session_state.full_name = None
if "page" not in st.session_state:
    st.session_state.page = "login"   # login | register | dashboard | predict | history


def go_to(page_name):
    st.session_state.page = page_name


# ============================================================
#  HEADER (shown on every page)
# ============================================================
st.markdown("""
<div class="header-box">
    <h1>🌱 AI Crop Recommendation System</h1>
    <p>Helping farmers choose the right crop using Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  PREDICTION PAGE (kept here — depends directly on the ML model)
# ============================================================
def show_predict_page():
    if st.button("⬅ Back to Dashboard"):
        go_to("dashboard")
        st.rerun()

    if not model_files_exist():
        st.error("""
        ⚠️ Model files not found in this folder.

        Make sure these files are present:
        - crop_model.pkl
        - crop_label_encoder.pkl
        - feature_columns.json

        Run the Colab notebook first to generate these files.
        """)
        return

    model, le, feature_cols = load_model()

    st.markdown('<div class="info-pill">📋 Enter Your Field Parameters</div>', unsafe_allow_html=True)

    st.markdown("##### 🧪 Soil Nutrients")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="section-label">Nitrogen (N) — kg/ha</p>', unsafe_allow_html=True)
        N = st.number_input("N", min_value=0.0, max_value=150.0, value=50.0, step=1.0, label_visibility="collapsed")
    with col2:
        st.markdown('<p class="section-label">Phosphorus (P) — kg/ha</p>', unsafe_allow_html=True)
        P = st.number_input("P", min_value=0.0, max_value=150.0, value=50.0, step=1.0, label_visibility="collapsed")
    with col3:
        st.markdown('<p class="section-label">Potassium (K) — kg/ha</p>', unsafe_allow_html=True)
        K = st.number_input("K", min_value=0.0, max_value=210.0, value=50.0, step=1.0, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("##### 🌤️ Climate Conditions")
    col4, col5 = st.columns(2)
    with col4:
        st.markdown('<p class="section-label">Temperature — °C</p>', unsafe_allow_html=True)
        temperature = st.slider("Temperature", 5.0, 50.0, 25.0, 0.1, label_visibility="collapsed")
        st.markdown(f'<div class="slider-row"><span>5°C</span><span><b>{temperature}°C</b></span><span>50°C</span></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<p class="section-label">Humidity — %</p>', unsafe_allow_html=True)
        humidity = st.slider("Humidity", 10.0, 100.0, 65.0, 0.1, label_visibility="collapsed")
        st.markdown(f'<div class="slider-row"><span>10%</span><span><b>{humidity}%</b></span><span>100%</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🌧️ Soil & Rainfall")
    col6, col7 = st.columns(2)
    with col6:
        st.markdown('<p class="section-label">Soil pH</p>', unsafe_allow_html=True)
        ph = st.slider("pH", 3.0, 10.0, 6.5, 0.1, label_visibility="collapsed")
        ph_label = "🔴 Acidic" if ph < 5.5 else ("🟢 Neutral" if ph <= 7.0 else "🔵 Alkaline")
        st.markdown(f'<div class="slider-row"><span>3.0</span><span><b>{ph} — {ph_label}</b></span><span>10.0</span></div>', unsafe_allow_html=True)
    with col7:
        st.markdown('<p class="section-label">Rainfall — mm</p>', unsafe_allow_html=True)
        rainfall = st.slider("Rainfall", 20.0, 300.0, 100.0, 1.0, label_visibility="collapsed")
        st.markdown(f'<div class="slider-row"><span>20mm</span><span><b>{rainfall}mm</b></span><span>300mm</span></div>', unsafe_allow_html=True)

    # ---- FEATURE 2: Input validation warnings ----
    validation_warnings = []
    if ph < 4.5 or ph > 8.5:
        validation_warnings.append(f"pH of {ph} is unusually extreme — please double-check your soil test result.")
    if rainfall < 30:
        validation_warnings.append(f"Rainfall of {rainfall}mm is very low — verify this is your correct regional average.")
    if humidity < 20:
        validation_warnings.append(f"Humidity of {humidity}% is unusually low for most crops — please re-check this value.")
    if N == 0 and P == 0 and K == 0:
        validation_warnings.append("All soil nutrient values (N, P, K) are zero — this is unusual for real soil. Please verify your soil test readings.")

    if validation_warnings:
        warning_text = "<br>".join([f"• {w}" for w in validation_warnings])
        st.markdown(f"""
        <div class="warn-card">
            ⚠️ <b>Please double-check your inputs:</b><br>{warning_text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🌾 Get Crop Recommendation"):
        input_data = {'N': N, 'P': P, 'K': K, 'temperature': temperature,
                       'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
        input_df = pd.DataFrame([input_data]).reindex(columns=feature_cols, fill_value=0)

        pred_encoded = model.predict(input_df)[0]
        crop_name    = le.inverse_transform([pred_encoded])[0]
        proba        = model.predict_proba(input_df)[0].max()

        info = crop_info.get(crop_name, {"emoji": "🌱", "tip": ""})

        if proba >= 0.80:
            conf_color, conf_label = "#2d6a4f", "High confidence"
        elif proba >= 0.60:
            conf_color, conf_label = "#f9a825", "Moderate confidence"
        else:
            conf_color, conf_label = "#c62828", "Low confidence — consider soil testing"

        st.markdown(f"""
        <div class="result-card">
            <h2>{info['emoji']} {crop_name.upper()}</h2>
            <div class="confidence" style="color:{conf_color};">
                ✅ {conf_label} &nbsp;|&nbsp; Confidence Score: {proba:.1%}
            </div>
            <div class="tip">💡 <b>Farming tip:</b> {info['tip']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- FEATURE 1: Top-3 crop confidence bar chart ----
        all_proba = model.predict_proba(input_df)[0]
        top3_idx  = np.argsort(all_proba)[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)
        top3_scores = all_proba[top3_idx]

        st.markdown("##### 🔍 Top 3 Likely Crops")
        chart_df = pd.DataFrame({
            "Crop": [f"{crop_info.get(c, {}).get('emoji','🌱')} {c.title()}" for c in top3_crops],
            "Confidence (%)": (top3_scores * 100).round(1)
        }).set_index("Crop")
        st.bar_chart(chart_df, color="#52b788")

        # ---- FEATURE 3: "Why this crop?" explanation using feature importances ----
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(feature_cols, model.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]

            feature_display_names = {
                'N': 'Nitrogen level', 'P': 'Phosphorus level', 'K': 'Potassium level',
                'temperature': 'Temperature', 'humidity': 'Humidity',
                'ph': 'Soil pH', 'rainfall': 'Rainfall'
            }
            feature_values = {
                'N': f"{N} kg/ha", 'P': f"{P} kg/ha", 'K': f"{K} kg/ha",
                'temperature': f"{temperature}°C", 'humidity': f"{humidity}%",
                'ph': f"{ph}", 'rainfall': f"{rainfall}mm"
            }

            reason_lines = []
            for feat, importance in top_features:
                name = feature_display_names.get(feat, feat)
                val  = feature_values.get(feat, "")
                reason_lines.append(f"<li><b>{name}</b> ({val}) — influence score: {importance:.1%}</li>")

            st.markdown(f"""
            <div class="result-card">
                <h4 style="margin:0 0 0.5rem 0; color:#1b4332;">🧠 Why this crop?</h4>
                <p style="color:#555; font-size:0.9rem; margin-bottom:0.5rem;">
                    The model relied most heavily on these factors for this prediction:
                </p>
                <ul style="color:#555; font-size:0.9rem;">
                    {''.join(reason_lines)}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Save this prediction into the user's history
        auth.save_prediction_to_history(st.session_state.username, {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "crop":       crop_name,
            "confidence": round(float(proba) * 100, 1),
            "inputs":     input_data
        })

        if proba < 0.60:
            st.markdown("""
            <div class="warn-card">
                ⚠️ <b>Low confidence prediction.</b> The entered values fall in a region
                where multiple crops overlap. We recommend getting a proper soil test done.
            </div>
            """, unsafe_allow_html=True)


# ============================================================
#  PREDICTION HISTORY PAGE (kept here — depends on crop_info)
# ============================================================
def show_history_page():
    if st.button("⬅ Back to Dashboard"):
        go_to("dashboard")
        st.rerun()

    st.markdown('<div class="info-pill">📜 Your Prediction History</div>', unsafe_allow_html=True)

    history = auth.get_user_history(st.session_state.username)

    if not history:
        st.info("You haven't made any predictions yet. Go to 'New Prediction' to get started.")
        return

    for record in reversed(history):
        emoji = crop_info.get(record["crop"], {}).get("emoji", "🌱")
        st.markdown(f"""
        <div class="history-row">
            <b>{emoji} {record['crop'].upper()}</b> — Confidence: {record['confidence']}%
            <br><span style="color:#999; font-size:0.8rem;">{record['timestamp']}</span>
            <br><span style="color:#666; font-size:0.85rem;">
                N={record['inputs']['N']}, P={record['inputs']['P']}, K={record['inputs']['K']},
                Temp={record['inputs']['temperature']}°C, Humidity={record['inputs']['humidity']}%,
                pH={record['inputs']['ph']}, Rainfall={record['inputs']['rainfall']}mm
            </span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
#  ROUTER - decides which page to show
#  Calls the imported functions from login.py, register.py, dashboard.py
# ============================================================
if not st.session_state.logged_in:
    if st.session_state.page == "register":
        show_register_page()
    else:
        show_login_page()
else:
    if st.session_state.page == "predict":
        show_predict_page()
    elif st.session_state.page == "history":
        show_history_page()
    else:
        show_dashboard_page()


# -- Footer -------------------------------------------------------
st.markdown("""
<div class="footer">
    AI Crop Recommendation System &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; Model: Random Forest Classifier
</div>
""", unsafe_allow_html=True)
