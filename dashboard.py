# ============================================================
#  dashboard.py
#  DASHBOARD PAGE MODULE
#
#  This file contains only the Dashboard Page UI and logic.
#  It is imported and used by app.py — do not run this file directly.
# ============================================================

import streamlit as st
import auth


def model_files_exist():
    import os
    required = ["crop_model.pkl", "crop_label_encoder.pkl", "feature_columns.json"]
    return all(os.path.exists(f) for f in required)


def logout():
    """Clears session state and sends the user back to the login page."""
    st.session_state.logged_in = False
    st.session_state.username  = None
    st.session_state.full_name = None
    st.session_state.page      = "login"


def show_dashboard_page():
    """
    Renders the Dashboard page for a logged-in user.
    Shows quick stats and navigation buttons to Prediction / History / Logout.
    """
    st.markdown(f"""
    <div class="welcome-card">
        <h3 style="margin:0; color:#1b4332;">👋 Welcome back, {st.session_state.full_name}!</h3>
        <p style="margin:0.3rem 0 0 0; color:#777;">Username: {st.session_state.username}</p>
    </div>
    """, unsafe_allow_html=True)

    history = auth.get_user_history(st.session_state.username)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="stat-box"><div class="num">{len(history)}</div>
        <div class="label">Total Predictions</div></div>""", unsafe_allow_html=True)
    with col2:
        last_crop = history[-1]["crop"].title() if history else "—"
        st.markdown(f"""<div class="stat-box"><div class="num">{last_crop}</div>
        <div class="label">Last Recommended Crop</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="stat-box"><div class="num">22</div>
        <div class="label">Crops Supported</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🌾 New Prediction"):
            st.session_state.page = "predict"
            st.rerun()
    with col_b:
        if st.button("📜 View History"):
            st.session_state.page = "history"
            st.rerun()
    with col_c:
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

    if not model_files_exist():
        st.markdown("""
        <div class="warn-card">
            ⚠️ Model files not found. Run the Colab notebook first and place
            crop_model.pkl, crop_label_encoder.pkl, feature_columns.json
            in this folder to enable predictions.
        </div>
        """, unsafe_allow_html=True)
