# ============================================================
#  login.py
#  LOGIN PAGE MODULE
#
#  This file contains only the Login Page UI and logic.
#  It is imported and used by app.py — do not run this file directly.
# ============================================================

import streamlit as st
import auth


def show_login_page():
    """
    Renders the Login page.
    On successful login, updates st.session_state and redirects
    to the dashboard.
    """
    st.markdown('<div class="info-pill">🔐 Login</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            success, result = auth.login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username  = username.strip().lower()
                st.session_state.full_name = result["full_name"]
                st.session_state.page      = "dashboard"
                st.success("✅ Login successful! Redirecting to dashboard...")
                st.rerun()
            else:
                st.error(f"❌ {result}")

    st.markdown("Don't have an account?")
    if st.button("Create New Account"):
        st.session_state.page = "register"
        st.rerun()
