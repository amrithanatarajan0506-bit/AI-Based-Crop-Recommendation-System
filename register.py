# ============================================================
#  register.py
#  REGISTRATION PAGE MODULE
#
#  This file contains only the Registration Page UI and logic.
#  It is imported and used by app.py — do not run this file directly.
# ============================================================

import streamlit as st
import auth


def show_register_page():
    """
    Renders the Registration page.
    On successful registration, redirects back to the Login page.
    """
    st.markdown('<div class="info-pill">📝 Register</div>', unsafe_allow_html=True)

    with st.form("register_form"):
        full_name = st.text_input("Full Name")
        location  = st.text_input("Location / Village (optional)")
        username  = st.text_input("Choose a Username")
        password  = st.text_input("Choose a Password", type="password")
        confirm   = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if password != confirm:
                st.error("❌ Passwords do not match.")
            elif not full_name.strip():
                st.error("❌ Please enter your full name.")
            else:
                success, message = auth.register_user(username, password, full_name, location)
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

    st.markdown("Already have an account?")
    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()
