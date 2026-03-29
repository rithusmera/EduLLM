import streamlit as st

def require_login():
    if not st.session_state.get("logged_in", False):
        st.switch_page("pages/login_page.py")