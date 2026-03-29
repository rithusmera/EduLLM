import streamlit as st
import login

login.init_user_db()

st.title("EduLLM Login")

tab1, tab2 = st.tabs(["Login", "Register"])

with tab1:

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if login.verify_user(username, password):

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success("Login successful")
            st.switch_page("app.py")

        else:
            st.error("Invalid credentials")


with tab2:

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")

    if st.button("Create Account"):

        if login.create_user(new_user, new_pass):
            st.success("Account created")

        else:
            st.error("Username already exists")