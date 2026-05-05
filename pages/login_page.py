import streamlit as st
import login

login.init_user_db()

# 1. PAGE CONFIG & GLOBAL STYLE
st.set_page_config(page_title="EduLLM Login", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B !important;
    }

    /* Force visibility for all text */
    .stMarkdown, .stText, p, span, label, div, h1, h2, h3 {
        color: #1E293B !important;
    }

    /* Hide Sidebar entirely on Login */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global Rounded Corners & Input Styles */
    .stButton>button, .stTextInput>div>div>input {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        height: 3rem !important;
    }
    
    /* Indigo Pill Buttons */
    div.stButton > button {
        background-color: #22C55E !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    
    div.stButton > button:hover {
        background-color: #16A34A !important;
        color: white !important;
    }
    
    /* Card Container */
    .login-container {
        background-color: white !important;
        padding: 3rem;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        max-width: 450px;
        margin: 2rem auto;
    }

    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-title {
        font-size: 32px;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.025em;
    }
    .login-subtitle {
        font-size: 15px;
        color: #64748B !important;
        margin-top: 0.5rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="login-container">
        <div class="login-header">
            <div class="login-title">EduLLM</div>
            <div class="login-subtitle">Offline Science Tutor</div>
        </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Login", "Register"])

with tab1:
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    if st.button("Sign In"):
        if login.verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Welcome back")
            st.switch_page("app.py")
        else:
            st.error("Invalid credentials")

with tab2:
    new_user = st.text_input("New Username", key="reg_user")
    new_pass = st.text_input("New Password", type="password", key="reg_pass")

    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    if st.button("Create Account"):
        if login.create_user(new_user, new_pass):
            st.success("Account created successfully")
        else:
            st.error("Username already exists")

st.markdown('</div>', unsafe_allow_html=True)
