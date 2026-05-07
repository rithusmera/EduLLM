import streamlit as st
import login
import theme as th

login.init_user_db()

st.set_page_config(page_title="EduLLM Login", layout="centered")

# Login page: inject CSS but override button width for full-width login btns
extra = """
/* Full-width buttons on login page */
div.stButton > button {
    width: 100% !important;
    padding: 0.7rem 1.5rem !important;
    font-size: 15px !important;
}
"""
st.markdown(th.get_css(extra=extra), unsafe_allow_html=True)

# Hide sidebar on login
st.markdown('<style>[data-testid="stSidebar"]{display:none!important;}</style>', unsafe_allow_html=True)

a = th.get_accent()

# ── Header ───────────────────────────────────────────
st.markdown(f"""
    <div style="text-align:center;margin-bottom:2rem;margin-top:2rem;">
        <div style="display:inline-flex;align-items:center;justify-content:center;
                    width:56px;height:56px;border-radius:18px;
                    background:linear-gradient(135deg,{a},{th._darken(a,24)});
                    font-size:26px;margin-bottom:1rem;">📚</div>
        <div style="font-size:30px;font-weight:800;color:#e2e8f0;letter-spacing:-0.025em;">EduLLM</div>
        <div style="font-size:14px;color:#6b7280;margin-top:4px;">Offline AI Tutor — Class 11–12 Science</div>
    </div>
""", unsafe_allow_html=True)

# ── Card wrapper ─────────────────────────────────────
st.markdown("""
    <div style="background:#1e2030;border-radius:20px;border:1px solid #2a2b3d;
                box-shadow:0 20px 40px rgba(0,0,0,0.4);padding:2.5rem 2.5rem 1.5rem;
                max-width:420px;margin:0 auto;">
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  Sign In  ", "  Register  "])

with tab1:
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    username = st.text_input("Username", key="login_user", placeholder="Enter your username")
    password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    if st.button("Sign In", key="signin_btn", use_container_width=True):
        if login.verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Welcome back!")
            st.switch_page("app.py")
        else:
            st.error("Invalid credentials")

with tab2:
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    new_user = st.text_input("New Username", key="reg_user", placeholder="Choose a username")
    new_pass = st.text_input("New Password", type="password", key="reg_pass", placeholder="Choose a password")
    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    if st.button("Create Account", key="register_btn", use_container_width=True):
        if login.create_user(new_user, new_pass):
            st.success("Account created! You can now sign in.")
        else:
            st.error("Username already exists")

st.markdown('</div>', unsafe_allow_html=True)
