import streamlit as st
import chapter_quiz
import auth

auth.require_login()

st.set_page_config(page_title="Chapter Quiz", layout="wide")

# 1. PAGE CONFIG & GLOBAL STYLE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B !important;
    }

    /* Force visibility for ALL text elements */
    .stMarkdown, .stText, .stHeader, p, span, label, div, h1, h2, h3 {
        color: #1E293B !important;
    }

    /* Hide default sidebar navigation */
    [data-testid="stSidebarNav"] {display: none;}

    /* Global Background */
    .stApp {
        background-color: #F8FAFC !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #475569 !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global Rounded Corners & Input Styles */
    .stButton>button, .stTextInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Indigo Pill Buttons - Primary */
    div.stButton > button {
        background-color: #22C55E !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2.5rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    div.stButton > button:hover {
        background-color: #16A34A !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        color: white !important;
    }

    /* Radio and Checkbox visibility */
    .stRadio label, .stCheckbox label {
        color: #334155 !important;
    }
    
    /* Card/Container Style */
    .edu-card {
        background-color: white !important;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #F1F5F9;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        background-color: #DCFCE7 !important;
        border-radius: 6px;
        font-size: 11px;
        color: #22C55E !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        margin-bottom: 1rem;
    }

    /* Logo area */
    .sidebar-logo {
        padding: 1.5rem 0.5rem;
    }
    .logo-text {
        font-size: 22px;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.025em;
    }
    .logo-sub {
        font-size: 13px;
        color: #64748B;
        margin-top: 2px;
    }
    
    /* Footer */
    .sidebar-footer {
        padding: 1rem 0.5rem;
        border-top: 1px solid #F1F5F9;
        margin-top: 2rem;
        font-size: 11px;
        color: #94A3B8;
    }

    /* Sidebar Divider */
    hr {
        margin: 1.5rem 0 !important;
        border: 0 !important;
        border-top: 1px solid #F1F5F9 !important;
    }

    /* Remove padding from top of main area */
    .block-container {
        padding-top: 3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-text">EduLLM</div>
            <div class="logo-sub">Intelligent Tutor System</div>
        </div>
        <hr>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 14px; font-weight: 500; color: #1E293B; margin-bottom: 12px; padding: 0 0.5rem;">Navigation</div>', unsafe_allow_html=True)
    
    st.page_link("app.py", label="App")
    st.page_link("pages/chapter_quiz_page.py", label="Chapter Quiz")
    
    st.markdown("""
        <div class="sidebar-footer">
            <b>EduLLM v1.0</b><br>
            Class 11-12 Science Curriculum<br>
            RAG Pipeline · Offline Llama
        </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------

st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <div style="font-size: 32px; font-weight: 700; color: #0F172A; letter-spacing: -0.025em;">Chapter Quiz</div>
        <div style="font-size: 15px; color: #64748B; margin-top: 4px;">Comprehensive review of science curriculum.</div>
    </div>
""", unsafe_allow_html=True)

subjects = chapter_quiz.get_available_subjects()

st.markdown('<div class="edu-card">', unsafe_allow_html=True)
st.markdown('<div style="font-size: 18px; font-weight: 700; color: #1E293B; margin-bottom: 1.5rem;">Quiz Configuration</div>', unsafe_allow_html=True)

selected_subject = st.selectbox(
    "Select Subject",
    subjects
)

chapters = chapter_quiz.get_available_chapters(selected_subject)

selected_chapter = st.selectbox(
    "Select Chapter",
    chapters
)

st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
if st.button("Start Quiz", use_container_width=True):

    chapter_file = chapter_quiz.get_chapter_file(
        selected_subject,
        selected_chapter
    )

    questions = chapter_quiz.load_questions(chapter_file)

    chapter_questions = chapter_quiz.filter_by_chapter(
        questions,
        subject=selected_subject,
        chapter=selected_chapter
    )

    quiz = chapter_quiz.sample_quiz_questions(chapter_questions)

    st.session_state.chapter_quiz = quiz
    st.session_state.chapter_index = 0
    st.session_state.chapter_score = 0
    st.session_state.show_next = False

    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


if "chapter_quiz" in st.session_state:

    quiz = st.session_state.chapter_quiz
    index = st.session_state.chapter_index

    if index < len(quiz):

        q = quiz[index]

        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="badge">Question {index+1}/{len(quiz)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 18px; font-weight: 700; color: #1E293B; margin-bottom: 1.5rem;">Chapter Assessment</div>', unsafe_allow_html=True)

        st.write(q["question"])

        options_list = [
            f"{k}) {v}" for k, v in q["options"].items()
        ]

        user_choice = st.radio(
            "Select the correct answer:",
            options_list,
            key=f"chapter_radio_{index}"
        )

        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
        if st.button("Submit Answer"):

            selected_letter = user_choice[0]

            if selected_letter == q["correct_option"]:
                st.success("Correct!")
                st.session_state.chapter_score += 1
            else:
                st.error(f"Incorrect. The correct answer was {q['correct_option']}.")

            st.session_state.show_next = True

        if st.session_state.show_next:

            if st.button("Continue to Next Question", use_container_width=True):

                st.session_state.chapter_index += 1
                st.session_state.show_next = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:

        score = st.session_state.chapter_score

        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 20px; font-weight: 700; color: #1E293B; margin-bottom: 1rem;">Quiz Completed</div>', unsafe_allow_html=True)

        st.write(f"Your final score is **{score} out of {len(quiz)}**.")

        if st.button("Restart Quiz", use_container_width=True):

            del st.session_state.chapter_quiz
            del st.session_state.chapter_index
            del st.session_state.chapter_score
            del st.session_state.show_next

            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
