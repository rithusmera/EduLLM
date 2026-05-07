import streamlit as st
import chapter_quiz
import auth
import theme as th
import ui_format

auth.require_login()

st.set_page_config(page_title="Chapter Quiz", layout="wide")
st.markdown(th.get_css(), unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    a = th.get_accent()
    st.markdown(f"""
        <div style="padding:1.5rem 0.5rem 0.5rem;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                <div style="width:34px;height:34px;border-radius:10px;
                            background:linear-gradient(135deg,{a},{th._darken(a,24)});
                            display:flex;align-items:center;justify-content:center;font-size:16px;">📚</div>
                <div>
                    <div class="logo-text">EduLLM</div>
                    <div class="logo-sub">Offline AI Tutor</div>
                </div>
            </div>
        </div>
        <hr>
        <div class="section-label">PAGES</div>
    """, unsafe_allow_html=True)

    st.page_link("app.py", label="Ask a question")
    st.markdown(f"""
        <div class="nav-item active">
            <span class="nav-dot" style="background:{a};"></span>
            Chapter Quiz
        </div>
    """, unsafe_allow_html=True)

    # ── Theme picker ──
    th.render_theme_picker()

    st.markdown("""
        <div class="sidebar-footer">
            <b>EduLLM v1.0</b><br>
            Class 11–12 Science<br>
            RAG Pipeline · Offline Llama
        </div>
    """, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────
st.markdown("""
    <div class="page-header">
        <div>
            <div class="page-title">Chapter Quiz</div>
            <div class="page-sub">Comprehensive review of science curriculum</div>
        </div>
        <div class="rag-badge">Offline</div>
    </div>
""", unsafe_allow_html=True)

# ── QUIZ CONFIG ───────────────────────────────────────
subjects = chapter_quiz.get_available_subjects()

st.markdown('<div class="edu-card">', unsafe_allow_html=True)
st.markdown('<div style="font-size:18px;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem;">Quiz Configuration</div>', unsafe_allow_html=True)

selected_subject = st.selectbox("Select Subject", subjects)
chapters = chapter_quiz.get_available_chapters(selected_subject)
selected_chapter = st.selectbox("Select Chapter", chapters)

st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
if st.button("Start Quiz", use_container_width=True):
    chapter_file = chapter_quiz.get_chapter_file(selected_subject, selected_chapter)
    questions = chapter_quiz.load_questions(chapter_file)
    chapter_questions = chapter_quiz.filter_by_chapter(questions, subject=selected_subject, chapter=selected_chapter)
    quiz = chapter_quiz.sample_quiz_questions(chapter_questions)
    st.session_state.chapter_quiz = quiz
    st.session_state.chapter_index = 0
    st.session_state.chapter_score = 0
    st.session_state.show_next = False
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ── QUIZ QUESTIONS ────────────────────────────────────
if "chapter_quiz" in st.session_state:
    quiz = st.session_state.chapter_quiz
    index = st.session_state.chapter_index

    if index < len(quiz):
        q = quiz[index]

        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="badge">Question {index+1}/{len(quiz)}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:18px;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem;">Chapter Assessment</div>', unsafe_allow_html=True)

        ui_format.render_markdown(q["question"])
        options_list = [ui_format.normalize_math_text(f"{k}) {v}") for k, v in q["options"].items()]
        user_choice = st.radio("Select the correct answer:", options_list, key=f"chapter_radio_{index}")

        st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
        if st.button("Submit Answer"):
            selected_letter = user_choice[0]
            if selected_letter == q["correct_option"]:
                st.success("Correct! ✓")
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
        st.markdown('<div style="font-size:20px;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">Quiz Completed 🎉</div>', unsafe_allow_html=True)
        st.write(f"Your final score is **{score} out of {len(quiz)}**.")
        if st.button("Restart Quiz", use_container_width=True):
            del st.session_state.chapter_quiz
            del st.session_state.chapter_index
            del st.session_state.chapter_score
            del st.session_state.show_next
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
