import streamlit as st
import auth

auth.require_login()

import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
import llm_client
import RAGPipeline
import concept_quiz
import student_state
import theme as th
import ui_format

DB_PATH = "edu_chunks.db"
IDX_PATH = "edu_index.faiss"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
STUDENT_ID = st.session_state.username


@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(IDX_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return embedder, index, conn


embedder, index, conn = load_resources()
student_state.init_db()

st.set_page_config(page_title="EduLLM Tutor", layout="wide")
st.markdown(th.get_css(), unsafe_allow_html=True)

# SIDEBAR 
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

    st.markdown(f"""
        <div class="nav-item active">
            <span class="nav-dot" style="background:{a};"></span>
            Ask a question
        </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/chapter_quiz_page.py", label="Chapter Quiz")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">KNOWLEDGE CHECK</div>', unsafe_allow_html=True)

    if st.button("Take Concept Quiz", use_container_width=True):
        if "last_context" not in st.session_state:
            st.error("Please search for a topic first.")
        elif not st.session_state.get("last_chunks"):
            st.error("Please generate an answer with textbook context first.")
        else:
            first_chunk = list(st.session_state.last_chunks.values())[0]
            with st.spinner("Crafting quiz..."):
                quiz_data = concept_quiz.generate_concept_quiz(
                    st.session_state.last_context, first_chunk
                )
            if quiz_data and "questions" in quiz_data:
                st.session_state.concept_questions = quiz_data["questions"]
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answer_submitted = False
                st.rerun()
            else:
                st.error("Unable to generate quiz at this time.")
                if concept_quiz.LAST_ERROR:
                    st.caption(concept_quiz.LAST_ERROR)

    #Theme picker
    th.render_theme_picker()

    st.markdown("""
        <div class="sidebar-footer">
            <b>EduLLM v1.0</b><br>
            Class 11–12 Science<br>
            RAG Pipeline · Offline Llama
        </div>
    """, unsafe_allow_html=True)

#HEADER    
st.markdown(f"""
    <div class="page-header">
        <div>
            <div class="page-title">Ask a question</div>
            <div class="page-sub">Physics · Class 11–12 Science Textbooks</div>
        </div>
        <div class="rag-badge">RAG · Offline</div>
    </div>
""", unsafe_allow_html=True)

#SUGGESTION CHIPS
study_plan = student_state.get_study_plan(STUDENT_ID)
if study_plan:
    st.markdown('<div class="edu-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:18px;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">Dynamic Study Plan</div>', unsafe_allow_html=True)
    cols = st.columns(min(3, len(study_plan)))
    for plan_index, item in enumerate(study_plan[:3]):
        topic_parts = item["topic_id"].split("|")
        topic_label = " · ".join(part for part in topic_parts if part and part != "Unknown")
        mastery_pct = round(item["mastery_score"] * 100)
        with cols[plan_index]:
            st.markdown(f"""
                <div style="border:1px solid #2a2b3d;border-radius:10px;padding:1rem;background:#171927;">
                    <div class="badge">{item["priority"]} priority</div>
                    <div style="font-weight:700;color:#e2e8f0;margin-bottom:0.35rem;">{topic_label or item["topic_id"]}</div>
                    <div style="font-size:13px;color:#a8b0c8;margin-bottom:0.5rem;">Mastery: {mastery_pct}% · Attempts: {item["attempts"]}</div>
                    <div style="font-size:13px;color:#cbd5e1;">{item["recommendation"]}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="placeholder-box">Answer a concept quiz to unlock a dynamic study plan.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Suggested Topics</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Dimensional Analysis", use_container_width=True, key="chip1"):
        st.session_state.chip_query = "What is dimensional analysis?"
with col2:
    if st.button("Newton's Laws", use_container_width=True, key="chip2"):
        st.session_state.chip_query = "Explain Newton's laws"
with col3:
    if st.button("Ohm's Law", use_container_width=True, key="chip3"):
        st.session_state.chip_query = "What is Ohm's law?"
with col4:
    if st.button("Electric Flux", use_container_width=True, key="chip4"):
        st.session_state.chip_query = "Define electric flux"

default_query = st.session_state.get('chip_query', "")

query = st.text_area(
    "Ask a question:",
    value=default_query,
    height=110,
    placeholder="Describe a concept or ask about a specific figure…",
    label_visibility="collapsed"
)

#GENERATE ANSWER
if st.button("Generate Answer", use_container_width=True):
    if not query.strip():
        st.warning("Please provide a question.")
    else:
        with st.spinner("Analyzing curriculum data..."):
            context = ""
            retrieved_chunks = {}

            title_match = RAGPipeline.detect_direct_ref(query)

            if title_match:
                retrieved_chunks = RAGPipeline.retrieve_by_title(title_match, conn)
                if retrieved_chunks:
                    context = "\n".join(chunk["content"] for chunk in retrieved_chunks.values())
                    for chunk in retrieved_chunks.values():
                        if chunk["parent_section_id"]:
                            parent = RAGPipeline.retrieve_by_id(chunk["parent_section_id"], conn)
                            if parent:
                                context += "\n" + "\n".join(parent.values())
                else:
                    ids = RAGPipeline.search_faiss(index, embedder, query, TOP_K)
                    retrieved_chunks = RAGPipeline.retrieve_similar_chunks(conn, ids)
            else:
                ids = RAGPipeline.search_faiss(index, embedder, query, TOP_K)
                retrieved_chunks = RAGPipeline.retrieve_similar_chunks(conn, ids)

            if retrieved_chunks:
                context += "\n\n".join(chunk["content"] for chunk in retrieved_chunks.values())
                for chunk in retrieved_chunks.values():
                    if chunk["parent_section_id"]:
                        parent = RAGPipeline.retrieve_by_id(chunk["parent_section_id"], conn)
                        if parent:
                            context += "\n" + "\n".join(parent.values())
                st.session_state.last_chunks = retrieved_chunks

            if context:
                full_prompt = f"""Get context for the question from the given text. Don't limit yourself to the text.{context} Question: {query}"""
                st.session_state.last_context = context
            else:
                full_prompt = f"Answer this question: {query}"

            answer = llm_client.run_ollama(full_prompt)

        st.markdown('<div style="font-size:11px;font-weight:600;color:#4a5070;text-transform:uppercase;letter-spacing:0.08em;margin-top:1.2rem;margin-bottom:6px;">EduLLM</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        ui_format.render_markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="placeholder-box">Type your question above to see the AI response.</div>', unsafe_allow_html=True)

#CONCEPT QUIZ
if "concept_questions" in st.session_state:
    questions = st.session_state.concept_questions
    idx = st.session_state.quiz_index

    if idx < len(questions):
        q = questions[idx]
        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="badge">Assessment {idx+1}/{len(questions)}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:18px;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem;">Quick Concept Check</div>', unsafe_allow_html=True)
        ui_format.render_markdown(q["question"])
        options_list = [ui_format.normalize_math_text(f"{k}) {v}") for k, v in q["options"].items()]
        user_choice = st.radio("Select the most accurate option:", options_list, key=f"concept_radio_{idx}")
        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

        if st.checkbox("Need a hint?", key=f"hint_{idx}"):
            st.info(ui_format.normalize_math_text(q["hint"]))

        if not st.session_state.quiz_answer_submitted:
            if st.button("Submit My Answer"):
                selected_letter = user_choice[0]
                if selected_letter == q["correct_option"]:
                    st.session_state.quiz_feedback = "correct"
                    st.session_state.quiz_score += 1
                    correctness = 1
                else:
                    st.session_state.quiz_feedback = "incorrect"
                    correctness = 0
                student_state.record_attempt(
                    STUDENT_ID,
                    q.get("question_id", f"concept_{idx}"),
                    q.get("topic_id", "Unknown|Unknown|Unknown"),
                    correctness
                )
                st.session_state.quiz_answer_submitted = True
                st.rerun()
        else:
            if st.session_state.quiz_feedback == "correct":
                st.success("Correct! Great understanding.")
            else:
                st.error(f"Incorrect. The correct answer was {q['correct_option']}.")
            if st.button("Continue to Next Question"):
                st.session_state.quiz_index += 1
                st.session_state.quiz_answer_submitted = False
                st.session_state.quiz_feedback = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        score = st.session_state.quiz_score
        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:20px;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">Quiz Completed 🎉</div>', unsafe_allow_html=True)
        st.write(f"You scored **{score} out of {len(questions)}** on this concept check.")
        if st.button("Finish Assessment"):
            del st.session_state.concept_questions
            del st.session_state.quiz_index
            del st.session_state.quiz_score
            del st.session_state.quiz_answer_submitted
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
