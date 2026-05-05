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
    .stButton>button, .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stTextArea textarea:focus {
        border-color: #4ADE80 !important;
        box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.2) !important;
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

    /* Chip Buttons Style Override */
    div[data-testid="stColumn"] button {
        background-color: #FFFFFF !important;
        color: #475569 !important;
        border: 1px solid #E2E8F0 !important;
        padding: 0.4rem 0.8rem !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        white-space: nowrap !important;
    }

    div[data-testid="stColumn"] button:hover {
        border-color: #4ADE80 !important;
        color: #4ADE80 !important;
        background-color: #F0FDF4 !important;
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
    
    .answer-box {
        background-color: #FFFFFF !important;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #F1F5F9;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
        line-height: 1.6;
        color: #1E293B !important;
    }

    .answer-box * {
        color: #1E293B !important;
    }
    
    .placeholder-box {
        border: 2px dashed #E2E8F0;
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
        color: #94A3B8 !important;
        font-weight: 500;
        background-color: #F8FAFC !important;
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

# 2. HEADER
st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <div style="font-size: 32px; font-weight: 700; color: #0F172A; letter-spacing: -0.025em;">EduLLM Tutor</div>
        <div style="font-size: 15px; color: #64748B; margin-top: 4px;">RAG-powered intelligence for Class 11-12 Science. Offline & Secure.</div>
    </div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

# 3. MAIN INPUT AREA - Chips
st.markdown('<div style="font-size: 13px; font-weight: 600; margin-bottom: 12px; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Suggested Topics</div>', unsafe_allow_html=True)
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
    height=120,
    placeholder="Describe a concept or ask about a specific figure (e.g., 'What is Figure 2.3 in Physics?')",
    label_visibility="collapsed"
)


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

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

                    context = "\n".join(
                        chunk["content"] for chunk in retrieved_chunks.values()
                    )

                    for chunk in retrieved_chunks.values():

                        if chunk["parent_section_id"]:

                            parent = RAGPipeline.retrieve_by_id(
                                chunk["parent_section_id"],
                                conn
                            )

                            if parent:
                                context += "\n" + "\n".join(parent.values())

                else:

                    ids = RAGPipeline.search_faiss(
                        index,
                        embedder,
                        query,
                        TOP_K
                    )

                    retrieved_chunks = RAGPipeline.retrieve_similar_chunks(
                        conn,
                        ids
                    )

            else:

                ids = RAGPipeline.search_faiss(
                    index,
                    embedder,
                    query,
                    TOP_K
                )

                retrieved_chunks = RAGPipeline.retrieve_similar_chunks(
                    conn,
                    ids
                )

            if retrieved_chunks:

                context += "\n\n".join(
                    chunk["content"] for chunk in retrieved_chunks.values()
                )

                for chunk in retrieved_chunks.values():

                    if chunk["parent_section_id"]:

                        parent = RAGPipeline.retrieve_by_id(
                            chunk["parent_section_id"],
                            conn
                        )

                        if parent:
                            context += "\n" + "\n".join(parent.values())

                st.session_state.last_chunks = retrieved_chunks

            if context:
                full_prompt = f"""Get context for the question from the given text. Don't limit yourself to the text.{context} Question: {query}"""

                st.session_state.last_context = context

            else:

                full_prompt = f"Answer this question: {query}"

            answer = llm_client.run_ollama(full_prompt)

        # 4. ANSWER DISPLAY
        st.markdown('<div style="margin-top: 2rem; margin-bottom: 0.5rem; font-size: 13px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">AI Response</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # 7. ANSWER WIPE BEHAVIOR - Placeholder
    st.markdown('<div class="placeholder-box">Type your question above to see the AI analysis.</div>', unsafe_allow_html=True)


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
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 14px; font-weight: 500; color: #1E293B; margin-bottom: 12px; padding: 0 0.5rem;">Knowledge Check</div>', unsafe_allow_html=True)

    if st.button("Take Concept Quiz", use_container_width=True):

        if "last_context" not in st.session_state:

            st.error("Please search for a topic first to generate a quiz.")

        else:

            first_chunk = list(st.session_state.last_chunks.values())[0]

            with st.spinner("Crafting quiz..."):

                quiz_data = concept_quiz.generate_concept_quiz(
                    st.session_state.last_context,
                    first_chunk
                )

            if quiz_data and "questions" in quiz_data:

                st.session_state.concept_questions = quiz_data["questions"]
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answer_submitted = False

                st.rerun()

            else:

                st.error("Unable to generate quiz at this time.")
    
    st.markdown("""
        <div class="sidebar-footer">
            <b>EduLLM v1.0</b><br>
            Class 11-12 Science Curriculum<br>
            RAG Pipeline · Offline Llama
        </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# CONCEPT QUIZ UI
# --------------------------------------------------

if "concept_questions" in st.session_state:

    questions = st.session_state.concept_questions
    index = st.session_state.quiz_index

    if index < len(questions):

        q = questions[index]

        # 6. CONCEPT QUIZ UI - Wrapper
        st.markdown('<div class="edu-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="badge">Assessment {index+1}/{len(questions)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 18px; font-weight: 700; color: #1E293B; margin-bottom: 1.5rem;">Quick Concept Check</div>', unsafe_allow_html=True)

        st.write(q["question"])

        options_list = [
            f"{k}) {v}" for k, v in q["options"].items()
        ]

        user_choice = st.radio(
            "Select the most accurate option:",
            options_list,
            key=f"concept_radio_{index}"
        )

        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        
        if st.checkbox("Need a hint?", key=f"hint_{index}"):
            st.info(q["hint"])

        if not st.session_state.quiz_answer_submitted:

            if st.button("Submit My Answer"):

                selected_letter = user_choice[0]

                if selected_letter == q["correct_option"]:
                    st.session_state.quiz_feedback = "correct"
                    st.session_state.quiz_score += 1

                else:
                    st.session_state.quiz_feedback = "incorrect"

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
        st.markdown(f'<div style="font-size: 20px; font-weight: 700; color: #1E293B; margin-bottom: 1rem;">Quiz Completed</div>', unsafe_allow_html=True)

        st.write(f"You scored **{score} out of {len(questions)}** on this concept check.")

        if st.button("Finish Assessment"):

            del st.session_state.concept_questions
            del st.session_state.quiz_index
            del st.session_state.quiz_score
            del st.session_state.quiz_answer_submitted

            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
