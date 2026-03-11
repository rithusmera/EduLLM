import streamlit as st
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
import llm_client
import RAGPipeline
import concept_quiz
import student_state

DB_PATH = 'edu_chunks.db'
IDX_PATH = 'edu_index.faiss'
MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K_DEFAULT = 3
STUDENT_ID = "default_student"

difficulty_points = {
    "easy": 1,
    "medium": 2,
    "hard": 3
}

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(IDX_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return embedder, index, conn

embedder, index, conn = load_resources()

student_state.init_db()

st.set_page_config(page_title="EduLLM Tutor", layout="wide")
st.title("🎓 EduLLM — Offline AI Tutor")
st.caption("RAG-based Intelligent Tutoring System (Class 11–12 Science)")

st.sidebar.header("Settings")
TOP_K = st.sidebar.slider("Number of retrieved chunks", 1, 10, TOP_K_DEFAULT)
show_context = st.sidebar.checkbox("Show retrieved context", True)

query = st.text_area("Ask a question:", placeholder="e.g., Explain Figure 2.3 in Physics or What is dimensional analysis?")

if st.button("Generate Answer"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            context = ''
            title_match = RAGPipeline.detect_direct_ref(query)
            retrieved_chunks = {}

            if title_match:
                retrieved_chunks = RAGPipeline.retrieve_by_title(title_match, conn)
                if retrieved_chunks:
                    context = "\n".join(chunk['content'] for chunk in retrieved_chunks.values())
                    for chunk in retrieved_chunks.values():
                        if chunk['parent_section_id']:
                            parent = RAGPipeline.retrieve_by_id(chunk['parent_section_id'], conn)
                            if parent:
                                context += "\n" + "\n".join(parent.values())
                else:
                    ids = RAGPipeline.search_faiss(index, embedder, query, TOP_K)
                    retrieved_chunks = RAGPipeline.retrieve_similar_chunks(conn, ids)
            else:
                ids = RAGPipeline.search_faiss(index, embedder, query, TOP_K)
                retrieved_chunks = RAGPipeline.retrieve_similar_chunks(conn, ids)

            if retrieved_chunks:
                context += "\n\n".join(chunk['content'] for chunk in retrieved_chunks.values())
                for chunk in retrieved_chunks.values():
                    if chunk['parent_section_id']:
                        parent = RAGPipeline.retrieve_by_id(chunk['parent_section_id'], conn)
                        if parent:
                            context += "\n" + "\n".join(parent.values())
                
                st.session_state.last_chunks = retrieved_chunks

            if context:
                full_prompt = f"Get context for the question from the given text. Don't limit yourself to the text:\n\n{context}\n\nQuestion: {query}"
                st.session_state.last_context = context
            else:
                full_prompt = f"Answer this question: {query}"

            answer = llm_client.run_ollama(full_prompt)

        st.subheader("Answer")
        st.write(answer)

        if show_context and context:
            st.subheader("Retrieved Context")
            st.text_area("Context", context, height=300)


#quiz generation
st.sidebar.markdown("---")
st.sidebar.text('Take a quick quiz to check if you understood the concept!')
if st.sidebar.button("📝Take quiz"):
    if "last_context" in st.session_state and st.session_state.last_context:

        first_chunk = list(st.session_state.last_chunks.values())[0]
        
        with st.spinner("Generating question..."):
            quiz_data = concept_quiz.generate_concept_quiz(
                st.session_state.last_context,
                first_chunk
            )

            if quiz_data:
                st.session_state.quiz = quiz_data
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answered = False
                st.session_state.quiz_feedback = None
                st.session_state.quiz_answer_submitted = False

                st.rerun()

            else:
                st.error("🤖 The AI is having trouble formatting the question. Please try clicking the button again!")
    else:
        st.error("Please search for a topic first to provide context for the quiz!")

if "quiz" in st.session_state:

    quiz = st.session_state.quiz
    index = st.session_state.quiz_index
    questions = quiz["questions"]

    if index < len(questions):

        q = questions[index]

        st.markdown("---")
        st.subheader(f"Concept Check ({index+1}/3)")
        st.write(q["question"])

        options_list = [f"{k}) {v}" for k, v in q["options"].items()]

        user_choice = st.radio(
            "Select the correct answer:",
            options_list,
            key=f"quiz_radio_{index}"
        )

        if st.checkbox("Show hint"):
            st.info(q["hint"])

        if st.button("Submit Answer"):

            selected_letter = user_choice[0]

            if selected_letter == q["correct_option"]:
                st.session_state.quiz_feedback = "correct"
                st.session_state.quiz_score += difficulty_points[q["difficulty"]]
                correctness = 1
            else:
                st.session_state.quiz_feedback = "incorrect"
                correctness = 0

            topic_id = st.session_state.quiz["topic_id"]
            question_id = str(index)  # temporary ID since questions are dynamic

            student_state.log_attempt(
                STUDENT_ID,
                question_id,
                topic_id,
                correctness
            )

            old_mastery = student_state.get_mastery(
                STUDENT_ID,
                topic_id
            )

            new_mastery = student_state.update_mastery_score(
                old_mastery,
                correctness
            )

            student_state.save_mastery(
                STUDENT_ID,
                topic_id,
                new_mastery
            )

            st.session_state.quiz_answer_submitted = True

        if st.session_state.quiz_answer_submitted:

            if st.session_state.quiz_feedback == "correct":
                st.success("Correct!")

            else:
                st.error(f"Incorrect. Correct answer: {q['correct_option']}")

            if st.button("Next Question"):
                st.session_state.quiz_index += 1
                st.session_state.quiz_answer_submitted = False
                st.session_state.quiz_feedback = None
                st.rerun()

    else:

        score = st.session_state.quiz_score
        clarity = score / 6

        st.markdown("---")
        st.subheader("Quiz Completed")

        st.write(f"Score: {score}/6")
        st.write(f"Concept clarity score: **{clarity:.2f}**")

        if clarity >= 0.8:
            st.success("Strong understanding.")
        elif clarity >= 0.5:
            st.warning("Moderate understanding.")
        else:
            st.error("Concept needs more review.")

        if st.button("Clear Quiz"):
            del st.session_state.quiz
            del st.session_state.quiz_index
            del st.session_state.quiz_score
            st.rerun()

st.markdown("---")
st.caption("Powered by local FAISS, SQLite, and Ollama for fully offline learning.")