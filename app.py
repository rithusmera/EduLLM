import streamlit as st
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
STUDENT_ID = "default_student"


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


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

query = st.text_area(
    "Ask a question:",
    placeholder="e.g., Explain Figure 2.3 in Physics or What is dimensional analysis?"
)


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

if st.button("Generate Answer"):

    if not query.strip():
        st.warning("Enter a question first.")

    else:

        with st.spinner("Retrieving and generating answer..."):

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

        st.subheader("Answer")
        st.write(answer)


# --------------------------------------------------
# CONCEPT QUIZ BUTTON
# --------------------------------------------------
st.sidebar.text("Take a quick quiz to check if you understood the concept!")

if st.sidebar.button("📝 Take Concept Quiz"):

    if "last_context" not in st.session_state:

        st.error("Search for a topic first.")

    else:

        first_chunk = list(st.session_state.last_chunks.values())[0]

        with st.spinner("Generating quiz..."):

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

            st.error("AI could not format the quiz.")


# --------------------------------------------------
# CHAPTER QUIZ SELECTION
# --------------------------------------------------
st.sidebar.markdown("---")

st.sidebar.page_link(
    "pages/chapter_quiz_page.py",
    label="Chapter Quiz",
    icon="📚"
)

# --------------------------------------------------
# CONCEPT QUIZ UI
# --------------------------------------------------

if "concept_questions" in st.session_state:

    questions = st.session_state.concept_questions
    index = st.session_state.quiz_index

    if index < len(questions):

        q = questions[index]

        st.markdown("---")
        st.subheader(f"Concept Quiz ({index+1}/{len(questions)})")

        st.write(q["question"])

        options_list = [
            f"{k}) {v}" for k, v in q["options"].items()
        ]

        user_choice = st.radio(
            "Select the correct answer:",
            options_list,
            key=f"concept_radio_{index}"
        )

        if st.checkbox("Show Hint", key=f"hint_{index}"):
            st.info(q["hint"])

        if not st.session_state.quiz_answer_submitted:

            if st.button("Submit Answer"):

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

        st.markdown("---")
        st.subheader("Concept Quiz Completed")

        st.write(f"Score: {score}/{len(questions)}")

        if st.button("Clear Concept Quiz"):

            del st.session_state.concept_questions
            del st.session_state.quiz_index
            del st.session_state.quiz_score
            del st.session_state.quiz_answer_submitted

            st.rerun()