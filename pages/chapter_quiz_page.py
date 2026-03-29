import streamlit as st
import chapter_quiz
import auth

auth.require_login()

st.set_page_config(page_title="Chapter Quiz", layout="wide")

st.title("📚 Chapter Quiz")

subjects = chapter_quiz.get_available_subjects()

selected_subject = st.selectbox(
    "Select Subject",
    subjects
)

chapters = chapter_quiz.get_available_chapters(selected_subject)

selected_chapter = st.selectbox(
    "Select Chapter",
    chapters
)


if st.button("Start Quiz"):

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


if "chapter_quiz" in st.session_state:

    quiz = st.session_state.chapter_quiz
    index = st.session_state.chapter_index

    if index < len(quiz):

        q = quiz[index]

        st.subheader(f"Question {index+1}/{len(quiz)}")
        st.write(q["question"])

        options_list = [
            f"{k}) {v}" for k, v in q["options"].items()
        ]

        user_choice = st.radio(
            "Select answer",
            options_list,
            key=f"chapter_radio_{index}"
        )

        if st.button("Submit Answer"):

            selected_letter = user_choice[0]

            if selected_letter == q["correct_option"]:
                st.success("Correct")
                st.session_state.chapter_score += 1
            else:
                st.error(f"Incorrect. Correct answer: {q['correct_option']}")

            st.session_state.show_next = True

        if st.session_state.show_next:

            if st.button("Next Question"):

                st.session_state.chapter_index += 1
                st.session_state.show_next = False
                st.rerun()

    else:

        score = st.session_state.chapter_score

        st.subheader("Quiz Completed")
        st.write(f"Score: {score}/{len(quiz)}")

        if st.button("Restart Quiz"):

            del st.session_state.chapter_quiz
            del st.session_state.chapter_index
            del st.session_state.chapter_score
            del st.session_state.show_next

            st.rerun()