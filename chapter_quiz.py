import json
import random
import os

QUIZ_BANK = "quiz_bank"

def load_questions(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If file contains {"questions": [...]}
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]

    return data


def filter_by_chapter(questions, subject, chapter):

    filtered = []

    for q in questions:

        if not isinstance(q, dict):
            continue

        if q.get("subject") == subject and q.get("chapter") == chapter:
            filtered.append(q)

    return filtered


def sample_quiz_questions(questions, num_easy=2, num_medium=2, num_hard=1):

    easy = [q for q in questions if q["difficulty"] == 1]
    medium = [q for q in questions if q["difficulty"] == 2]
    hard = [q for q in questions if q["difficulty"] == 3]

    quiz = []

    quiz += random.sample(easy, min(num_easy, len(easy)))
    quiz += random.sample(medium, min(num_medium, len(medium)))
    quiz += random.sample(hard, min(num_hard, len(hard)))

    random.shuffle(quiz)

    return quiz


def run_quiz(quiz_questions):

    score = 0

    for i, q in enumerate(quiz_questions, 1):

        print(f"\nQuestion {i}")
        print(q["question"])

        for key, value in q["options"].items():
            print(f"{key}. {value}")

        answer = input("Your answer: ").strip().upper()

        if answer == q["correct_option"]:
            print("Correct")
            score += 1
        else:
            print("Wrong")
            print("Hint:", q["hint"])

    print(f"\nFinal Score: {score}/{len(quiz_questions)}")

def get_available_subjects():

    subjects = []

    for folder in os.listdir(QUIZ_BANK):
        path = os.path.join(QUIZ_BANK, folder)

        if os.path.isdir(path):
            subjects.append(folder.capitalize())

    return subjects


def get_available_chapters(subject):

    subject_folder = os.path.join(QUIZ_BANK, subject.lower())

    chapters = []

    for file in os.listdir(subject_folder):

        if file.endswith(".json"):

            file_path = os.path.join(subject_folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            questions = data["questions"]

            if questions:
                chapters.append(questions[0]["chapter"])

    return chapters


def get_chapter_file(subject, chapter):

    subject_folder = os.path.join(QUIZ_BANK, subject.lower())

    for file in os.listdir(subject_folder):

        file_path = os.path.join(subject_folder, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data["questions"][0]["chapter"] == chapter:
            return file_path

    return None