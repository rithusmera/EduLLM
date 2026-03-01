import sqlite3

DB_PATH = "student_state.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS topic_mastery (
        student_id TEXT,
        topic_id TEXT,
        mastery_score REAL,
        last_updated INTEGER,
        PRIMARY KEY (student_id, topic_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attempts (
        student_id TEXT,
        question_id TEXT,
        topic_id TEXT,
        correctness REAL,
        time_taken INTEGER,
        timestamp INTEGER
    )
    """)

    conn.commit()
    conn.close()

init_db()