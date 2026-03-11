import sqlite3
import time

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
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
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

def update_mastery_score(old_score, correctness, alpha = 0.2):
    return round((old_score*(1 - alpha) + correctness*alpha), 3) #Exponential smoothing to calculate new mastery score

def get_mastery(student_id, topic_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
    SELECT mastery_score FROM topic_mastery
    WHERE student_id = ? AND topic_id = ?''',
    (student_id, topic_id))

    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]
    return 0.5 #Default mastery

def save_mastery(student_id, topic_id, new_score):
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = int(time.time())

    cursor.execute('''
    INSERT OR REPLACE INTO topic_mastery
    (student_id, topic_id, mastery_score, last_updated)
    VALUES (?,?,?,?)''',
    (student_id, topic_id, new_score, timestamp))

    conn.commit()
    conn.close()

def log_attempt(student_id, question_id, topic_id, correctness, time_taken=0):
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = int(time.time())

    cursor.execute("""
    INSERT INTO attempts
    (student_id, question_id, topic_id, correctness, time_taken, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    (student_id, question_id, topic_id, correctness, time_taken, timestamp))

    conn.commit()
    conn.close()