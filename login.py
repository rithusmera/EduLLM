import sqlite3
import hashlib

DB = "users.db"

def init_user_db():

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password):

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    hashed = hash_password(password)

    try:
        cur.execute(
            "INSERT INTO users(username,password) VALUES (?,?)",
            (username, hashed)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def verify_user(username, password):

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    hashed = hash_password(password)

    cur.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hashed)
    )

    user = cur.fetchone()

    conn.close()

    return user is not None