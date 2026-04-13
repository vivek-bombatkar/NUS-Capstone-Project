import sqlite3
from datetime import datetime

DB_PATH = "data/feedback.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feedback_text TEXT,
        sentiment TEXT,
        product_id TEXT,
        source TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_feedback(feedback_text, sentiment="unknown", product_id="P1", source="user"):
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    cursor.execute("""
    INSERT INTO feedback (feedback_text, sentiment, product_id, source, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (feedback_text, sentiment, product_id, source, timestamp))

    conn.commit()
    conn.close()


def fetch_all_feedback():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM feedback")
    rows = cursor.fetchall()

    conn.close()
    return rows