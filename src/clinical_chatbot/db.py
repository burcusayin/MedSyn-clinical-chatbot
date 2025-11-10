import os, sqlite3

DB_PATH = os.getenv("USERS_DB_PATH")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY,
      username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'user'
    )
    """)
    return conn