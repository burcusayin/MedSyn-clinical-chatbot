# seed_user.py
import sqlite3
from passlib.context import CryptContext

# Choose ONE of these contexts:
# pwd_ctx = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
# OR:
pwd_ctx = CryptContext(schemes=["argon2"], deprecated="auto")

DB = "users.db"
USERNAME = "admin"
PLAINTEXT = "changeme"  # change after first login
ROLE = "admin"

conn = sqlite3.connect(DB)
conn.execute("""
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT DEFAULT 'user'
)
""")
password_hash = pwd_ctx.hash(PLAINTEXT)
conn.execute(
    "INSERT OR REPLACE INTO users(username, password_hash, role) VALUES(?,?,?)",
    (USERNAME, password_hash, ROLE),
)
conn.commit()
print(f"Created {USERNAME} / {PLAINTEXT}")
