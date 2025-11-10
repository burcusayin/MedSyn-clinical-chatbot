# auth.py
import os
from passlib.context import CryptContext
from .db import get_conn

# Argon2 = modern & safe (no 72-byte limit issues)
pwd_ctx = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(p: str) -> str:
    return pwd_ctx.hash(p)

def verify_password(p: str, h: str) -> bool:
    return pwd_ctx.verify(p, h)

def ensure_admin():
    """
    Seed or update the admin from .env at startup.
    Uses ADMIN_PASSWORD_HASH if provided; otherwise hashes ADMIN_PASSWORD.
    """
    admin_user = os.getenv("ADMIN_USERNAME")
    admin_pw = os.getenv("ADMIN_PASSWORD")
    admin_hash = os.getenv("ADMIN_PASSWORD_HASH")

    if not admin_user or not (admin_pw or admin_hash):
        return  # nothing to do

    final_hash = admin_hash if admin_hash else hash_password(admin_pw)

    conn = get_conn()
    cur = conn.execute("SELECT password_hash FROM users WHERE username=?", (admin_user,))
    row = cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
            (admin_user, final_hash, "admin"),
        )
        conn.commit()
    else:
        # If ADMIN_PASSWORD is set, keep DB in sync (won't change if only ADMIN_PASSWORD_HASH was set)
        if admin_pw:
            conn.execute(
                "UPDATE users SET password_hash=?, role='admin' WHERE username=?",
                (final_hash, admin_user),
            )
            conn.commit()
    conn.close()
