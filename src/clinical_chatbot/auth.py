# src/clinical_chatbot/auth.py
import os
from typing import Optional

from passlib.context import CryptContext

from .db import get_conn

# Argon2 = modern & safe (no 72-byte limit issues)
pwd_ctx = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plain-text password using Argon2."""
    return pwd_ctx.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify that a plain-text password matches the stored hash."""
    return pwd_ctx.verify(password, hashed)


def ensure_admin() -> None:
    """
    Ensure there is an admin user in the local SQLite DB.

    Reads the following environment variables:
      - ADMIN_USERNAME (required)
      - ADMIN_PASSWORD or ADMIN_PASSWORD_HASH (at least one required)

    Behaviour:
      - If the user does not exist, it is created with role='admin'.
      - If the user exists and ADMIN_PASSWORD is set, the hash is updated.
      - If only ADMIN_PASSWORD_HASH is set for an existing user, the hash is left
        unchanged (we assume the DB value is already correct).
    """
    admin_user: Optional[str] = os.getenv("ADMIN_USERNAME")
    admin_pw: Optional[str] = os.getenv("ADMIN_PASSWORD")
    admin_hash: Optional[str] = os.getenv("ADMIN_PASSWORD_HASH")

    # Nothing to do if we lack a username or any password source
    if not admin_user or not (admin_pw or admin_hash):
        return

    # Prefer a precomputed hash if given; otherwise hash the plain password
    final_hash = admin_hash if admin_hash else hash_password(admin_pw)  # type: ignore[arg-type]

    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE username=?",
            (admin_user,),
        )
        row = cur.fetchone()

        if row is None:
            # Create admin user if it does not exist yet
            conn.execute(
                "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
                (admin_user, final_hash, "admin"),
            )
            conn.commit()
        else:
            # If ADMIN_PASSWORD is set, keep DB in sync
            if admin_pw:
                conn.execute(
                    "UPDATE users SET password_hash=?, role='admin' WHERE username=?",
                    (final_hash, admin_user),
                )
                conn.commit()
    finally:
        conn.close()
