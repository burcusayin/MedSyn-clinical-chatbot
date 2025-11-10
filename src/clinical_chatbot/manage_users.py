# manage_users.py
"""
# Add a user
python manage_users.py add alice --role=user

# Change password
python manage_users.py passwd alice

# Delete
python manage_users.py del alice

# List
python manage_users.py list

"""
import argparse, getpass, sys
from .db import get_conn
from .auth import hash_password

def add_user(username: str, role: str):
    pwd = getpass.getpass("Password: ")
    pwd2 = getpass.getpass("Confirm: ")
    if pwd != pwd2:
        print("Passwords do not match.", file=sys.stderr)
        return 1
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO users(username, password_hash, role) VALUES(?,?,?)",
        (username, hash_password(pwd), role)
    )
    conn.commit()
    print(f"Created/updated user '{username}' with role '{role}'.")
    return 0

def passwd(username: str):
    pwd = getpass.getpass("New password: ")
    pwd2 = getpass.getpass("Confirm: ")
    if pwd != pwd2:
        print("Passwords do not match.", file=sys.stderr)
        return 1
    conn = get_conn()
    cur = conn.execute("SELECT id FROM users WHERE username=?", (username,))
    if not cur.fetchone():
        print(f"User '{username}' not found.", file=sys.stderr)
        return 1
    conn.execute(
        "UPDATE users SET password_hash=? WHERE username=?",
        (hash_password(pwd), username)
    )
    conn.commit()
    print(f"Password updated for '{username}'.")
    return 0

def delete_user(username: str):
    conn = get_conn()
    cur = conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    if cur.rowcount:
        print(f"Deleted '{username}'.")
        return 0
    print(f"User '{username}' not found.", file=sys.stderr)
    return 1

def list_users():
    conn = get_conn()
    rows = conn.execute("SELECT username, role FROM users ORDER BY username").fetchall()
    for u, r in rows:
        print(f"{u}\t{r}")
    return 0

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Add or update a user")
    a.add_argument("username")
    a.add_argument("--role", default="user", choices=["user", "admin"])

    pw = sub.add_parser("passwd", help="Change a user's password")
    pw.add_argument("username")

    d = sub.add_parser("del", help="Delete a user")
    d.add_argument("username")

    sub.add_parser("list", help="List users")

    args = p.parse_args()
    if args.cmd == "add":
        sys.exit(add_user(args.username, args.role))
    if args.cmd == "passwd":
        sys.exit(passwd(args.username))
    if args.cmd == "del":
        sys.exit(delete_user(args.username))
    if args.cmd == "list":
        sys.exit(list_users())

if __name__ == "__main__":
    main()
