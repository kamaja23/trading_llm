"""
Authentication module for TradeBot.
Handles user registration, login, JWT sessions, and saved stocks.
"""

import sqlite3
import hashlib
import datetime
import os
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tradebot.db"
JWT_SECRET = os.environ.get("JWT_SECRET", "tradebot-default-secret-change-me")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_DAYS = 30


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            company_name TEXT DEFAULT '',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, ticker)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _make_token(username: str, user_id: int) -> str:
    raw = f"{user_id}:{username}:{datetime.datetime.utcnow().isoformat()}:{JWT_SECRET}"
    return hashlib.sha256(raw.encode()).hexdigest()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 2:
        return False, "Username must be at least 2 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, _hash_password(password)),
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> tuple[bool, str, dict | None]:
    conn = get_db()
    row = conn.execute(
        "SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
        (username.strip(), _hash_password(password)),
    ).fetchone()
    conn.close()

    if row is None:
        return False, "Invalid username or password.", None
    return True, "Login successful!", dict(row)


def create_session(user_id: int, username: str) -> str:
    token = _make_token(username, user_id)
    expires = datetime.datetime.utcnow() + datetime.timedelta(days=TOKEN_EXPIRY_DAYS)
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user_id, token, expires.isoformat()),
    )
    conn.commit()
    conn.close()
    return token


def verify_session(token: str) -> tuple[bool, dict | None]:
    conn = get_db()
    row = conn.execute(
        """
        SELECT s.user_id, s.expires_at, u.username
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token = ?
        """,
        (token,),
    ).fetchone()
    conn.close()

    if row is None:
        return False, None
    row = dict(row)
    expires = datetime.datetime.fromisoformat(row["expires_at"])
    if expires < datetime.datetime.utcnow():
        return False, None
    return True, {"user_id": row["user_id"], "username": row["username"]}


def delete_session(token: str):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def save_stock(user_id: int, ticker: str, company_name: str = ""):
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO saved_stocks (user_id, ticker, company_name) VALUES (?, ?, ?)",
        (user_id, ticker.upper(), company_name),
    )
    conn.commit()
    conn.close()


def remove_stock(user_id: int, ticker: str):
    conn = get_db()
    conn.execute(
        "DELETE FROM saved_stocks WHERE user_id = ? AND ticker = ?",
        (user_id, ticker.upper()),
    )
    conn.commit()
    conn.close()


def get_saved_stocks(user_id: int) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT ticker, company_name FROM saved_stocks WHERE user_id = ? ORDER BY added_at ASC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
