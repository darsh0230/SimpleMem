"""
User storage for SimpleMem MCP Server
Uses SQLite for user metadata persistence
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from ..auth.models import User


class UserStore:
    """SQLite-based user storage"""

    def __init__(self, db_path: str = "./data/users.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    openrouter_api_key_encrypted TEXT NOT NULL DEFAULT '',
                    litellm_api_key_encrypted TEXT NOT NULL DEFAULT '',
                    table_name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_table_name
                ON users(table_name)
            """)
            # Migration: Add litellm_api_key_encrypted column if it doesn't exist
            try:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN litellm_api_key_encrypted TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists
            # Migration: Add embedding_api_key_encrypted column if it doesn't exist
            try:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN embedding_api_key_encrypted TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_user(self, user: User) -> User:
        """Create a new user"""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, openrouter_api_key_encrypted, litellm_api_key_encrypted, table_name, created_at, last_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.openrouter_api_key_encrypted,
                    user.litellm_api_key_encrypted,
                    user.table_name,
                    user.created_at.isoformat(),
                    user.last_active.isoformat(),
                ),
            )
            conn.commit()
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if row:
                return User(
                    user_id=row["user_id"],
                    openrouter_api_key_encrypted=row["openrouter_api_key_encrypted"]
                    or "",
                    litellm_api_key_encrypted=row["litellm_api_key_encrypted"]
                    if "litellm_api_key_encrypted" in row.keys()
                    else "",
                    embedding_api_key_encrypted=row["embedding_api_key_encrypted"]
                    if "embedding_api_key_encrypted" in row.keys()
                    else "",
                    table_name=row["table_name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_active=datetime.fromisoformat(row["last_active"]),
                )
            return None

    def get_user_by_table(self, table_name: str) -> Optional[User]:
        """Get user by table name"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE table_name = ?",
                (table_name,),
            ).fetchone()

            if row:
                return User(
                    user_id=row["user_id"],
                    openrouter_api_key_encrypted=row["openrouter_api_key_encrypted"]
                    or "",
                    litellm_api_key_encrypted=row["litellm_api_key_encrypted"]
                    if "litellm_api_key_encrypted" in row.keys()
                    else "",
                    embedding_api_key_encrypted=row["embedding_api_key_encrypted"]
                    if "embedding_api_key_encrypted" in row.keys()
                    else "",
                    table_name=row["table_name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_active=datetime.fromisoformat(row["last_active"]),
                )
            return None

    def update_last_active(self, user_id: str):
        """Update user's last active timestamp"""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET last_active = ? WHERE user_id = ?",
                (datetime.utcnow().isoformat(), user_id),
            )
            conn.commit()

    def update_api_key(self, user_id: str, encrypted_api_key: str):
        """Update user's encrypted API key"""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET openrouter_api_key_encrypted = ?, last_active = ? WHERE user_id = ?",
                (encrypted_api_key, datetime.utcnow().isoformat(), user_id),
            )
            conn.commit()

    def update_api_keys(
        self,
        user_id: str,
        openrouter_api_key_encrypted: Optional[str] = None,
        litellm_api_key_encrypted: Optional[str] = None,
        embedding_api_key_encrypted: Optional[str] = None,
    ) -> bool:
        """
        Update one or more of the user's encrypted API keys.

        Only the columns for which a non-None value is supplied are touched.
        Returns True if the user was found and updated, False otherwise.
        """
        updates = []
        params = []

        if openrouter_api_key_encrypted is not None:
            updates.append("openrouter_api_key_encrypted = ?")
            params.append(openrouter_api_key_encrypted)

        if litellm_api_key_encrypted is not None:
            updates.append("litellm_api_key_encrypted = ?")
            params.append(litellm_api_key_encrypted)

        if embedding_api_key_encrypted is not None:
            # embedding_api_key_encrypted is a newer column; add it via migration if needed
            updates.append("embedding_api_key_encrypted = ?")
            params.append(embedding_api_key_encrypted)

        if not updates:
            return False

        updates.append("last_active = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(user_id)

        sql = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"

        with self._get_connection() as conn:
            # Ensure the embedding_api_key_encrypted column exists (migration guard)
            if embedding_api_key_encrypted is not None:
                try:
                    conn.execute(
                        "ALTER TABLE users ADD COLUMN embedding_api_key_encrypted TEXT NOT NULL DEFAULT ''"
                    )
                except sqlite3.OperationalError:
                    pass  # Column already exists

            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0

    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM users WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """List all users with pagination"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

            return [
                User(
                    user_id=row["user_id"],
                    openrouter_api_key_encrypted=row["openrouter_api_key_encrypted"]
                    or "",
                    litellm_api_key_encrypted=row["litellm_api_key_encrypted"]
                    if "litellm_api_key_encrypted" in row.keys()
                    else "",
                    table_name=row["table_name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_active=datetime.fromisoformat(row["last_active"]),
                )
                for row in rows
            ]

    def count_users(self) -> int:
        """Get total user count"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()
            return row["count"]
