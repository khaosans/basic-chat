import sqlite3
import time
from typing import List, Dict

class ChatDB:
    def __init__(self, db_path: str = "chat.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def load_messages(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT role, content FROM messages ORDER BY id ASC")
        rows = c.fetchall()
        conn.close()
        return [{"role": r, "content": c} for r, c in rows]

    def save_message(self, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)", (role, content, time.time()))
        conn.commit()
        conn.close()

    def delete_message(self, idx: int):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM messages ORDER BY id ASC")
        ids = [row[0] for row in c.fetchall()]
        if 0 <= idx < len(ids):
            c.execute("DELETE FROM messages WHERE id = ?", (ids[idx],))
            conn.commit()
        conn.close()

    def clear_messages(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM messages")
        conn.commit()
        conn.close() 