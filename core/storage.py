import os
import sqlite3
import pandas as pd
from datetime import datetime


class StorageManager:
    def __init__(self, excel_path='attendance.xlsx', db_path='history.db', datasets_dir='datasets'):
        self.excel_path = excel_path
        self.db_path = db_path
        self.datasets_dir = datasets_dir
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def is_marked_today(self, name):
        date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('SELECT id FROM attendance WHERE name=? AND date=?', (name, date))
        row = cur.fetchone()
        conn.close()
        return row is not None

    def mark_presence(self, name, status='Present'):
        if self.is_marked_today(name):
            return False
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)',
                     (name, date, time, status))
        conn.commit()
        conn.close()
        self._write_excel(name, date, time, status)
        return True

    def _write_excel(self, name, date, time, status):
        """Append a new row then rewrite the whole file from SQLite to stay in sync."""
        # Always rebuild from the source of truth (SQLite) to avoid drift
        self.rebuild_excel()

    def rebuild_excel(self):
        """Rewrite attendance.xlsx completely from SQLite. Call after any delete/clear."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT name AS Name, date AS Date, time AS Time, status AS Status '
            'FROM attendance ORDER BY date, time',
            conn
        )
        conn.close()
        df.to_excel(self.excel_path, index=False)

    def get_today(self):
        date = datetime.now().strftime('%Y-%m-%d')
        return self.get_by_date(date)

    def get_by_date(self, date):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute('SELECT name, time, status FROM attendance WHERE date=? ORDER BY time DESC', (date,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_users(self):
        if not os.path.exists(self.datasets_dir):
            return []
        return sorted([d for d in os.listdir(self.datasets_dir)
                       if os.path.isdir(os.path.join(self.datasets_dir, d))])

    def mark_absentees(self):
        registered = self.get_users()
        for user in registered:
            if not self.is_marked_today(user):
                self.mark_presence(user, status='Absent')
        return len(registered)

    def get_stats(self):
        today = self.get_today()
        return {
            'total_users': len(self.get_users()),
            'present': sum(1 for r in today if r['status'] == 'Present'),
            'absent': sum(1 for r in today if r['status'] == 'Absent'),
        }
