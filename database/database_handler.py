import pymysql
import sys

# Global connection
_global_db_connection = None

def get_db_connection():
    global _global_db_connection
    print("🔌 [DB] Trying to connect to MySQL...")

    if _global_db_connection is None or not _global_db_connection.open:
        try:
            _global_db_connection = pymysql.connect(
                host="localhost",
                user="root",
                password="5284912015",
                database="mental_health_chatbot",
                autocommit=True
            )
            print("✅ [DB] Connected using PyMySQL.")
        except pymysql.MySQLError as err:
            print(f"❌ [DB ERROR] {err}")
            sys.exit(1)
    return _global_db_connection


def init_schema():
    print("🛠 [Schema] Creating tables if not exist...")
    db = get_db_connection()
    with db.cursor() as cur:
        # ✅ Updated: sessions now includes title
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title VARCHAR(255) DEFAULT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT NOT NULL,
                sender ENUM('user','bot') NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """)
    print("✅ [Schema] Ready.")


def create_session():
    print("📝 [Session] Inserting new session...")
    db = get_db_connection()
    with db.cursor() as cur:
        cur.execute("INSERT INTO sessions () VALUES ()")
        cur.execute("SELECT LAST_INSERT_ID()")
        session_id = cur.fetchone()[0]
    print(f"✅ [Session] ID = {session_id}")
    return session_id


def log_message(session_id, sender, content):
    db = get_db_connection()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (session_id, sender, content) VALUES (%s, %s, %s)",
            (session_id, sender, content)
        )
    print(f"✅ [Log] {sender} message saved.")


def fetch_sessions():
    db = get_db_connection()
    with db.cursor() as cur:
        cur.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC")
        sessions = cur.fetchall()
    print(f"✅ [Fetch] {len(sessions)} sessions found.")
    return sessions


def fetch_messages(session_id):
    db = get_db_connection()
    with db.cursor() as cur:
        cur.execute(
            "SELECT sender, content FROM messages WHERE session_id = %s ORDER BY timestamp",
            (session_id,)
        )
        messages = cur.fetchall()
    print(f"✅ [Fetch] {len(messages)} messages found.")
    return messages


def update_session_title(session_id, title):
    db = get_db_connection()
    with db.cursor() as cur:
        cur.execute("UPDATE sessions SET title = %s WHERE id = %s", (title, session_id))
    print(f"📝 [Session] Title updated: {title}")
