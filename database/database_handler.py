# database/database_handler.py
import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="5284912015",
        database="mental_health_chatbot"
    )

def init_schema():
    """Call once at startup to create tables if they don’t exist."""
    db = get_db_connection()
    cur = db.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS sessions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS messages (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id INT NOT NULL,
        sender ENUM('user','bot') NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
          ON DELETE CASCADE
      );
    """)
    db.commit()
    cur.close()
    db.close()

def create_session():
    db = get_db_connection(); cur = db.cursor()
    cur.execute("INSERT INTO sessions () VALUES ();")
    session_id = cur.lastrowid
    db.commit(); cur.close(); db.close()
    return session_id

def log_message(session_id: int, sender: str, content: str):
    db = get_db_connection(); cur = db.cursor()
    cur.execute(
      "INSERT INTO messages (session_id, sender, content) VALUES (%s,%s,%s)",
      (session_id, sender, content)
    )
    db.commit(); cur.close(); db.close()

def fetch_sessions():
    db = get_db_connection(); cur = db.cursor()
    cur.execute("SELECT id, created_at FROM sessions ORDER BY created_at DESC;")
    rows = cur.fetchall()
    cur.close(); db.close()
    return rows  # list of (session_id, timestamp)

def fetch_messages(session_id: int):
    db = get_db_connection(); cur = db.cursor()
    cur.execute(
      "SELECT sender, content FROM messages "
      "WHERE session_id=%s ORDER BY timestamp;",
      (session_id,)
    )
    msgs = cur.fetchall()
    cur.close(); db.close()
    return msgs  # list of (sender, content)
