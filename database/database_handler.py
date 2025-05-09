# database_handler.py
import mysql.connector

# Database connection settings
db_config = {
    'host': 'localhost',
    'user': 'root',  # Your MySQL username
    'password': '5284912015',  # Your MySQL password
    'database': 'mental_health_chatbot'
}

def connect_db():
    """Connects to the MySQL database."""
    return mysql.connector.connect(**db_config)

def save_conversation(user_message, bot_reply):
    """Saves a user message and bot reply into the database."""
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO conversations (user_message, bot_reply) VALUES (%s, %s)"
    cursor.execute(query, (user_message, bot_reply))
    conn.commit()
    cursor.close()
    conn.close()

def fetch_all_conversations():
    """Fetches all conversation records."""
    conn = connect_db()
    cursor = conn.cursor()
    query = "SELECT * FROM conversations ORDER BY timestamp ASC"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

# Example usage
if __name__ == "__main__":
    save_conversation("I feel great!", "That's wonderful to hear!")
    chats = fetch_all_conversations()
    for chat in chats:
        print(chat)
