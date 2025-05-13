from database import database_handler

print("🔍 Attempting DB connection...")
conn = database_handler.get_db_connection()
print("✅ It worked!")
schema=database_handler.init_schema()
print("it too")

conn.close()
