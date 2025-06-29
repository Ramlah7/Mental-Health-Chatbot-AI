# 🧠 MindMate: Mental Health Chatbot

**MindMate** is an intelligent, hybrid chatbot designed to offer compassionate, context-aware mental health support using a combination of semantic search (SBERT + FAISS), pattern-based intent detection, and optional generative models (BlenderBot). It is built in Python with a PyQt5 GUI and MySQL backend.

---

## 💡 Features

- ✅ Emotion-aware response engine (anxiety, depression, insomnia, panic, etc.)
- ✅ Real-time semantic retrieval with FAISS and SBERT
- ✅ Regex-based intent detection with 10+ categories
- ✅ PyQt5-based interactive desktop GUI
- ✅ Session history and message logging in MySQL
- ✅ Easily extendable with your own datasets or generative models

---

## 📂 Project Structure

```
Mental-Health-Chatbot-AI/
├── chatbot/
│   ├── generator.py
│   ├── intent.py
│   ├── faq_query.py
│   ├── chatbot_engine.py
│   ├── model/
│   │   ├── faiss_index/
│   │   └── mindmate_dialo/
│   ├── data/
│   │   └── mental_health_tagged.csv
├── gui/
├── database/
│   └── database_handler.py
```

---

## 🧠 How It Works

1. **User Input** → Intent is detected using `intent.py`
2. If FAISS match score is high → SBERT semantic reply is returned
3. Otherwise → fallback response from `generator.py` is used
4. All interactions are saved to MySQL (`sessions`, `messages` tables)

---

## 💾 Installation

```bash
git clone https://github.com/yourusername/Mental-Health-Chatbot-AI.git
cd Mental-Health-Chatbot-AI

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Requirements

```
sentence-transformers
faiss-cpu
transformers
pandas
numpy
PyQt5
pymysql
```

---

## 🗄️ MySQL Schema

```sql
CREATE TABLE sessions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  title VARCHAR(255)
);

CREATE TABLE messages (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT,
  sender ENUM('user','bot'),
  content TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

---

## 🚀 Running the App

### 🧠 1. Build the FAISS index:
```bash
python chatbot/build_faiss.py
```

### 💬 2. Launch the GUI:
```bash
python gui/main_window.py
```

---

## 🧪 Sample Queries

```
User: I can't sleep because of racing thoughts
Bot: Racing thoughts are common with insomnia. Try the 4-7-8 breathing...
```

```
User: What is anxiety?
Bot: Anxiety is your body's natural response to stress. It's often...
```


---

## 📚 References

- [SBERT](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [BlenderBot](https://huggingface.co/facebook/blenderbot_small-90M)

---

> 💚 You are not alone. This project was built to listen, support, and stand by anyone going through a hard time.
