# 🧠 MindMate – AI-Powered Mental Health Chatbot

MindMate is a desktop-based mental health chatbot designed to provide emotional support through empathetic, intent-aware conversations. It integrates semantic search, intent classification, and generative AI in a polished PyQt5 GUI.

---

## 🧩 Features

- 🔍 **Intent Detection**: Uses regex-based rule sets to detect emotional categories (e.g., sadness, rant, overwhelm).
- 🧠 **Semantic Retrieval**: Matches user messages to pre-written replies using Sentence-BERT + FAISS.
- 🤖 **AI Generation**: Fallback to Blenderbot-small for natural replies when no match is found.
- 💾 **Session Logging**: MySQL backend for storing chat sessions and messages.
- 🎨 **GUI**: Modern PyQt5 GUI with chat bubbles, history list, and animated splash screen.
- 🏋️ **Model Training**: Colab-compatible script to fine-tune Blenderbot on a mental health empathy dataset.

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install -r requirements.txt
```

Install MySQL and set up a local database with:
```sql
CREATE DATABASE mental_health_chatbot;
```

### 🗂️ Folder Structure

```text
.
├── main_window.py
├── loading_window.py
├── chatbot_engine.py
├── generator.py
├── intent.py
├── faq_query.py
├── database_handler.py
├── build fiass.py
├── build mindmate_dialo.py
├── train_formatted.jsonl
├── model/
│   ├── mindmate_dialo/
│   └── faiss_index/
│       ├── index.bin
│       └── responses.csv
├── data/
│   └── mental_health_tagged.csv
├── gui/
│   ├── main_window_ui.py
│   └── loading_window_ui1.py
```

---

## 💡 How It Works

1. **User types a message**
2. `intent.py` classifies the emotional intent.
3. `faq_query.py` uses Sentence-BERT to search FAISS index.
4. If no high-confidence match is found → `generator.py` builds an intent-aware prompt for Blenderbot.
5. The GUI displays the bot's reply and logs the message in MySQL.

---

## 🧪 Train Your Own Blenderbot Model

Run `build mindmate_dialo.py` in Google Colab and upload your training file `train_formatted.jsonl`.

Example data format:
```json
{ "instruction": "User says something sad", "output": "It's okay to feel this way." }
```

---

## 📊 Database Schema

**Table: `sessions`**
```sql
id INT PRIMARY KEY AUTO_INCREMENT,
created_at TIMESTAMP,
title VARCHAR(255)
```

**Table: `messages`**
```sql
id INT PRIMARY KEY AUTO_INCREMENT,
session_id INT FOREIGN KEY,
sender ENUM('user', 'bot'),
content TEXT,
timestamp TIMESTAMP
```

---

## 👩‍💻 Developed By

> Ramlah Munir  
> BS Computer Science – COMSATS University  
> GitHub: [itsEkramah](https://github.com/itsEkramah)
