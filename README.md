# MindMate: Empathetic Mental Health Chatbot

A PyQt5-based intelligent chatbot that combines emotion-aware dialogue management, FAISS-based semantic search using Sentence-BERT, intent classification, and a fine-tuned Blenderbot-small model to provide accessible mental health support.

---

## 📌 Overview

**MindMate** is a multi-strategy mental health chatbot designed to offer 24/7 empathetic conversation, detect emotional distress, and provide appropriate support including referral to professional help in crisis situations.

This project was developed by **Ramlah Munir**, CS Student at COMSATS University, Islamabad.

---

## 🔧 Features

* 💬 Hybrid Chatbot: Combines Rule-based + Retrieval + Generative approaches
* 🎯 Intent Classification: Regex + rule-based logic (in `intent.py`)
* 🔎 Semantic Retrieval: Sentence-BERT + FAISS (via `semantic.py`)
* 🧠 Generative Model: Fine-tuned `facebook/blenderbot_small-90M` on mental health JSONL dataset
* 📚 Session Logging: All conversations stored securely in MySQL
* 🌐 GUI: PyQt5-based chat interface with rich formatting
* 🛑 Crisis Detection: Auto-response if user shows distress or uses trigger phrases

---

## 🗂️ Directory Structure

```
MindMate/
├── build mindmate_dialo.py       # Blenderbot-small fine-tuning script
├── build fiass.py                # FAISS index builder from SBERT embeddings
├── generator.py                  # Inference handler using HuggingFace
├── intent.py                     # Intent classification logic
├── semantic.py                   # Sentence-BERT + FAISS semantic search
├── router.py                     # Hybrid strategy selector (rule / semantic / generative)
├── train_formatted.jsonl         # Dataset for fine-tuning
├── model/
│   └── mindmate_finetuned/       # Trained Blenderbot-small model directory
├── faiss_index/
│   ├── index.bin                 # FAISS vector search index
│   └── responses.csv             # Response mapping
├── data/
│   └── mental_health_tagged.csv # Raw input and response pairs
├── gui/
│   └── mainwindow.py             # PyQt5 GUI logic
├── requirements.txt              # All Python dependencies
└── README.md
```

---

## 🧠 Model Training (Blenderbot-small)

Fine-tuned `facebook/blenderbot_small-90M` using custom JSONL with mental health dialogues.

✅ Script: [`build mindmate_dialo.py`](./build%20mindmate_dialo.py)
✅ Dataset: [`train_formatted.jsonl`](./train_formatted.jsonl)
✅ Output Model: `model/mindmate_finetuned/`

Each training sample includes `prompt`, `instruction`, and `response` fields. Training was performed using HuggingFace `Trainer` API on Colab with GPU/TPU.

Sample JSON:

```json
{
  "prompt": "I'm feeling really anxious before exams",
  "instruction": "Provide supportive and calming response",
  "response": "It's completely okay to feel this way before exams. Deep breathing and planning can help. Would you like me to guide you through some techniques?"
}
```

Use `generator.py` to load and generate responses using the fine-tuned model.

````

Trained model is saved in: `model/mindmate_finetuned/`

---

## 🧠 FAISS Semantic Search Index

**Script:** `build fiass.py`
This script builds a dense vector index using **Sentence-BERT** embeddings for user input queries and saves it using **FAISS** for fast similarity search.

### 🔧 Workflow:
1. Loads a dataset from: `data/mental_health_tagged.csv`
2. Extracts `user_input` → encodes with SBERT: `all-mpnet-base-v2`
3. Creates FAISS index (`IndexFlatL2`)
4. Saves:
   - FAISS vector index → `faiss_index/index.bin`
   - Associated bot replies → `faiss_index/responses.csv`

### 💃 Expected CSV Format:
```csv
user_input,bot_reply
"I'm feeling lonely","I'm here for you. You're not alone..."
"I can't sleep lately","Sleep issues are common. Want to try a calming routine?"
````

### 🏁 Output Files:

* `faiss_index/index.bin` → FAISS vector search structure
* `faiss_index/responses.csv` → Matched response lookup

This FAISS index is later queried by `faq_query.py` during inference.

---

## 💻 Setup Instructions

```bash
# Clone this repository
$ git clone https://github.com/yourusername/mindmate-chatbot.git
$ cd mindmate-chatbot

# Install requirements
$ pip install -r requirements.txt

# Download Sentence-BERT
$ pip install sentence-transformers

# Run GUI
$ python gui/mainwindow.py
```

---

## ✅ Requirements

```
torch
transformers
sentence-transformers
faiss-cpu
pyqt5
mysql-connector-python
```

---

## 📊 MySQL Logging Schema

### `sessions` Table

```sql
CREATE TABLE sessions (
  session_id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(50),
  start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  message_count INT,
  ...
);
```

### `messages` Table

```sql
CREATE TABLE messages (
  message_id INT PRIMARY KEY AUTO_INCREMENT,
  session_id VARCHAR(36),
  sender ENUM('user', 'bot'),
  message_text TEXT,
  intent_classified VARCHAR(50),
  response_method ENUM('rule_based', 'semantic_search', 'generated')
);
```

---

## 📷 Screenshots

* Happy interaction session ✅
* Crisis support session 🖘
* FAISS fallback and model response 🧠

---

## 🚀 Future Enhancements

* Voice-to-text integration
* Emotion detection using vision/audio
* More language support (Urdu, Arabic)
* Mobile app (Flutter frontend)

---

## 📄 Credits

Developed by:
**Ramlah Munir**

Model Finetuning Help: HuggingFace Transformers + Colab TPU
Data Cleaning: Self-curated empathetic dialogues (based on DailyDialog + MentalHealthReddit)

---
