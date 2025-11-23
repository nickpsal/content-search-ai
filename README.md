
# 🎓 Content-Based Search in Multimedia Digital Archives using Artificial Intelligence

This repository is part of a university thesis project focused on **multimodal semantic search** inside **digital multimedia archives** (Images, PDFs, Audio) using **Artificial Intelligence** models such as **CLIP**, **M-CLIP**, and **Whisper**.

The system supports **text-based search**, **image similarity**, **PDF semantic retrieval**, **audio semantic + emotion-based search**, **real-time filesystem indexing**, and a unified **SQLite-powered** embedding database.  
All functionalities are exposed through a modern **Streamlit web interface**.

---

## 📁 Updated Project Structure (with Watchdogs + Database Integration)

```
content-search-ai/
├── data/
│   ├── images/
│   │   ├── coco/                
│   │   └── other/               # 🆕 Watchdog-monitored folder for new images
│   ├── pdfs/                    # 🆕 Watchdog-monitored folder for PDFs
│   ├── audio/
│   │   ├── AudioWAV/            # Main dataset (RAVDESS, CREMA-D etc.)
│   │   └── audio_other/         # 🆕 Watchdog-monitored folder for .wav files
│   ├── transcripts/             # Auto-generated (legacy – now replaced by DB)
│   ├── emotions/                # Cached emotion predictions (legacy)
│   └── embeddings/              # Cached transcript embeddings (legacy)
│
├── core/
│   ├── image_search.py          # CLIP/M-CLIP image retrieval
│   ├── pdf_search.py            # PDF semantic page search
│   ├── audio_search.py          # Whisper + MCLIP + Emotion Search
│   ├── emotion_model_v5.py      # Fine-tuned emotion classifier
│   ├── db/
│   │   └── database_helper.py   # 🆕 Unified DB handler (images, pdfs, audio)
│   └── watchdog/
│       ├── watch_images_other.py  # 🆕 Realtime IMAGE watcher
│       ├── watch_pdfs.py          # 🆕 Realtime PDF watcher
│       └── watch_audio_other.py   # 🆕 Realtime AUDIO watcher
│
├── app.py                       # Streamlit UI
├── main.py                      # Starts 3 watchdogs + Streamlit
├── environment.yml              # Conda environment
├── requirements.txt             # pip environment
└── README.md                    # This file
```

---

# 🚀 New Features Added

## 🔥 1. Real-Time Watchdog System (Images + PDFs + Audio)
All three folders are now monitored live:

| Folder | Watchdog File | Action |
|--------|----------------|--------|
| `data/images/other` | `watch_images_other.py` | Extract CLIP embedding → store in DB |
| `data/pdfs` | `watch_pdfs.py` | Extract page text + embedding → store in DB |
| `data/audio/audio_other` | `watch_audio_other.py` | Whisper transcription → M-CLIP → Emotion → store in DB |

### ✔ What happens automatically:
- Add new file → instantly indexed  
- Delete file → instantly removed from database  
- No manual embedding scripts anymore  
- No transcripts CSV files needed  
- No emotion cache JSON needed (stored in DB)

Everything is handled by SQLite.

---

# 🧠 Database Structure (Updated)

### `images`
```
id | filename | image_path | embedding (BLOB)
```

### `pdf_pages`
```
id | pdf_path | page_number | text_content | embedding (BLOB)
```

### `audio_embeddings`
```
id | audio_path | embedding (BLOB)
```

### `audio_emotions`
```
id | audio_path | emotion | emotion_scores_json
```

Your new system is now a **full multimodal search engine** with **continuous, real-time indexing**.

---

# ⚙️ Installation Guide (Unified – one place only)

## 1️⃣ Conda Installation (recommended)

```
conda env create -f environment.yml
conda activate content-search-ai
```

## 2️⃣ pip Installation (alternative)

```
pip install -r requirements.txt
```

---

# ▶️ How to Run the System

### **Start the full multimodal system:**
```
python main.py
```

This launches:

- 🖼 Watchdog for Images  
- 📄 Watchdog for PDFs  
- 🎧 Watchdog for Audio  
- 🌐 Streamlit UI

Access UI:  
👉 http://localhost:8501

---

# 🔥 Roadmap (Future)

- Video indexing (frame sampling + transcript + embeddings)
- Large-scale FAISS migration (GPU)
- Improved PDF OCR for scanned documents

---

# 👨‍💻 Author
**Thesis by:** Nikolaos Psaltakis  
University of West Attica – Department of Computer Science

---

# 📜 License
Academic use only.
