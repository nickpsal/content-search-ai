# 🎓 Content-Based Search in Multimedia Digital Archives using Artificial Intelligence (v1.8)

This repository is part of a university thesis project focused on **content-based multimodal search** in **digital multimedia archives** (Images, PDFs, Audio) using **Artificial Intelligence**.

The system supports:
- **Text → Image search**
- **Image → Image similarity**
- **Text → PDF semantic retrieval**
- **PDF → PDF document similarity**
- **Text / Emotion → Audio search**
- **Real-time filesystem indexing**
- **Unified SQLite database**
- **Explainable retrieval results**

All functionalities are exposed through a **Streamlit web interface**.

---

## 🧠 Core Design Principles

- **Pure embedding-based retrieval**
- **No hard rules / no keyword filters in the core**
- **Explainability layer separated from retrieval**
- **Stable retrieval core (not modified once validated)**

---

## 📁 Project Structure (Current – Clean & Stable)

```
content-search-ai/
├── data/
│   ├── images/                 # Indexed image archive
│   ├── pdfs/                   # Indexed PDF archive
│   ├── audio/                  # Indexed audio archive (.wav)
│   ├── transcripts/            # Audio transcripts (if present/used by your pipeline)
│   ├── query/                  # Uploaded query PDFs (runtime)
│   └── query_images/           # Uploaded query images (runtime)
│
├── core/
│   ├── image_search.py         # CLIP / M-CLIP image retrieval
│   ├── pdf_search.py           # PDF page-level semantic retrieval + PDF→PDF similarity
│   ├── audio_search.py         # Audio search (transcript keywords + emotion)
│   ├── emotion_model_v5.py     # Fine-tuned audio emotion classifier (v5)
│   └── db/
│       └── database_helper.py  # Unified SQLite handler
│
├── app.py                      # Streamlit UI
├── main.py                     # App entry point
├── content_search_ai.db        # ✅ SQLite database (images/pdfs/audio embeddings & metadata)
├── environment.yml             # Conda environment
├── requirements.txt            # pip environment
└── README.md                   # This file
```

---

## 🔍 Supported Search Modes

### 🖼 Image Search
- **Text → Image** (CLIP / M-CLIP embeddings)
- **Image → Image similarity**
- Confidence score based on similarity distribution (UI explainability only)

### 📄 PDF Search
- **Text → PDF page retrieval**
- **PDF → PDF similarity** (document-level semantic similarity)
- Semantic similarity between text embeddings
- Explainability via **most similar paragraph per page**
- Confidence score for UI explainability only

### 🎧 Audio Search
- **Text → Audio** (via transcript keyword search)
- **Emotion → Audio** (emotion-only search)
- Emotion detection using **Emotion Model v5**
- No audio embeddings used
- Emotion probabilities available for explainability

---

## 🧠 Explainability Layer

Each modality provides:
- **Computational Summary** (counts / scale)
- **Top-K numerical table**
- **Confidence score** (does NOT affect ranking)
- **Explainable evidence**
  - Images: similarity strength + confidence label
  - PDFs: most similar paragraph within page
  - Audio: detected emotion + probabilities

Explainability **never affects ranking**, only UI transparency.

---

## 🗄️ Database (SQLite)

Database file:
- `content_search_ai.db`

Tables (current):
- `images` (image metadata + embeddings)
- `pdf_pages` (pdf page text + embeddings)
- `audio_files` (audio metadata, transcript text, emotion + emotion probabilities)

> Note: Table names are important — the system assumes the above schema.

---

## ⚙️ Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate content-search-ai
```

### pip (alternative)
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the System

```bash
python main.py
```

Then open:
👉 http://localhost:8501

---

## 📊 Current Dataset Composition

- **Images**: COCO subset + curated generic images + your custom images (target ~100 images)
- **PDFs**: Academic and technical documents
- **Audio**: WAV files with transcripts & emotion labels

---

## 🚧 Future Extensions (Planned)

- Video search (frame-based + transcript)
- FAISS-based large-scale indexing
- OCR for scanned PDFs
- Advanced multimodal fusion (late fusion layer)

---

## 👨‍💻 Author

**Nikolaos Psaltakis**  
University of West Attica  
Department of Computer Science

---

## 📜 License

Academic use only.
