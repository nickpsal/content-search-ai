# 🎓 Content-Based Search in Multimedia Digital Archives using Artificial Intelligence

This repository is part of a university thesis project focused on **semantic search** within **multimedia digital 
archives** — primarily **images** and **PDF documents** — using **Artificial Intelligence** models such as **CLIP** 
and **M-CLIP (multilingual CLIP)**.

The system allows users to perform **text-to-image**, **image-to-image**, **text-to-PDF**, and **PDF-to-PDF** 
similarity searches through a unified Streamlit web interface.

---

## 📁 Project Structure

```
content-search-ai/
├── data/                   # Datasets (COCO images, PDFs, embeddings)
│   ├── images/
│   ├── pdfs/
│   ├── annotations/
│   └── embeddings/
├── core/                   # Core logic (ImageSearcher, PDFSearcher)
├── app.py                  # Streamlit web interface
├── main.py                 # CLI execution script
├── environment.yml         # Conda environment definition
├── requirements.txt        # PIP installation file
└── README.md               # Project documentation
```

---

## 🚀 Implemented Features

| Category                        | Description                                                       | Status        |
|---------------------------------|-------------------------------------------------------------------|---------------|
| 🖼️ **Text → Image Search**     | Search for similar images using a text prompt (via CLIP & M-CLIP) | ✅ Implemented |
| 🖼️ **Image → Image Search**    | Visual similarity detection using image embeddings                | ✅ Implemented |
| 📚 **PDF → PDF Search**         | Compare PDF documents on a per-page semantic level (M-CLIP)       | ✅ Implemented |
| 💬 **Text → PDF Search**        | Semantic text-to-document search inside PDF archives              | ✅ Implemented |
| 🎧 **Audio Phrase Search**      | Speech-to-text and semantic similarity using Whisper              | 🚧 Planned    |
| 🎥 **Video Content Search**     | Frame-based and transcript-based video indexing                   | 🚧 Planned    |

---

## 🧠 Technologies Used

- **OpenAI CLIP / M-CLIP** – for multilingual image-text embeddings  
- **Sentence Transformers (SBERT)** – for PDF and text similarity  
- **PyTorch** – model inference and tensor computation  
- **FAISS** – vector-based similarity search  
- **Streamlit** – interactive user interface  
- **Deep Translator** – automatic language translation (Greek ↔ English)  
- **PyMuPDF (fitz)** – PDF parsing and text extraction  

---

## ⚙️ How It Works

The system computes embeddings for all supported media types and stores them in vector form (`.pt` files).  
When a new query (text, image, or PDF) is provided, the model encodes it into the same vector space and compares 
it against the stored embeddings using **cosine similarity**.

---

## 🧩 Execution Modes

### 1️⃣ CLI Mode (Command Line)
Run the backend directly from the terminal:

```bash
python main.py
```

This will:
- Download the COCO dataset if not available  
- Generate embeddings for images and captions  
- Perform text-based image search via command-line output  

---

### 2️⃣ Streamlit Web Interface (Recommended)

Launch the full web app:
```bash
streamlit run app.py
```

Then open your browser:
```
http://localhost:8501
```

The interface includes:
- Tabs for text-to-image, image-to-image, and PDF searches  
- Download buttons for dataset preparation  
- Real-time similarity scoring and visual results display  

---

## ☁️ Deployment Notes

- For **local experiments**, the app runs with the full COCO dataset and user-provided PDFs.  
- For **Streamlit Cloud / Hugging Face Spaces**, use a reduced dataset (e.g., 200–500 samples) with precomputed 
embeddings.  
- GPU acceleration is **automatically enabled** when CUDA is available.

---

## 🧭 Current Development Progress

| Phase       | Description                                               | Status         |
|-------------|-----------------------------------------------------------|----------------|
| **Phase 1** | Text & image semantic search (CLIP/M-CLIP)                | ✅ Completed    |
| **Phase 2** | PDF semantic similarity search (text-to-PDF & PDF-to-PDF) | ✅ Completed    |
| **Phase 3** | Audio similarity using Whisper                            | 🚧 Not started |
| **Phase 4** | Video content search and indexing                         | 🚧 Not started |

---

## 🧪 Example Queries

| Input Type    | Example Query                        | Output                                                  |
|---------------|--------------------------------------|---------------------------------------------------------|
| Text → Image  | “A cat on green grass”               | Returns top-5 COCO images with cosine similarity scores |
| Image → Image | Upload any photo                     | Finds visually similar images                           |
| PDF → PDF     | Upload a thesis PDF                  | Finds other PDFs with semantically related text         |
| Text → PDF    | “Neural networks for classification” | Locates documents discussing deep learning topics       |

---

## 🧰 Installation

### Using Conda
```bash
conda env create -f environment.yml
conda activate content-search-ai
```

### Using pip
```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author
**Thesis by:** Nikolaos Psaltakis  
**Institution:** University of West Attica  
**Department:** Computer Science  
**Year:** 2025  

---

## 📜 License
This project is developed exclusively for **academic and research purposes**.  
Redistribution or commercial use is not permitted without written permission.
