# 🎓 Content-Based Search in Multimedia Archives using AI

This repository is part of a thesis project focused on **semantic search** in multimedia digital files — specifically **images** and **audio** — using **Artificial Intelligence** models such as CLIP and Whisper.

---

## 📁 Project Structure

```
content-search-ai/
├── data/                   # COCO dataset (images, captions, embeddings)
│   ├── images/
│   ├── annotations/
│   └── embeddings/
├── core/                   # Core logic (image_search.py)
├── app.py                  # Streamlit web application
├── main.py                 # CLI execution script
├── environment.yml         # Conda environment setup
└── README.md               # Project documentation
```

---

## 📌 Key Features

- ✅ **Text → Image Search** using CLIP  
- ✅ **COCO Dataset integration** for image-caption pairs  
- ✅ **Dual Execution Modes**  
  - **CLI mode:** Run from terminal with `python main.py`  
  - **Web UI mode:** Run interactive app with `streamlit run app.py`  
- 🚧 Coming soon: **Audio phrase search** using Whisper  
- 💡 Future goal: **Video content indexing**

---

## 🧠 Technologies Used

- [OpenAI CLIP](https://github.com/openai/CLIP) – joint image-text embeddings  
- [PyTorch](https://pytorch.org/) – model inference  
- [COCO Dataset](https://cocodataset.org/) – images & captions  
- [OpenAI Whisper](https://github.com/openai/whisper) – (planned) for audio transcription  
- [Streamlit](https://streamlit.io/) – web interface  
- [Deep Translator](https://pypi.org/project/deep-translator/) – automatic query translation  

---

## 🚀 How to Run

You can run this project in **two modes**:

### 1️⃣ CLI Mode (Command Line)

Run the core pipeline directly through the terminal:
```bash
python main.py
```

This mode will:
- Download the COCO dataset (if missing)
- Generate image and text embeddings
- Execute a text query and display results in the console

---

### 2️⃣ Streamlit Web App (Recommended)

Run the interactive version:
```bash
streamlit run app.py
```

Then open your browser at:
```
http://localhost:8501
```

Through the web interface you can:
- Download the COCO dataset  
- Generate embeddings via buttons  
- Type a natural language query (Greek or English)  
- View the top-5 most relevant images with similarity scores  

---

## ☁️ Deployment Notes

- For **local use**, the app runs with the full COCO dataset.
- For **online demos (Streamlit Cloud)**, use a small subset (e.g., 100–200 images) and precomputed embeddings.
- For **full deployment with GPU**, use a VPS or Hugging Face Space.

---

## 📅 Project Timeline

- **Phase 1:** Image search (In Progress)  
- **Phase 2:** Audio phrase detection with Whisper (Future)  
- **Phase 3:** Full Web UI & video content analysis (Future)

---

## 👨‍💻 Author

**Thesis by:** Nikolaos Psaltakis  
**Institution:** University of West Attica  
**Date:** 2025

---

## 📜 License

This project is developed for academic and research purposes.
