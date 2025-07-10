# 🎓 Content-Based Search in Multimedia Archives using AI

This repository is part of a thesis project focused on **semantic search** in multimedia digital files — specifically **images** and **audio** — using **Artificial Intelligence** models such as CLIP and Whisper.

---

## 📁 Project Structure

```
content_retrieval_ai/
├── data/                  # COCO dataset (images, captions) + extracted embeddings
│   ├── images/
│   ├── annotations/
│   └── embeddings/
├── models/                # Pretrained or fine-tuned models (e.g., CLIP, Whisper)
├── notebooks/             # Jupyter notebooks for experimentation
├── results/               # Output results (retrieved images, evaluation)
├── scripts/               # Python scripts for processing and searching
├── utils/                 # Utility functions
├── environment.yml        # Conda environment setup
└── README.md              # Project description and usage
```

---

## 📌 Key Features

- ✅ Search images using natural language (`text → image`)
- ✅ Search similar images using image query (`image → image`)
- 🚧 Coming soon: Search audio by phrase using Whisper (`text → audio segment`)
- 💡 Future support for video content analysis

---

## 🧠 Technologies Used

- [OpenAI CLIP](https://github.com/openai/CLIP) – for joint image-text embeddings
- [PyTorch](https://pytorch.org/) – for model inference
- [COCO Dataset](https://cocodataset.org/) – for images and captions
- [OpenAI Whisper](https://github.com/openai/whisper) – (planned) for speech-to-text in audio files
- `tkinter` – (planned) for a simple desktop GUI interface

---

## 🚀 Setup Instructions

1. Clone the repository and install dependencies:
    ```bash
    conda env create -f environment.yml
    conda activate content-search-ai
    ```

2. Run scripts in order:
    ```bash
    python scripts/download_coco_val_images.py
    python scripts/download_coco_val_captions.py
    python scripts/extract_image_embeddings.py
    python scripts/extract_caption_embeddings.py
    ```

3. Perform search by text:
    ```bash
    python scripts/search_by_text.py
    ```

---

## 📅 Project Timeline

- **Phase 1**: Image search (Complete ✅)
- **Phase 2**: Audio indexing & phrase detection (In Progress)
- **Phase 3**: UI interface and potential video extension

---

## 👨‍💻 Author

Thesis by: Nikolaos Psaltakis  
Date: 2025-07-10

---

## 📜 License

This project is developed for academic purposes.
