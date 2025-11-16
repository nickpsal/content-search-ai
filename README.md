
# 🎓 Content-Based Search in Multimedia Digital Archives using Artificial Intelligence

This repository is part of a university thesis project focused on **multimodal semantic search** inside **digital multimedia archives** (Images, PDFs, Audio) using **Artificial Intelligence** models such as **CLIP**, **M-CLIP**, and **Whisper**.

The system supports **text-based search**, **image similarity**, **PDF semantic retrieval**, and **audio semantic/emotion-based search**, all unified into a single Streamlit interface.

---

## 📁 Project Structure

```
content-search-ai/
├── data/
│   ├── images/                
│   ├── pdfs/                  
│   ├── audio/                 
│   ├── transcripts/           
│   ├── embeddings/            
│   └── emotions/              
│
├── core/
│   ├── image_search.py        
│   ├── pdf_search.py          
│   ├── audio_search.py        
│   ├── emotion_model_v5.py    
│   └── tools.py               
│
├── app.py                     
├── main.py                    
├── environment.yml            
├── requirements.txt
└── README.md
```

---

## 🚀 Implemented Features

| Category                        | Description                                                                     | Status        |
|--------------------------------|---------------------------------------------------------------------------------|---------------|
| 🖼️ **Text → Image**           | Text prompt to COCO/M-CLIP retrieval                                            | ✅ Implemented |
| 🖼️ **Image → Image**          | Visual similarity search using CLIP embeddings                                  | ✅ Implemented |
| 📚 **PDF → PDF**              | Semantic document comparison using M-CLIP                                       | ✅ Implemented |
| 💬 **Text → PDF**             | Text-to-document semantic search                                                | ✅ Implemented |
| 🎧 **Audio Semantic Search**  | Whisper transcription + MCLIP semantic search on transcripts                    | ✅ Implemented |
| 🎭 **Emotion Detection**      | Fine-tuned Emotion Model V5                                                     | ✅ Implemented |
| 🔊 **Keyword Spotting**       | Word-level timestamp detection via Whisper                                      | ✅ Implemented |
| 🎨 **Audio Visualization**    | Waveform, spectrogram, emotion overlay, query highlight                         | ✅ Implemented |
| 🎥 **Video Content Search**   | Frame-based & transcript-based indexing                                         | 🚧 Planned    |

---

## 🧠 Technologies Used

- **CLIP / M-CLIP (multilingual)**
- **Sentence-Transformers**
- **Whisper & Faster-Whisper**
- **Emotion Model V5 (fine-tuned)**
- **PyTorch**
- **FAISS**
- **Librosa + Matplotlib**
- **Streamlit**
- **PyMuPDF**

---

## ⚙️ How It Works

The system computes embeddings for:
- Images  
- PDFs  
- Audio transcripts  

Audio module supports:
- Word-level timestamps  
- Query-based segment highlighting  
- Emotion classification  
- Waveform + spectrogram visualization  

Similarity uses **cosine similarity**.

---

## 🧩 Execution Modes

### 1️⃣ CLI Mode
```
python main.py
```

### 2️⃣ Streamlit Web App
```
streamlit run app.py
```
Open browser:
```
http://localhost:8501
```

---

## 🧪 Example Queries

| Type           | Example Query                     | Output                                         |
|----------------|----------------------------------|------------------------------------------------|
| Text → Image   | “People on bicycles at sunset”   | COCO images ranked by similarity              |
| Image → Image  | Upload any portrait               | Similar portraits                              |
| Text → PDF     | “Neural networks”                | Relevant PDF sections                          |
| Audio Search   | “καλησπέρα”                      | Highlighted audio segment                      |
| Emotion Search | “happy”                          | Audio clips with happy emotion                 |

---

## 🧭 Development Progress

| Phase       | Description                     | Status        |
|-------------|---------------------------------|---------------|
| Phase 1     | Image search                    | ✅ Completed  |
| Phase 2     | PDF search                      | ✅ Completed  |
| Phase 3     | Audio semantic + emotion search | ✅ Completed  |
| Phase 4     | Video indexing                  | 🚧 Pending    |

---

## 🧰 Installation

### Conda
```
conda env create -f environment.yml
conda activate content-search-ai
```

### pip
```
pip install -r requirements.txt
```

---

## 👨‍💻 Author
**Thesis by:** Nikolaos Psaltakis  
**University of West Attica**  
**Department of Computer Science**  
**Year:** 2025  

---

## 📜 License
Academic use only. Commercial use requires permission.
