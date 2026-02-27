Here is a clean `README.md` generated specifically based on your **Streamlit + Gemini Multimodal RAG implementation**:

---

# ⚡ Multimodal RAG (Streamlit + Gemini)

A lightweight **Multimodal Retrieval-Augmented Generation (RAG)** application built with:

- 🖥️ Streamlit (UI)
- 🤖 Google Google GenAI (Gemini models)
- 📄 Text + PDF ingestion
- 🖼 Image understanding
- 🔎 Vector similarity search (NumPy-based)

This app allows users to upload **PDF, TXT, JPG, and PNG** files, index them into an in-memory vector database, and perform cross-modal search with grounded answer generation.

---

## 🚀 Features

### ✅ Supported Modalities

- **PDF**

  - Text extracted using Gemini
  - Chunked and embedded

- **Text files (.txt)**

  - Recursive chunking with overlap

- **Images (.jpg / .png)**

  - Automatically described by Gemini
  - Description embedded for retrieval

### ❌ Not Supported

- Audio
- Video
- External vector databases
- Persistent storage (in-memory only)

---

## 🏗 System Architecture

```id="arch1"
User Upload (PDF / TXT / Image)
        │
        ▼
Gemini Content Extraction
        │
        ▼
Text Chunking
        │
        ▼
Gemini Embeddings (gemini-embedding-001)
        │
        ▼
In-Memory Vector Store (NumPy)
        │
        ▼
Cosine Similarity Retrieval (Top-K)
        │
        ▼
Gemini Generation (Selected Model)
        │
        ▼
Grounded Final Answer
```

---

## ⚙️ Configuration Panel

From the sidebar, users can:

- 🔑 Enter Gemini API Key
- 🤖 Select Model:

  - `gemini-3-flash-preview`
  - `gemini-2.5-flash`
  - `gemini-2.5-pro`

- 🧩 Adjust RAG Parameters:

  - Chunk Size
  - Chunk Overlap
  - Top-K Retrieval

- 🗑 Clear Index

Connection is validated before enabling indexing.

---

## 🔍 How It Works

### 1️⃣ Indexing Phase

When files are uploaded:

#### 📄 PDF Processing

- PDF sent to Gemini
- Model extracts text
- Text is chunked
- Each chunk embedded using `gemini-embedding-001`
- Stored in session state vector DB

#### 📝 Text Files

- Loaded directly
- Split using recursive chunking
- Embedded and stored

#### 🖼 Images

- Sent to Gemini for description
- Description used as embedding input
- Raw image bytes stored for later generation

All vectors are stored in:

```python
st.session_state.vector_db = [
    {
        "type": "text" | "image",
        "vec": np.array(...),
        "data": text_or_image_bytes,
        "src": source_name,
        "desc": optional_image_description
    }
]
```

---

### 2️⃣ Retrieval Phase

When a query is submitted:

1. Query is embedded
2. Cosine similarity computed using NumPy:

   ```python
   scores = dot(A, B) / (||A|| * ||B||)
   ```

3. Top-K matches selected
4. Retrieved items displayed in Inspector view
5. Retrieved content passed to Gemini for grounded answer generation

---

### 3️⃣ Generation Phase

The model receives:

```
Query: <user question>
Answer based on this context:
<retrieved text chunks>
<retrieved images>
```

The selected Gemini model generates a final grounded response.

---

## 🧠 RAG Tuning Parameters

| Parameter       | Purpose                              |
| --------------- | ------------------------------------ |
| Chunk Size      | Controls text chunk length           |
| Chunk Overlap   | Preserves context continuity         |
| Top-K           | Number of retrieved chunks           |
| Model Selection | Controls generation quality vs speed |

---

## 🖥 Inspector Mode

Two visual debugging views:

### 📚 Initial Index Viewer

- Shows indexed chunks in a 5-column grid
- Text preview preserves whitespace
- Images rendered directly

### 🎯 Retrieved Matches Viewer

- Displays similarity score
- Shows exact content passed to model
- Helps debug retrieval quality

---

## 📦 Dependencies

- `streamlit`
- `google-genai`
- `numpy`
- `io`

Install:

```bash
pip install streamlit google-genai numpy
```

Run:

```bash
streamlit run app.py
```

---

## 🎯 Use Cases

- PDF-based knowledge assistants
- Diagram-aware Q&A
- Document + image search
- Lightweight multimodal prototyping
- RAG experimentation without external vector DB

---

## 🛡 Design Decisions

- In-memory vector store (simple + transparent)
- Gemini-based PDF text extraction (no external PDF parser)
- Image → description → embedding pipeline
- Cosine similarity via NumPy
- Debug-first UI with full inspection

---

## ⚠️ Limitations

- No persistent storage
- No batching optimizations
- No hybrid BM25 retrieval
- No metadata filtering
- PDF extraction quality depends on Gemini output
- All embeddings use a single embedding model

---

## 🔮 Possible Improvements

- Add persistent vector DB (e.g., FAISS)
- Add hybrid retrieval
- Add metadata filtering
- Support structured citations
- Add streaming responses
- Add reranking stage
- Add table-aware extraction

---

## 🏁 Summary

This project demonstrates a clean, transparent implementation of:

- ✅ Multimodal ingestion (PDF + Text + Images)
- ✅ Gemini-based embeddings
- ✅ NumPy cosine similarity retrieval
- ✅ Streamlit debugging interface
- ✅ Grounded multimodal answer generation

It is ideal for experimentation, demos, and understanding how Multimodal RAG works end-to-end without complex infrastructure.
