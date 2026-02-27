import streamlit as st
from google import genai
from google.genai import types
import numpy as np
import io

# --- 1. UI SETUP ---
st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("⚡ Multimodal RAG")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = []
if "connected" not in st.session_state:
    st.session_state.connected = False

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    model_id = st.selectbox(
        "Select Active Model:",
        ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )

    st.divider()
    st.subheader("🛠️ RAG Tuning")
    chunk_size = st.slider("Chunk Size (Characters)", 500, 5000, 500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 100, step=50)
    top_k = st.slider("Top-K (Chunks to Retrieve)", 1, 20, 5)
    st.divider()
    
    if st.button("🚀 Connect"):
        if api_key:
            try:
                client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
                client.models.get(model=model_id)
                st.session_state.connected = True
                st.success(f"Connected to {model_id}!")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("🗑️ Clear Index"):
        st.session_state.vector_db = []
        st.rerun()

# --- 2. UTILITIES ---
def split_text(text, max_chars, overlap):
    """Recursively splits text into smaller chunks without stripping whitespace."""
    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""
    for p in paragraphs:
        # Check if adding this paragraph exceeds limit
        if len(current_chunk) + len(p) < max_chars:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Start new chunk with overlap
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk += p + "\n\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def get_embedding(client, text=None, image_bytes=None):
    content_to_embed = text
    if image_bytes:
        try:
            res = client.models.generate_content(
                model=model_id,
                contents=["Describe this for search indexing:", 
                          types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
            )
            content_to_embed = res.text
        except: return None, None
    if content_to_embed:
        try:
            emb = client.models.embed_content(
                model="gemini-embedding-001",
                contents=[content_to_embed],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return np.array(emb.embeddings[0].values), content_to_embed
        except: return None, None
    return None, None

# --- 3. PROCESSING ---
if st.session_state.connected:
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    files = st.file_uploader("Upload Data (PDF/JPG/PNG/TXT)", type=["pdf", "jpg", "png", "txt"], accept_multiple_files=True)

    if files and not st.session_state.vector_db:
        with st.spinner("🔍 Indexing Multimodal Chunks..."):
            for f in files:
                fb = f.read()
                if f.type == "application/pdf":
                    # Send the PDF file to the model for text extraction instead of using PdfReader
                    try:
                        res = client.models.generate_content(
                            model=model_id,
                            contents=["Extract text from this PDF for search indexing:", 
                                      types.Part.from_bytes(data=fb, mime_type="application/pdf")]
                        )
                        extracted_text = res.text  # The model's response with the extracted text
                        if extracted_text:
                            page_chunks = split_text(extracted_text, chunk_size, chunk_overlap)
                            for chunk_idx, chunk_txt in enumerate(page_chunks):
                                vec, _ = get_embedding(client, text=chunk_txt)
                                if vec is not None:
                                    st.session_state.vector_db.append({"type":"text", "vec":vec, "data":chunk_txt, "src":f"{f.name}_extracted_text_{chunk_idx}"})
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                
                elif f.type == "text/plain":
                    full_text = fb.decode("utf-8", errors="ignore")
                    text_chunks = split_text(full_text, chunk_size, chunk_overlap)
                    for idx, chunk in enumerate(text_chunks):
                        vec, _ = get_embedding(client, text=chunk)
                        if vec is not None:
                            st.session_state.vector_db.append({"type":"text", "vec":vec, "data":chunk, "src":f"{f.name}_c{idx}"})
                
                elif f.type.startswith("image/"):
                    vec, desc = get_embedding(client, image_bytes=fb)
                    if vec is not None:
                        st.session_state.vector_db.append({"type":"image", "vec":vec, "data":fb, "src":f.name, "desc":desc})
        st.success(f"Indexed {len(st.session_state.vector_db)} chunks.")

    # 👀 VIEW 1: INITIAL CHUNKS (5 * n GRID)
    if st.session_state.vector_db:
        with st.expander("📚 [INSPECTOR] View Initial Index"):
            num_chunks = len(st.session_state.vector_db)
            for i in range(0, num_chunks, 5):
                cols = st.columns(5)
                for j in range(5):
                    if i + j < num_chunks:
                        item = st.session_state.vector_db[i + j]
                        with cols[j]:
                            st.caption(f"ID: {i+j} | {item['src']}")
                            if item["type"] == "image":
                                st.image(item["data"], use_container_width=True)
                            else:
                                # Using code block to show preserved whitespace in preview
                                st.code(item["data"][:150] + "...", language=None)

    # --- 4. SEARCH & RETRIEVED CHUNKS (FORM SUBMISSION) ---
    with st.form("query_form"):
        query = st.text_input("💬 Type your question here:")
        submit_button = st.form_submit_button("🔍 Search & Analyze")

    if submit_button and query and st.session_state.vector_db:
        with st.spinner("Analyzing context..."):
            q_vec, _ = get_embedding(client, text=query)
            db_vecs = np.array([x["vec"] for x in st.session_state.vector_db])
            scores = np.dot(db_vecs, q_vec) / (np.linalg.norm(db_vecs, axis=1) * np.linalg.norm(q_vec))
            
            actual_top_k = min(top_k, len(st.session_state.vector_db))
            top_idx = np.argsort(scores)[-actual_top_k:][::-1]

            st.subheader(f"🎯 [INSPECTOR] Top {actual_top_k} Retrieved Matches")
            
            retrieved_parts = []
            for row_start in range(0, actual_top_k, 5):
                cols = st.columns(5)
                for col_idx in range(5):
                    match_idx = row_start + col_idx
                    if match_idx < actual_top_k:
                        db_idx = top_idx[match_idx]
                        item = st.session_state.vector_db[db_idx]
                        with cols[col_idx]:
                            st.success(f"Match #{match_idx+1}\nScore: {scores[db_idx]:.3f}")
                            if item["type"] == "image":
                                st.image(item["data"], use_container_width=True)
                                retrieved_parts.append(types.Part.from_bytes(data=item["data"], mime_type="image/jpeg"))
                            else:
                                # Using code block to show preserved whitespace in retrieved view
                                st.code(item["data"], language=None)
                                retrieved_parts.append(item["data"])

            ans = client.models.generate_content(
                model=model_id,
                contents=[f"Query: {query}\nAnswer based on this context:"] + retrieved_parts
            )
            st.divider()
            st.markdown(f"### 🤖 Final Answer\n{ans.text}")