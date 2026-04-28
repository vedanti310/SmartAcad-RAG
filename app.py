import streamlit as st
import os
import shutil
import tempfile
from audio_recorder_streamlit import audio_recorder
from audio_loader import transcribe_audio
from rag_pipeline import query_rag
from document_loader import load_document
from text_splitter import split_documents
from vector_store import create_vector_store

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Academic RAG System",
    page_icon="🎓",
    layout="wide"
)
st.markdown("""
<style>
.floating-btn {
    position: fixed;
    bottom: 10px;
    right: 25px;
    z-index: 9999;
}
.chat-title {
    font-size: 35px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
# ---------------- HEADER ----------------
st.markdown('<div class="chat-title">🎓 AI-Based Academic Document Intelligence System using RAG</div>', unsafe_allow_html=True)
st.caption("Retrieval Augmented Generation (RAG) System")
st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Document Control")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or Audio Files",
        type=["pdf", "docx", "mp3", "wav", "m4a", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        all_documents = []

        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                file_path = uploaded_file.name

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                docs = load_document(file_path)
                if docs:
                    all_documents.extend(docs)

            if all_documents:
                chunks = split_documents(all_documents)
                create_vector_store(chunks)

        st.success("✅ Documents indexed successfully!")

    st.divider()

    if st.button("🗑 Clear Index"):
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
            st.success("Index cleared successfully!")

    st.divider()

    st.info("""
    🔹 Model: Ollama (Llama3)  
    🔹 Embeddings: Ollama  
    🔹 Vector Store: FAISS  
    🔹 Architecture: RAG Pipeline  
    """)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- MULTI CHAT INIT ----------------

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
# ---------------- DISPLAY CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- VOICE INPUT (Optional) ----------------


# ---------------- INPUT ROW ----------------

voice_input = None

col1, col2 = st.columns([1, 8])

with col1:
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=16000,
        icon_size="2x"
    )

    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            temp_audio_path = tmp_file.name

        voice_input = transcribe_audio(temp_audio_path)
# ---------------- CHAT INPUT (STAYS STICKY AUTOMATICALLY) ----------------
with col2:
    text_input = st.chat_input("Ask a question about your uploaded documents...")

user_input = voice_input if voice_input else text_input




# ---------------- PROCESS USER INPUT ----------------
if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_rag(user_input)

            answer = result.get("answer", "No response generated.")
            sources = result.get("sources", [])

            st.markdown(answer)

            if sources:
                with st.expander("📚 View Source Chunks"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.write(doc.page_content[:400])
                        st.markdown("---")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})