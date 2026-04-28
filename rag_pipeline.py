from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import os


def load_vector_store():
    if not os.path.exists("faiss_index"):
        raise ValueError("Vector store not found. Please upload and index documents first.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


def query_rag(query):
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = OllamaLLM(
        model="llama3",
        temperature=0
    )

    prompt = f"""
You are an AI-powered Academic Analysis System built using RAG.

You analyze:
1. Student Result Sheets
2. Question Papers

Rules:
- Use ONLY the provided context.
- Do NOT hallucinate.
- If information is missing, say:
  "The information is not available in the provided documents."
- Perform numerical reasoning carefully.
- Provide structured professional answers.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response,
        "sources": docs   # return original document objects
    }