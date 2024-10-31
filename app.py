# app.py

# Import necessary libraries
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example document chunks with metadata (replace with actual content)
# This assumes `document_metadata` is defined as a list of dicts with "text", "page", and "ano" keys.
document_texts = [item["text"] for item in document_metadata]
document_metadata = [{"page": item["page"], "ano": item["ano"]} for item in document_metadata]

# Step 1: Generate embeddings and initialize FAISS index
document_embeddings = model.encode(document_texts)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

# Step 2: Define the retrieve_answer function
def retrieve_answer(query):
    # Generate embedding for the query
    query_embedding = model.encode([query])

    # Search FAISS index for similar chunks
    _, indices = index.search(np.array(query_embedding), k=3)

    # Retrieve results with metadata
    results = []
    for idx in indices[0]:
        text = document_texts[idx]
        metadata = document_metadata[idx]
        results.append((text, metadata))
    return results

# Streamlit UI
st.title("Document Query System")
st.write("Enter a question to search the document for relevant information.")

# Input box for user query
query = st.text_input("Your Query", "")

if query:
    results = retrieve_answer(query)
    st.write("### Results")
    for i, (text, metadata) in enumerate(results):
        st.write(f"**Result {i+1}:**")
        st.write(text)
        st.write(f"**Source:** {metadata['page']} | {metadata['ano']}")
        st.write("---")
