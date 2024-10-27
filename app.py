import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import requests

# GroqLLM class definition
class GroqLLM:
    def __init__(self, api_key):
        self.api_key = api_key

    def query(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta-llama-3-8b-instruct",
            "prompt": prompt,
            "temperature": 0.5
        }
        response = requests.post("https://api.groq.com/v1/meta-llama-3-8b-instruct/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            raise Exception(f"Groq API request failed: {response.text}")

# Embedding function to retrieve embeddings via the Groq API
def get_groq_embeddings(texts, batch_size=5):
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    embeddings = []
    for i in range(0, len(texts), batch_size):
        payload = {"input": texts[i:i + batch_size]}
        response = requests.post("https://api.groq.com/v1/meta-llama-3-8b-instruct/embeddings", headers=headers, json=payload)
        if response.status_code == 200:
            embeddings.extend(response.json()["data"])
        else:
            raise Exception(f"Groq API request failed: {response.text}")
    return embeddings

# Retrieve the embeddings and store them in ChromaDB
def setup_chroma_db(documents):
    embeddings = get_groq_embeddings([doc.page_content for doc in documents])
    db = Chroma.from_documents(documents, embeddings)
    return db

# RAG setup
def setup_rag(db):
    llm = GroqLLM(api_key=groq_api_key)
    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
    return qa_chain

# Streamlit interface
st.title("RAG-based Query System")
st.write("Ask a question about the organization's rules and regulations.")

groq_api_key = "your_groq_api_key_here"  # Replace with your actual Groq API key

# Assuming the documents are preloaded and processed
# Use a small sample of documents for testing
documents = [...]  # Replace with your document objects after chunking and embedding

# Initialize ChromaDB and RAG model
db = setup_chroma_db(documents)
rag_model = setup_rag(db)

# User input
user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question:
        answer = rag_model({"query": user_question})["result"]
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please enter a question.")
