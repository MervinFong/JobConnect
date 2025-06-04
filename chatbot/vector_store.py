from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import streamlit as st

# --- Load environment variables (.env) ---
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

# --- Build vector store from .txt files ---
def build_vector_store(doc_folder="jobconnect_docs"):
    docs = []

    for filepath in Path(doc_folder).glob("*.txt"):
        loader = TextLoader(str(filepath), encoding="utf-8")
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(raw_docs)
        docs.extend(split_docs)

        print(f"📄 Processed: {filepath.name} ({len(split_docs)} chunks)")

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("jobconnect_index")
    print("✅ Vector store built and saved to: jobconnect_index")

# --- Load existing vector store ---
def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    return FAISS.load_local("jobconnect_index", embeddings, allow_dangerous_deserialization=True)
