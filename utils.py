import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# -----------------------------
# Load PDF and split into chunks
# -----------------------------
def load_pdf_and_split(pdf_path):
    """Load a PDF file and split it into smaller chunks."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    return docs

# -----------------------------
# Get OpenAI embeddings
# -----------------------------
def get_embeddings():
    """Return OpenAI embedding model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")

# -----------------------------
# Create FAISS vectorstore
# -----------------------------
def create_vectorstore(docs, embeddings):
    """Create and return a FAISS vectorstore."""
    if not docs:
        raise ValueError("No documents provided to create vectorstore.")
    return FAISS.from_documents(docs, embeddings)

# -----------------------------
# Download PDF from URL
# -----------------------------
def download_pdf_from_url(url, save_folder):
    """Download a PDF from a URL and save it locally."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = os.path.basename(url)
        save_path = os.path.join(save_folder, filename)
        with open(save_path, "wb") as f:
            f.write(response.content)
        return filename
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
