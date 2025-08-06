import os
import streamlit as st
from utils import *
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import fitz  # PyMuPDF

# Ensure data folders exist before accessing
os.makedirs("sample_data", exist_ok=True)
os.makedirs("html_data", exist_ok=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("MANISH SINGH RAG Application with PDF + Links")

st.markdown("### Step 1: Download and index official PDFs and pages")
if st.button("Download & Index Data"):
    ensure_data()  # downloads and saves PDFs and HTML text files
    pdf_docs = load_pdfs_from_folder("sample_data")
    html_docs = load_html_from_folder("html_data")
    create_vectorstore(pdf_docs, html_docs)
    st.success("Data downloaded and indexed.")

st.markdown("### Step 2: Upload your own PDFs (optional)")
uploaded_files = st.file_uploader("Upload multiple PDFs", accept_multiple_files=True, type=["pdf"])
user_docs = []
if uploaded_files:
    for file in uploaded_files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        user_docs.extend(loader.load())

if st.button("Add uploaded PDFs to index"):
    if user_docs:
        vectorstore = load_vectorstore()
        embeddings = OpenAIEmbeddings()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_user_docs = splitter.split_documents(user_docs)
        vectorstore.add_documents(split_user_docs)
        vectorstore.save_local("vectorstore")
        st.success("Uploaded PDFs added to index.")
    else:
        st.warning("Upload PDFs first.")

st.markdown("### Step 3: Query the knowledge base")
query = st.text_input("Enter your question here:")
if query:
    vectorstore = load_vectorstore()
    llm = ChatOpenAI(temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))
    st.write("**Answer:**")
    st.write(result["answer"])

    st.markdown("### Chat history")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")

st.markdown("### Step 4: View indexed PDFs")

# Safely get list of PDFs; if none, show message
pdf_files = []
if os.path.exists("sample_data"):
    pdf_files = [f for f in os.listdir("sample_data") if f.endswith(".pdf")]

if pdf_files:
    selected_pdf = st.selectbox("Choose a PDF to view", pdf_files)
    if selected_pdf:
        pdf_path = os.path.join("sample_data", selected_pdf)
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            st.image(img_bytes)
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
else:
    st.info("No PDFs found in sample_data folder. Please download data first.")

