import os
import streamlit as st
import requests
from utils import load_pdf_and_split, get_embeddings, create_vectorstore, download_pdf_from_url
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# -----------------------------
# Initialize session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📄 MANISH SINGH RAG Application with PDF + Links")
st.markdown("Upload PDFs, paste PDF links, or use existing ones in `sample_data`, then ask questions about them.")

# -----------------------------
# Ensure sample_data folder exists
# -----------------------------
pdf_folder = "sample_data"
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
    st.warning(f"📂 Created folder '{pdf_folder}'. You can upload or link PDFs now.")

# -----------------------------
# 1. Upload PDFs
# -----------------------------
st.markdown("### 📂 Step 1: Add Your PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(pdf_folder, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded and saved {len(uploaded_files)} PDF(s) to '{pdf_folder}'.")

# -----------------------------
# 2. Add PDFs from URLs
# -----------------------------
st.markdown("### 🌐 Step 2: Add PDFs via URL")
pdf_url = st.text_input("Enter PDF URL (e.g., https://example.com/file.pdf)")

if st.button("Download PDF from URL"):
    if pdf_url.lower().endswith(".pdf"):
        try:
            filename = download_pdf_from_url(pdf_url, pdf_folder)
            st.success(f"✅ Downloaded PDF from URL and saved as '{filename}'.")
        except Exception as e:
            st.error(f"❌ Error downloading PDF: {e}")
    else:
        st.error("URL must end with `.pdf`")

# -----------------------------
# 3. Load all PDFs from folder
# -----------------------------
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

if not pdf_files:
    st.error("❌ No PDFs found. Please upload files or provide links to continue.")
    st.stop()
else:
    st.success(f"📄 Found {len(pdf_files)} PDF(s): {pdf_files}")

# -----------------------------
# 4. Process PDFs
# -----------------------------
docs = []
for pdf in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf)
    docs.extend(load_pdf_and_split(pdf_path))

# -----------------------------
# 5. Create Vectorstore + Retriever
# -----------------------------
embeddings = get_embeddings()
vectorstore = create_vectorstore(docs, embeddings)
retriever = vectorstore.as_retriever()

# -----------------------------
# 6. Create QA Chain
# -----------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# -----------------------------
# 7. Chat Interface
# -----------------------------
st.markdown("### 💬 Step 3: Ask questions about your documents")
user_question = st.text_input("Enter your question:")

if user_question:
    result = qa_chain({
        "question": user_question,
        "chat_history": st.session_state.chat_history
    })
    st.session_state.chat_history.append((user_question, result["answer"]))
    st.markdown(f"**Answer:** {result['answer']}")

# -----------------------------
# 8. Show Chat History
# -----------------------------
if st.session_state.chat_history:
    st.markdown("### 📜 Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
