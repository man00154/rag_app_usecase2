import streamlit as st
from utils import *
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("RAG Application with PDF + Links")

st.markdown("### Step 1: Download and index official PDFs and pages")
if st.button("Download & Index Data"):
    ensure_data()
    pdf_docs = load_pdfs_from_folder("sample_data")
    html_docs = load_html_from_folder("html_data")
    create_vectorstore(pdf_docs, html_docs)
    st.success("Data downloaded and indexed.")

st.markdown("### Step 2: Upload your own PDFs (optional)")
uploaded_files = st.file_uploader("Upload multiple PDFs", accept_multiple_files=True, type=["pdf"])
user_docs = []
if uploaded_files:
    for file in uploaded_files:
        with open(f"temp_{file.name}", "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(f"temp_{file.name}")
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
import fitz  # PyMuPDF
pdf_files = [f for f in os.listdir("sample_data") if f.endswith(".pdf")]
selected_pdf = st.selectbox("Choose a PDF to view", pdf_files)
if selected_pdf:
    pdf_path = os.path.join("sample_data", selected_pdf)
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    st.image(img_bytes)
