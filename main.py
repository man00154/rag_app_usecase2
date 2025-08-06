from utils import (
    ensure_data,
    load_pdf_and_split,
    load_html_from_folder,
    create_vectorstore,
    load_vectorstore
)

def main():
    # Step 1: Download PDFs and HTML files if not already downloaded
    print("Downloading PDF and HTML files (if needed)...")
    ensure_data()

    # Step 2: Load and split PDFs
    print("Loading and splitting PDFs...")
    pdf_docs = load_pdf_and_split("sample_data")

    # Step 3: Load HTML documents
    print("Loading HTML documents...")
    html_docs = load_html_from_folder("html_data")

    # Step 4: Create vectorstore (embedding + index)
    print("Creating vectorstore...")
    create_vectorstore(pdf_docs, html_docs, store_path="vectorstore")

    # Step 5: Load vectorstore to verify it works
    print("Loading vectorstore from disk...")
    vectorstore = load_vectorstore("vectorstore")

    print("Vectorstore loaded successfully! Number of vectors:", len(vectorstore.index))

if __name__ == "__main__":
    main()
