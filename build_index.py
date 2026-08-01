"""
build_index.py — run this LOCALLY (once, and again whenever course content
changes) to pre-build the FAISS knowledge index before deploying app.py.

WHY
---
Without this, the deployed app scrapes the vendor website live every time
it wakes up from sleep on Streamlit Community Cloud — slower for the first
visitor, and it fails if the website happens to be down at that moment.
Pre-building the index here and committing the result means the deployed
app just loads a file — fast, and independent of the live site's uptime.

USAGE
-----
    pip install -r requirements.txt
    python build_index.py

This creates a `faiss_index/` folder. Commit it to your git repo alongside
app.py, requirements.txt, and (optionally) a `knowledge/` folder of extra
PDFs/notes.

IMPORTANT: Keep KNOWLEDGE_URLS and KNOWLEDGE_FOLDER below identical to the
CONFIG section at the top of app.py, or the index won't match what you
think you deployed.
"""

import os

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ==== keep this identical to the CONFIG section at the top of app.py ====
KNOWLEDGE_URLS = [
    "https://timmins-consulting.com/",
    "https://timmins-consulting.com/about-us/",
    "https://timmins-consulting.com/our-approach/",
    "https://timmins-consulting.com/our-solution/",
    "https://timmins-consulting.com/domain/embedded-lnux/",
    "https://timmins-consulting.com/training-calendar/public-classes",
    "https://timmins-consulting.com/case-study/",
    "https://timmins-consulting.com/contact-us/",
]
KNOWLEDGE_FOLDER = "knowledge"
FAISS_INDEX_DIR = "faiss_index"
# ==========================================================================


def load_folder_documents(folder_path: str) -> list[Document]:
    docs: list[Document] = []
    if not os.path.isdir(folder_path):
        return docs
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            if fname.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(fpath).load())
            elif fname.lower().endswith(".txt"):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
        except Exception as e:
            print(f"Could not read {fname}: {e}")
    return docs


def main():
    docs: list[Document] = []

    if KNOWLEDGE_URLS:
        print(f"Scraping {len(KNOWLEDGE_URLS)} URL(s)...")
        loader = WebBaseLoader(KNOWLEDGE_URLS)
        loader.requests_kwargs = {"timeout": 20}
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} web page(s).")

    folder_docs = load_folder_documents(KNOWLEDGE_FOLDER)
    print(f"Loaded {len(folder_docs)} file(s) from '{KNOWLEDGE_FOLDER}/'.")
    docs.extend(folder_docs)

    if not docs:
        raise SystemExit(
            "No documents found — check KNOWLEDGE_URLS and the knowledge/ folder."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks. Loading embedding model (first run downloads it)...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)

    print(f"\n✅ Saved index to '{FAISS_INDEX_DIR}/'.")
    print("Commit this folder to your git repo, then push and deploy/redeploy.")


if __name__ == "__main__":
    main()
