import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_TYPES = {"pdf", "txt", "docx", "doc"}


def load_document(file_path: str, file_type: str) -> list:
    """Load a document from disk and return list of Document objects."""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_type in {"docx", "doc"}:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支援的檔案格式: {file_type}（支援: {', '.join(SUPPORTED_TYPES)}）")
    return loader.load()


def split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 80) -> list:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " ", ""],
    )
    return splitter.split_documents(docs)


def process_uploaded_file(uploaded_file) -> list:
    """
    Process a Streamlit UploadedFile object.
    Returns a list of document chunks ready for embedding.
    """
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if file_ext not in SUPPORTED_TYPES:
        raise ValueError(f"不支援的檔案格式: .{file_ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path, file_ext)
        chunks = split_documents(docs)
        return chunks
    finally:
        os.unlink(tmp_path)
