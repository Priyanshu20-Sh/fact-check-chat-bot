import tempfile
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyMuPDFLoader
import os

def pdf_loader(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    docs = loader.load()
    return docs

def word_documentt(word_file):
    loader = UnstructuredWordDocumentLoader(word_file, mode="elements")
    docs = loader.load()
    text = "".join([doc.page_content for doc in docs])
    return text

def load_document(upload_file):
    upload_file.seek(0)

    file_ext = os.path.splitext(upload_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name

    try:
        if file_ext == ".pdf":
            return pdf_loader(temp_file_path)


        elif file_ext == ".docx":
            return word_documentt(temp_file_path)
        else:
            raise ValueError("Unsupported file type")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)