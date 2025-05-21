import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

DATA_PATH = "data/"
PDF_FILENAME = "llama2.pdf"

def load_documents():
    pdf_path =  os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {pdf_path}")    
    return documents