from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract text from PDF files
def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# filtering the data

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of document objects 
    containing only the 'source' in the metadata and the original page content.
    """
    minimal_docs = []
    for doc in docs:
        # Create a new Document object with only the 'source' and 'page_content'
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "unknown")}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs

# split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20, 
    )
    split_docs = text_splitter.split_documents(minimal_docs)
    return split_docs

# downloading model for text embeddings
from langchain.embeddings import HuggingFaceEmbeddings
def get_embeddings():
    """Download and return HuggingFace embeddings model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embeddings

embedding= get_embeddings()