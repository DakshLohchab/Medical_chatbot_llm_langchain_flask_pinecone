from dotenv import load_dotenv
import os
from src.helper import extract_text_from_pdf, filter_to_minimal_docs, text_split, get_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Fixed path - go up one directory to find data folder
extracted_data = extract_text_from_pdf("../data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chuncks = text_split(filter_data)
embedding = get_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

# Create a Pinecone index if it doesn't exist
index_name = "medical-chatbot"
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )   
index = pc.Index(index_name)

# storing text chunks in the index
vector_store = PineconeVectorStore.from_documents(
    documents=text_chuncks,
    embedding=embedding,
    index_name=index_name,    
)