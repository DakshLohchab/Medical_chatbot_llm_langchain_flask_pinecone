from flask import Flask, render_template,jsonify, request
from src.helper import extract_text_from_pdf, filter_to_minimal_docs, text_split, get_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embedding = get_embeddings()
index_name = "medical-chatbot"
# Embedded each chuck and upsert the embeddings into your pinecone index
vector_store = PineconeVectorStore(
    embedding=embedding,
    index_name=index_name,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Adjust k as needed
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    convert_system_message_to_human=True 

)
prompt = ChatPromptTemplate.from_messages([
    ("human", prompt_text)
])
question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})  # Fixed: changed brackets to curly braces
    print("Response: ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)