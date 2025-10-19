from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI 
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt 
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')


if not PINECONE_API_KEY or not GOOGLE_API_KEY:
     print("Error: Missing PINECONE_API_KEY or GOOGLE_API_KEY in .env file.")
     

embeddings = download_hugging_face_embeddings()
index_name = "scholar-ai"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":20})


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.4, 
    max_tokens=500,
    google_api_key=GOOGLE_API_KEY 
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), # Use the loaded system_prompt
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"]) 
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "Error: No message received.", 400
    
    print(f"User Input: {msg}")
    
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found. Check your .env file.", 500
        
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "No answer could be generated.")
        print(f"Response: {answer}")
        
        return str(answer) 
        
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return f"Sorry, an internal error occurred while processing your request.", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)