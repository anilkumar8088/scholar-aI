from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI 
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt # Assuming system_prompt is defined in src.prompt
import os

app = Flask(__name__)

# 1. Load environment variables from .env file
load_dotenv()

# 2. Retrieve keys from environment variables
# These variables should be defined in your .env file like:
# PINECONE_API_KEY="pcsk_..."
# GOOGLE_API_KEY="AIzaSy..."
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# ⚠️ CRITICAL FIX: The lines below are removed as they incorrectly set environment variables.
# os.environ["pcsk_6w34RU_SJY478JxddiVhBaPxwB4yGb5D2MFpD7cZ8YRWB4XPGc18dAMCrnqSQngGWythU3"] = PINECONE_API_KEY 
# os.environ["AIzaSyCblvDS5Z6QNzzAAFGxulWiRsqy1Bor3Mw"] = GOOGLE_API_KEY 

# Check for essential API keys before proceeding
if not PINECONE_API_KEY or not GOOGLE_API_KEY:
     print("Error: Missing PINECONE_API_KEY or GOOGLE_API_KEY in .env file.")
     # Optionally exit or handle this more robustly
     
# --- RAG Setup ---

# 3. Initialize Embeddings and Vector Store
embeddings = download_hugging_face_embeddings()
index_name = "scholar-ai"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    # ⚠️ Note: For LangChain, Pinecone API key is usually picked from env or needs to be passed explicitly.
    # If the key is not in os.environ, you might need to use Pinecone(api_key=...)
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":8})


# 4. Initialize LLM (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.4, 
    max_tokens=500,
    google_api_key=GOOGLE_API_KEY # Explicitly pass the key (good practice)
)

# 5. Setup Chains
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), # Use the loaded system_prompt
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- Flask Routes ---

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"]) # Changed to POST only, matching the JS
def chat():
    # Use request.form.get to safely retrieve data
    msg = request.form.get("msg")
    if not msg:
        return "Error: No message received.", 400
    
    print(f"User Input: {msg}")
    
    # The check below is redundant since we check at startup, but good for safety
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found. Check your .env file.", 500
        
    try:
        # 6. Invoke the RAG Chain
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "No answer could be generated.")
        print(f"Response: {answer}")
        
        # Flask expects a string response
        return str(answer) 
        
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        # Return a user-friendly error message
        return f"Sorry, an internal error occurred while processing your request.", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)