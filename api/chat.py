# api/index.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pinecone import Pinecone

# --- Initialize FastAPI App ---
app = FastAPI()

# --- Add CORS Middleware for browser security ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define the data structure for incoming chat queries ---
class ChatQuery(BaseModel):
    query: str

# --- Main Chat Endpoint ---
@app.post("/api/chat")
async def handle_chat(chat_query: ChatQuery):
    # 1. Securely get API Keys from Vercel's Environment Variables
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        pinecone_index_name = "physical-therapy-index"

        if not all([gemini_api_key, pinecone_api_key, pinecone_environment]):
            raise ValueError("One or more API keys are not configured in the Vercel environment.")
    except Exception as e:
        return {"answer": f"Server Configuration Error: {e}"}

    # 2. Configure the AI and Database services
    try:
        genai.configure(api_key=gemini_api_key)
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
    except Exception as e:
        return {"answer": f"Error initializing AI services: {e}"}

    user_query = chat_query.query

    # This is a placeholder response.
    # After you load the clinic's data, this will be replaced by the real AI logic.
    # For now, it confirms that the backend can successfully connect to the services.
    try:
        # We can test our connection by describing the index.
        index_stats = index.describe_index_stats()
        vector_count = index_stats.get('total_vector_count', 0)

        if vector_count == 0:
             ai_response = f"The AI is connected, but the knowledge base is empty. Ready to load clinic data. You asked: '{user_query}'"
        else:
             ai_response = f"The AI is connected, and the knowledge base has {vector_count} documents. Ready for real queries. You asked: '{user_query}'"

        return {"answer": ai_response}

    except Exception as e:
        return {"answer": f"An error occurred during processing: {e}"}

