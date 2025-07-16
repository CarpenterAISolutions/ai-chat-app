# api/chat.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pinecone import Pinecone
from typing import List, Dict, Any

# --- Pydantic Models ---
# Defines the structure for a single message in the history
class Message(BaseModel):
    role: str
    parts: List[Dict[str, Any]]

# Defines the structure of the entire request from the frontend
class ChatRequest(BaseModel):
    history: List[Message]

# --- FastAPI App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Main Chat Endpoint ---
@app.post("/")
async def handle_chat(chat_request: ChatRequest):
    # --- 1. Get API Keys and Configure Services ---
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = "physical-therapy-index"

        if not all([gemini_api_key, pinecone_api_key]):
            raise ValueError("API keys are not configured.")

        genai.configure(api_key=gemini_api_key)
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
    except Exception as e:
        return {"answer": f"Server Configuration Error: {e}"}

    # --- 2. Extract the latest user query and the full history ---
    full_history = [message.dict() for message in chat_request.history]
    latest_user_query = full_history[-1]['parts'][0]['text']

    # --- 3. Perform RAG to find relevant context ---
    try:
        # Embed the latest user query to find relevant documents
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=latest_user_query,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]

        # Search Pinecone for relevant context
        search_results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        context = " ".join([match['metadata']['text'] for match in search_results['matches']])
    except Exception as e:
        context = f"Could not retrieve context due to an error: {e}"

    # --- 4. Generate the Final Conversational Response ---
    try:
        # Create a new generative model instance that understands conversation history
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        chat_session = model.start_chat(history=full_history[:-1]) # Load history *without* the latest query

        # This is the corrected prompt. It's less restrictive and more effective.
        final_prompt = f"""
        Based on the following context, answer the user's question.
        Context: "{context}"
        Question: "{latest_user_query}"
        """

        response = chat_session.send_message(final_prompt)
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"An error occurred with the AI service: {e}"}