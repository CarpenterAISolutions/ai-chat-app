# api/chat.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pinecone import Pinecone
from typing import List, Dict, Any

# --- Pydantic Models for Robust Data Validation ---
class Part(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    parts: List[Part]

class ChatRequest(BaseModel):
    history: List[Message]

# --- FastAPI App Initialization ---
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
    # --- 1. Securely Initialize Services with Robust Error Handling ---
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = "physical-therapy-index"

        if not all([gemini_api_key, pinecone_api_key]):
            raise HTTPException(status_code=500, detail="Server Configuration Error: API keys are not set.")

        genai.configure(api_key=gemini_api_key)
        pc = Pinecone(api_key=pinecone_api_key)

        if pinecone_index_name not in pc.list_indexes().names():
            raise HTTPException(status_code=500, detail="Knowledge Base Error: Pinecone index not found.")

        index = pc.Index(pinecone_index_name)

    except Exception as e:
        # This will catch any error during initialization and report it.
        raise HTTPException(status_code=500, detail=f"Initialization Error: {str(e)}")

    # --- 2. Safely Extract History and Latest Query ---
    try:
        # Convert Pydantic models to clean dictionaries for the AI library
        full_history = [message.dict() for message in chat_request.history]
        latest_user_query = full_history[-1]['parts'][0]['text']
    except (IndexError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")

    # --- 3. Perform RAG Search with Error Handling ---
    context = ""
    try:
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=latest_user_query,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]

        search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        context = " ".join([match['metadata']['text'] for match in search_results['matches']])
    except Exception as e:
        # If the search fails, we can still proceed, but we log it (on the server)
        # and inform the AI that context is unavailable.
        print(f"RAG Search Warning: {str(e)}")
        context = "No context could be retrieved from the knowledge base."

    # --- 4. Generate Conversational Response with a Refined Prompt ---
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # The history for the chat session must not include the latest user query
        chat_session = model.start_chat(history=full_history[:-1])

        # This improved prompt gives the AI better instructions for conversation
        final_prompt = f"""
        You are "Clinibot," a helpful AI assistant for a physical therapy clinic.
        Your personality is professional, empathetic, and clear.
        First, consider the user's latest message: "{latest_user_query}".
        Next, consider the relevant context from our clinic's documents: "{context}".
        Finally, consider the previous conversation history.

        Synthesize all of this information to provide the best possible response.
        If the context does not contain enough information to answer the question,
        state that you cannot find specific information in the clinic's documents and advise consulting a therapist.
        Do not make up information.
        """

        response = chat_session.send_message(final_prompt)
        return {"answer": response.text}
    except Exception as e:
        # This will catch errors from the Gemini API call itself.
        raise HTTPException(status_code=500, detail=f"AI Generation Error: {str(e)}")