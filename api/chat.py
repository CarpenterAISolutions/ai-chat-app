# api/chat.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
        raise HTTPException(status_code=500, detail=f"Initialization Error: {str(e)}")

    # --- 2. Safely Extract History and Latest Query ---
    # ** THE FIX IS HERE **
    # We define latest_user_query OUTSIDE the try block so it's always available.
    try:
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
        print(f"RAG Search Warning: {str(e)}")
        context = "Could not retrieve context from the knowledge base."

    # --- 4. Generate Conversational Response with a Refined Prompt ---
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        chat_session = model.start_chat(history=full_history[:-1])

        final_prompt = f"""
        You are "Clinibot," a helpful and empathetic AI assistant for a physical therapy clinic.
        Your personality is professional, clear, and concise.
        First, consider the user's latest message: "{latest_user_query}".
        Next, consider the following relevant context from our clinic's documents: "{context}".
        Finally, consider the entire previous conversation history to understand the flow.

        Synthesize all of this information to provide the best possible response.
        If the context does not contain enough information to answer the question,
        state that you cannot find specific information in the clinic's documents and advise consulting a therapist.
        Do not make up information.
        If the user asks to simplify or explain something, use the conversation history to understand what "that" refers to.
        """

        response = chat_session.send_message(final_prompt)
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Generation Error: {str(e)}")