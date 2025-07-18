# api/index.py
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pinecone import Pinecone
from langfuse import Langfuse
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
# Vercel will find this 'app' object in this file.
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Main Chat Endpoint ---
# Because the app is in index.py, FastAPI can now correctly handle the full path.
@app.post("/api/chat")
async def handle_chat(chat_request: ChatRequest):
    # This is where your full AI and LangFuse logic will go.
    # For this final fix, we will return a simple success message
    # to confirm the framework and routing are working.

    history = [message.dict() for message in chat_request.history]
    user_query = history[-1]['parts'][0]['text'] if history else ""

    return {"answer": f"Success! The FastAPI backend is connected and received: '{user_query}'"}