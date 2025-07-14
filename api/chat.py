# api/chat.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    query: str

# This is the main function that Vercel will route to from /api/chat
@app.post("/")
async def handle_chat(chat_query: ChatQuery):
    # This code will now check for the environment variables
    # and report back on what it finds.

    gemini_key = os.getenv("GEMINI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

    # Create a status report
    gemini_status = "Found" if gemini_key else "MISSING"
    pinecone_status = "Found" if pinecone_key else "MISSING"
    pinecone_env_status = "Found" if pinecone_env else "MISSING"

    # Send the report back to the frontend
    report = (
        f"API Key Status Report:\n"
        f"GEMINI_API_KEY: {gemini_status}\n"
        f"PINECONE_API_KEY: {pinecone_status}\n"
        f"PINECONE_ENVIRONMENT: {pinecone_env_status}"
    )

    return {"answer": report}