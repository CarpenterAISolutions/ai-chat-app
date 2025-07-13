# api/index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# This is the crucial middleware that was missing.
# It allows your frontend website to securely talk to your backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all websites to talk to it
    allow_credentials=True,
    allow_methods=["*"],  # Allows all types of requests
    allow_headers=["*"],  # Allows all headers
)

class ChatQuery(BaseModel):
    query: str

@app.post("/api/chat")
async def handle_chat(chat_query: ChatQuery):
    user_message = chat_query.query
    ai_response = f"Success! The backend is connected and received: '{user_message}'"
    return {"answer": ai_response}