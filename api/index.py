# backend/api/index.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create the main FastAPI application
app = FastAPI()

# This is the crucial security middleware that was missing.
# It tells the server to accept requests from any other website.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# This defines the data structure we expect from the frontend
class ChatQuery(BaseModel):
    query: str

# This is our main chat endpoint that listens for POST requests
@app.post("/api/chat")
async def handle_chat(chat_query: ChatQuery):
    user_message = chat_query.query
    # This is the success response we want to see
    ai_response = f"Success! The backend is connected and received: '{user_message}'"
    return {"answer": ai_response}