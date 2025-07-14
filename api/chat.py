# api/chat.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create the main FastAPI application
app = FastAPI()

# This is the crucial security middleware that was missing.
# It tells the server to accept requests from any other website.
# This directly fixes the CORS error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now. We can lock this down later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data structure for the incoming message
class ChatQuery(BaseModel):
    query: str

# This is the main function that handles the chat request.
@app.post("/")
async def handle_chat(chat_query: ChatQuery):
    user_message = chat_query.query
    ai_response = f"Success! The backend is connected and received: '{user_message}'"
    return {"answer": ai_response}