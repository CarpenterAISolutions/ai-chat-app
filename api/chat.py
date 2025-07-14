# api/chat.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create the main FastAPI application
app = FastAPI()

# Add the security middleware to allow the frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data structure for the incoming message
class ChatQuery(BaseModel):
    query: str

# This is the main function. Because the file is named chat.py,
# Vercel automatically routes requests from /api/chat to this file.
# The "@app.post('/')" tells FastAPI to handle the request at the root of this file.
@app.post("/")
async def handle_chat(chat_query: ChatQuery):
    user_message = chat_query.query
    ai_response = f"Success! The backend is connected and received: '{user_message}'"
    return {"answer": ai_response}