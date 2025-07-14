# api/index.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create the main FastAPI application. Vercel will find this 'app' object.
app = FastAPI()

# Add the security middleware to allow the frontend to connect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data structure for the incoming message.
class ChatQuery(BaseModel):
    query: str

# This is the main function. Because the file is named index.py,
# FastAPI will correctly handle the full /api/chat route.
@app.post("/api/chat")
async def handle_chat(chat_query: ChatQuery):
    user_message = chat_query.query
    ai_response = f"Success! The backend is connected and received: '{user_message}'"
    return {"answer": ai_response}