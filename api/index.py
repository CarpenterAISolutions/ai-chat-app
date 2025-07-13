# api/index.py
from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum

# Create the main FastAPI application
app = FastAPI()

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

# This is the crucial part that was missing.
# The 'handler' is what Vercel will now talk to directly.
# Mangum acts as the translator between Vercel and our FastAPI app.
handler = Mangum(app)