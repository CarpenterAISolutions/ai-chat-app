# api/index.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# This is the main application object that Vercel will find and run.
app = FastAPI()

# This adds the security middleware that allows your frontend to talk to the backend.
# This was missing in some previous versions and is critical.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This defines the data structure we expect from the frontend.
class ChatQuery(BaseModel):
    query: str

# This is the endpoint that will handle the chat requests.
@app.post("/api/chat")
async def handle_chat(chat_query: ChatQuery):
    user_message = chat_query.query
    ai_response = f"Success! The backend is connected and received: '{user_message}'"
    return {"answer": ai_response}