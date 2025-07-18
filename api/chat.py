# api/chat.py
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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
app = FastAPI()

# --- Master Error Handler ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected server error occurred: {exc}"},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Main Chat Endpoint ---
# Because the file is named chat.py, Vercel routes /api/chat here.
# The "@app.post('/')" tells FastAPI to handle the request at the root of this file.
@app.post("/")
async def handle_chat(chat_request: ChatRequest):
    # --- 1. Securely Initialize ALL Services ---
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_host = os.getenv("LANGFUSE_HOST")
        pinecone_index_name = "physical-therapy-index"

        if not all([gemini_api_key, pinecone_api_key, langfuse_secret_key, langfuse_public_key, langfuse_host]):
            raise HTTPException(status_code=500, detail="Server Configuration Error: One or more API keys (GEMINI, PINECONE, or LANGFUSE) are not set in the Vercel environment.")

        genai.configure(api_key=gemini_api_key)
        pc = Pinecone(api_key=pinecone_api_key)
        langfuse = Langfuse(secret_key=langfuse_secret_key, public_key=langfuse_public_key, host=langfuse_host)

        if pinecone_index_name not in pc.list_indexes().names():
            raise HTTPException(status_code=500, detail="Knowledge Base Error: Pinecone index not found.")

        index = pc.Index(pinecone_index_name)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization Error: {str(e)}")

    # --- 2. Safely Extract History and Latest Query ---
    history = [message.dict() for message in chat_request.history]
    user_query = history[-1]['parts'][0]['text'] if history else ""
    trace = langfuse.trace(name="rag-pipeline", user_id="end-user-123", input={"query": user_query})

    try:
        # --- 3. Perform RAG Search ---
        retrieval_span = trace.span(name="retrieval", input={"query": user_query})
        query_embedding = genai.embed_content(model="models/text-embedding-004", content=user_query, task_type="RETRIEVAL_QUERY")["embedding"]
        search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        context = " ".join([match['metadata']['text'] for match in search_results['matches']])
        retrieval_span.end(output={"retrieved_context": context})

        # --- 4. Generate Conversational Response ---
        generation_span = trace.span(name="generation", input={"history": history, "context": context})
        chat_session = model.start_chat(history=history[:-1])
        final_prompt = f"Based on the following context: '{context}', and our previous conversation, answer the user's latest question: '{user_query}'"
        response = chat_session.send_message(final_prompt)
        ai_answer = response.text
        generation_span.end(output={"answer": ai_answer})

        trace.update(output={"final_answer": ai_answer})
        langfuse.flush()
        return {"answer": ai_answer}

    except Exception as e:
        error_message = f"An error occurred during RAG: {e}"
        trace.update(output={"error": error_message}, level="ERROR")
        langfuse.flush()
        raise HTTPException(status_code=500, detail=error_message)