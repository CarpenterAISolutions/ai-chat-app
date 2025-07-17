import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- Final AI Persona (Simplified for clarity) ---
AI_PERSONA = "You are 'CliniBot,' an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic."
SIMILARITY_THRESHOLD = 0.70

# --- Helper Functions for the new multi-step logic ---

def classify_intent(history: List[Dict[str, Any]], llm_model) -> str:
    """Uses the LLM to classify the user's intent."""
    user_query = history[-1]['content']
    
    # Simple keyword check for performance
    if user_query.lower() in ['hi', 'hello', 'hey', 'thanks', 'thank you']:
        return "GREETING"
    if any(cmd in user_query.lower() for cmd in ['simplify', 'explain', 'summarize', 'rephrase']):
        return "META_COMMAND"

    # If not a simple keyword, use the LLM for nuanced understanding
    prompt = f"""
    Classify the user's "Latest Message" into one of the following categories based on the "Chat History":
    1. GREETING: A simple greeting or social response (e.g., "hi", "thanks").
    2. META_COMMAND: A command about your previous response (e.g., "simplify that", "tell me more").
    3. INFORMATION_REQUEST: A new question or statement seeking information.

    Chat History:
    {history[:-1]}

    User's Latest Message: "{user_query}"

    Category:
    """
    try:
        response = llm_model.generate_content(prompt)
        intent = response.text.strip().upper()
        print(f"Intent classified as: {intent}")
        if "GREETING" in intent: return "GREETING"
        if "META_COMMAND" in intent: return "META_COMMAND"
        return "INFORMATION_REQUEST"
    except Exception:
        return "INFORMATION_REQUEST" # Default to information request on error

def handle_meta_command(history: List[Dict[str, Any]], llm_model) -> str:
    """Handles commands like 'simplify that'."""
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    prompt = f"{AI_PERSONA}\n\nCONVERSATIONAL HISTORY:\n{history}\n\n**Instruction:** The user has issued a command about your last response. Fulfill their command now."
    response = llm_model.generate_content(prompt)
    return response.text

def get_rag_response(history: List[Dict[str, Any]], llm_model, pinecone_index) -> str:
    """Handles the full RAG process for information requests."""
    user_query = history[-1]['content']
    search_query = user_query # Start with the direct query

    # For follow-ups, create a better search query
    if len(history) > 2:
        try:
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])
            rewrite_prompt = f"Based on the chat history, rewrite the user's latest message into a clear, standalone search query.\n\nChat History:\n{formatted_history}\n\nUser's Latest Message: \"{user_query}\"\n\nRewritten Search Query:"
            response = llm_model.generate_content(rewrite_prompt)
            search_query = response.text.strip()
            print(f"Rewritten search query: '{search_query}'")
        except Exception:
            pass # Use original query if rewrite fails

    # Search Pinecone
    query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
    search_results = pinecone_index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    context_docs = ""
    if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
        context_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])
    
    # Generate final answer
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    final_context = f"RELEVANT DOCUMENTS:\