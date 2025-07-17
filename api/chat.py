import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- Final AI Persona (Simplified for clarity) ---
AI_PERSONA = "You are 'CliniBot,' an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic."
SIMILARITY_THRESHOLD = 0.70

# --- Helper function to determine user intent ---
def classify_intent(user_query: str) -> str:
    """Classifies user intent into one of three categories using simple keywords."""
    query = user_query.lower()
    # Check for simple greetings first
    greeting_keywords = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'ok']
    if query in greeting_keywords:
        return "GREETING"
    
    # Check for commands about the previous response
    meta_keywords = ['simplify', 'explain', 'summarize', 'rephrase', 'what you just said', 'in other words']
    if any(keyword in query for keyword in meta_keywords):
        return "META