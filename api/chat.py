import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- A SIMPLIFIED, DIRECT, AND ROBUST PERSONA FOR CLINIBOT ---
AI_PERSONA_AND_RULES = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic.
Your persona is professional, knowledgeable, and empathetic.
- Your primary goal is to provide helpful information based ONLY on the verified `CONTEXT FROM DOCUMENTS`.
- When you receive a `CONVERSATIONAL HISTORY`, use it to understand the user's follow-up questions.
- NEVER diagnose, prescribe, or give medical advice that is not directly from the `CONTEXT`.
- NEVER ask for personal health information or patient history.
- Be natural and do not greet the user if a conversation is in progress.
"""

SIMILARITY_THRESHOLD = 0.70 # We can use a higher threshold with better search queries

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Helper function and initial setup (no changes)
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query', "").strip()
        history: List[Dict[str, Any]] = request_body.get('history', [])

        if not user_query:
            send_json_response(200, {"answer": "Please type a message."})
            return
            
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = "physical-therapy-index"
            genai.configure(api_key=gemini_api_key)
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
        except Exception as e:
            send_json_response(500, {"answer": f"Server Error: Could not initialize AI services. Details: {e}"})
            return

        # --- REBUILT AND SIMPLIFIED LOGIC FLOW ---
        try:
            is_meta_command = any(user_query.lower().startswith(cmd) for cmd in ['simplify', 'explain', 'summarize', 'rephrase'])
            
            # If it's a meta-command and there's history, we handle it specially
            if is_meta_command and len(history) > 1:
                # We don't need to search, just act on the last message
                context = "" # No document search needed
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Instruction:** The user has issued a command about your last response. Fulfill their request directly.\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            
            else:
                # It's a normal query, so we run the RAG process
                search_query = user_query
                # For follow-ups, add the last AI response to the search query for better context
                if len(history) > 1:
                    last_ai_response = history[-2].get('content', '')
                    search_query = f"{last_ai_response}\n\nUser Question: {user_query}"
                
                query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
                search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                
                context = ""
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

                if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                    context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                    prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n{context}\n\n**Instruction:** Based on the history and context, respond to the \"USER'S LATEST MESSAGE\".\n\nUSER'S LATEST MESSAGE:\n{user_query}"
                else:
                    prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Instruction:** No specific documents were found. Respond to the user's latest message naturally based on your persona and rules.\n\nUSER'S LATEST MESSAGE:\n{user_query}"

            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})