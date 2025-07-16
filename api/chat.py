# api/chat.py (Final Business-Ready Version)
import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- Final AI Persona and Rules ---
AI_PERSONA_AND_RULES = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.
**Core Directives:**
- Your primary goal is to provide helpful information based ONLY on the `CONTEXT FROM DOCUMENTS`.
- Use the `CONVERSATIONAL HISTORY` to understand follow-up questions.
- If the user's message is a command about your previous response (e.g., "simplify that"), fulfill it.
- If no relevant context is found, gracefully state that the topic is outside your current scope.
- **CRITICAL SAFETY RULE: NEVER diagnose, prescribe, or ask for personal health information.**
- Be natural and avoid repetitive greetings.
"""
SIMILARITY_THRESHOLD = 0.70

def rewrite_query_for_search(history: List[Dict[str, Any]], llm_model) -> str:
    user_query = history[-1]['content']
    if len(history) == 1:
        return user_query

    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])
    prompt = f"Based on the chat history, rewrite the user's latest message into a clear, standalone search query.\n\nChat History:\n{formatted_history}\n\nUser's Latest Message: \"{user_query}\"\n\nRewritten Search Query:"
    
    try:
        response = llm_model.generate_content(prompt)
        rewritten_query = response.text.strip()
        print(f"Original query: '{user_query}' -> Rewritten query: '{rewritten_query}'")
        return rewritten_query if rewritten_query else user_query
    except Exception:
        return user_query # Fallback on error

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))

        # Standard request handling
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query', "").strip()
        history = request_body.get('history', [])
        if not user_query:
            send_json_response(200, {"answer": "Please type a message."})
            return

        # Service initialization
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = "physical-therapy-index"
            genai.configure(api_key=gemini_api_key)
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
        except Exception as e:
            send_json_response(500, {"answer": f"Server Error: Could not initialize services: {e}"})
            return

        # --- Final, Robust Logic Flow ---
        try:
            # Step 1: Check for Meta-Command
            is_meta_command = any(user_query.lower().startswith(cmd) for cmd in ['simplify', 'explain', 'summarize', 'rephrase'])
            if is_meta_command and len(history) > 1:
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Instruction:** The user has issued a command about your last response. Fulfill their request directly."
            else:
                # Step 2: Rewrite query for better search
                search_query = rewrite_query_for_search(history, model)
                
                # Step 3: Search documents
                query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
                search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                
                # Step 4: Build final prompt based on search results
                context = ""
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
                
                if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                    context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                    prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n{context}\n\n**Instruction:** Based on the history and context, respond to the user's latest message."
                else:
                    prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Instruction:** No specific documents were found. Respond to the user's latest message naturally based on your persona and rules."

            # Step 5: Generate final answer
            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})