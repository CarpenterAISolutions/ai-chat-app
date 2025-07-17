import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- Final Production Persona and Rules ---
SYSTEM_INSTRUCTION = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.
Your primary goal is to answer the user's question based on the provided `CONTEXT`. The `CONTEXT` contains relevant documents from the clinic and the recent conversation history.

**Your Rules:**
- You MUST base your answers on the information found in the `CONTEXT`.
- If the `CONTEXT` says no relevant documents were found, state that the topic is outside the scope of your available information, and then proactively suggest topics you CAN discuss.
- NEVER diagnose, prescribe, or give medical advice that is not explicitly in the `CONTEXT`.
- NEVER ask for personal health information.
"""
SIMILARITY_THRESHOLD = 0.70

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))

        # Standard request handling and service initialization...
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query', "").strip()
        history = request_body.get('history', [])

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

        try:
            # --- THE FINAL FIX: A more flexible meta-command check ---
            meta_keywords = ['simplify', 'explain', 'summarize', 'rephrase', 'in other words', 'what you just said']
            is_meta_command = any(keyword in user_query.lower() for keyword in meta_keywords)
            
            if is_meta_command and len(history) >= 2:
                # Path A: Handle the meta-command directly
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
                prompt_to_use = f"{SYSTEM_INSTRUCTION}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Instruction:** The user has issued a command about your last response ('{user_query}'). Fulfill this command now by acting on your previous message."
            else:
                # Path B: Standard RAG process for new questions
                search_query = user_query
                # A simple and reliable way to add context for follow-ups
                if len(history) >= 2:
                    last_ai_response = history[-2].get('content', '')
                    search_query = f"The user is asking about '{user_query}' in the context of the last response: '{last_ai_response}'"
                
                query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
                search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                
                context = ""
                if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                    context = "\n".join([match['metadata']['text'] for match in search_results['matches']])
                
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
                final_context = f"CONTEXT FROM DOCUMENTS:\n{context if context else 'No relevant documents were found.'}"
                prompt_to_use = f"{SYSTEM_INSTRUCTION}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n{final_context}\n\n**Instruction:** Based on the history and context, provide a direct and helpful response to the user's latest message."

            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})