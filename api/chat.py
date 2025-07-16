import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

SYSTEM_INSTRUCTION = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic.
Your persona is professional, knowledgeable, and empathetic.
Your primary goal is to answer the user's question based on the provided `CONTEXT`.
The `CONTEXT` contains relevant documents and the recent conversation history.

**Your Rules:**
- You MUST base your answers on the information found in the `CONTEXT`.
- If the `CONTEXT` does not contain the answer, state that the topic is outside the scope of your available information.
- NEVER diagnose, prescribe, or give medical advice that is not explicitly in the `CONTEXT`.
- NEVER ask for personal health information.
- Be natural and conversational. Avoid repetitive greetings.
"""
# --- CHANGE 1: LOWER THE THRESHOLD ---
SIMILARITY_THRESHOLD = 0.60

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # ... (The first part of the file is unchanged) ...
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))

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
            search_query = user_query
            if len(history) > 1:
                contextual_history = "\n".join([msg['content'] for msg in history[-2:] if msg.get('content')])
                search_query = f"Based on the conversation below, what is a good search query for the user's last message?\n\n{contextual_history}"

            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            # --- CHANGE 2: ADD A PRINT STATEMENT FOR DEBUGGING ---
            print(f"PINEcone Search Results: {search_results}")

            retrieved_docs = ""
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                retrieved_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])
            
            # ... (The rest of the file is unchanged) ...
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            combined_context = f"CONVERSATIONAL HISTORY:\n{formatted_history}\n\nRELEVANT DOCUMENTS:\n{retrieved_docs if retrieved_docs else 'No relevant documents found.'}"
            final_prompt = f"{SYSTEM_INSTRUCTION}\n\nCONTEXT:\n{combined_context}\n\nBased on all the information above, provide a direct and helpful answer to the user's latest message.\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            
            response = model.generate_content(final_prompt)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})