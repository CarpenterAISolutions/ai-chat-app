import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# A clear and direct set of instructions for the AI
SYSTEM_INSTRUCTION = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional and helpful.
Your primary goal is to answer the user's question using the `RELEVANT DOCUMENTS` provided.

**Your Rules:**
- Base your answers strictly on the `RELEVANT DOCUMENTS`.
- If the documents section says "No relevant documents were found," you MUST state that the topic is outside your scope of knowledge and suggest other topics you can help with.
- Use the `CONVERSATIONAL HISTORY` to understand the context of the user's latest message.
- NEVER diagnose, prescribe, or ask for personal health information. This is a critical safety rule.
"""
SIMILARITY_THRESHOLD = 0.70

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))

        try:
            # --- Standard Request Handling & Service Initialization ---
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data)
            user_query = request_body.get('query', "").strip()
            history = request_body.get('history', [])

            if not user_query:
                send_json_response(200, {"answer": "Please type a message."})
                return

            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = "physical-therapy-index"
            genai.configure(api_key=gemini_api_key)
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')

            # --- Simplified and Stable RAG Pipeline ---
            # The search query is always the user's direct query. It's simple and reliable.
            search_query = user_query
            
            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            retrieved_docs = ""
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                retrieved_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])
            
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            
            # A single, clean prompt is sent to the AI
            final_prompt = f"""
            {SYSTEM_INSTRUCTION}

            ### CONVERSATIONAL HISTORY ###
            {formatted_history}

            ### RELEVANT DOCUMENTS ###
            {retrieved_docs if retrieved_docs else "No relevant documents were found for this query."}

            ### USER'S LATEST MESSAGE ###
            {user_query}

            Please provide your response now:
            """
            
            # A single, fast API call
            response = model.generate_content(final_prompt)
            ai_answer = response.text

        except Exception as e:
            # A robust catch-all to report any errors instead of crashing
            print(f"--- [ERROR] An exception occurred: {e} ---") # This will now log to Vercel
            ai_answer = f"An error occurred on the server. Please try again later."

        send_json_response(200, {"answer": ai_answer})