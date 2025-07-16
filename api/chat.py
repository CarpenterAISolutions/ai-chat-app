import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- FINAL, MORE NUANCED SYSTEM INSTRUCTION ---
SYSTEM_INSTRUCTION = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.

**Your Action Protocol:**
1.  **Analyze the User's Latest Message:** First, determine the user's intent. Is it a simple conversational message (e.g., "hi", "thank you"), or is it a question seeking information?
2.  **Handle Conversation:** If the message is a simple greeting or social nicety, respond naturally and conversationally in your persona.
3.  **Handle Information Requests:** If the message is a question, use the `CONTEXT` provided below to formulate your answer.
    a. **If `RELEVANT DOCUMENTS` are provided in the `CONTEXT`**, you MUST base your answer on that information.
    b. **If the `CONTEXT` says "No relevant documents were found,"** you MUST state that the topic is outside the scope of your available information. Do not use your general knowledge.
4.  **Follow Safety Guardrails:** NEVER diagnose, prescribe, or ask for personal health information. Avoid repetitive greetings.
"""
SIMILARITY_THRESHOLD = 0.70

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Helper function and initial setup
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
            send_json_response(500, {"answer": f"Server Error: Could not initialize AI services. Details: {e}"})
            return

        # The rest of the logic is stable and requires no changes.
        try:
            search_query = user_query
            if len(history) > 1:
                contextual_history = "\n".join([msg['content'] for msg in history[-2:] if msg.get('content')])
                search_query = f"Based on the conversation below, what is a good search query for the user's last message?\n\n{contextual_history}"

            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            retrieved_docs = ""
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                retrieved_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])
            
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            combined_context = f"CONVERSATIONAL HISTORY:\n{formatted_history}\n\nRELEVANT DOCUMENTS:\n{retrieved_docs if retrieved_docs else 'No relevant documents were found for this query.'}"

            final_prompt = f"{SYSTEM_INSTRUCTION}\n\nCONTEXT:\n{combined_context}\n\nBased on the CONTEXT and your rules, provide a direct and helpful answer to the user's latest message:\n{user_query}"
            
            response = model.generate_content(final_prompt)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})