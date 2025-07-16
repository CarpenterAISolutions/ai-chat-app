# api/chat.py (Final Production Version)
import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- Final AI Persona and Rules ---
AI_PERSONA_AND_RULES = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.
Your primary directive is to synthesize a helpful, conversational, and accurate response based on the information provided.

**Your Rules of Engagement:**
- **Use all Provided Information:** Base your response on the `CONVERSATIONAL HISTORY`, `CONTEXT FROM DOCUMENTS`, and the `USER'S LATEST MESSAGE`.
- **Prioritize Documents:** If relevant `CONTEXT FROM DOCUMENTS` is available, you MUST use it as the primary source for your answer.
- **Acknowledge History:** Use the `CONVERSATIONAL HISTORY` to understand follow-up questions and avoid repeating information.
- **Handle Lack of Documents:** If the `CONTEXT FROM DOCUMENTS` states that none were found, answer the user's question conversationally using your general knowledge, but you MUST include the phrase "Please note that this information is general and not from our clinic's specific documents."
- **Critical Safety Guardrail:** NEVER diagnose, prescribe, or ask for personal health information. If asked, you must politely decline and state that this requires a consultation with a qualified physical therapist.
"""
SIMILARITY_THRESHOLD = 0.70 # A stable threshold that works well with clean search queries

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

        # --- Final, Stabilized RAG Logic ---
        try:
            # 1. The search query is ALWAYS the user's direct query. This is simple and reliable.
            search_query = user_query
            print(f"Searching for: '{search_query}'")

            # 2. Search Pinecone
            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            # 3. Build the final prompt for the AI
            retrieved_docs = ""
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                retrieved_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])
            
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

            final_prompt = f"""
            {AI_PERSONA_AND_RULES}

            ### CONTEXT FROM DOCUMENTS ###
            {retrieved_docs if retrieved_docs else "No relevant documents were found."}
            
            ### CONVERSATIONAL HISTORY ###
            {formatted_history}

            ### USER'S LATEST MESSAGE ###
            {user_query}

            Please provide your response now:
            """
            
            # 4. Generate the final response
            response = model.generate_content(final_prompt)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})