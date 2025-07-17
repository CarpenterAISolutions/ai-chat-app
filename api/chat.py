import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- Final Production Persona and Rules ---
SYSTEM_INSTRUCTION = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.
Your goal is to provide a helpful and accurate answer based on the information provided to you.

**Your Action Protocol:**
1.  Review the `CONVERSATIONAL HISTORY` to understand the flow of the conversation.
2.  Review the `RELEVANT DOCUMENTS`. This is your primary source of truth.
3.  Formulate a direct answer to the `USER'S LATEST MESSAGE` using the documents.
4.  If the documents section states "No relevant documents were found," you MUST inform the user that this topic is outside your scope and then suggest topics you can help with.
5.  You MUST follow your safety rules at all times: NEVER diagnose, prescribe, or ask for personal health information.
"""
SIMILARITY_THRESHOLD = 0.70

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))

        # Standard request handling and service initialization
        try:
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
        except Exception as e:
            send_json_response(500, {"answer": f"Server Error: Could not initialize services. Details: {e}"})
            return

        try:
            # --- STABLE RAG PIPELINE ---
            # 1. The search query is ALWAYS the user's direct, clean query.
            search_query = user_query
            print(f"Searching Pinecone for: '{search_query}'")

            # 2. Search Pinecone for relevant documents
            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            # 3. Build the context for the AI
            retrieved_docs = ""
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                print(f"Found relevant context with score: {search_results['matches'][0]['score']:.2f}")
                retrieved_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])
            
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            
            combined_context = f"""
            CONVERSATIONAL HISTORY:
            {formatted_history}

            RELEVANT DOCUMENTS:
            {retrieved_docs if retrieved_docs else "No relevant documents were found for this query."}
            """

            # 4. Construct the final prompt
            final_prompt = f"""
            {SYSTEM_INSTRUCTION}

            ---
            BEGIN CONTEXT
            {combined_context}
            END CONTEXT
            ---

            Based on the context above, respond to the user's latest message.

            USER'S LATEST MESSAGE: {user_query}
            """
            
            # 5. Generate the final response
            response = model.generate_content(final_prompt)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})