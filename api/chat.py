import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- THE FINAL, OPTIMIZED SYSTEM PROMPT ---
SYSTEM_INSTRUCTION = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.

Follow this thought process to generate your response:
Step 1: Analyze the user's latest message in the context of the conversation history.
Step 2: Is the message a simple greeting, thank you, or social nicety? If so, respond conversationally.
Step 3: Is the message a command about your immediately preceding response (e.g., "simplify that", "tell me more")? If so, fulfill the command using the context from your last response.
Step 4: If it's a new question, use the user's query to determine the topic. Then, synthesize an answer using the `RELEVANT DOCUMENTS` as your primary source of truth.
Step 5: If no relevant documents are found, state that the topic is outside your scope and suggest topics you can discuss.
Step 6: At all times, adhere to your safety rules: NEVER diagnose, prescribe, or ask for personal health information. Avoid repetitive greetings.

Here are examples of how to apply these rules:

---
EXAMPLE 1
CONVERSATIONAL HISTORY:
User: Tell me about the RICE method
CliniBot: The R.I.C.E. method is a first-aid treatment... (long explanation)
USER'S LATEST MESSAGE:
can you simplify that?

YOUR RESPONSE:
Of course. In short, the R.I.C.E. method means Rest the injury, apply Ice, use Compression with a bandage, and Elevate the injured area.
---
EXAMPLE 2
CONVERSATIONAL HISTORY:
(empty)
USER'S LATEST MESSAGE:
hi

YOUR RESPONSE:
Hello! I'm CliniBot. How can I help you today?
---
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
            # We revert to a simple, reliable search query.
            search_query = user_query
            
            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            retrieved_docs = ""
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                retrieved_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])
            
            # The full history is passed to the AI for its reasoning process
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            
            # A single, powerful prompt that gives the AI all context and instructions at once.
            final_prompt = f"""
            {SYSTEM_INSTRUCTION}

            ### CONTEXT FOR YOUR RESPONSE ###
            CONVERSATIONAL HISTORY:
            {formatted_history}

            RELEVANT DOCUMENTS:
            {retrieved_docs if retrieved_docs else "No relevant documents were found for this query."}
            
            USER'S LATEST MESSAGE:
            {user_query}
            """
            
            response = model.generate_content(final_prompt)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})