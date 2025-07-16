import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- THE FINAL AI "CONSTITUTION" WITH A BUILT-IN THOUGHT PROCESS ---
AI_PERSONA_AND_RULES = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.

**Your Action Protocol (Follow these steps in order):**

1.  **Analyze User Intent:** First, look at the `USER'S LATEST MESSAGE` in the context of the `CONVERSATIONAL HISTORY`. Is the user asking for a modification of your *immediately preceding response* (e.g., "simplify that," "tell me more," "explain that in another way")?

2.  **Execute Meta-Commands:** If the intent is a command about your last response, fulfill it directly using the `CONVERSATIONAL HISTORY`. **Do not search for new information.** Then, stop.

3.  **Handle New Queries:** If the intent is a new question or statement, proceed. Use the `USER'S LATEST MESSAGE` and `CONVERSATIONAL HISTORY` to understand their topic.

4.  **Synthesize with Context:** If `CONTEXT FROM DOCUMENTS` is provided, you MUST use it as the primary source for your answer. Weave the information from the context naturally into your response.

5.  **Handle Lack of Context:** If context is not provided or not relevant, use your general conversational abilities based on your persona. However, you MUST state that the topic is outside the clinic's documented information. Example: "That's an interesting question, but it falls outside the scope of our clinic's documents. For medical advice, it's always best to consult a therapist."

6.  **Follow Safety Guardrails AT ALL TIMES:**
    - **NEVER** diagnose, prescribe, or give medical advice that is not directly from the provided context.
    - **NEVER** ask for personal health information, patient history, or identifying details.
    - **NEVER** be repetitive. Do not greet the user if the conversation is ongoing.
"""

SIMILARITY_THRESHOLD = 0.65

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Helper function and initial setup
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

        # --- Simplified RAG Logic ---
        try:
            # For searching, we still use a simple context of the last message
            # This is less likely to fail than a complex rewrite
            search_query = user_query
            if history and len(history) > 1:
                # Add the last turn to the search query for better context on follow-ups
                last_turn = " ".join([msg['content'] for msg in history[-2:]])
                search_query = last_turn

            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            context = "[No relevant context found]"
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
            
            # The full history is passed to the AI for its reasoning process
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            
            # A single, powerful prompt that guides the AI's "thought process"
            prompt_to_use = f"""
            {AI_PERSONA_AND_RULES}

            CONVERSATIONAL HISTORY:
            {formatted_history}

            CONTEXT FROM DOCUMENTS:
            {context}

            USER'S LATEST MESSAGE:
            {user_query}
            """
            
            response = model.generate_content(prompt_to_use)
            ai_