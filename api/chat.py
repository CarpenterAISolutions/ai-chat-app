import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- AI Configuration and Threshold (No changes) ---
AI_PERSONA_AND_RULES = """
You are an expert AI assistant from a physical therapy clinic. Your name is "CliniBot".
Your persona is professional, knowledgeable, confident, and empathetic.

**Your Core Directives:**
1.  **Synthesize and Share:** When relevant context is available, your primary goal is to synthesize a helpful response that directly integrates the information from the context.
2.  **Use Conversational History:** Use the provided "CONVERSATIONAL HISTORY" to understand follow-up questions. Your response should flow naturally from the previous turn.
3.  **DO NOT BE REPETITIVE:** Do not greet the user with "Hello" or "Hi" if a conversational history is present. Get straight to the user's point.
4.  **Adhere Strictly to Context:** You must base your answers exclusively on the provided context when it's available. Do not use external knowledge.
5.  **Handle Lack of Context:** If no relevant context is found, use the conversational history and your general knowledge to respond naturally.
6.  **NEVER Diagnose or Prescribe:** You must never diagnose a condition or create a new treatment plan.
7.  **DO NOT Request Personal Data:** You are strictly forbidden from asking for personal health information or patient history.
"""
SIMILARITY_THRESHOLD = 0.68

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        
        # --- NEW: INPUT VALIDATION ---
        # Get the user's query and remove any leading/trailing whitespace
        user_query = request_body.get('query', "").strip()

        # If the query is empty after stripping whitespace, stop here.
        if not user_query:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response_data = {"answer": "Please type a message to get started."}
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return # Exit the function early

        # --- The rest of the function remains the same ---
        history: List[Dict[str, Any]] = request_body.get('history', [])

        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = "physical-therapy-index"
            if not all([gemini_api_key, pinecone_api_key]):
                self.send_error(500, "API keys are not configured.")
                return
            genai.configure(api_key=gemini_api_key)
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
        except Exception as e:
            self.send_error(500, f"Error initializing AI services: {e}")
            return

        try:
            history_for_search = [msg['content'] for msg in history[-4:]]
            contextual_search_query = "\n".join(history_for_search)

            query_embedding = genai.embed_content(model="models/text-embedding-004", content=contextual_search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            context = ""
            prompt_to_use = ""
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])

            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                prompt_to_use = f"""
                {AI_PERSONA_AND_RULES}
                CONVERSATIONAL HISTORY:\n{formatted_history}\n
                CONTEXT FROM DOCUMENTS:\n{context}\n
                **Final Instruction:** Based on the history and the new context, provide a direct and helpful response to the "USER'S LATEST MESSAGE".
                USER'S LATEST MESSAGE:\n{user_query}
                """
            else:
                prompt_to_use = f"""
                {AI_PERSONA_AND_RULES}
                CONVERSATIONAL HISTORY:\n{formatted_history}\n
                **Final Instruction:** The user has sent the following message. Respond naturally according to your core directives.
                USER'S LATEST MESSAGE:\n{user_query}
                """

            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()