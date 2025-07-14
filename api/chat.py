# api/chat.py (New version with memory)

import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any # Import new types for validation
import google.generativeai as genai
from pinecone import Pinecone

# --- AI Configuration (No changes) ---
AI_PERSONA_AND_RULES = """
You are an expert AI assistant from a physical therapy clinic. Your name is "CliniBot".
Your persona is professional, knowledgeable, confident, and empathetic.
Your purpose is to provide helpful information based ONLY on the verified documents from the clinic.

**Your Core Directives:**
1.  **Synthesize and Share:** When a user states a problem or asks a question and relevant context is available, your primary goal is to synthesize a helpful response that directly integrates the information from the context.
2.  **Use Conversational History:** Use the provided "CONVERSATIONAL HISTORY" to understand follow-up questions and maintain context. Refer to previous turns in the conversation when it's relevant.
3.  **Ask Guiding Questions:** After providing information, you can ask a relevant follow-up question.
4.  **Adhere Strictly to Context:** You must base your answers exclusively on the provided context when it's available. Do not use external knowledge.
5.  **Handle Lack of Context:** If no relevant context is found for a specific question, use the conversational history and your general knowledge to respond naturally and helpfully.
6.  **NEVER Diagnose or Prescribe:** You must never diagnose a condition or prescribe a new treatment plan.
7.  **CRITICAL NEGATIVE CONSTRAINT: DO NOT REQUEST PERSONAL DATA.** You are strictly forbidden from asking for personal health information or patient history.
"""

SIMILARITY_THRESHOLD = 0.55

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        
        # --- NEW: Get the query AND the history from the request ---
        user_query = request_body.get('query')
        history: List[Dict[str, Any]] = request_body.get('history', [])

        # --- Initialize Services (No changes) ---
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

        # --- The Logic Flow (with history integration) ---
        try:
            query_embedding = genai.embed_content(model="models/text-embedding-004", content=user_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

            context = ""
            prompt_to_use = ""

            # --- NEW: Format the chat history for the prompt ---
            # We remove the last message from history, as it's the current user_query
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])

            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                print(f"✅ Found relevant context with score: {search_results['matches'][0]['score']:.2f}")
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                prompt_to_use = f"""
                {AI_PERSONA_AND_RULES}

                CONVERSATIONAL HISTORY:
                {formatted_history}

                CONTEXT FROM DOCUMENTS:
                {context}
                
                Based on the history and the new context, provide a helpful, synthesized response to the user's latest message.
                
                USER'S LATEST MESSAGE:
                {user_query}
                """
            else:
                print("⚠️ No relevant context found. Responding based on general rules and history.")
                prompt_to_use = f"""
                {AI_PERSONA_AND_RULES}

                CONVERSATIONAL HISTORY:
                {formatted_history}

                The user has sent the following message. Respond according to your core directives, keeping the conversation natural and acknowledging the history.

                USER'S LATEST MESSAGE:
                {user_query}
                """

            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        # --- Send the final response (No changes) ---
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"answer": ai_answer}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return