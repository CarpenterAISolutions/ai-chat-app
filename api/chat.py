# api/chat.py (Final robust version)
import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

AI_PERSONA_AND_RULES = """
You are an expert AI assistant from a physical therapy clinic. Your name is "CliniBot". Your persona is professional, knowledgeable, confident, and empathetic. Your purpose is to provide helpful information based ONLY on the verified documents from the clinic. Your core directives are to synthesize and share information from the context, use conversational history to understand follow-up questions, avoid being repetitive, adhere strictly to the provided context, handle lack of context gracefully by using your general conversational abilities, never diagnose or prescribe, and never ask for personal data.
"""
SIMILARITY_THRESHOLD = 0.68

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            request_body = json.loads(post_data)
        except json.JSONDecodeError:
            send_json_response(400, {"answer": "Error: Invalid request format."})
            return

        user_query = request_body.get('query', "").strip()
        if not user_query:
            send_json_response(200, {"answer": "Please type a message to get started."})
            return
            
        history: List[Dict[str, Any]] = request_body.get('history', [])

        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = "physical-therapy-index"
            if not all([gemini_api_key, pinecone_api_key]):
                send_json_response(500, {"answer": "Server Error: API keys are not configured correctly on Vercel."})
                return
            
            genai.configure(api_key=gemini_api_key)
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
        except Exception as e:
            error_message = f"Server Error: Could not initialize AI services. Details: {e}"
            send_json_response(500, {"answer": error_message})
            return

        try:
            # --- THIS BLOCK IS NOW MORE ROBUST ---
            history_for_search = [msg['content'] for msg in history[-4:] if msg.get('content')]
            if not history_for_search:
                history_for_search.append(user_query)
            contextual_search_query = "\n".join(history_for_search)
            # --- END OF ROBUST BLOCK ---

            query_embedding = genai.embed_content(model="models/text-embedding-004", content=contextual_search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            context = ""
            prompt_to_use = ""
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])

            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n{context}\n\n**Final Instruction:** Based on the history and the new context, provide a direct and helpful response to the \"USER'S LATEST MESSAGE\".\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            else:
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Final Instruction:** The user has sent the following message. Respond naturally according to your core directives.\n\nUSER'S LATEST MESSAGE:\n{user_query}"

            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})
        return