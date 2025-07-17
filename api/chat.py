import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

AI_PERSONA_AND_RULES = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.
**Your Core Directives:**
- Your primary goal is to provide helpful information based ONLY on the verified `CONTEXT FROM DOCUMENTS`.
- Use the `CONVERSATIONAL HISTORY` to understand follow-up questions.
- If the user's message is a command about your previous response (e.g., "simplify that"), fulfill it.
- If the `CONTEXT FROM DOCUMENTS` states that no relevant documents were found, gracefully state that the topic is outside your current scope and proactively suggest topics you can discuss.
- **CRITICAL SAFETY RULE: NEVER diagnose, prescribe, or ask for personal health information.**
- Be natural and avoid repetitive greetings if a conversation is in progress.
"""
SIMILARITY_THRESHOLD = 0.70

def rewrite_query_for_search(history: List[Dict[str, Any]], llm_model) -> str:
    """Uses the LLM to rewrite a user's query to be a standalone question for better search results."""
    # --- THE DEFINITIVE FIX: Add a robust safety check for empty history ---
    if not history:
        return "" # If history is empty, we cannot rewrite, so return an empty string.

    user_query = history[-1]['content']
    # If this is the first real question, no need to rewrite.
    if len(history) <= 2:
        return user_query

    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])
    prompt = f"Based on the chat history, rewrite the user's latest message into a clear, standalone search query.\n\nChat History:\n{formatted_history}\n\nUser's Latest Message: \"{user_query}\"\n\nRewritten Search Query:"
    
    try:
        response = llm_model.generate_content(prompt)
        rewritten_query = response.text.strip()
        return rewritten_query if rewritten_query else user_query
    except Exception as e:
        print(f"Error during query rewriting, falling back to original query. Error: {e}")
        return user_query

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
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
            meta_keywords = ['simplify', 'explain', 'summarize', 'rephrase', 'what you just said']
            is_meta_command = any(keyword in user_query.lower() for keyword in meta_keywords)
            
            if is_meta_command and len(history) >= 2:
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Instruction:** The user has issued a command about your last response ('{user_query}'). Fulfill this command now."
            else:
                search_query = rewrite_query_for_search(history, model)
                # --- Add a fallback to ensure search_query is never empty ---
                if not search_query:
                    search_query = user_query
                
                query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
                search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                
                context = ""
                if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                    context = "\n".join([match['metadata']['text'] for match in search_results['matches']])
                
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
                final_context = f"CONTEXT FROM DOCUMENTS:\n{context if context else 'No relevant documents were found.'}"
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n{final_context}\n\n**Instruction:** Based on the history and context, provide a direct, helpful response to the user's latest message."

            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})