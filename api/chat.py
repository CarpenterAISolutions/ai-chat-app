import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- THE FIX: A NEW "THOUGHT PROCESS" FOR THE AI ---
AI_PERSONA_AND_RULES = """
You are "CliniBot," a professional, knowledgeable, and empathetic AI assistant for a physical therapy clinic.

**Your Thought Process and Rules of Engagement:**

1.  **Analyze the User's Intent:** First, examine the "USER'S LATEST MESSAGE" in the context of the "CONVERSATIONAL HISTORY". Determine the user's intent. Is it a new question, or is it a command related to your *immediately preceding* response (e.g., "simplify that," "tell me more," "explain that differently")?

2.  **Fulfill Meta-Commands:** If the user's intent is a command about your last message, fulfill it directly using the conversational history. **Do not perform a new search.** For example, if you just gave a long explanation and the user says "simplify that," you must provide a simplified version of your last response.

3.  **Handle New Queries:** If the user's intent is a new question or statement, proceed with the following rules:
    a. **Use Provided Context:** If relevant "CONTEXT FROM DOCUMENTS" is provided, you MUST base your answer exclusively on it. Synthesize the information into a helpful, detailed response. Do not mention your own limitations or talk about the documents themselves.
    b. **Handle Lack of Context:** If no relevant context is found, respond naturally and conversationally. If the user asks for information outside your scope, state your limitations gracefully. Example: "That's a great question, but it falls outside the scope of the clinical information I have available. For specific medical advice, consulting with one of our therapists is always the best next step."
    c. **Follow Safety Protocols:** NEVER diagnose, prescribe, or ask for personal health information. Avoid repetitive greetings.
"""

SIMILARITY_THRESHOLD = 0.65

def rewrite_query_with_history(history: List[Dict[str, Any]], llm_model) -> str:
    if len(history) <= 1:
        return history[0]['content'] if history and history[0].get('content') else ""
        
    user_query = history[-1]['content']
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])
    prompt = f"Based on the chat history, rewrite the user's latest message into a clear, standalone search query.\n\nChat History:\n{formatted_history}\n\nUser's Latest Message: \"{user_query}\"\n\nRewritten Search Query:"
    try:
        response = llm_model.generate_content(prompt)
        rewritten_query = response.text.strip()
        if not rewritten_query: return user_query
        print(f"Original query: '{user_query}' -> Rewritten query: '{rewritten_query}'")
        return rewritten_query
    except Exception as e:
        print(f"Error during query rewriting, falling back to original query. Error: {e}")
        return user_query

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Helper function and initial setup (no changes)
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

        try:
            search_query = user_query
            if len(history) > 1:
                search_query = rewrite_query_with_history(history, model)

            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            context = ""
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]) # Pass full history now

            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n{context}\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            else:
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n[No relevant context found]\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            
            response = model.generate_content(prompt_to_use)
            ai_answer = response.text
        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})