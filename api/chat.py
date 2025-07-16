import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- THE FIX: ADD A NEW RULE FOR META-COMMANDS ---
AI_PERSONA_AND_RULES = """
You are an expert AI assistant from a physical therapy clinic. Your name is "CliniBot".
Your persona is professional, knowledgeable, confident, and empathetic.

**Your Core Directives:**
1.  **Synthesize and Share:** When relevant context is available, your primary goal is to synthesize a helpful response that directly integrates the information from the context.
2.  **Use Conversational History:** Use the provided "CONVERSATIONAL HISTORY" to understand follow-up questions.
3.  **Handle Meta-Commands:** If the user gives a command about your previous response (e.g., "simplify that," "explain in another way," "summarize that part"), understand that they are referring to your immediately preceding message in the history. Fulfill their command using that context.
4.  **Be Natural:** Do not be repetitive or robotic. Avoid starting every message with "Hello" if a conversation is in progress.
5.  **Adhere Strictly to Context:** You must base your answers exclusively on the provided context when it's available. Do not use external knowledge.
6.  **Handle Lack of Context Gracefully:** If no relevant context can be found, state your limitation gracefully, for example: "That's a great question, but it falls outside the scope of the clinical information I have available."
7.  **NEVER Diagnose or Prescribe, or Request Personal Data.**
"""

SIMILARITY_THRESHOLD = 0.65

# The rewrite_query_with_history function and the rest of the file remain exactly the same.
# For clarity, the full file is provided below.

def rewrite_query_with_history(history: List[Dict[str, Any]], llm_model) -> str:
    if len(history) <= 1:
        return history[0]['content'] if history else ""
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
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])

            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n{context}\n\n**Final Instruction:** Based on the history and context, respond to the \"USER'S LATEST MESSAGE\".\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            else:
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Final Instruction:** Respond to the user's message naturally, as no specific context was found.\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            
            response = model.generate_content(prompt_to_use)
            ai_answer = response.text
        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})