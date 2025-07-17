import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- Final Production Persona and Rules ---
SYSTEM_INSTRUCTION = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.
Your primary goal is to answer the user's question based on the provided `CONTEXT`. The `CONTEXT` contains relevant documents from the clinic and the recent conversation history.

**Your Rules:**
- You MUST base your answers on the information found in the `CONTEXT`.
- If the `CONTEXT` says no relevant documents were found, state that the topic is outside the scope of your available information and proactively suggest topics you can discuss.
- NEVER diagnose, prescribe, or give medical advice that is not explicitly in the `CONTEXT`.
- NEVER ask for personal health information.
"""
SIMILARITY_THRESHOLD = 0.70

# --- HELPER FUNCTION TO MAKE THE SEARCH SMARTER ---
def rewrite_query_for_search(history: List[Dict[str, Any]], llm_model) -> str:
    """Uses the LLM to rewrite a user's query to be a standalone question for better search results."""
    user_query = history[-1]['content']
    # If this is the first real question, no need to rewrite.
    if len(history) <= 2: # First from AI ("Hello"), then first from User
        return user_query

    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])
    prompt = f"Based on the chat history, rewrite the user's latest message into a clear, standalone search query that can be understood without the previous context. If the message is already a clear question, you can return it as-is.\n\nChat History:\n{formatted_history}\n\nUser's Latest Message: \"{user_query}\"\n\nRewritten Search Query:"
    
    try:
        response = llm_model.generate_content(prompt)
        rewritten_query = response.text.strip()
        print(f"Original query: '{user_query}' -> Rewritten query: '{rewritten_query}'")
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
            # --- THE NEW, INTELLIGENT LOGIC FLOW ---
            # Step 1: Rewrite the user's query to include conversational context.
            search_query = rewrite_query_for_search(history, model)
            
            # Step 2: Search Pinecone using the new, smarter query.
            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            # Step 3: Build the final prompt with all context.
            retrieved_docs = ""
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                retrieved_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])
            
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            combined_context = f"CONVERSATIONAL HISTORY:\n{formatted_history}\n\nRELEVANT DOCUMENTS:\n{retrieved_docs if retrieved_docs else 'No relevant documents were found for this query.'}"

            final_prompt = f"{SYSTEM_INSTRUCTION}\n\nCONTEXT:\n{combined_context}\n\nBased on the CONTEXT, provide a direct and helpful answer to the user's latest message:\n{user_query}"
            
            # Step 4: Generate the final, intelligent response.
            response = model.generate_content(final_prompt)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})