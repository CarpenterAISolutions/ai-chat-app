import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- 1. REFINE THE AI's "CONSTITUTION" FOR MORE NATURAL RESPONSES ---
AI_PERSONA_AND_RULES = """
You are an expert AI assistant from a physical therapy clinic. Your name is "CliniBot".
Your persona is professional, knowledgeable, confident, and empathetic.

**Your Core Directives:**
1.  **Synthesize and Share:** When relevant context is available, your primary goal is to synthesize a helpful response that directly integrates the information from the context.
2.  **Use Conversational History:** Use the provided "CONVERSATIONAL HISTORY" to understand follow-up questions.
3.  **Be Natural:** Do not be repetitive or robotic. Avoid starting every message with "Hello". Get straight to the user's point if a conversation is in progress.
4.  **Adhere Strictly to Context:** You must base your answers exclusively on the provided context when it's available. Do not use any external knowledge.
5.  **Handle Lack of Context Gracefully:** If no relevant context can be found for a question, state your limitation gracefully. Do not say "I don't have access to documents." Instead, say something like, "That's a great question, but it falls outside the scope of the clinical information I have available. For specific medical advice, consulting with one of our therapists is always the best next step."
6.  **NEVER Diagnose or Prescribe, or Request Personal Data.**
"""

SIMILARITY_THRESHOLD = 0.65 # We can raise this slightly with a better search query

# --- 2. NEW FUNCTION FOR QUERY REWRITING ---
def rewrite_query_with_history(history: List[Dict[str, Any]], llm_model) -> str:
    """Uses the LLM to rewrite a user's query to be a standalone question."""
    if not history:
        return ""

    # Only use the last few turns of conversation for brevity
    history_for_rewriting = history[-5:]
    
    # The last message is the one we want to rewrite
    user_query = history_for_rewriting[-1]['content']
    
    # Format previous turns for the prompt
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_for_rewriting[:-1]])

    prompt = f"""
    Based on the following chat history, rewrite the "User's Latest Message" into a single, clear, standalone search query that can be understood without the previous context.

    Chat History:
    {formatted_history}

    User's Latest Message: "{user_query}"

    Rewritten Search Query:
    """
    
    try:
        response = llm_model.generate_content(prompt)
        rewritten_query = response.text.strip()
        print(f"Original query: '{user_query}' -> Rewritten query: '{rewritten_query}'")
        return rewritten_query
    except Exception as e:
        print(f"Error during query rewriting: {e}")
        return user_query # Fallback to the original query on error


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
            send_json_response(200, {"answer": "Please type a message to get started."})
            return
            
        # Initialize Services (no changes)
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

        # --- 3. THE NEW LOGIC FLOW WITH QUERY REWRITING ---
        try:
            # Step A: Rewrite the query using conversation history
            search_query = rewrite_query_with_history(history, model)
            
            # Step B: Embed the rewritten query and search Pinecone
            query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            context = ""
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])

            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                # Path 1: Context found, use it to answer
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n{context}\n\n**Final Instruction:** Based on the history and the new context, provide a direct and helpful response to the \"USER'S LATEST MESSAGE\".\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            else:
                # Path 2: No context, use general rules
                prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\n**Final Instruction:** The user has sent the following message. Respond naturally according to your core directives.\n\nUSER'S LATEST MESSAGE:\n{user_query}"
            
            # Step C: Generate the final answer
            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})
        return