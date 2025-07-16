import os
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone

# --- 1. THE AI's CORE "CONSTITUTION" ---
# This is now focused on its persona and high-level rules.
AI_PERSONA_AND_RULES = """
You are "CliniBot," a professional, knowledgeable, confident, and empathetic AI assistant for a physical therapy clinic.

**Your Core Directives:**
- **CRITICAL SAFETY RULE: YOU ARE EXPLICITLY FORBIDDEN FROM REQUESTING PERSONAL DATA.** Never ask for personal health information, patient history, or specific details about a user's symptoms (e.g., "when did it start," "what makes it worse"). Your role is to provide information from documents, not to gather information.
- **Synthesize and Share:** When context is available, your primary goal is to synthesize a helpful response that directly integrates the information.
- **Handle Frustration Gracefully:** If the user expresses frustration (e.g., "that's not what I asked"), apologize for the misunderstanding and try to guide them by offering examples of questions you can answer.
- **Be Natural:** Do not be repetitive. Avoid starting every message with "Hello" if a conversation is in progress.
- **NEVER Diagnose or Prescribe.**
"""

SIMILARITY_THRESHOLD = 0.65

# --- 2. NEW HELPER FUNCTIONS FOR THE AI'S "THOUGHT PROCESS" ---

def is_meta_command(history: List[Dict[str, Any]], llm_model) -> bool:
    """Determines if the user's last message is a command about the conversation itself."""
    if len(history) < 2:
        return False # Not enough history to be a meta-command

    user_query = history[-1]['content']
    last_ai_response = history[-2]['content']

    prompt = f"""
    Analyze the user's latest message in the context of the AI's last response.
    Is the user's message a "meta-command" asking to modify, simplify, summarize, or elaborate on the AI's previous statement?
    Respond with only the word "YES" or "NO".

    AI's Last Response: "{last_ai_response[:200]}..."
    User's Latest Message: "{user_query}"

    Is this a meta-command? (YES/NO):
    """
    try:
        response = llm_model.generate_content(prompt)
        decision = response.text.strip().upper()
        print(f"Meta-command check for '{user_query}': {decision}")
        return "YES" in decision
    except Exception as e:
        print(f"Error in meta-command check: {e}")
        return False

def handle_meta_command(history: List[Dict[str, Any]], llm_model) -> str:
    """Handles meta-commands like 'simplify that'."""
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    
    prompt = f"""
    {AI_PERSONA_AND_RULES}
    
    You are in a conversation. The user has just given you a command about your previous response.
    Analyze the full "CONVERSATIONAL HISTORY" and fulfill the user's latest command.

    CONVERSATIONAL HISTORY:
    {formatted_history}
    """
    response = llm_model.generate_content(prompt)
    return response.text

def rewrite_query_for_search(history: List[Dict[str, Any]], llm_model) -> str:
    """Rewrites a conversational query into a standalone search query."""
    user_query = history[-1]['content']
    if len(history) == 1:
        return user_query # No history to rewrite from

    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[:-1]])
    prompt = f"Based on the chat history, rewrite the user's latest message into a clear, standalone search query.\n\nChat History:\n{formatted_history}\n\nUser's Latest Message: \"{user_query}\"\n\nRewritten Search Query:"
    
    try:
        response = llm_model.generate_content(prompt)
        rewritten_query = response.text.strip()
        if not rewritten_query: return user_query
        print(f"Original query: '{user_query}' -> Rewritten query: '{rewritten_query}'")
        return rewritten_query
    except Exception as e:
        print(f"Error during query rewriting, falling back to original. Error: {e}")
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

        # --- 3. THE NEW DECISION-MAKING LOGIC FLOW ---
        try:
            ai_answer = ""
            # Step A: Decide if the user's message is a meta-command
            if is_meta_command(history, model):
                # Path 1: Handle the meta-command directly
                ai_answer = handle_meta_command(history, model)
            else:
                # Path 2: It's a new question, so run the RAG process
                # B-1: Rewrite the query for a better search
                search_query = rewrite_query_for_search(history, model)

                # B-2: Embed and search Pinecone
                query_embedding = genai.embed_content(model="models/text-embedding-004", content=search_query, task_type="RETRIEVAL_QUERY")["embedding"]
                search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                
                context = ""
                formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

                if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                    # B-3a: If context is found, build a prompt with it
                    context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                    prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n{context}\n\n**Instruction:** Based on the history and context, respond to the user's latest message."
                else:
                    # B-3b: If no context, build a prompt without it
                    prompt_to_use = f"{AI_PERSONA_AND_RULES}\n\nCONVERSATIONAL HISTORY:\n{formatted_history}\n\nCONTEXT FROM DOCUMENTS:\n[No relevant context found]\n\n**Instruction:** Respond to the user's latest message naturally based on your persona and rules."
                
                # B-4: Generate the final answer
                response = model.generate_content(prompt_to_use)
                ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        send_json_response(200, {"answer": ai_answer})