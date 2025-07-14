import os
import json
from http.server import BaseHTTPRequestHandler
import google.generativeai as genai
from pinecone import Pinecone

# --- 1. REFINE THE AI's "CONSTITUTION" ---
AI_PERSONA_AND_RULES = """
You are an expert AI assistant from a physical therapy clinic. Your name is "CliniBot".
Your persona is professional, knowledgeable, confident, and empathetic.
Your purpose is to provide helpful information based ONLY on the verified documents from the clinic.

**Your Core Directives:**
1.  **Synthesize and Share:** When a user states a problem (e.g., "my back hurts") and relevant context is available, your primary goal is to synthesize a helpful response that directly integrates the information from the context. Explain the concepts from the documents clearly.
2.  **Ask Guiding Questions:** After providing information from the context, ask a relevant follow-up question to see if the user wants more detail. For example: "Would you like me to elaborate on those core exercises?" or "Would you like a more detailed description of the R.I.C.E. method?"
3.  **Adhere Strictly to Context:** You must base your answers exclusively on the provided context. Do not use any external knowledge.
4.  **Handle Lack of Context:** If no relevant context is found for a specific question, state that the topic is outside the scope of the clinic's documents.
5.  **Be Conversational:** Engage in general conversation naturally. Only introduce yourself as CliniBot if the user asks who you are or at the very beginning of a new conversation.
6.  **NEVER Diagnose or Prescribe:** You must never diagnose a condition or create a new treatment plan. If asked, politely state that this requires a direct evaluation by a qualified physical therapist.
"""

SIMILARITY_THRESHOLD = 0.68

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query')

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

        # --- The Logic Flow (with a new final instruction) ---
        try:
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=user_query,
                task_type="RETRIEVAL_QUERY"
            )["embedding"]

            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

            context = ""
            prompt_to_use = ""

            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                print(f"✅ Found relevant context with score: {search_results['matches'][0]['score']:.2f}")
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                # --- 2. THE NEW, MORE DIRECT INSTRUCTION ---
                prompt_to_use = f"""
                {AI_PERSONA_AND_RULES}

                **Final Instruction:** Synthesize a helpful and empathetic response to the user's statement/question. Directly integrate the key details from the `CONTEXT` below into your answer, explaining them clearly as if you are an expert assistant. Do not just state that the information exists; explain the information itself. Conclude by asking a relevant follow-up question.

                CONTEXT:
                {context}

                USER'S QUESTION:
                {user_query}
                """
            else:
                print("⚠️ No relevant context found. Responding based on general rules.")
                prompt_to_use = f"""
                {AI_PERSONA_AND_RULES}

                **Final Instruction:** The user has sent the following message. Respond according to your core directives, keeping the conversation natural.

                USER'S MESSAGE:
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