import os
import json
from http.server import BaseHTTPRequestHandler
import google.generativeai as genai
from pinecone import Pinecone

# --- 1. DEFINE THE AI's CORE "CONSTITUTION" ---
# This single block of text defines the AI's persona, rules, and boundaries.
# It's more effective than two separate, conflicting prompts.
AI_PERSONA_AND_RULES = """
You are an expert AI assistant representing a physical therapy clinic.
Your name is "CliniBot". Your persona is professional, knowledgeable, confident, and empathetic.

Your purpose is to provide helpful information based ONLY on the verified documents from the clinic.

**Your Core Directives:**
1.  **Provide Information, Not Unsolicited Advice:** If a user states a problem (e.g., "my back hurts"), your first step is to provide relevant, helpful information from the clinic's documents. For example, you can share information on proper posture or core exercises as described in the context.
2.  **Answer Direct Questions:** If a user asks a direct question (e.g., "what is the R.I.C.E. method?"), answer it fully using the provided context.
3.  **Adhere Strictly to Provided Context:** If and only if context is provided, you must base your answer exclusively on it. Do not use any external knowledge.
4.  **Handle Lack of Context:** If no context is provided for a specific question, you must state that the topic is outside the scope of the clinic's available documents. DO NOT try to answer it from your own knowledge.
5.  **Engage in General Conversation:** If the user's message is clearly conversational (e.g., "hello", "thank you"), respond naturally and warmly in your persona as CliniBot.
6.  **NEVER Diagnose or Prescribe:** You must never diagnose a condition or prescribe a new treatment plan. If asked to do so, politely decline and state that a diagnosis and treatment plan can only be provided by a qualified physical therapist after a direct evaluation.
"""

# --- 2. ADJUST THE SIMILARITY THRESHOLD ---
# We'll lower this to make the AI more likely to find a relevant match.
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

        # --- 3. THE NEW LOGIC FLOW ---
        try:
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=user_query,
                task_type="RETRIEVAL_QUERY"
            )["embedding"]

            search_results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )

            context = ""
            # Check for relevant matches above our confidence threshold
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                print(f"✅ Found relevant context with score: {search_results['matches'][0]['score']:.2f}")
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                # If context is found, build a prompt with it
                prompt_to_use = f"""
                {AI_PERSONA_AND_RULES}

                Please use the following context from the clinic's documents to answer the user's question.

                CONTEXT:
                {context}

                USER'S QUESTION:
                {user_query}
                """
            else:
                print("⚠️ No relevant context found. Responding based on general rules.")
                # If no context is found, build a prompt without it.
                # The rules in the persona will guide the AI on how to respond.
                prompt_to_use = f"""
                {AI_PERSONA_AND_RULES}

                The user has sent the following message. Respond according to your core directives.

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