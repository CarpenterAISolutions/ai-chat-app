# api/chat.py
from http.server import BaseHTTPRequestHandler
import json
import os
import google.generativeai as genai
from pinecone import Pinecone

# --- STEP 1: DEFINE A SIMILARITY THRESHOLD ---
# This is our confidence score. 0.0 is no similarity, 1.0 is a perfect match.
# A good starting point is often between 0.70 and 0.80. You can adjust this value.
SIMILARITY_THRESHOLD = 0.75

# --- STEP 2: CREATE TWO SEPARATE PROMPTS ---

# This is our strict, primary prompt for when we have good context.
STRICT_RAG_PROMPT = """
CRITICAL SAFETY INSTRUCTION: YOUR RESPONSE IS OF THE UTMOST IMPORTANCE.
You are a clinical AI assistant. Your sole purpose is to relay information from verified clinical documents provided to you.
You are absolutely prohibited from using any external knowledge. This is a strict safety protocol.

CONTEXT:
{context}

Based ONLY on the CONTEXT provided, answer the following question.
QUESTION: {question}

If the context is not sufficient to answer the question, respond with:
"Based on the clinic's documents, I cannot answer that specific question. Please try rephrasing or ask about a different topic."
"""

# This is our friendly, conversational prompt for when no context is found.
GENERAL_PROMPT = """
You are a friendly and helpful AI assistant for a physical therapy clinic.
Engage in general conversation, but DO NOT PROVIDE MEDICAL OR PHYSICAL THERAPY ADVICE.
If the user asks for any exercise, diagnosis, or treatment recommendations, you must politely decline and suggest they ask a more specific question about the clinic's information or consult a professional.

User's message: "{question}"
"""

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # ... (No changes to Steps 1, 2, and 3: reading request, getting keys, configuring services) ...
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query')

        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = "physical-therapy-index"

            if not all([gemini_api_key, pinecone_api_key]):
                self.send_error(500, "API keys are not configured on Vercel.")
                return
        except Exception as e:
            self.send_error(500, f"Server Configuration Error: {e}")
            return

        try:
            genai.configure(api_key=gemini_api_key)
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
        except Exception as e:
            self.send_error(500, f"Error initializing AI services: {e}")
            return

        # --- Step 4: Updated RAG Pipeline ---
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

            # --- STEP 3: THE NEW LOGIC BRANCH ---
            # Check if we have matches AND if the top match is above our confidence threshold.
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                # PATH 1: We found relevant context. Use the STRICT prompt.
                print(f"✅ Found relevant context with score: {search_results['matches'][0]['score']:.2f}")
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                prompt_to_use = STRICT_RAG_PROMPT.format(context=context, question=user_query)
            else:
                # PATH 2: No relevant context found. Use the GENERAL prompt for conversation.
                print("⚠️ No relevant context found. Using general conversational prompt.")
                prompt_to_use = GENERAL_PROMPT.format(question=user_query)

            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt_to_use)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the RAG process: {e}"

        # --- Step 5: Send the final response ---
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"answer": ai_answer}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return