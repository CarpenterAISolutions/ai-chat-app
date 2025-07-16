# api/chat.py
from http.server import BaseHTTPRequestHandler
import json
import os
import google.generativeai as genai
from pinecone import Pinecone

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # --- Step 1: Read the user's message ---
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query')

        # --- Step 2: Securely get API Keys ---
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = "physical-therapy-index"

            if not all([gemini_api_key, pinecone_api_key]):
                raise ValueError("API keys are not configured correctly on Vercel.")
        except Exception as e:
            self.send_error(500, f"Server Configuration Error: {e}")
            return

        # --- Step 3: Configure AI services ---
        try:
            genai.configure(api_key=gemini_api_key)
            pc = Pinecone(api_key=pinecone_api_key)

            if pinecone_index_name not in pc.list_indexes().names():
                raise ValueError("Pinecone index not found. Please run the ingestion script.")

            index = pc.Index(pinecone_index_name)
        except Exception as e:
            self.send_error(500, f"Error initializing AI services: {e}")
            return

        # --- Step 4: Full RAG Pipeline ---
        try:
            # A. Convert user's question into a vector
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=user_query,
                task_type="RETRIEVAL_QUERY"
            )["embedding"]

            # B. Search Pinecone for relevant documents
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

            # C. Construct a detailed prompt for the AI
            context = " ".join([match['metadata']['text'] for match in search_results['matches']])
            prompt_template = f"""
            You are a helpful AI assistant for a physical therapy clinic.
            Answer the user's question based ONLY on the following context provided from the clinic's documents.
            If the context doesn't contain the answer, say "I do not have information on that topic based on the provided documents. Please consult with one of our physical therapists directly."

            CONTEXT: {context}
            QUESTION: {user_query}
            ANSWER:
            """

            # D. Generate the final answer
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt_template)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred during the AI process: {e}"

        # --- Step 5: Send the final response ---
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"answer": ai_answer}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return