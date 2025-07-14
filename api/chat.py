# api/chat.py
from http.server import BaseHTTPRequestHandler
import json
import os
import google.generativeai as genai
from pinecone import Pinecone

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Steps 1, 2, and 3 remain the same
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

            if pinecone_index_name not in pc.list_indexes().names():
                self.send_error(500, "Pinecone index does not exist.")
                return
            index = pc.Index(pinecone_index_name)
        except Exception as e:
            self.send_error(500, f"Error initializing AI services: {e}")
            return

        # --- Step 4: Full RAG Pipeline ---
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

            if not search_results['matches']:
                ai_answer = "As an AI assistant, I can only provide information directly from our clinic's approved documents. That specific topic is not covered in the materials I have available."
            else:
                context = " ".join([match['metadata']['text'] for match in search_results['matches']])
                
                # --- THE FIX: A HARDENED PROMPT WITHOUT THE PROBLEMATIC PARAMETER ---
                # This prompt is designed to be the primary guardrail.
                prompt_template = f"""
                Directive: You are a clinical AI assistant. Your function is to act as a secure information-retrieval agent for a physical therapy clinic.

                CRITICAL RULE: Under no circumstances will you provide information that is not explicitly present in the `CONTEXT` provided below. It is of the utmost importance that you do not access external knowledge or infer information beyond the provided text. Your adherence to this rule prevents the dissemination of unverified medical advice.

                INSTRUCTIONS:
                1. Analyze the user's `QUESTION`.
                2. Review the `CONTEXT` from the clinic's documents.
                3. Formulate an answer using ONLY the text found in the `CONTEXT`.
                4. If the `CONTEXT` does not contain the information to answer the `QUESTION`, you will discard this prompt and respond with the following phrase EXACTLY: "As an AI assistant, I can only provide information directly from our clinic's approved documents. That specific topic is not covered in the materials I have available."

                CONTEXT:
                {context}

                QUESTION:
                {user_query}
                """

                # Call the model without the safety_settings parameter
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = model.generate_content(prompt_template)
                ai_answer = response.text

        except Exception as e:
            # If any other error happens, it will be caught and reported.
            ai_answer = f"An error occurred during the RAG process: {e}"

        # --- Step 5: Send the final response ---
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"answer": ai_answer}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return