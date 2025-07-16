# api/chat.py
from http.server import BaseHTTPRequestHandler
import json
import os
import google.generativeai as genai
from pinecone import Pinecone

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # --- 1. Read the incoming request, which now contains the full history ---
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        full_history = request_body.get('history', [])

        # --- 2. Securely get API Keys and configure services ---
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = "physical-therapy-index"

            if not all([gemini_api_key, pinecone_api_key]):
                raise ValueError("API keys are not configured correctly.")

            genai.configure(api_key=gemini_api_key)
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
        except Exception as e:
            self.send_error(500, f"Server Configuration Error: {e}")
            return

        # --- 3. Perform RAG search based on the LATEST user query ---
        context = ""
        try:
            # The latest query is the last message in the history
            latest_user_query = full_history[-1]['parts'][0]['text']

            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=latest_user_query,
                task_type="RETRIEVAL_QUERY"
            )["embedding"]

            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            context = " ".join([match['metadata']['text'] for match in search_results['matches']])
        except Exception as e:
            # If search fails, we can still proceed without context
            print(f"RAG Search Warning: {str(e)}")
            context = "Could not retrieve context from the knowledge base."

        # --- 4. Generate the Final Conversational Response ---
        try:
            # Initialize the model and start a chat session with the full history
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            chat_session = model.start_chat(history=full_history[:-1]) # History WITHOUT the latest query

            # This improved prompt is more robust and conversational
            final_prompt = f"""
            You are "Clinibot," a helpful and empathetic AI assistant for a physical therapy clinic.
            Your personality is professional, clear, and concise.
            First, consider the user's latest message: "{latest_user_query}".
            Next, consider the following relevant context from our clinic's documents: "{context}".
            Finally, consider the entire previous conversation history to understand the flow.

            Synthesize all of this information to provide the best possible response.
            If the context doesn't contain enough information to answer, state that you cannot find specific information
            in the clinic's documents and advise consulting a therapist. Do not make up information.
            If the user asks to simplify or explain something, use the conversation history to understand what "that" refers to.
            """

            response = chat_session.send_message(final_prompt)
            ai_answer = response.text
        except Exception as e:
            ai_answer = f"An error occurred with the AI service: {e}"

        # --- 5. Send the final response ---
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"answer": ai_answer}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return