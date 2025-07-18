
# --- Constants and System Instructions ---
SYSTEM_INSTRUCTION = """
You are "CliniBot," an expert AI assistant for a physical therapy clinic. Your persona is professional, knowledgeable, and empathetic.
Your primary goal is to answer the user's question based on the provided `CONTEXT`.

**Your Rules:**
- You MUST base your answers on the information found in the `CONTEXT`.
- If the `CONTEXT` says no relevant documents were found, state that the topic is outside your scope and then proactively suggest topics you can help with.
- NEVER diagnose, prescribe, or give medical advice that is not explicitly in the `CONTEXT`.
- NEVER ask for personal health information.
"""
SIMILARITY_THRESHOLD = 0.70

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Helper function to send JSON responses
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))
        try:
            # --- 1. Initialize Langfuse ---
            # Securely get LangFuse keys from environment variables
            langfuse = Langfuse(
                secret_key=os.getenv("sk-lf-80f208d4-9393-40cd-ba05-d3f80c2bc53b"),
                public_key=os.getenv("pk-lf-a542df46-3a4e-4469-818e-100e1170cae5"),
                host=os.getenv("https://us.cloud.langfuse.com")
            )
except Exception as e:
            send_json_response(500, {"answer": f"Server Error: Could not initialize Langfuse. {e}"})
            return

        # --- Process Incoming Request ---
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query', "").strip()
        history = request_body.get('history', [])

        # --- Create the Root Trace ---
        trace = langfuse.trace(
            name="rag-pipeline",
            user_id="end-user-123", # Replace with a dynamic user ID in production
            input={"query": user_query}
        )

        try:
            if not user_query:
                send_json_response(200, {"answer": "Please type a message."})
                return

            # --- Initialize AI & DB Services ---
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
            pinecone_index_name = "physical-therapy-index"

            if not all([gemini_api_key, pinecone_api_key, pinecone_environment]):
                raise ValueError("Server Error: One or more API configuration variables are missing.")

            pc = Pinecone(api_key=pinecone_api_key)
            genai.configure(api_key=gemini_api_key)
            index = pc.Index(pinecone_index_name)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')

            # --- Retrieval Span ---
            retrieval_span = trace.span(
                name="retrieval",
                input={"query": user_query}
            )
            query_embedding = genai.embed_content(model="models/text-embedding-004", content=user_query, task_type="RETRIEVAL_QUERY")["embedding"]
            search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

            retrieved_docs = ""
            if search_results['matches'] and search_results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
                retrieved_docs = "\n".join([match['metadata']['text'] for match in search_results['matches']])

            retrieval_span.end(output={"retrieved_docs": retrieved_docs})

            # --- Generation Span ---
            generation_span = trace.span(
                name="generation",
                input={"history": history, "retrieved_docs": retrieved_docs}
            )
            formatted_history = "\n".join([f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" for msg in history])
            combined_context = f"CONVERSATIONAL HISTORY:\n{formatted_history}\n\nRELEVANT DOCUMENTS:\n{retrieved_docs if retrieved_docs else 'No relevant documents were found for this query.'}"
            final_prompt = f"{SYSTEM_INSTRUCTION}\n\nCONTEXT:\n{combined_context}\n\nBased on the CONTEXT, provide a direct and helpful answer to the user's latest message:\n{user_query}"

            response = model.generate_content(final_prompt)
            ai_answer = response.text
            generation_span.end(output={"answer": ai_answer})

            # Update the overall trace with the final output
            trace.update(output={"final_answer": ai_answer})

        except Exception as e:
            ai_answer = f"An error occurred: {e}"
            # Log the error to the trace before sending it to the user
            trace.update(output={"error": ai_answer}, level="ERROR")
            print(f"Error: {e}")

        # --- Send Final Response and Flush Langfuse ---
        send_json_response(200, {"answer": ai_answer})
        langfuse.flush() # Ensure all traces are sent
        return