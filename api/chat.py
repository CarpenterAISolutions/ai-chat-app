# api/chat.py
from http.server import BaseHTTPRequestHandler
import json
import os
import google.generativeai as genai

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # --- Step 1: Read the user's message from the request ---
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query')

        # --- Step 2: Securely get the Gemini API Key from Vercel ---
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                self.send_error(500, "GEMINI_API_KEY is not configured on Vercel.")
                return
        except Exception as e:
            self.send_error(500, f"Server Configuration Error: {e}")
            return

        # --- Step 3: Configure and call the Gemini model ---
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')

            # For this test, we send the user's query directly to the AI
            response = model.generate_content(user_query)
            ai_answer = response.text

        except Exception as e:
            # If the AI call fails, send back a specific error message
            ai_answer = f"An error occurred with the AI service: {e}"

        # --- Step 4: Send the AI's response back to the frontend ---
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"answer": ai_answer}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return