# api/chat.py
from http.server import BaseHTTPRequestHandler
import json
import os
import google.generativeai as genai

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)
        user_query = request_body.get('query')

        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                self.send_error(500, "GEMINI_API_KEY is not configured on Vercel.")
                return
        except Exception as e:
            self.send_error(500, f"Server Configuration Error: {e}")
            return

        try:
            genai.configure(api_key=gemini_api_key)
            # This line has been corrected to use the current model name
            model = genai.GenerativeModel('gemini-1.5-flash-latest')

            response = model.generate_content(user_query)
            ai_answer = response.text

        except Exception as e:
            ai_answer = f"An error occurred with the AI service: {e}"

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"answer": ai_answer}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return