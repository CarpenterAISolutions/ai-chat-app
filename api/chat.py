import os
import json
from http.server import BaseHTTPRequestHandler

# This is a temporary script for diagnostics only.
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        def send_json_response(status_code, content):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(content).encode('utf-8'))

        # Get the API key from Vercel's environment
        pinecone_key = os.getenv("PINECONE_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        # Create a diagnostic message
        if pinecone_key:
            diagnostic_message = f"Vercel is using a Pinecone key ending in: ...{pinecone_key[-6:]}"
        else:
            diagnostic_message = "Vercel cannot find the PINECONE_API_KEY environment variable."
        
        print(f"--- [DIAGNOSTIC LOG] ---")
        print(diagnostic_message)
        print(f"Gemini Key Found: {'Yes' if gemini_key else 'No'}")
        print(f"----------------------")

        send_json_response(200, {"answer": diagnostic_message})