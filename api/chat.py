# api/chat.py
from http.server import BaseHTTPRequestHandler
import json
import os

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # This diagnostic code will run because the server itself is stable.
        # It checks for the environment variables and reports what it finds.

        gemini_key = os.getenv("GEMINI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")

        # Create a status report
        gemini_status = "Found" if gemini_key else "MISSING"
        pinecone_status = "Found" if pinecone_key else "MISSING"

        # Send the report back to the frontend
        report = (
            f"API Key Status Report:\n"
            f"GEMINI_API_KEY: {gemini_status}\n"
            f"PINECONE_API_KEY: {pinecone_status}"
        )

        # --- Send the response back to the frontend ---
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"answer": report}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return