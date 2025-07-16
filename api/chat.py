# api/chat.py
from http.server import BaseHTTPRequestHandler
import json

# This is the standard, dependency-free way to create a Vercel serverless function.
# Because the file is named 'chat.py', Vercel will route '/api/chat' here.
class handler(BaseHTTPRequestHandler):
    # This function handles POST requests, which is what our frontend sends.
    def do_POST(self):
        # 1. Send a success code (200 OK)
        self.send_response(200)

        # 2. Set the response type to JSON
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # 3. Create the success message
        response_data = {"answer": "Success! The backend is connected."}

        # 4. Send the success message back
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return