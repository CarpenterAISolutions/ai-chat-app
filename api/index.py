# api/index.py
from http.server import BaseHTTPRequestHandler
import json

# Vercel looks for a class named 'handler' that inherits from BaseHTTPRequestHandler.
# This is the most basic way to create a serverless function.
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Send a success response code
        self.send_response(200)

        # Set the content type to application/json
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # Create the success message
        response_data = {"answer": "Success! The basic Python server is working."}

        # Write the response back to the browser
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return