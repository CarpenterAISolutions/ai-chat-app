import unittest
from unittest.mock import MagicMock
import json

# We'll define a minimal mock handler to test send_json_response in isolation.
class MockHandler:
    def __init__(self):
        self.sent_status = None
        self.headers = {}
        self.ended = False
        self.wfile = MagicMock()
    def send_response(self, code):
        self.sent_status = code
    def send_header(self, key, value):
        self.headers[key] = value
    def end_headers(self):
        self.ended = True

# The function to test, extracted for testability.
def send_json_response(self, status_code, content):
    self.send_response(status_code)
    self.send_header('Content-type', 'application/json')
    self.end_headers()
    self.wfile.write(json.dumps(content).encode('utf-8'))

class TestSendJsonResponse(unittest.TestCase):
    def test_send_json_response_success(self):
        handler = MockHandler()
        content = {"answer": "test"}
        send_json_response(handler, 200, content)
        # Check status code
        self.assertEqual(handler.sent_status, 200)
        # Check headers
        self.assertIn('Content-type', handler.headers)
        self.assertEqual(handler.headers['Content-type'], 'application/json')
        # Check end_headers called
        self.assertTrue(handler.ended)
        # Check wfile.write called with correct JSON
        handler.wfile.write.assert_called_once_with(json.dumps(content).encode('utf-8'))

    def test_send_json_response_with_different_status(self):
        handler = MockHandler()
        content = {"error": "not found"}
        send_json_response(handler, 404, content)
        self.assertEqual(handler.sent_status, 404)
        handler.wfile.write.assert_called_once_with(json.dumps(content).encode('utf-8'))

if __name__ == "__main__":
    unittest.main()