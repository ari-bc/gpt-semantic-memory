import unittest
from flask import json
from app import app

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_send_message(self):
        user_input = "Hello, how are you?"
        response = self.app.post(
            '/send_message',
            data=dict(user_input=user_input),
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("response", data)
        self.assertIsInstance(data["response"], str)

if __name__ == '__main__':
    unittest.main()
