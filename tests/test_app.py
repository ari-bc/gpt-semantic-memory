import unittest
from unittest.mock import MagicMock, patch
from flask import json
import numpy as np

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Mock so that the word2vec model is not loaded
        cls.patcher_api_load = patch('gensim.downloader.load')
        cls.mock_api_load = cls.patcher_api_load.start()

        # Create a mock word2vec model with correct dimensions
        mocked_word2vec = MagicMock()
        mocked_word2vec.vector_size = 300
        mocked_word2vec.get_vector = MagicMock(side_effect=lambda _: np.zeros(300))
        cls.mock_api_load.return_value = mocked_word2vec

    @classmethod
    def tearDownClass(cls):
        cls.patcher_api_load.stop()

    def setUp(self):
        # Import app after patching
        from front_ends.app import app

        # Mock the openai request system to avoid actually calling the API
        self.patcher = patch('app.GPTCommunication.send_message')
        self.mock_send_message = self.patcher.start()
        self.mock_send_message.return_value = "Hello, how are you?"

        self.app = app.test_client()
        self.app.testing = True

    def tearDown(self):
        self.patcher.stop()

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
