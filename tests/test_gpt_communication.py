import gensim.downloader as api
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

from gpt_communication import GPTCommunication


class TestGPTCommunication(unittest.TestCase):

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
        self.api_key = "placeholder_key"
        self.db_file = ":memory:"
        self.gpt_comm = GPTCommunication(self.api_key, self.db_file)

        # Mock the openai.ChatCompletion.create method to avoid actual API calls
        self.patcher_openai_create = patch('openai.ChatCompletion.create')
        self.mock_openai_create = self.patcher_openai_create.start()
        self.mock_openai_create.return_value = {
            'choices': [
                {
                    'message': {
                        'content': "Hello, how are you?"
                    }
                }
            ]
        }

    def tearDown(self):
        self.patcher_openai_create.stop()

    def test_add_message(self):
        self.gpt_comm.add_message("user", "Hello!")
        self.assertEqual(len(self.gpt_comm.messages), 2)
        self.assertEqual(self.gpt_comm.messages[-1], {"role": "user", "content": "Hello!"})

    def test_clear_messages(self):
        self.gpt_comm.add_message("user", "Hello!")
        self.gpt_comm.clear_messages()
        self.assertEqual(len(self.gpt_comm.messages), 1)
        self.assertIn("system", self.gpt_comm.messages[0]["role"])

    @patch("openai.ChatCompletion.create")
    def test_send_message(self, mock_chat_completion):
        mock_chat_completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="r: Hello! I am your assistant.\nsummary: Greeting\ni: 1.0\nc: greeting, hello")
                )
            ]
        )

        user_input = "Hello, how are you?"
        assistant_response = self.gpt_comm.send_message(user_input)

        self.assertEqual(assistant_response, "Hello! I am your assistant.")
        self.assertEqual(len(self.gpt_comm.messages), 6)
        self.assertIn("assistant", self.gpt_comm.messages[-1]["role"])

        mock_chat_completion.assert_called_once()
        messages = mock_chat_completion.call_args[1]["messages"]
        self.assertEqual(messages[-1], {"role": "assistant", "content": "Hello! I am your assistant."})


if __name__ == "__main__":
    unittest.main()
