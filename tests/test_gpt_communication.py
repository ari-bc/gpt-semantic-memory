import unittest
from types import SimpleNamespace
from unittest.mock import patch
from gpt_communication import GPTCommunication


class TestGPTCommunication(unittest.TestCase):

    def setUp(self):
        self.api_key = "placeholder_key"
        self.db_file = ":memory:"
        self.gpt_comm = GPTCommunication(self.api_key, self.db_file)

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
                    message=SimpleNamespace(content="Hello! I am your assistant.\n5.0 greeting, assistant, meet")
                )
            ]
        )

        user_input = "Hello, how are you?"
        assistant_response = self.gpt_comm.send_message(user_input)

        self.assertEqual(assistant_response, "Hello! I am your assistant.")
        self.assertEqual(len(self.gpt_comm.messages), 4)
        self.assertIn("assistant", self.gpt_comm.messages[-1]["role"])

        mock_chat_completion.assert_called_once()
        messages = mock_chat_completion.call_args[1]["messages"]
        self.assertEqual(messages[-1], {"role": "assistant", "content": "Hello! I am your assistant."})


if __name__ == "__main__":
    unittest.main()
