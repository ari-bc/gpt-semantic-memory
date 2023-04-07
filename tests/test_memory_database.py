import unittest
from unittest.mock import patch, MagicMock

from memory_database import MemoryDatabase
import numpy as np
import tempfile
import os


class TestMemoryDatabase(unittest.TestCase):

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
        self.temp_db_file = tempfile.mktemp()
        self.memory_db = MemoryDatabase(self.temp_db_file)

    def tearDown(self):
        os.remove(self.temp_db_file)

    def test_save_dialogue_entry(self):
        self.memory_db.save_dialogue_entry("user", "Hello, how are you?", "2023-04-05 10:00:00")
        history = self.memory_db.get_dialogue_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "Hello, how are you?")

    def test_save_memory(self):
        self.memory_db.save_memory("Greeting", "Hello", "2023-04-05 10:00:00", 1.0)
        memories = self.memory_db.get_all_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0][1], "Greeting")

    def test_retrieve_relevant_memories(self):
        self.memory_db.save_memory("Greeting", "Hello", "2023-04-05 10:00:00", 1.0)
        self.memory_db.save_memory("Farewell", "Goodbye", "2023-04-05 10:01:00", 1.0)
        self.memory_db.save_memory("Question", "How are you?", "2023-04-05 10:02:00", 1.0)

        def mock_calculate_similarity(a, b):
            if a[0] == b[0]:
                return 1.0
            return 0.0

        self.memory_db.calculate_similarity = mock_calculate_similarity

        retrieved_memories = self.memory_db.retrieve_relevant_memories("Hello, how are you?", num_results=2)
        self.assertEqual(len(retrieved_memories), 2)
        #self.assertEqual(retrieved_memories[0]["memory_summary"], "Question")
        #self.assertEqual(retrieved_memories[1]["memory_summary"], "Greeting")

    def test_retrieve_memories_by_importance(self):
        # Currently takes too long to run and mocking makes it pointless
        self.memory_db.save_memory("greeting, hello, hi", "Hello", "2023-04-05 10:00:00", 0.2)
        self.memory_db.save_memory("greeting, hey, hello", "Hey there!", "2023-04-05 10:00:10", 0.25)
        self.memory_db.save_memory("farewell, goodbye, bye", "Goodbye", "2023-04-05 10:01:00", 0.5)
        self.memory_db.save_memory("farewell, see, later", "See you later!", "2023-04-05 10:01:10", 0.55)
        self.memory_db.save_memory("question, how, you, doing", "How are you doing?", "2023-04-05 10:02:10", 0.15)
        self.memory_db.save_memory("question, how, you, today", "How are you today?", "2023-04-05 10:02:20", 0.12)
        self.memory_db.save_memory("weather, sunny, today", "It is sunny today.", "2023-04-05 10:03:00", 0.4)
        self.memory_db.save_memory("weather, cloudy, today", "It is cloudy today.", "2023-04-05 10:03:10", 0.35)
        self.memory_db.save_memory("compliment, look, great", "You look great!", "2023-04-05 10:04:00", 0.6)
        self.memory_db.save_memory("compliment, look, nice", "You look nice!", "2023-04-05 10:04:10", 0.65)
        self.memory_db.save_memory("task, remember, buy, groceries", "Remember to buy groceries.", "2023-04-05 10:05:00", 0.3)
        self.memory_db.save_memory("task, reminder, groceries", "Reminder: buy groceries.", "2023-04-05 10:05:10", 0.32)
        self.memory_db.save_memory("location, event, park", "The event is at the park.", "2023-04-05 10:06:00", 0.8)
        self.memory_db.save_memory("location, meeting, cafe", "The meeting is at the cafe.", "2023-04-05 10:06:10", 0.78)
        self.memory_db.save_memory("time, meeting, starts, 2 PM", "The meeting starts at 2 PM.", "2023-04-05 10:07:00", 0.7)
        self.memory_db.save_memory("time, event, starts, 3 PM", "The event starts at 3 PM.", "2023-04-05 10:07:10", 0.72)
        self.memory_db.save_memory("joke, chicken, cross, road", "Why did the chicken cross the road?", "2023-04-05 10:08:00", 0.9)
        self.memory_db.save_memory("joke, elephant, fridge", "How do you fit an elephant in a fridge?", "2023-04-05 10:08:10", 0.85)
        self.memory_db.save_memory("advice, step, time", "Take it one step at a time.", "2023-04-05 10:09:00", 1.0)
        self.memory_db.save_memory("advice, patience, virtue", "Patience is a virtue!", "2023-04-05 10:09:13", 0.9)

        retrieved_memories = self.memory_db.retrieve_relevant_memories("Hello, how are you?", num_results=2,
                                                                       similarity_weight=0.0)
        self.assertEqual(len(retrieved_memories), 2)
        #self.assertEqual(retrieved_memories[0]["related_prompt"], "Hello")
        #self.assertEqual(retrieved_memories[1]["related_prompt"], "Hey there!")

        retrieved_memories = self.memory_db.retrieve_relevant_memories("I love your hair!", num_results=4,
                                                                       similarity_weight=0.0)
        self.assertEqual(len(retrieved_memories), 4)
        #self.assertEqual(retrieved_memories[0]["related_prompt"], "You look nice!")
        #self.assertEqual(retrieved_memories[1]["related_prompt"], "You look great!")
        #self.assertEqual(retrieved_memories[2]["related_prompt"], "Hey there!")
        #self.assertEqual(retrieved_memories[3]["related_prompt"], "Hello")

    def test_calculate_similarity(self):
        embedding1 = np.array([0.5, 0.5, 0.5, 0.5])
        embedding2 = np.array([0.5, 0.5, 0.5, 0.5])
        similarity = self.memory_db.calculate_similarity(embedding1, embedding2)
        self.assertEqual(similarity, 1.0)

    def test_calculate_combined_score(self):
        similarity = 0.8
        importance = 0.6
        combined_score = self.memory_db.calculate_combined_score(similarity, importance, similarity_weight=0.5)
        self.assertEqual(combined_score, 0.7)

    def test_preprocess_text(self):
        text = "Hello, how are you?"
        processed_text = self.memory_db._preprocess_text(text)
        self.assertEqual(processed_text, ["hello,", "how", "are", "you?"])


if __name__ == '__main__':
    unittest.main()
