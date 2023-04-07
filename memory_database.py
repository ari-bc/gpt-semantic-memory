import threading
import time
from typing import Dict, List, Sequence, Optional
from urllib.parse import unquote

import gensim.downloader as api
import numpy as np
from annoy import AnnoyIndex
from sqlalchemy import Column, Integer, String, Float, LargeBinary, MetaData, func
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class DialogueHistory(Base):
    __tablename__ = 'dialogue_history'

    id = Column(Integer, primary_key=True)
    content = Column(String, nullable=False)
    speaker = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)


class Memories(Base):
    __tablename__ = 'memories'
    id = Column(Integer, primary_key=True)
    memory_summary = Column(String)
    related_prompt = Column(String)
    embedding = Column(LargeBinary)
    timestamp = Column(String)
    importance = Column(Float)


def delete_unimportant_memories():
    engine = create_engine(f"sqlite:///memories.db",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool,
                           echo=False)
    session = scoped_session(sessionmaker(bind=engine))

    Base.metadata.create_all(bind=engine)
    session.query(Memories).filter(Memories.importance < 10.0).delete()
    session.commit()
    session.close()

#delete_unimportant_memories()

class MemoryDatabase:

    def __init__(self, db_file: str):
        print("Loading word2vec data... ", end='')
        begin_time = time.time()
        self.word2vec = api.load("word2vec-google-news-300")
        print(f"Loaded in {time.time() - begin_time} seconds")
        self.embedding_dim = self.word2vec.vector_size
        self._db_lock = threading.Lock()

        engine = create_engine(f"sqlite:///{db_file}",
                               connect_args={"check_same_thread": False},
                               poolclass=StaticPool,
                               echo=False)
        self.Session = scoped_session(sessionmaker(bind=engine))

        Base.metadata.create_all(bind=engine)

        print("Building AnnoyIndex... ", end='')
        begin_time = time.time()
        self.annoy_index = AnnoyIndex(self.embedding_dim, 'angular')
        self._build_annoy_index()
        print(f"Loaded in {time.time() - begin_time} seconds")

    def _build_annoy_index(self):
        memories = self.get_all_memories()
        ID_COLUMN = 0
        SUMMARY_COLUMN = 1

        for memory in memories:
            embedding = self._generate_embedding(memory[SUMMARY_COLUMN])
            self.annoy_index.add_item(memory[ID_COLUMN], embedding)

        self.annoy_index.build(10)  # You can adjust the number of trees (10) for better accuracy

    def save_dialogue_entry(self, speaker: str, content: str, timestamp: str):
        session = self.Session()
        decoded_content = unquote(content)
        new_dialogue = DialogueHistory(speaker=speaker, content=decoded_content, timestamp=timestamp)
        session.add(new_dialogue)
        session.commit()
        session.close()

    def get_dialogue_history(self, num_results: int = None, max_length: int = 2000) -> List[Dict]:
        session = self.Session()
        query = session.query(DialogueHistory).order_by(DialogueHistory.timestamp.desc())

        if num_results is not None:
            query = query.limit(num_results)

        dialogue_history = query.all()
        session.close()

        total_dialogue_length = 0
        dialogue_to_return = []
        for entry in reversed(dialogue_history):
            content_length = len(entry.content)
            total_dialogue_length += content_length
            if total_dialogue_length > max_length:
                break
            dialogue_to_return.insert(0, entry)

        return [{'id': entry.id, 'speaker': entry.speaker, 'content': entry.content, 'timestamp': entry.timestamp} for
                entry in dialogue_to_return]

    def save_memory(self, memory_summary: str, related_prompt: str, timestamp: str, importance: float):
        #embedding = self._generate_embedding(memory_summary)
        #same_memory = self.find_same_memory(embedding)

        #if same_memory:
        #    memory_id = same_memory.id
        #    existing_importance = same_memory.importance
        #    # Add a small increment to the importance so that repeated exposure gradually increases it
        #    self.update_memory(memory_id, timestamp, existing_importance + 0.1)
        #else:
        scaled_importance = pow(importance, 3)/100.0
        if scaled_importance < 2.0:
            # Don't save memories that are too low importance
            return
        self.insert_memory(memory_summary, related_prompt, timestamp, scaled_importance)

    def insert_memory(self, memory_summary: str, related_prompt: str, timestamp: str, importance: float):
        session = self.Session()

        print(f"ADDING MEMORY: s:{memory_summary}\nr:{related_prompt}\nt:{timestamp}\ni:{importance}")
        new_memory = Memories(memory_summary=memory_summary, related_prompt=related_prompt,
                              embedding=None, timestamp=timestamp, importance=importance)
        session.add(new_memory)
        session.commit()

        embedding = self._generate_embedding(memory_summary)
        self.annoy_index.unbuild()
        self.annoy_index.add_item(new_memory.id, embedding)
        self.annoy_index.build(10)

        session.close()

    def update_memory(self, memory_id: int, timestamp: str, new_importance: float):
        session = self.Session()

        memory = session.query(Memories).filter(Memories.id == memory_id).first()
        if memory:
            memory.timestamp = timestamp
            memory.importance = new_importance
            session.commit()

        session.close()

    def find_same_memory(self, new_embedding: np.ndarray, threshold: float = 0.99) -> Optional[Memories]:
        """
        Find a memory in the database with a similar embedding to the given embedding.

        :param new_embedding: The embedding to compare with the existing memories.
        :param threshold: The similarity threshold above which a memory is considered similar.
        :return: The similar memory if found, otherwise None.
        """
        closest_memory_ids, closest_memory_distances = self.annoy_index.get_nns_by_vector(new_embedding, 1,
                                                                                          include_distances=True)
        session = self.Session()

        if not closest_memory_ids:
            return None

        memory = session.query(Memories).filter(Memories.id == closest_memory_ids[0]).first()

        if memory:
            existing_embedding = np.frombuffer(memory.embedding, dtype=np.float32)
            similarity = self.calculate_similarity(new_embedding, existing_embedding)
            if similarity > threshold:
                return memory

        return None

    def get_all_memories(self) -> Sequence:
        session = self.Session()
        cursor = session.connection().connection.cursor()
        cursor.execute("SELECT * FROM memories")
        memories = cursor.fetchall()
        session.close()
        return memories

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two embeddings.

        :param embedding1: The first embedding.
        :param embedding2: The second embedding.
        :return: The cosine similarity between the embeddings.
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def retrieve_relevant_memories(self, user_input: str, num_results: int = 5, similarity_weight: float = 0.5) -> List[Dict]:
        """
        Retrieve memories relevant to the user input.

        :param user_input: The user input to find relevant memories.
        :param num_results: The number of results to return.
        :return: A list of relevant memories.
        """
        user_input_embedding = self._generate_embedding(user_input)

        closest_memory_ids, closest_memory_distances = self.annoy_index.get_nns_by_vector(
            user_input_embedding, num_results, include_distances=True)

        closest_memories = []
        session = self.Session()

        for memory_id, distance in zip(closest_memory_ids, closest_memory_distances):
            memory = session.query(Memories).filter(Memories.id == memory_id).first()

            if memory is not None:
                closest_memories.append({
                    'memory_id': memory.id,
                    'memory_summary': memory.memory_summary,
                    'related_prompt': memory.related_prompt,
                    'timestamp': memory.timestamp,
                    'importance': memory.importance,
                    'distance': distance
                })

        session.close()

        # Sort the memories by their combined scores
        sorted_memories = sorted(
            closest_memories,
            key=lambda x: self.calculate_combined_score(1 - x['distance'], x['importance'], similarity_weight),
            reverse=True
        )

        return sorted_memories

    def calculate_combined_score(self, similarity: float, importance: float, similarity_weight: float) -> float:
        """
        Calculate the combined score based on similarity and importance.

        :param similarity: The similarity score.
        :param importance: The importance score.
        :param similarity_weight: The weight of the similarity score in the combined score.
        :return: The combined score.
        """
        return similarity_weight * similarity + (1 - similarity_weight) * importance

    def _generate_embedding(self, text: str) -> np.ndarray:
        words = self._preprocess_text(text)
        word_embeddings = [self.word2vec[word.strip(',').strip(' ')] for word in words if word in self.word2vec]
        if not word_embeddings:
            return np.zeros(self.word2vec.vector_size)
        return np.mean(word_embeddings, axis=0)

    def _preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        words = text.split()
        return words

