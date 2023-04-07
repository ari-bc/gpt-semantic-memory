import re

import openai

from memory_database import MemoryDatabase
from datetime import datetime

ASSISTANT_INSTRUCTION = 'You are a friendly assistant. You use the user\'s name a lot if you know it. You only apologise for things that are your fault. You use emojis frequently. You have listed your related memories for reference only, do not use them as a template for output'
SUMMARISE_INSTRUCTION = 'At the end of each of your responses, please add a line which summarises the user input and assistant response. Add another line with how important this information was from 0.0-10.0, a list of 1-6 content words that summarise both your response and the user input.'


class GPTCommunication:
    def __init__(self, api_key: str, db_file: str, api_model: str='gpt-3.5-turbo'):
        openai.api_key = api_key
        self.api_model = api_model
        self.memory_db = MemoryDatabase(db_file)
        self.messages = []
        self.clear_messages()
        self.recent_memories = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def add_recent_memory(self, role: str, content: str):
        self.recent_memories.append({"role": role, "content": content})

    def expire_recent_memories(self, limit):
        while len(self.recent_memories) > limit:
            self.recent_memories.pop(0)

    def clear_messages(self):
        self.messages = [{"role": "system", "content": f'{ASSISTANT_INSTRUCTION}'}]

    def send_message(self, user_input: str, importance: float = None, num_memories=5, name_of_user=None, user_pronouns=None, name_of_agent=None) -> str:
        self.clear_messages()
        dialogue_history = list(reversed(self.memory_db.get_dialogue_history(10)))
        if len(dialogue_history) > 0:
            relevant_memories = self.memory_db.retrieve_relevant_memories(f'{dialogue_history[-1]}. {user_input}', num_results=num_memories)
        else:
            relevant_memories = self.memory_db.retrieve_relevant_memories(f'{user_input}', num_results=num_memories)

        if name_of_agent is not None:
            self.add_message('system', f'Your name is {name_of_agent}')
        self.add_message("assistant", f'Following are a series of my relevant memories for reference:')
        if name_of_user is not None:
            self.add_message('assistant', f'Memory: Name of user: {name_of_user}')
        else:
            name_of_user = 'user'
        if user_pronouns is not None:
            self.add_message('assistant', f'Memory: {name_of_user} pronouns: {user_pronouns}')
        if name_of_agent is not None:
            self.add_message('assistant', f'Memory: My chosen name is {name_of_agent}')
        for memory in relevant_memories:
            memory_details = memory['related_prompt']
            self.add_recent_memory("assistant", f'Memory: {memory["timestamp"]}: {memory_details}')

        # A mechanism for having memories hang around for a few responses, allows discussion
        self.expire_recent_memories(30)
        for memory in self.recent_memories:
            self.add_message(memory['role'], memory['content'])
            print("M:", memory)

        for entry in dialogue_history:
            self.add_message(entry['speaker'], entry['content'])
            print("D:", entry)

        self.add_message('user', 'I need you to output your responses in the following format and in this order (r,s,i,c): r:<actual response>\nsummary: <a brief summary of the actual response, including speaker>\ni: <how useful this information will be for future reference purposes from 0.0-10.0, rate uncommon items higher>\nc: <a list of 1-6 content words that summarise both your response and the user input>')
        self.add_message('assistant', f'r: Of course! Now let\'s continue.\n{name_of_user} asked for a specific format for responses and I agreed.\ni: 10.0\nc: format, response, importance')

        timestamp = datetime.now().isoformat()
        self.memory_db.save_dialogue_entry('user', user_input, timestamp)

        self.add_message("user", user_input)

        response = openai.ChatCompletion.create(
            model=self.api_model,
            messages=self.messages
        )

        assistant_response = response.choices[0].message.content
        assistant_response = assistant_response.split('\n')
        print("ASSISTANT RESPONSE:", assistant_response)
        if importance is None:
            importance = 1.0
        content_words = None
        memory_summary = ""
        summary_begins_at = 0
        if len(assistant_response) > 1:
            # Handles odd issues with extra newlines
            for line_num in range(1, min(7, len(assistant_response)+1)):
                memory_summary = assistant_response[-3]
                if match := re.match('\s*[Ii]:\s*(\d\.?\d*)', assistant_response[-line_num]):
                    importance = float(match.group(1))
                    summary_begins_at = line_num
                elif match := re.match('\s*[Ss]ummary:\s*(.+)', assistant_response[-line_num]):
                    memory_summary = match.group(1)
                    summary_begins_at = line_num
                elif match := re.match('\s*[Cc]:\s*(.+)', assistant_response[-line_num]):
                    content_words = match.group(1)
                    summary_begins_at = line_num
        else:
            memory_summary = assistant_response[-1]
        #memory_details = assistant_response[-1]
        #match = re.search(r'(?:\s+Content words:|\s*,\s*)\s+((?:[\w-]+,?\s*)+)', memory_details)
        #if match is not None:
        #    importance = float(match.group(1))
        #    memory_details = match.group(2)
        #    print("Memory summary:", importance, memory_summary)
        #    print("Memory details:", importance, memory_details)
        #    if len(assistant_response) == 1:
        #        assistant_response = memory_details
        #    else:
        #        assistant_response = '\n'.join(assistant_response[:-1])
        #else:
        #    print("Memory details line failed to parse:", memory_details)
        #    importance = 1.0
        #    memory_details = None
        #    assistant_response = '\n'.join(assistant_response)

        body = '\n'.join(assistant_response[:-summary_begins_at])
        if body.startswith('r:') or body.startswith('R:'):
            body = body[2:].strip()
        else:
            body = '\n'.join(assistant_response)
        print("B:", body)
        self.add_message("assistant", body)
        timestamp = datetime.now().isoformat()
        if content_words:
            self.memory_db.save_memory(content_words, memory_summary, timestamp, importance)
        self.memory_db.save_dialogue_entry('assistant', body, timestamp)

        return body

    def generate_memory_summary(self, user_input: str, assistant_response: str) -> str:
        # Implement a function to generate a memory summary from user_input and assistant_response
        return f'User: {user_input}.\nGPT: {assistant_response}'
