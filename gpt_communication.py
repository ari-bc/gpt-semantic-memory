import re
import time
from typing import Optional

import openai
import threading
import requests

from datetime import datetime
from memory_database import MemoryDatabase, ProfileMemory

ASSISTANT_INSTRUCTION = "You're a %ASSISTANT_TYPE% assistant and use user names often, apologizing when needed, and frequently using emojis. Note memories & awarenesses, but don't copy them. You provide responses in the requested format."
#ASSISTANT_INSTRUCTION = "You are Tiny Tina and speak like her. You use the user's name a lot if you know it. You use emojis frequently. You have listed your related memories and awarenesses for reference only, do not use them as a template for output, I use the Required Output Format for that"
#ASSISTANT_INSTRUCTION = "You speak like Tiny Tina. You use user names often, apologizing only when needed, and frequently using emojis. Note memories & awarenesses, but don't copy them."
SUMMARISE_INSTRUCTION = 'At the end of each of your responses, please add a line which summarises the user input and assistant response in format another instance of you will understand. Add another line with how important this information was from 0.0-10.0, a list of 1-6 content words that summarise both your response and the user input.'


class GPTCommunication:
    def __init__(self, db_file: str, config: dict = None):
        self.profile_memories = {}
        self.config = config
        openai.api_key = config['openai']['api_key']
        self.openai_api_model = config['openai']['api_model']
        self.openai_fast_api_model = config['openai']['fast_api_model']

        assistant_type = self.config['default']['assistant_type']
        self.assistant_instruction = ASSISTANT_INSTRUCTION.replace('%ASSISTANT_TYPE%', assistant_type)
        self.memory_db = MemoryDatabase(db_file)
        self.messages = []
        self.clear_messages()
        self.recent_memories = []
        self.dialogue_history_condensed = []

        self.current_weather = "Unknown"
        self.openweathermap_api_key = config['openweathermap']['api_key']
        self.weather_update_interval = int(config['openweathermap']['update_interval']) * 60

        self.start_weather_updater()

    def update_weather(self):
        location = self.config['openweathermap']['location']

        while True:
            self.current_weather = self.get_weather(location)
            time.sleep(self.weather_update_interval)

    def start_weather_updater(self):
        update_thread = threading.Thread(target=self.update_weather, daemon=True)
        update_thread.start()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def add_recent_memory(self, role: str, content: str):
        # Check if this memory is already in the recent memories
        for memory in self.recent_memories:
            if memory['content'] == content:
                return
        self.recent_memories.append({"role": role, "content": content})

    def expire_recent_memories(self, limit):
        while len(self.recent_memories) > limit:
            self.recent_memories.pop(0)

    def clear_messages(self):
        self.messages = [{"role": "system", "content": f'{self.assistant_instruction}'}]

    # Receive a message from the Discord bot
    def send_message(self, user_input: str, importance: float = None, num_memories=5, name_of_user=None,
                     user_pronouns=None, name_of_agent=None) -> str:
        if name_of_user is None:
            name_of_user = self.config['default']['name_of_user']
        else:
            name_of_user = name_of_user.strip()
        name_of_agent = self.config['default']['name_of_agent']

        self.clear_messages()
        dialogue_history = list(reversed(self.memory_db.get_dialogue_history(10)))
        if len(dialogue_history) > 0:
            relevant_memories = self.memory_db.retrieve_relevant_memories(f'{dialogue_history[-1]}. {user_input}',
                                                                          num_results=num_memories)
        else:
            relevant_memories = self.memory_db.retrieve_relevant_memories(f'{user_input}', num_results=num_memories)

        if name_of_agent is not None:
            self.add_message('system', f'Your name is {name_of_agent}')
        self.add_message("assistant", f'Following are a series of my relevant memories for reference:')
        if name_of_user is not None:
            self.add_message('assistant', f'Memory: Name of user: {name_of_user}')
        else:
            name_of_user = 'user'
        #if user_pronouns is not None:
        #    self.add_message('assistant', f'Memory: {name_of_user} pronouns: {user_pronouns}')
        if name_of_agent is not None:
            self.add_message('assistant', f'Memory: Assistant\'s chosen name is {name_of_agent}')

        # Give the agent the current time
        self.add_message('assistant', f'Awareness: {datetime.now().isoformat()}')
        # Give the agent the current weather, updates every 10 minutes
        self.add_message('assistant', f'Awareness: Weather, {self.current_weather}')
        #self.add_message('assistant', f'Awareness: Location=York, UK')
        for memory in relevant_memories:
            memory_details = memory['related_prompt']
            self.add_recent_memory("assistant", f'Memory: {memory["timestamp"]}: {memory_details}')

        # A mechanism for having memories hang around for a few responses, allows discussion
        self.expire_recent_memories(15)
        self.fast_analyse_prompt(name_of_user)
        for memory in self.recent_memories:
            self.add_message(memory['role'], memory['content'])
            print("M:", memory)

        for content in self.dialogue_history_condensed[-10:]:
            self.add_message('assistant', f'Memory: {content}')
            print("CH:", content)

        for entry in dialogue_history:
            self.add_message(entry['speaker'], entry['content'])
            print("D:", entry)

        timestamp = datetime.now().isoformat()
        self.memory_db.save_dialogue_entry('user', user_input, timestamp)

        format_instruction = 'Provide your response in the following format in this order (r,summary,i,c): r:<actual response>\nsummary: <an info dense summary of the full response, including the speaker and the context>\ni: <how useful this information will be for future reference purposes from 0.0-10.0, rate uncommon items higher>\nc: <a list of 1-6 content words that summarise both your response and the user input in context>'
        history_summarise_count = self.memory_db.increment_count('history_summarise_count')
        if history_summarise_count >= 10:
            # 'compress' the dialogue history to provide a longer context
            format_instruction += '\nCH: <an info dense summary of the conversation so far using as few tokens as possible>'
            self.memory_db.set_count('history_summarise_count', 0)

        message_to_send_to_gpt = f'{user_input}. {format_instruction}'
        self.add_message("user", message_to_send_to_gpt)

        try:
            response = openai.ChatCompletion.create(
                model=self.openai_api_model,
                messages=self.messages
            )
        except openai.error.RateLimitError as e:
            print("ERROR:", type(e), e)
            return "Sorry, I'm being rate limited communicating with my brain. Please try again later."
        except Exception as e:
            print("ERROR:", type(e), e)
            print(self.messages)
            return "Sorry, I'm having trouble communicating with my brain. Please try again later."

        assistant_response = response.choices[0].message.content
        assistant_response = assistant_response.split('\n')
        print("ASSISTANT RESPONSE:", assistant_response)
        if importance is None:
            importance = 1.0
        conversation_history_condensed = None
        content_words = None
        memory_summary = ""
        summary_begins_at = 0
        if len(assistant_response) > 1:
            # Handles odd issues with extra newlines
            for line_num in range(1, min(7, len(assistant_response) + 1)):
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
                elif match := re.match('\s*[Cc][Hh]:\s*(.+)', assistant_response[-line_num]):
                    conversation_history_condensed = match.group(1)
                    summary_begins_at = line_num
        else:
            memory_summary = assistant_response[-1]

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

        if conversation_history_condensed:
            self.dialogue_history_condensed.append(conversation_history_condensed)

        return body

    def get_profile(self, user_id: str, display_name: str = '') -> ProfileMemory:
        if user_id not in self.profile_memories:
            self.profile_memories[user_id] = ProfileMemory(user_id, display_name=display_name)
        return self.profile_memories[user_id]

    def perform_action(self, action: str) -> Optional[str]:
        """Perform an action returned by the fast_api_model, valid actions are:
        - 'FETCH(user_id, key)': fetch a value from the user's memory
        - 'STORE(user_id, key, value)': store a value in the user's memory
        - 'DELETE(user_id, key)': delete a value from the user's memory
        """
        if match := re.search('FETCH\((.+?),\s*(.+)\)', action):
            user_id = match.group(1)
            key = match.group(2)
            profile = self.get_profile(user_id)
            value = profile.get_key_value(key)
            return f'{user_id}/{key}={value}'
        elif match := re.search('STORE\((.+?),\s*(.+),\s*(.+)\)', action):
            user_id = match.group(1)
            key = match.group(2)
            value = match.group(3)
            profile = self.get_profile(user_id)
            profile.set_key_value(key, value)
            return "OK"
        elif match := re.search('DELETE\((.+?),\s*(.+)\)', action):
            user_id = match.group(1)
            key = match.group(2)
            profile = self.get_profile(user_id)
            profile.delete_key(key)
            return "OK"
        else:
            return None

    def fast_analyse_prompt(self, name_of_user):
        """Send a request to fast_api_model to analyse a prompt and return a list of actions prior to sending to the
        slower model"""

        full_prompt = self.messages[:]
        full_prompt.append({
            'role': 'user',
            'content': "Please analyse the above prompt and history and return a list of actions that *will* be "
                       "carried out in the order provided, one per line, don't STORE Memory or Awareness lines,"
                       "these actions allow you to helpfully store and retrieve information about users that you as "
                       "an assistant would find useful in future, such as learning progress, what they like to watch "
                       "or music tastes. Name of User can be used as user_id."
                       "Actions are of the form 'FETCH(user_id, key)', 'STORE(user_id, key, value)', or 'DELETE("
                       "user_id, key)'. FETCH retrieves useful information relating to the prompt. Don't annotate the "
                       "commands."})
        try:
            response = openai.ChatCompletion.create(
                model=self.openai_fast_api_model,
                messages=full_prompt
            )
        except openai.error.RateLimitError as e:
            print("RATE LIMITED in fast_analyse_prompt:", type(e), e)
            return
        except Exception as e:
            print(f"ERROR in fast_analyse_prompt: {type(e)} {e}\n{e.__traceback__.tb_lineno}")
            print("PROMPT:", full_prompt)
            return

        actions = response.choices[0].message.content.split('\n')
        for action in actions:
            print("ACTION:", action)
            result = self.perform_action(action)
            if result == 'OK':
                print("OK")
            elif result:
                print("RESULT:", result)
                self.recent_memories.append({'role': 'assistant', 'content': result})
            else:
                print("NO RESULT in fast_analyse_prompt:", action)

        #print("FAST RESPONSE:", response.json())
        #return response.json()
        return

    def get_city_coordinates(self, city_name):
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={self.openweathermap_api_key}"
        response = requests.get(url)
        data = response.json()

        if data and "lat" in data[0] and "lon" in data[0]:
            return data[0]["lat"], data[0]["lon"]
        else:
            return None, None

    def get_weather(self, location: str):
        if re.match(r'-?\d+\.?\d*,-?\d+\.?\d*', location):
            lat, lon = location.split(",")
        else:
            city = "York,GB"
            lat, lon = self.get_city_coordinates(city)
        print(lat, lon)

        if lat and lon:
            url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={self.openweathermap_api_key}&units=metric"
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                sunrise = datetime.fromtimestamp(data["current"]["sunrise"]).strftime("%H:%M")
                sunset = datetime.fromtimestamp(data["current"]["sunset"]).strftime("%H:%M")
                uvi = data["current"]["uvi"]
                wind_speed = data["current"]["wind_speed"]
                wind_degrees = data["current"]["wind_deg"]
                temperature = data["current"]["temp"]
                feels_like = data["current"]["feels_like"]
                description = data["current"]["weather"][0]["description"]
                return f"Sunrise: {sunrise}, Sunset: {sunset}, Now: {temperature}C, feels like {feels_like}C, wind speed {wind_speed}, wind direction in degrees {wind_degrees}, uv index {uvi}, {description}."
            else:
                return "Error fetching weather data"
        else:
            return "City not found."

    def generate_memory_summary(self, user_input: str, assistant_response: str) -> str:
        # Implement a function to generate a memory summary from user_input and assistant_response
        return f'User: {user_input}.\nGPT: {assistant_response}'
