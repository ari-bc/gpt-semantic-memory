import re
import time

import openai
import threading
import python_weather
import requests

from memory_database import MemoryDatabase
from datetime import datetime, timedelta

ASSISTANT_INSTRUCTION = 'You are a %ASSISTANT_TYPE% assistant. You use the user\'s name a lot if you know it. You only apologise for things that are your fault. You use emojis frequently. You have listed your related memories and awarenesses for reference only, do not use them as a template for output'
SUMMARISE_INSTRUCTION = 'At the end of each of your responses, please add a line which summarises the user input and assistant response. Add another line with how important this information was from 0.0-10.0, a list of 1-6 content words that summarise both your response and the user input.'


class GPTCommunication:
    def __init__(self, api_key: str, db_file: str, api_model: str='gpt-3.5-turbo', assistant_type: str='friendly', openweathermap_api_key: str=None, weather_update_interval: int=10):
        openai.api_key = api_key
        self.api_model = api_model
        self.assistant_instruction = ASSISTANT_INSTRUCTION.replace('%ASSISTANT_TYPE%', assistant_type)
        self.memory_db = MemoryDatabase(db_file)
        self.messages = []
        self.clear_messages()
        self.recent_memories = []
        self.current_weather = "Unknown"

        self.weather_update_interval = weather_update_interval * 60  # 10 minutes
        self.openweathermap_api_key = openweathermap_api_key
        self.start_weather_updater()
    def update_weather(self):
        while True:
            self.current_weather = self.get_weather()
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
        # Give the agent the current time
        self.add_message('assistant', f'Awareness: {datetime.now().isoformat()}')
        # Give the agent the current weather, updates every 10 minutes
        self.add_message('assistant', f'Awareness: Weather, {self.current_weather}')
        self.add_message('assistant', f'Awareness: Location=York, UK')
        for memory in relevant_memories:
            memory_details = memory['related_prompt']
            self.add_recent_memory("assistant", f'Memory: {memory["timestamp"]}: {memory_details}')

        # A mechanism for having memories hang around for a few responses, allows discussion
        self.expire_recent_memories(15)
        for memory in self.recent_memories:
            self.add_message(memory['role'], memory['content'])
            print("M:", memory)

        for entry in dialogue_history:
            self.add_message(entry['speaker'], entry['content'])
            print("D:", entry)

        self.add_message('user', 'I need you to output your responses in the following format and in this order (r,s,i,c): r:<actual response>\nsummary: <a brief summary of the actual response, including the speaker and the context>\ni: <how useful this information will be for future reference purposes from 0.0-10.0, rate uncommon items higher>\nc: <a list of 1-6 content words that summarise both your response and the user input in context>')
        self.add_message('assistant', f'r: Of course! Now let\'s continue.\n{name_of_user} asked for a specific format for responses and I agreed.\ni: 10.0\nc: format, response, importance')

        timestamp = datetime.now().isoformat()
        self.memory_db.save_dialogue_entry('user', user_input, timestamp)

        self.add_message("user", user_input)

        try:
            response = openai.ChatCompletion.create(
                model=self.api_model,
                messages=self.messages
            )
        except openai.error.RateLimitError as e:
            print("ERROR:", type(e), e)
            return "Sorry, I'm being rate limited communicating with my brain. Please try again later."
        except Exception as e:
            print("ERROR:", type(e), e)
            return "Sorry, I'm having trouble communicating with my brain. Please try again later."


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

    def get_city_coordinates(self, city_name):
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={self.openweathermap_api_key}"
        response = requests.get(url)
        data = response.json()

        if data and "lat" in data[0] and "lon" in data[0]:
            return data[0]["lat"], data[0]["lon"]
        else:
            return None, None

    def get_weather(self):
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
