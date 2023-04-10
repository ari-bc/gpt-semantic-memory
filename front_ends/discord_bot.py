import configparser
import os

import discord
from discord.ext import commands

from pydispatch import dispatcher

from gpt_communication import GPTCommunication

MESSAGE_RECEIVED_SIGNAL = 'discord.message.received'

# This example requires the 'message_content' intent.

intents = discord.Intents.default()
intents.message_content = True
intents.typing = False
intents.presences = False


class DiscordBot:
    def __init__(self, config: configparser.ConfigParser = None, db_file: str = 'discord_memories.db'):
        #application_id = config['discord']['application_id']
        #public_key = config['discord']['public_key']
        #permissions_integer = int(config['discord']['permissions_integer'])
        self.agent_name = config['default']['name_of_agent']
        self.token = config['discord']['token']

        self.bot = commands.Bot(intents=intents, command_prefix='!')
        # Set the bot's name to the agent's name
        self.bot.username = self.agent_name

        self.gpt_communication = GPTCommunication(db_file, config=config)
        dispatcher.connect(self.process_discord_message, signal=MESSAGE_RECEIVED_SIGNAL, sender=dispatcher.Any)

        @self.bot.event
        async def on_ready():
            print(f'{self.bot.user.name} has logged in to Discord.')

        @self.bot.event
        async def on_message(message: discord.Message):
            print(1, message)
            if message.author == self.bot.user:
                return

            dispatcher.send(MESSAGE_RECEIVED_SIGNAL, message=message)

    def send_message(self, channel_id: int, message: str):
        print(channel_id, message)
        channel = self.bot.get_channel(channel_id)
        self.bot.loop.create_task(channel.send(message))

    def send_dm(self, channel: discord.DMChannel, message: str):
        print(channel, message)

        async def send_message_async():
            try:
                await channel.send(message)
                print("DM sent successfully.")
            except discord.Forbidden:
                print("I do not have permission to send a DM to this user.")
            except Exception as e:
                print("An error occurred while sending a DM to this user.")
                print(type(e), e)

        self.bot.loop.create_task(send_message_async())

    def process_discord_message(self, message: discord.Message):
        gpt_response = self.gpt_communication.send_message(message.content, name_of_user=message.author.display_name)
        print(message.author.display_name, message.channel, message.content, gpt_response)
        # If it's a DM, send it to the DM channel
        if isinstance(message.channel, discord.DMChannel):
            self.send_dm(message.channel, gpt_response)
        else:
            self.send_message(message.channel.id, gpt_response)

    def run(self):
        print(self.token)
        self.bot.run(self.token)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config.ini')
    config.read(config_path)
    discord_bot = DiscordBot(config=config)
    discord_bot.run()
