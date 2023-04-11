import configparser
import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from pydispatch import dispatcher

from gpt_communication import GPTCommunication

MESSAGE_RECEIVED_SIGNAL = 'slack.message.received'

class SlackBot:
    def __init__(self, config: configparser.ConfigParser = None, db_file: str = 'slack_memories.db'):
        self.agent_name = config['default']['name_of_agent']
        self.app_token = config['slack']['app_token']
        self.bot_token = config['slack']['bot_token']

        self.client = WebClient(token=self.bot_token)
        self.app = App(token=self.bot_token)
        self.handler = SocketModeHandler(self.app, self.app_token)

        self.gpt_communication = GPTCommunication(db_file, config=config)
        dispatcher.connect(self.process_slack_message, signal=MESSAGE_RECEIVED_SIGNAL, sender=dispatcher.Any)

        @self.app.event("app_mention")
        async def command_handler(body, say):
            text = body['event']['text']
            user_id = body['event']['user']
            user_info = self.client.users_info(user=user_id)
            user_name = user_info['user']['real_name']

            dispatcher.send(MESSAGE_RECEIVED_SIGNAL, message=text, user_name=user_name, reply_func=say)

    def process_slack_message(self, message: str, user_name: str, reply_func):
        gpt_response = self.gpt_communication.send_message(message, name_of_user=user_name)
        reply_func(gpt_response)

    def run(self):
        self.handler.start()

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config.ini')
    config.read(config_path)
    slack_bot = SlackBot(config=config)
    slack_bot.run()
