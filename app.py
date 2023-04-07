import urllib.parse

from flask import Flask, render_template, request, jsonify
from gpt_communication import GPTCommunication
import configparser
import os

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
config.read(config_path)

app = Flask(__name__)

api_key = config['default']['openai_api_key']
api_model = config['default']['openai_api_model']
name_of_user = config['default']['name_of_user']
name_of_agent = config['default']['name_of_agent']
assistant_type = config['default']['assistant_type']
db_file = 'memories.db'
gpt_comm = GPTCommunication(api_key, db_file, api_model=api_model, assistant_type=assistant_type)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    # Get URI encoded user input
    user_input = request.form['user_input']
    # Decode URI encoded user input using urllib.parse.unquote
    decoded_user_input = urllib.parse.unquote(user_input)
    assistant_response = gpt_comm.send_message(decoded_user_input, name_of_user=name_of_user, name_of_agent=name_of_agent, user_pronouns='she/her', num_memories=3)
    return jsonify({"response": assistant_response})

if __name__ == '__main__':
    app.run(debug=True)

