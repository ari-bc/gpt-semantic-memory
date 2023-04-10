import threading
import time

import requests


class GoogleNews:

    def __init__(self, api_key: str, query: str=None, page_size: int=10, language: str='en', news_update_interval: int=60):
        self.api_key = api_key
        self.query = query
        self.language = language
        self.page_size = page_size
        self.current_news = None
        self.news_update_interval = news_update_interval * 60

    def get_news(self):
        return self.current_news

    def update_news(self):
        while True:
            self.current_news = self.fetch_news()
            time.sleep(self.news_update_interval)

    def start_news_updater(self):
        update_thread = threading.Thread(target=self.update_news, daemon=True)
        update_thread.start()

    def fetch_news(self):
        base_url = "https://newsapi.org/v2/everything"
        params = {
            'q': self.query,
            'language': self.language,
            'pageSize': self.page_size,
            'apiKey': self.api_key
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data['articles']
            return articles
        else:
            print(f"Error fetching news: {response.status_code}")
            return None
