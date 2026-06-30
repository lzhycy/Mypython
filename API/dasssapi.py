import requests
import json
import os

class DasssAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.dasss.com/v1"

    def get_headers(self):
        return {
            "Authorization": f"Bearer
    {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_data(self, endpoint):

        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, headers=self.get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

            