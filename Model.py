import requests


def model(content):
    url = "http://208.78.220.192:4750/model"
    data = {
        "content": content
    }
    response = requests.post(url, data=data)

    return response.text
