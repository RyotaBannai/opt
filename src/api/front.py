"""
API へリクエストを投げる
> python front.py
"""
import requests

url = "http://127.0.0.1:5000/api"

files = {
    "students": open("../../data/api_prob/students.csv", "r"),
    "cars": open("../../data/api_prob/cars.csv", "r"),
}
reponse = requests.post(url=url, files=files)

with open("./solution.csv", "w") as f:
    f.write(reponse.text)
