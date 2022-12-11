"""
実務で想定されるAPI

起動
> FLASK_ENV=development FLASK_APP=api:app flask run
ターミナルからリクエスト
> cd .../the-py-optbook/src/api
> curl -X POST \
  -F students=@./data/api_prob/students.csv \
  -F cars=@./data/api_prob/cars.csv \
  -o ./src/api/solution.csv \
  http://127.0.0.1:5000/api
"""
from typing import Tuple

import pandas as pd
from flask import Flask, Request, make_response, request
from problem import CarGroupPrblem

app = Flask(__name__)


@app.route("/api", methods=["POST"])
def solve():
    student_df, car_df = preprocess(request)
    solution_df = CarGroupPrblem(student_df=student_df, car_df=car_df).solve()
    res = postprocess(solution_df)
    return res


def preprocess(request: Request) -> Tuple[pd.DataFrame, pd.DataFrame]:
    students = request.files["students"]
    cars = request.files["cars"]
    student_df = pd.read_csv(students)
    car_df = pd.read_csv(cars)
    return student_df, car_df


def postprocess(solution_df: pd.DataFrame):
    solution_csv = solution_df.to_csv(index=False)
    response = make_response()
    response.data = solution_csv
    response.headers["Content-Type"] = "text/csv"
    return response
