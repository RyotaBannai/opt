"""
solver をモジュール化したもの
"""
# %%
from typing import Dict

import pandas as pd
import pulp


class CarGroupPrblem:
    def __init__(
        self, car_df: pd.DataFrame, student_df: pd.DataFrame, name="ClubCarProblem"
    ) -> None:
        self.car_df = car_df
        self.student_df = student_df
        self.name = name
        self.prob = self._formulate()

    def _formulate(self) -> Dict:
        prob = pulp.LpProblem(self.name, sense=pulp.LpMinimize)
        S = self.student_df["student_id"].tolist()
        C = self.car_df["car_id"].to_list()
        G = [1, 2, 3, 4]
        SC = [(s, c) for c in C for s in S]
        S_license = self.student_df[self.student_df["license"] == 1]["student_id"].to_list()
        S_g = {g: self.student_df[self.student_df["grade"] == g]["student_id"] for g in G}
        S_male = self.student_df[self.student_df["gender"] == 0]["student_id"]
        S_female = self.student_df[self.student_df["gender"] == 1]["student_id"]
        # 定数
        U = self.car_df["capacity"].to_list()
        # 辺数
        x = pulp.LpVariable.dicts("x", SC, cat="Binary")
        # 制約
        for s in S:
            prob += pulp.lpSum([x[s, c] for c in C]) == 1
        for c in C:
            prob += pulp.lpSum([x[s, c] for s in S]) <= U[c]
        for c in C:
            prob += pulp.lpSum([x[s, c] for s in S_license]) >= 1
        for c in C:
            for g in G:
                prob += pulp.lpSum([x[s, c] for s in S_g[g]]) >= 1
        for c in C:
            prob += pulp.lpSum([x[s, c] for s in S_male]) >= 1
            prob += pulp.lpSum([x[s, c] for s in S_female]) >= 1
        return {"prob": prob, "variable": {"x": x, "list": {"S": S, "C": C}}}

    def solve(self) -> pd.DataFrame:
        self.prob["prob"].solve()
        x = self.prob["variable"]["x"]
        S = self.prob["variable"]["list"]["S"]
        C = self.prob["variable"]["list"]["C"]
        car2students = {c: [s for s in S if x[s, c].value() == 1] for c in C}
        student2car = {s: c for c, members in car2students.items() for s in members}
        solution_df = pd.DataFrame(list(student2car.items()), columns=["student_id", "car_id"])
        return solution_df


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path("../../data/api_prob/")
    car_df = pd.read_csv(data_dir / "cars.csv")
    student_df = pd.read_csv(data_dir / "students.csv")  # gender 0:male 1:female grade 学年1~4

    prob = CarGroupPrblem(
        car_df=car_df,
        student_df=student_df,
    )

    solution_df = prob.solve()
    print(solution_df)
