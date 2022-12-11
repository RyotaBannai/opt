"""
生徒をどの車に割り当てるか決定する. 変数が {0,1}(学生と車をキーにした変数を用意)
0-1整数計画問題

目的関数はなし

こちらは最適化モデリングの試験ファイル.
モジュール化したものは ./problem.py
"""
# %%
from pathlib import Path

import pandas as pd
import pulp

# %%
data_dir = Path("../../data/api_prob/")
car_df = pd.read_csv(data_dir / "cars.csv")
student_df = pd.read_csv(data_dir / "students.csv")  # gender 0:male 1:female grade 学年1~4

# %%
prob = pulp.LpProblem("ClubCarProblem", sense=pulp.LpMinimize)

S = student_df["student_id"].tolist()
C = car_df["car_id"].to_list()
G = [1, 2, 3, 4]  # 学年のリスト
SC = [(s, c) for c in C for s in S]  # 学生と車をペアのリスト
# 免許を持っている学生
S_license = student_df[student_df["license"] == 1]["student_id"].to_list()
# 学年ごとに整理
S_g = {g: student_df[student_df["grade"] == g]["student_id"] for g in G}
# 性別ごとに整理
S_male = student_df[student_df["gender"] == 0]["student_id"]
S_female = student_df[student_df["gender"] == 1]["student_id"]

# 定数
# 車の定員数
U = car_df["capacity"].to_list()

# 辺数
# どの生徒をどの車に割り当てるか
x = pulp.LpVariable.dicts("x", SC, cat="Binary")

# 制約
# 1. 各学生を１つの車に割り当てる
for s in S:
    prob += pulp.lpSum([x[s, c] for c in C]) == 1
# 2. 法規制に関する制約. 各車には乗車定員より多く割り当てない
for c in C:
    prob += pulp.lpSum([x[s, c] for s in S]) <= U[c]

# 3. 法規制に関する制約. 各車に運転免許証を持っている学生を最低１人割り当てる
for c in C:
    prob += pulp.lpSum([x[s, c] for s in S_license]) >= 1

# 4. 懇親を目的とした制約. 懇親：各車に各学年の生徒を最低１人割り当てる
for c in C:
    # ひとつの車両に対して、
    for g in G:
        # 各学年を一つずつ回して、それぞれの学年g の学生の総和が１以上になるよう組
        prob += pulp.lpSum([x[s, c] for s in S_g[g]]) >= 1


# 5. 懇親を目的とした制約. 懇親：ジェンダーバランスを考慮して、male,female 最低１人は割り当てる
for c in C:
    prob += pulp.lpSum([x[s, c] for s in S_male]) >= 1
    prob += pulp.lpSum([x[s, c] for s in S_female]) >= 1

status = prob.solve()
print(f"Status: {pulp.LpStatus[status]}")

# %%
"""
学生の車への割り当てが完了したら、データを整理
"""
# キーにしたいリストは for loop は同階層に持ってくる.
car2students = {c: [s for s in S if x[s, c].value() == 1] for c in C}
max_people = dict(zip(car_df["car_id"], car_df["capacity"]))
for c, members in car2students.items():
    print(f"車ID:{c}")
    print(f"定員:{max_people[c]}, 割り当て学生数:{len(members)}")
    print(f"学生リスト:{members}")

# %%
