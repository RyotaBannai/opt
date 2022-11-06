"""
- 学年で318 人、8 つのクラス.
- 男子 158 人、女子160人いる.
- 学力試験 500点満点で、平均点は303.6点
- 学年にリーダー気質な生徒は17人いる
- 特別な支援が必要な学生が４人いる(遅刻が多い、欠席が多い、保護者からの問い合わせが多い、など)
- 特定ペアが３人いる（同姓同名、双子、など）

条件
- 学年の全生徒をそれぞれ１つのクラスに割り当てる
- クラスの人数を均等に分けたいため、各クラスの人数は 39人以上40人以下とする
- 男女均等に分けたいため、各クラスの男女ともに人数は 20人以下とする
- 学力を均等にしたいため、各クラスの学力試験の平均点は、学力平均点+-10点とする
- 各クラスにリーダー気質な生徒を２人以上割り当てる
- 各クラスに特別な支援が必要な生徒は１人以下とする
- 特定ペアの生徒同士は同一のクラスに割り当てない

Tips:
- 操作する前にimport したデータの確認をとる.
  - 点数の平均 describe(), hist()
  - フラグ
  - 想定したデータ数あるかどうか、など
"""

# %%

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import pulp

# %%
data_dir = Path("../data/school/")
# データをimport
# 女子:0, 男子:1
student_df = pd.read_csv(data_dir / "students.csv")
pair_df = pd.read_csv(data_dir / "student_pairs.csv")

# %%

# senseには目的関数を最小化or最大化するかを指定する. 今回はm問題に目的関数がない場合はどちらでも良い.
cs = ["A", "B", "C", "D", "E", "F", "G", "H"]
ss = student_df["student_id"].tolist()
sc = [(s, c) for s in ss for c in cs]  # クラスと生徒の全組み合わせ
problem: pulp.LpProblem = pulp.LpProblem(name="ClassAssignmentProblem", sense=pulp.LpMaximize)

# ΣXs,c=1 生徒がどのクラスか１つにだけ所属する条件
xs: Dict = pulp.LpVariable.dicts(name="x", indices=sc, cat="Binary")  # dict と dicts を区別.
for s in ss:
    # 生徒一人に対して見たいから、外側のループにおいて、内側でクラスをループすると良い.
    problem += pulp.lpSum([xs[s, c] for c in cs]) == 1
# 各クラスの生徒の人数は39人以上40人以下
for c in cs:
    problem += pulp.lpSum([[xs[s, c] for s in ss]]) >= 39
    problem += pulp.lpSum([[xs[s, c] for s in ss]]) <= 40

# 男女のリストを作る
female = [row.student_id for row in student_df.itertuples() if row.gender == 0]
male = [row.student_id for row in student_df.itertuples() if row.gender == 1]
for c in cs:
    problem += pulp.lpSum([[xs[s, c] for s in female]]) <= 20  # 女子
    problem += pulp.lpSum([[xs[s, c] for s in male]]) <= 20  # 男子

# 学力テストのスコア
scores = {row.student_id: row.score for row in student_df.itertuples()}
score_mean = student_df["score"].mean()
for c in cs:
    # 非線形の制約を線形の制約に変換可能な場合は、迷わず線形に書き換える.
    problem += (score_mean - 10) * pulp.lpSum([xs[s, c] for s in ss]) <= pulp.lpSum(
        [xs[s, c] * scores[s] for s in ss]
    )
    problem += (score_mean + 10) * pulp.lpSum([xs[s, c] for s in ss]) >= pulp.lpSum(
        [xs[s, c] * scores[s] for s in ss]
    )

# このままでは、学力平均は均等になるが、クラスの中の学生それぞれの学力の分布には隔たりができてしまう。そのため、初めに初めに学力分布が均等になるようにクラスを編成し、そのクラス編成をもとに、それぞれの制約を満たすような組み分けを考える.
student_df["score_rank"] = student_df["score"].rank(ascending=False, method="first")
# 単純に学力テストのランキングの１位から順にクラスA~H に割り当てていく.
class_dict = {i: x for (i, x) in enumerate(cs)}
student_df["init_assigned_class"] = student_df["score_rank"].map(
    lambda rank_num: class_dict[(rank_num - 1) % 8]
)
init_flag = {(s, c): 0 for (s, c) in sc}
for row in student_df.itertuples():
    init_flag[(row.student_id, row.init_assigned_class)] = 1

# debug クラスに均等に生徒がアサインされているかどうか確認
# for c in cs:
#   print(len(student_df[student_df['init_assigned_class'] == c]))
# 初期の学力分布が考慮されたクラス編成の生徒のアサインと一致しているほど良いとする目的関数をセット
problem += pulp.lpSum([xs[s, c] * init_flag[s, c] for s, c in sc])

# リーダーは各クラス２人以上
leaders = {row.student_id: row.leader_flag for row in student_df.itertuples()}
for c in cs:
    problem += pulp.lpSum([xs[s, c] * leaders[s] for s in ss]) >= 2

# 特別な支援が必要な学生は各クラス１人以下
supportees = {row.student_id: row.support_flag for row in student_df.itertuples()}
for c in cs:
    problem += pulp.lpSum([xs[s, c] * supportees[s] for s in ss]) <= 1

# 特定ペアの生徒同士は同一のクラスに割り当てない
for c in cs:
    for row in pair_df.itertuples():
        problem += pulp.lpSum([xs[s, c] for s in [row.student_id1, row.student_id2]]) <= 1


# 生徒番号１をクラスA に必ず割り当てたい.
must_students = {(s, c): 0 for (s, c) in sc}
must_students[1, "A"] = 1
for c in cs:
    problem += pulp.lpSum([xs[(s, c)] for s in ss if must_students[s, c] == 1]) == len(
        [1 for s in ss if must_students[s, c] == 1]
    )


status = problem.solve()
print(status)  # 解が存在した場合は1(Solved)、そうでない場合は0(Not Solved)、そもそも問題にかいが存在しない場合には-1(Infeasible)
print(pulp.LpStatus[status])

c2s = {}
for c in cs:
    c2s[c] = [s for s in ss if xs[s, c].value() == 1]

# debug
# for c, s in c2s.items():
#     print("Class", c)
#     print("Num:", len(s))
#     print("Student", s)
#     print()

s2c = {s: c for s in ss for c in cs if xs[s, c].value() == 1}
result_df = student_df.copy()
result_df["assigned_class"] = result_df["student_id"].map(s2c)

# print(result_df.head())
# print(result_df.groupby("assigned_class")["student_id"].count())
# print(result_df.groupby(["assigned_class", "gender"])["student_id"].count()) # [('A', 0)]

# check pair
# for row in pair_df.itertuples():
#     print(row.student_id1, s2c[row.student_id1])
#     print(row.student_id2, s2c[row.student_id2])

# 初期のクラス編成で、学力テストに偏りがないことをチェック
init_class_df = student_df.copy()
fig = plt.figure(figsize=(12, 20))
for i, c in enumerate(cs):
    cls_df = init_class_df[init_class_df["init_assigned_class"] == c]
    ax = fig.add_subplot(
        4,
        2,
        i + 1,
        xlabel="score",
        ylabel="num",
        xlim=(0, 500),
        ylim=(0, 20),
        title="Class :{:s}".format(c),
    )
    ax.hist(cls_df["score"], bins=range(0, 500, 40))

# クラスごとの学力テストの結果の分布をグラフ化
fig = plt.figure(figsize=(12, 20))
for i, c in enumerate(cs):
    cls_df = result_df[result_df["assigned_class"] == c]
    ax = fig.add_subplot(
        4,
        2,
        i + 1,
        xlabel="score",
        ylabel="num",
        xlim=(0, 500),
        ylim=(0, 20),
        title="Class :{:s}".format(c),
    )
    ax.hist(cls_df["score"], bins=range(0, 500, 40))  # 都度描画

# %%
