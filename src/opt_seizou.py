# %%
# 「線形計画問題」を解く
from pathlib import Path

import pandas as pd
import pulp

# %%
data_dir = Path("../data/seizou/")
# データをimport
stock_df = pd.read_csv(data_dir / "stocks.csv")
require_df = pd.read_csv(data_dir / "requires.csv")
gain_df = pd.read_csv(data_dir / "gains.csv")
# リストを定義
ps = gain_df["p"].to_list()
ms = stock_df["m"].to_list()
# df から辞書を定義
stocks = {row.m: row.stock for row in stock_df.itertuples()}
requires = {(row.p, row.m): row.require for row in require_df.itertuples()}
gains = {row.p: row.gain for row in gain_df.itertuples()}

problem: pulp.LpProblem = pulp.LpProblem(name="LP2", sense=pulp.LpMaximize)

# 変数をリストから一度に定義
# val は辞書だから、変数名をkey にして、LpVariable を一つずつループで定義してもok

# 今回はKg を求めるから、Cotinuous で良いが、個数の場合は「整数計画問題」となるから、cat="Integer" とする.
xs: dict = pulp.LpVariable.dict(name="x", indices=ps, cat="Continuous")
# 制約式
# 全ての製品の製造量は 0 以上
for p in ps:
    problem += xs[p] >= 0
# 生産は在庫の範囲内で行う
for m in ms:
    # 全ての辺数x に紐づく条件を在庫の制約に紐付ける
    problem += pulp.lpSum([xs[p] * requires[p, m] for p in ps]) <= stocks[m]  # 等式・不等式は、lpSum の外！
# 複数変数の合計を計算したいときは、lpSum を使うと良い.
problem += pulp.lpSum([gains[p] * xs[p] for p in ps])

problem.solve()

for p in ps:
    print(p, xs[p].value())

print("obj= ", problem.objective.value())


# %%
"""
NOTE:
一般に、線形計画問題よりも整数計画問題の方が難しい問題になる.
線形計画問題では、100万変数の問題を解くことができるが、整数計画問題の場合ほぼ解けない.
さらに、整数数計画問題では１万変数の規模でも難しい場合があり、構造上幸運なケースを除いて安定して解くことは期待できない.
そのため、線形計画問題で解を丸める（#連続緩和）は、実務において非常に有効な方法となる.
"""
