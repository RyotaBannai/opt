"""
概要：
割引クーポン付与による来客増加数の最大化における、model1, model2 をもとに、
「公平性」を考慮したモデリングを行う

目的：
各Dm の各セグメントへの送付率の下限値を最大化する
= 各Dm の送付について、各セグメント間の不公平性を減らしたい(理由は以下のリストを参考)

- model1,model2 の送付率の結果から、クーポンを配布しなくてもすでに来店してくれる会員には配らない傾向がある
- このキャンペーンの効果として、短期的な来店増加数だけでなく、「顧客生涯価値」のような長期的な指標も考慮すると、
  あるセグメントに対して多く投資して、あるセグメントに対しては全く投資しない、というのはリスクがあるかもしれない

- 上記の問題点を考えると、
  - 各セグメントへの各Dm の送付率の下限値を最大化（下限値を設定するのではない<-定数）
  - 各Dm を求めた送付率の下限値以上送付する（目的変数、決定変数が一つ目になる。これを制約としても使うと良い.）

結果：
- 昨年の来店数に関係なく配布できることがわかる

"""
# %%
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import pulp
import seaborn as sns

# %%
data_dir = Path("../data/campaign/")
customer_df = pd.read_csv(data_dir / "customers.csv")
# prob_dm1,prob_dm2,prob_dm3 はダイレクトメールの３パターンを送付したときに来店する確率を表す
prob_df = pd.read_csv(data_dir / "visit_probability.csv")
# %%
# 年齢区分と昨年度の来店回数区分の組み合わせの人数について確認.
# どの年齢区分の会員が昨年どれくらい来店したかを見る
# pivot_table の index は row
customer_pivot_df = pd.pivot_table(
    data=customer_df, values="customer_id", columns="freq_cat", index="age_cat", aggfunc="count"
)
# それぞれのrow の順番を調整
customer_pivot_df = customer_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])

# %%
problem = pulp.LpProblem(name="DiscountCouponProblem3", sense=pulp.LpMaximize)
segs = prob_df["segment_id"].to_list()
dms = {"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
sm = [(s, dm) for dm in dms.values() for s in segs]

# 各パターンのDm をどの程度の割合で送付するかを [0,1] (0から1 の連続変数)で決定
xsm: Dict = pulp.LpVariable.dicts(name="xsm", indices=sm, lowBound=0, upBound=1, cat="Continuous")
# 各会員にいずれかのDm を送付する== 各セグメントに送付するDm パターンの送付率の和が100%  と置き換えることができる.
for s in segs:
    problem += pulp.lpSum(xsm[s, dm] for dm in dms.values()) == 1.0

# 各セグメントへのそれぞれのパターンのDm の送付率の下限値
y: pulp.LpVariable = pulp.LpVariable(name="y", lowBound=0, upBound=1, cat="Continuous")
# 各セグメントへのそれぞれのパターンのDm の送付率の下限値を最大化
problem += y

# 予算を100万円以下に設定
customer_prob_df_row_origin = pd.merge(customer_df, prob_df, on=["age_cat", "freq_cat"])
seg_nums = customer_prob_df_row_origin.groupby(["segment_id"])["customer_id"].count().to_dict()
sd_prob_df = prob_df.rename(columns=dms).melt(
    id_vars=["segment_id"],
    value_vars=dms.values(),
    var_name="dm",
    value_name="prob",
    col_level=None,
)
psm = sd_prob_df.set_index(["segment_id", "dm"])["prob"].to_dict()
# 予算消費期待値 CmPm Cm はdm パターンに対するクーポン額.
# Pm はセグメントの来店率
# 来店率を来店数に変換するために、来店率にセグメントの会員数をかける.
cm = {1: 0, 2: 1000, 3: 2000}
problem += (
    pulp.lpSum(cm[dm] * seg_nums[s] * psm[s, dm] * xsm[s, dm] for dm in [2, 3] for s in segs)
    <= 1000000
)

# 各パターンのDm を設定した送付率の下限値以上送付
# これまで10% と定数として設定していた部分が、決定変数y となる. mode2 の 85行目あたり.
for s in segs:
    for dm in dms.values():
        problem += xsm[s, dm] >= y

# %%
time_start = time.time()
status = problem.solve()
time_stop = time.time()
print(f"status: {pulp.LpStatus[status]}")
print(f"objective fns: {pulp.value(problem.objective):.4}")  # 0.1313 送付率の下限値
print(f"time to run: {(time_stop - time_start):.3}s")

# %%
# 各セグメントへの送付率を確認
send_dm_df = pd.DataFrame(
    [[xsm[s, dm].value() for dm in dms.values()] for s in segs], columns=dms.keys()
)
seg_send_df = pd.concat([prob_df[["segment_id", "age_cat", "freq_cat"]], send_dm_df], axis=1)
# デフォルトでセグメント（昨年の頻度、カテゴリ）に送付した割合（%）
ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(nrows=1, ncols=3, figsize=(20, 3))
for i, ptn in enumerate(dms.keys()):
    cust_send_pivot_df = pd.pivot_table(
        data=seg_send_df, values=ptn, columns="freq_cat", index="age_cat", aggfunc="mean"
    )
    cust_send_pivot_df = cust_send_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
    cust_send_pivot_df = sns.heatmap(
        data=cust_send_pivot_df, annot=True, fmt=".1%", cmap="Blues", vmin=0, vmax=1, ax=ax[i]
    )
    ax[i].set_title("{}_rate".format(ptn))
plt.show()

# %%
