"""
セグメント送付モデル
- セグメントとDm をキーにした決定変数を定義してモデリングする
- この決定変数は一つのセグメントに対するDm の送付率を表し、一つのセグメントのDm の全パターンの送付率は1.0 になる.

目的：
割引クーポン付与による来客増加数の最大化

要件；
- 各会員に対してどのパターンのダイレクトメールを送付するか決定
  - ダイレクトメールのパターン
    - キャンペーンのチラシ
    - キャンペーンのチラシ+10%クーポン券
    - キャンペーンのチラシ+20%クーポン券
- 各会員に対して送付するダイレクトメールは上記のパターンから一パターンのみ、一回だけ送付
- クーポン付与による来客増加数を最大化する（来客数増加数:=実際の来客数ではなく、ダイレクトメールを送付した効果による増加）
- 会員の予算消費期待値の合計は 100万円以下 （配ったクーポンの合計ではなく、実際に使われるであろう期待値）
- 各パターンのダイレクトメールをそれぞれのセグメントに属する会員の10% 以上に送付
  - セグメント数は 年齢区分「19歳以下、20~34歳、35~49歳、50歳以上」の4カテゴリと昨年度利用回数区分「0,1,2,3回以上」の4カテゴリで合計16

tags: #決定変数 #二値辺数 #離散変数 #縦持ち #横持ち
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
problem = pulp.LpProblem(name="DiscountCouponProblem2", sense=pulp.LpMaximize)
segs = prob_df["segment_id"].to_list()
dms = {"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
sm = [(s, dm) for dm in dms.values() for s in segs]
# 各パターンのDm をどの程度の割合で送付するかを [0,1] (0から1 の連続変数)で決定
xsm: Dict = pulp.LpVariable.dicts(name="xsm", indices=sm, lowBound=0, upBound=1, cat="Continuous")
# 各会員にいずれかのDm を送付する== 各セグメントに送付するDm パターンの送付率の和が100%  と置き換えることができる.
for s in segs:
    problem += pulp.lpSum(xsm[s, dm] for dm in dms.values()) == 1.0

# 各セグメントにDm を一定割合送るときに、来客増加数の期待値は、
# セグメントの来客率*セグメントに所属する会員数*連続変数　で表せる
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
problem += pulp.lpSum(
    seg_nums[s] * (psm[s, dm] - psm[s, 1]) * xsm[s, dm] for dm in [2, 3] for s in segs
)

# 予算を100万円以下に設定 予算消費期待値 CmPm Cm はdm パターンに対するクーポン額.
# Pm はセグメントの来店率
# 来店率を来店数に変換するために、来店率にセグメントの会員数をかける.
cm = {1: 0, 2: 1000, 3: 2000}
problem += (
    pulp.lpSum(cm[dm] * seg_nums[s] * psm[s, dm] * xsm[s, dm] for dm in [2, 3] for s in segs)
    <= 1000000
)

# 各セグメントに送るDm の割合は 10% 以上
for s in segs:
    for dm in dms.values():
        problem += xsm[s, dm] >= 0.1

# %%
time_start = time.time()
status = problem.solve()
time_stop = time.time()
print(f"status: {pulp.LpStatus[status]}")
print(f"objective fns: {pulp.value(problem.objective):.4}")  # 326.1 来客数の増加
print(f"time to run: {(time_stop - time_start):.3}s")

# %%
# 各セグメントにDm パターンを送った時のそれぞれの割合（送付率）を見る（合計1.0 になっているかなど）
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
# 送付数を計算 決定変数*セグメントの会員数
seg_send_df["num_cust"] = seg_send_df["segment_id"].apply(lambda x: seg_nums[x])
for dm in dms.values():
    # 列を取り出して、行同士の積を計算できる
    seg_send_df[f"send_num_dm{dm}"] = seg_send_df[f"prob_dm{dm}"] * seg_send_df["num_cust"]

ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(nrows=1, ncols=3, figsize=(20, 3))
for i, ptn in enumerate([f"send_num_dm{dm}" for dm in dms.values()]):
    cust_send_pivot_df = pd.pivot_table(
        data=seg_send_df, values=ptn, columns="freq_cat", index="age_cat", aggfunc="sum"
    )
    cust_send_pivot_df = cust_send_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
    cust_send_pivot_df = sns.heatmap(
        data=cust_send_pivot_df, annot=True, fmt=".1f", cmap="Blues", vmax=800, ax=ax[i]
    )
    ax[i].set_title("{}_rate".format(ptn))
plt.show()

# %%
