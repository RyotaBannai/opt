"""
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
# 年齢区分と昨年度来店回数区分の組み合わせの人数について確認.
# どの年齢区分のお客さんが昨年どれくらい来店してるかをみる
# pivot_table の index は row
customer_pivot_df = pd.pivot_table(
    data=customer_df, values="customer_id", columns="freq_cat", index="age_cat", aggfunc="count"
)
# それぞれのrow の順番を調整
customer_pivot_df = customer_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
# customer_pivot_df
sns.heatmap(data=customer_pivot_df, annot=True, fmt="d", cmap="Blues")

# ダイレクトメールのパターンにおける、年齢と来店数における来店確率を見る
ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(nrows=1, ncols=3, figsize=(20, 3))
for i, ptn in enumerate(["prob_dm1", "prob_dm2", "prob_dm3"]):
    plob_pivot_df = pd.pivot_table(data=prob_df, values=ptn, columns="freq_cat", index="age_cat")
    plob_pivot_df = plob_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
    plob_pivot_df = sns.heatmap(data=plob_pivot_df, annot=True, fmt=".0%", cmap="Blues", ax=ax[i])
    ax[i].set_title("Visit Probability of {}".format(ptn))
plt.show()

# %%
# 会員個別送付モデル：各会員にダイレクトメールを送るか送らないかを決定するモデル
problem = pulp.LpProblem(name="DiscountCouponProblem1", sense=pulp.LpMaximize)
customer_ids = customer_df["customer_id"].to_list()
dms = {"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
im = [(i, dm) for dm in dms.values() for i in customer_ids]
xim: Dict = pulp.LpVariable.dicts(name="xim", indices=im, cat="Binary")

# 送付する各クーポンのパターンはいずれか１つのみ
for i in customer_ids:
    problem += pulp.lpSum([xim[i, dm] for dm in dms.values()]) == 1

# それぞれのクーポンを配布した時の来客増加数の最大化をモデリング
# クーポン配布した時としなかった時の変化は、あらかじめ与えられているあるクーポンを配布した時の各セグメントの来店率の差から求める.
# 上の来店率にカスタマー,DM を決定変数とした（Binary）をかけて最大化モデリングを解くと良い
# まずはカスタマー,DM をキーとして、来店率を簡単に求めるためのデータ構造を用意する.(pim)
keys = ["age_cat", "freq_cat"]
customer_prob_df_row_origin = pd.merge(customer_df, prob_df, on=keys)
# melt, pivot の話 https://www.salesanalytics.co.jp/datascience/datascience021/
# 来店率の横持ちの状態を縦持ちにする
customer_prob_df_column_origin = customer_prob_df_row_origin.rename(
    columns={"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
).melt(
    id_vars=["customer_id"],
    value_vars=[1, 2, 3],
    var_name="dm",
    value_name="prob",
    col_level=None,
)
pim = customer_prob_df_column_origin.set_index(["customer_id", "dm"])["prob"].to_dict()
problem += pulp.lpSum((pim[i, dm] - pim[i, 1]) * xim[i, dm] for dm in [2, 3] for i in customer_ids)

# 予算を100万円以下に設定 予算消費期待値 CmPm Cm はdm パターンに対するクーポン額. Pm はセグメントの来店率
# 決定変数 xim を忘れない.
cm = {1: 0, 2: 1000, 3: 2000}
problem += (
    pulp.lpSum(cm[dm] * pim[i, dm] * xim[i, dm] for dm in [2, 3] for i in customer_ids) <= 1000000
)

# 各パターンのDM をそれぞれのセグメントに属する会員に 10% 以上に送付したい.
# 10%=セグメントに属する会員がわかるから、そのセグメント全体の会員数がわかる
seg_nums = customer_prob_df_row_origin.groupby(["segment_id"])["customer_id"].count().to_dict()
# カスタマーid を指定したときにどのセグメントに属するかの辞書.
si = customer_prob_df_row_origin.set_index(["customer_id"])["segment_id"].to_dict()
for (s, n) in seg_nums.items():
    for dm in dms.values():
        # 全ての会員Id を見る代わりに、もしセグメントに属するなら変数を加えるようにする
        # (セグメントに属することでしか1加算する作用がない)
        problem += pulp.lpSum(xim[i, dm] for i in customer_ids if si[i] == s) >= n * 0.1

# %%
time_start = time.time()
status = problem.solve()
time_stop = time.time()
print(f"status: {pulp.LpStatus[status]}")
print(f"objective fns: {pulp.value(problem.objective):.4}")  # 326.1 来客数の増加
print(f"time to run: {(time_stop - time_start):.3}s")

# %%
# 獲得費用（CPA：cost per action）：来客者１人獲得するためにかかる費用
# モデルを解いた後にどの会員にどのDM を送付したか確認
send_dm_df = pd.DataFrame(
    [[xim[i, dm].value() for dm in dms.values()] for i in customer_ids], columns=dms.keys()
)
cust_send_df = pd.concat([customer_df[["customer_id", "age_cat", "freq_cat"]], send_dm_df], axis=1)
# 各セグメントに対するそれぞれのダイレクトメールの送付率と送付数を確認
# デフォルトでセグメント（昨年の頻度、カテゴリ）に送付した割合（%）
ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(nrows=1, ncols=3, figsize=(20, 3))
for i, ptn in enumerate(dms.keys()):
    cust_send_pivot_df = pd.pivot_table(
        data=cust_send_df, values=ptn, columns="freq_cat", index="age_cat", aggfunc="mean"
    )
    cust_send_pivot_df = cust_send_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
    cust_send_pivot_df = sns.heatmap(
        data=cust_send_pivot_df, annot=True, fmt=".1%", cmap="Blues", vmin=0, vmax=1, ax=ax[i]
    )
    ax[i].set_title("{}_rate".format(ptn))
plt.show()

ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(nrows=1, ncols=3, figsize=(20, 3))
for i, ptn in enumerate(dms.keys()):
    cust_send_pivot_df = pd.pivot_table(
        data=cust_send_df, values=ptn, columns="freq_cat", index="age_cat", aggfunc="sum"
    )
    cust_send_pivot_df = cust_send_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
    cust_send_pivot_df = sns.heatmap(
        data=cust_send_pivot_df, annot=True, fmt=".1f", cmap="Blues", vmax=800, ax=ax[i]
    )
    ax[i].set_title("{}_rate".format(ptn))
plt.show()
# %%

# %%
