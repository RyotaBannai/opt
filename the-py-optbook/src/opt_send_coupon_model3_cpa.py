"""
概要：
割引クーポン付与による来客増加数の最大化における、model1, model2 をもとに、
「投資対効果」の観点を考察する

目的：
投資対効果の評価（どれくらい予算を使えば効率的に来店増加率を上げられるか）
"""
# %%
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import pulp

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
cost_list = []
cpa_list = []
inc_action_list = []

print("ステータス, キャンペーン費用, 来客増加数, CPA")
# 加減は 各セグメントに各Dm の送付率を10% 以上に設定したときに必要な予算最低額p135 を参考
for cost in range(761850, 3000000, 100000):
    problem = pulp.LpProblem(name="DiscountCouponProblem2", sense=pulp.LpMaximize)
    segs = prob_df["segment_id"].to_list()
    dms = {"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
    sm = [(s, dm) for dm in dms.values() for s in segs]
    # 各パターンのDm をどの程度の割合で送付するかを [0,1] (0から1 の連続変数)で決定
    xsm: Dict = pulp.LpVariable.dicts(
        name="xsm", indices=sm, lowBound=0, upBound=1, cat="Continuous"
    )
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
        <= cost
    )

    # 各セグメントに送るDm の割合は 10% 以上
    for s in segs:
        for dm in dms.values():
            problem += xsm[s, dm] >= 0.1

    status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
    inc_action = pulp.value(problem.objective)  # 来客増加数
    cpa = cost / inc_action  # 来店者一人の増加に必要な費用

    cost_list.append(cost)
    cpa_list.append(cpa)
    inc_action_list.append(inc_action)

    print(f"{pulp.LpStatus[status]}, {cost}, {inc_action:.4}, {cpa:.5}")


# %%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter(cost_list, inc_action_list, marker="x", label="Incremental vistor")
ax2.scatter(cost_list, cpa_list, marker=".", label="CPA")

# 数値区切りなどを設定 1000 -> 1,000
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax1.set_xlabel("Cost")
ax1.set_ylabel("Incremental vistor")
ax2.set_ylabel("CPA")

# 複数プロットした図のレジェンドをまとめる
reg1, lable1 = ax1.get_legend_handles_labels()
reg2, lable2 = ax2.get_legend_handles_labels()
ax2.legend(reg1 + reg2, lable1 + lable2, loc="upper center")
plt.show()

# %%
