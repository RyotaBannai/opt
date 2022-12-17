"""
商品水栓のための興味スコアリング

ユーザーの商品閲覧履歴を活用

Recency に関する単調整：ユーザーは最近閲覧した商品ほど興味がある
Frequency に関する単調整：ユーザーは何度も閲覧した商品ほど興味がある

実務においてアルゴリズムを中心に商品水栓のロジックを考えてはいけない. 
商品推薦は、ユーザー体験を起点に、どのようなユーザーがどのタイミングでどんなニーズがあるのか？を考えることが重要. 238
"""

# %%
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
from mpl_toolkits.mplot3d import Axes3D

# %%
data_dir = Path("../data/recommendation/")
log_df = pd.read_csv(data_dir / "access_log.csv")
log_df["date"] = log_df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
# log_df.dtypes # check the data types

# %%
# log_df['user_id'].value_counts().describe() # 「一人当たりの平均閲覧履歴は10件」など
start_date = datetime(2015, 7, 1)
end_date = datetime(2015, 7, 7)
target_date = datetime(2015, 7, 8)

x_df = log_df[(start_date <= log_df["date"]) & (log_df["date"] <= end_date)]
y_df = log_df[log_df["date"] == target_date]
y_df = y_df.drop_duplicates()
# check if sucess to split data by date
# log_df.drop_duplicates(subset=['date'])
# x_df.drop_duplicates(subset=['date'])

# freq: ユーザーが商品を閲覧した総数
# rec: ユーザーが最後に商品を閲覧した日　を基準日から引いたもの.
#   基準日: その日にユーザーが再閲覧するかどうか推測する日.
#   rec はユーザーが最後に閲覧した日が最近であればあるほど、小さな値をとる.
# pv_flag: 再閲覧フラグ. ユーザーが基準日に閲覧していれば1 そうでなければ0

# %%
# 各ユーザーに関して、ある商品を閲覧しているならば、それを基準日から何日前に閲覧したかを整理する
UI2Recs: Dict[str, Dict[str, List[int]]] = {}
for row in x_df.itertuples():
    rec = (target_date - row.date).days

    UI2Recs.setdefault(row.user_id, {})
    UI2Recs[row.user_id].setdefault(row.item_id, [])
    UI2Recs[row.user_id][row.item_id].append(rec)

Rows1 = []
for user_id, I2Recs in UI2Recs.items():
    for item_id, Rcens in I2Recs.items():
        rec = min(Rcens)
        freq = len(Rcens)
        Rows1.append((user_id, item_id, rec, freq))
UI2RF_df = pd.DataFrame(Rows1, columns=["user_id", "item_id", "rec", "freq"])

# 予測値をもとのデータにマージ
y_df["pv_flag"] = 1
UI2RF_df = pd.merge(
    UI2RF_df, y_df[["user_id", "item_id", "pv_flag"]], how="left", on=["user_id", "item_id"]
)
UI2RF_df["pv_flag"].fillna(0, inplace=True)
UI2RF_df["pv_flag"] = UI2RF_df["pv_flag"].astype({"pv_flag": "int64"})

# わかりやすさのため、定義域をrec の規模感に合わせて, freq が7以下となるようにフィルタリングする
tar_df = UI2RF_df[UI2RF_df["freq"] <= 7]

# %%
# freq, rec と再閲覧率が単調整を持つことを確認
rec_df = pd.crosstab(index=tar_df["rec"], columns=tar_df["pv_flag"]).rename(
    columns={0: "neg", 1: "pos"}
)
rec_df["N"] = rec_df["neg"] + rec_df["pos"]
rec_df["prob"] = rec_df["pos"] / rec_df["N"]
rec_df[["prob"]].plot.bar()

freq_df = pd.crosstab(index=tar_df["freq"], columns=tar_df["pv_flag"]).rename(
    columns={0: "neg", 1: "pos"}
)
freq_df["N"] = freq_df["neg"] + freq_df["pos"]
freq_df["prob"] = freq_df["pos"] / freq_df["N"]
freq_df[["prob"]].plot.bar()

# %%
RF2N: Dict[Tuple[int, int], int] = {}  # rec,freq ペアに対する総数
RF2PV: Dict[Tuple[int, int], int] = {}  # rec,freq ペアに対する再閲覧件数

# user_id, item_id の行から、閲覧回数と最近閲覧した日をキーにして集計をする
for row in tar_df.itertuples():
    RF2N.setdefault((row.rec, row.freq), 0)
    RF2PV.setdefault((row.rec, row.freq), 0)
    RF2N[row.rec, row.freq] += 1
    RF2PV[row.rec, row.freq] += row.pv_flag

RF2Prob = {}
for rf, N in RF2N.items():
    RF2Prob[rf] = RF2PV[rf] / N

Rows3 = []
for rf, N in sorted(RF2N.items()):
    pv = RF2PV[rf]
    prob = RF2Prob[rf]
    row = (rf[0], rf[1], N, pv, prob)
    Rows3.append(row)

# 閲覧回数と最近閲覧した日をもとに基準日に再閲覧する確率を求めた結果をまとめたdf
rf_df = pd.DataFrame(Rows3, columns=["rec", "freq", "N", "pv", "prob"])
# 縦持ちから横持ち(テーブル形式)にする
# rf_df.pivot_table(index="rec", columns="freq", values="prob")

# 3d plot
Freq = rf_df["freq"].unique().tolist()
Rec = rf_df["rec"].unique().tolist()
Z = [
    rf_df[(rf_df["freq"] == freq) & (rf_df["rec"] == rec)]["prob"].iloc[0]
    for freq in Freq
    for rec in Rec
]
Z = np.array(Z).reshape((len(Freq), len(Rec)))  # y,x(i行,j列だから)
X, Y = np.meshgrid(Rec, Freq)  # x,y
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d", xlabel="rec", ylabel="freq", zlabel="prob")
ax.plot_wireframe(X, Y, Z)
