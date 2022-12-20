"""
商品推薦のための興味スコアリング
確率推定問題（あるデータをもとに何か確率を求めたい）を凸二次計画問題にモデリングする
凸二次計画問題: 目的関数が凸な二次関数であり、制約式が線形の不等式で書ける最適化問題

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

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# %%
# 1) rec と freq に対して再閲覧確率 pred_prob を推定する
R = sorted(tar_df["rec"].unique().tolist())
F = sorted(tar_df["freq"].unique().tolist())

# cvxopt では辺数ベクトルと積をとる行列やベクトルを直接定義するため、変数を明示的に定義する必要はない.
# その代わり、数理モデルにおける変数 pred_prob_{r,f} が何番目の変数か索引をつける必要がある.
Idx = []
RF2Idx = {}
index = 0
for r in R:
    for f in F:
        Idx.append(index)
        RF2Idx[r, f] = index
        index += 1

G_list = []  # 制約式に現れる変数の係数行列
h_list = []  # 制約式に現れる定数項のベクトルを作るためのリスト
# 変数の係数行列を作成するためのデフォルトの変数の係数ベクトル
# 制約式や、目的関数を定義する際にデフォルトで利用するベクトル
var_vec = [0.0] * len(Idx)

# pred_prob は確率なので、0以上1以下の値をとる変数
# -pred_prob[r,f] <=0 の実装.
for r in R:
    for f in F:
        idx = RF2Idx[r, f]
        G_row = var_vec[:]
        G_row[idx] = -1  # pred_prob[r,f] の係数は-1 (<=0 としている)
        G_list.append(G_row)
        h_list.append(0)  # 右辺の定数項 0

# pred_prob[r,f] <=1 の実装.
for r in R:
    for f in F:
        idx = RF2Idx[r, f]
        G_row = var_vec[:]
        G_row[idx] = 1  # pred_prob[r,f] の係数は1
        G_list.append(G_row)
        h_list.append(1)  # 右辺の定数項 1

# 2) pred_prob はrec について単調減少する
# より最近に商品をみた場合の確率の方が、再閲覧確率が高くなることを表現する
# 3) pred_prob はfreq について単調増加する
# より頻繁に商品をみた場合の確率の方が、再閲覧確率が高くなることを表現する
# cvxoptでの制約式 Gx<=h となるよう、右辺に定数項を置く

# -pred_prob[r,f] + pred_prob[r+1,f] <= 0 の実装
# r+1 しているから、定義域からはみ出さないように注意.
for r in R[:-1]:
    for f in F:
        idx1 = RF2Idx[r, f]
        idx2 = RF2Idx[r + 1, f]  # 最後に閲覧した日付が１日だけさらに前
        G_row = var_vec[:]
        G_row[idx1] = -1  # pred_prob[r,f] の係数は-1
        G_row[idx2] = 1  # pred_prob[r,f] の係数は1
        G_list.append(G_row)
        h_list.append(0)  # 右辺の定数項 0

# pred_prob[r,f] - pred_prob[r,f+1] <= 0 の実装
for r in R:
    for f in F[:-1]:
        idx1 = RF2Idx[r, f]
        idx2 = RF2Idx[r, f + 1]  # 閲覧回数が１回だけ多い
        G_row = var_vec[:]
        G_row[idx1] = 1  # pred_prob[r,f] の係数は1
        G_row[idx2] = -1  # pred_prob[r,f] の係数は-1
        G_list.append(G_row)
        h_list.append(0)  # 右辺の定数項 0


# %%
"""
Recency に関する凸性（下に凸）: 過去に閲覧すればするほど、再閲覧確率の下降幅は逓減する
Frequency に関する凸性（上に凸）: 閲覧数が増えれば増えるほど、再閲覧確率の増加幅は逓減する

実装要件が前後逆になるが..
モデルブラッシュアップ最終項.p278 あたり
"""
# Recency について
rec_df["prob"].diff().plot.bar()
rec_df["prob"].diff().plot
# %%
# Frequency について
freq_df["prob"].diff().plot.bar()
freq_df["prob"].diff().plot

# %%
# 凸性
# prob_pred[r+1,f] - prob_pred[r,f] <= prob_pred[r+2,f] - prob_pred[r,f]
# - prob_pred[r,f] + 2 * prob_pred[r+1,f] - prob_pred[r+2,f] <= 0
# 凸二次計画問題の制約式の右辺は定数にする
for r in R[:-2]:
    for f in F:
        idx1 = RF2Idx[r, f]
        idx2 = RF2Idx[r + 1, f]
        idx3 = RF2Idx[r + 2, f]
        G_row = var_vec[:]
        G_row[idx1] = -1
        G_row[idx2] = 2
        G_row[idx3] = -1
        G_list.append(G_row)
        h_list.append(0)


# 凹性
# prob_pred[r,f+1] - prob_pred[r,f] =< prob_pred[r,f+2] - prob_pred[r,f+1]
# - prob_pred[r,f] + 2 * prob_pred[r,f+1] - prob_pred[r,f+1] =< 0
# prob_pred[r,f] - 2 * prob_pred[r,f+1] + prob_pred[r,f+1] <= 0
# 「右辺以下」となるようにする
for r in R:
    for f in F[:-2]:
        idx1 = RF2Idx[r, f]
        idx2 = RF2Idx[r, f + 1]
        idx3 = RF2Idx[r, f + 2]
        G_row = var_vec[:]
        G_row[idx1] = 1
        G_row[idx2] = -2
        G_row[idx3] = 1
        G_list.append(G_row)
        h_list.append(0)


# 4) pred_prob と prob の二乗誤差を後見すうの重み付きで最小化する

P_list = []  # 目的関数の変数の２次の項の係数行列を作るためのリスト
q_list = []  # 目的関数の変数の１次の項の係数ベクトルを作るためのリスト

# N[r,f] * pred_prob[r,f] ^ 2 - 2 *N[r,f] * pred_prob[r,f] の実装
# pred_prob[r,f] が変数だから、定数項は削除してある.

for r in R:
    for f in F:
        idx = RF2Idx[r, f]
        N = RF2N[r, f]
        prob = RF2Prob[r, f]  # 実測値
        P_row = var_vec[:]
        P_row[idx] = 2 * N  # (1/2) を打ち消すために２をかける.
        P_list.append(P_row)
        q_list.append(-2 * N * prob)


# 行列の作成
G = cvxopt.matrix(np.array(G_list), tc="d")
h = cvxopt.matrix(np.array(h_list), tc="d")
P = cvxopt.matrix(np.array(P_list), tc="d")
q = cvxopt.matrix(np.array(q_list), tc="d")

solve = cvxopt.solvers.qp(P, q, G, h)
status = solve["status"]

# %%
RF2PredProb = {}
X = solve["x"]
for r in R:
    for f in F:
        idx = RF2Idx[r, f]
        pred_prob = X[idx]
        RF2PredProb[r, f] = pred_prob
rf_df["pred_prob"] = rf_df.apply(lambda x: RF2PredProb[x["rec"], x["freq"]], axis=1)
rf_df.head()

# %%

# テーブル形式
# rf_df.pivot_table(index="rec", columns="freq", values="pred_prob")

# 3d plot
Freq = rf_df.freq.unique().tolist()
Rcen = rf_df.rec.unique().tolist()
Z = [
    rf_df[(rf_df["freq"] == freq) & (rf_df["rec"] == rcen)]["pred_prob"].iloc[0]
    for freq in Freq
    for rcen in Rcen
]
Z = np.array(Z).reshape((len(Freq), len(Rcen)))
X, Y = np.meshgrid(Rcen, Freq)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d", xlabel="rec", ylabel="freq", zlabel="pred_prob")
ax.plot_wireframe(X, Y, Z)


# %%
# サンプルデータで再閲覧確率を推定
# どの組み合わせの場合に確率が高いのか判断がつかない場合で一目瞭然となる.
Rows4 = [
    ("item1", 1, 6),
    ("item2", 2, 2),
    ("item3", 1, 2),
    ("item4", 1, 1),
]
sample_df = pd.DataFrame(Rows4, columns=["item_name", "rec", "freq"])
pd.merge(sample_df, rf_df, on=["rec", "freq"])
