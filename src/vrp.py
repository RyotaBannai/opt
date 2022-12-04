"""
コストを最小化する輸送車両の配送計画
"""
# %%
import os
import sys
from itertools import combinations_with_replacement, product
from typing import List, Literal, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
from IPython.core.display import display
from joblib import Parallel, delayed

# %%
np.random.seed(10)
num_places = 10  # 地点の数
num_days = 30  # 計画対象日数
num_requests = 120  # 荷物の数

mean_travel_time_to_destinations = 100  # 自社から平均的に100 分程度の距離に配達先候補があるとしてデータを作る
H_regular = 8 * 60  # 8時間が所定労働時間
H_max_overtime = 3 * 60  # 残業 3 時間まで
c = 3000 // 60  # 残業による経費60分3000 円
W = 4000  # 4トントラック
delivery_outsourcing_unit_cost = 4600  # 100kg あたり4600 円の配送費用
deilvery_time_window = 3  # 連続する3 日が配達可能のな候補日となる
avg_weight = 1000  # 荷物の平均的な重さを 1000kg とする

K = range(num_places)  # 地点の集合
o = 0  # 自社拠点を表す地点
K_minus_o = K[1:]  # 配達先の集合
_K = np.random.normal(0, mean_travel_time_to_destinations, size=(len(K), 2))  # 各地点の（平面xy）座標を設定
_K[o, :] = 0  # 自社拠点は原点とする

t = np.array([[np.floor(np.linalg.norm(_K[k] - _K[l])) for k in K] for l in K])  # 各地点間の移動時間行列（分）

D = range(num_days)  # 日付の集合
R = range(num_requests)  # 荷物の集合
# k[r] は荷物r の配送先を表す
k = np.random.choice(K_minus_o, size=len(R))
# d_0[r] は荷物r n配送可日の初日を表す.
d_0 = np.random.choice(D, size=len(R))
# d_1[r] は荷物r n配送可日の最終日を表す
d_1 = d_0 + deilvery_time_window - 1
# w[r] は荷物r のp重さ（kg）を表す（10 を平均としていい感じに荷物の重さをランダムに手に入れたい）
w = np.floor(np.random.gamma(10, avg_weight / 10, size=len(R)))
# f[r] は荷物r の外部委託時の配送料を表す（送料コストが100 kg あたりだからいい感じに発送料を調整.）
f = np.ceil(w / 100) * delivery_outsourcing_unit_cost
# %%
# 拠点と配送先の関係を可視化
a = plt.subplot()
a.scatter(_K[1:, 0], _K[1:, 1], marker="x")
a.scatter(_K[0, 0], _K[0, 1], marker="o")
a.set_aspect("equal")
plt.show()
# %%
# 日毎にスケジュールの列挙
# 自社拠点から指定された配送先を訪問する最短のルートを算出する
# 一つの配送先集合に対して最も移動時間が短い移動経路だけを残すとした方が効率が良い.


def simulate_route(z: List[Union[Literal[0], Literal[1]]]):
    # enumerate_routes の中でのみ用いる関数
    # z は k_minus_o の部分集合を意味する長さ num_places の 0 または１の値のリストで、
    # z[k]==1 (k in K) が k への訪問があることを意味する.
    if z[0] == 0:
        # 自社拠点を通らないルートは不適切なので None を返し、後段で除去する
        return None

    # 巡回セールスマン問題を解く
    daily_route_prob = pulp.LpProblem(sense=pulp.LpMinimize)
    #  k->l へ移動の有無
    # LpAffineExpression は一次式としての0 を意味する.
    x = {
        (k, l): pulp.LpVariable(f"x_{k}_{l}", cat="Binary") if k != l else pulp.LpAffineExpression()
        for k, l in product(K, K)
    }

    # MTZ 定式化のための補助変数u
    u = {
        k: pulp.LpVariable(
            f"u_{k}",
            lowBound=1,
            upBound=len(K) - 1,
        )
        for k in K_minus_o
    }
    # MTZ 定式化の補助変数の説明では、訪問順序であることを意識して、u[0] を変数かのように書いてあるが、
    # 実際には 0 に固定されている値だから、ここでは u[0] を辺数として定義しない.

    h = pulp.LpVariable("h", lowBound=0, cat="Contunuous")

    # 移動の構造
    for l in K:
        daily_route_prob += pulp.lpSum([x[k, l] for k in K]) <= 1
    for l in K:
        if z[l] == 1:
            # z で l への訪問が指定されている場合、必ず訪問するようにする. k->l->m
            daily_route_prob += pulp.lpSum(x[k, l] for k in K) == 1
            daily_route_prob += pulp.lpSum(x[l, k] for k in K) == 1
        else:
            # z で l への訪問が禁止されている場合、訪問できないようにx に制約を入れる.
            daily_route_prob += pulp.lpSum(x[k, l] for k in K) == 0
            daily_route_prob += pulp.lpSum(x[l, k] for k in K) == 0

    # サイクルの除去
    for k, l in product(K_minus_o, K_minus_o):
        daily_route_prob += u[k] + 1 <= u[l] + len(K_minus_o) * (1 - x[k, l])  # MTZ 定式化

    # 労働関係（巡回セールスマン問題にはない制約だが、これが満たされない場合実行不可能としたので追加）
    # 移動時間
    travel = pulp.lpSum([t[k, l] * x[k, l] for k, l in product(K, K)])
    daily_route_prob += (travel - H_regular) <= h
    daily_route_prob += h <= H_max_overtime

    # 目的関数
    daily_route_prob += travel
    daily_route_prob.solve()

    return {
        "z": z,
        "route": {(k, l): x[k, l].value() for k, l in product(K, K)},
        "optimal": daily_route_prob.status == 1,  # k から l への移動の有無を辞書で保持
        "移動時間": travel.value(),
        "残業時間": h.value(),
    }


def enumerate_routes():
    # 移動時間を列挙する
    # joblib を用いて計算並列化（16並列）して、K_minus_o の全ての部分集合に対する最短の移動経路を計算
    # これは次のコードを並列化したもの
    # routes=[]
    # for z in product([0,1], repeat=len(K)):
    #   routes.appened(minulate_route(z))
    routes = Parallel(n_jobs=16)(delayed(simulate_route)(z) for z in product([0, 1], repeat=len(K)))

    # 結果が None のもの（自社拠点を通らないもの）を除去
    routes = pd.DataFrame([x for x in routes if x is not None])

    # 結果が Optimal でないもの（ここでは移動時間が長すぎて実行不能となるもの）を削除
    routes = routes[routes.optimal].copy()
    return routes


# 移動経路一覧
routes_df = enumerate_routes()

# %%
