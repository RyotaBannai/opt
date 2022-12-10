"""
コストを最小化する輸送車両の配送計画
"""
# %%
import os
import sys
from itertools import combinations_with_replacement, product
from typing import Dict, List, Literal, Optional, Tuple, Union

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
plt.hist(w, bins=20, range=(0, 2000))
# %%
# 日毎にスケジュールの列挙
# 自社拠点から指定された配送先を訪問する最短のルートを算出する
# 一つの配送先集合に対して最も移動時間が短い移動経路だけを残すとした方が効率が良い.


def simulate_route(z: List[Union[Literal[0], Literal[1]]]):
    # enumerate_routes の中でのみ用いる関数
    # z は k_minus_o の部分集合を意味する, 長さ num_places の 0 または１の値のリストで
    # z[k]==1 (k in K) が k への訪問があることを意味する.
    if z[0] == 0:
        # 自社拠点を通らないルートは不適切なので None を返し、後段で除去する
        return None

    # 巡回セールスマン問題を解く
    daily_route_prob = pulp.LpProblem(sense=pulp.LpMinimize)
    # k->l へ移動の有無
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

    # 移動経路での移動時間の合計を表す変数
    h = pulp.LpVariable("h", lowBound=0, cat="Continuous")
    # 労働関係（巡回セールスマン問題にはない制約だが、これが満たされない場合実行不可能としたので追加）
    # 移動時間
    travel = pulp.lpSum([t[k, l] * x[k, l] for k, l in product(K, K)])
    daily_route_prob += (travel - H_regular) <= h
    daily_route_prob += h <= H_max_overtime

    # 目的関数.
    # ある拠点の集合z を全て通りたい時の最短経路問題を解く.
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
    """
    拠点の集合z における全ての部分集合の移動時間が最短となる最短を列挙する
    """

    # joblib を用いて計算並列化（16並列）して、K_minus_o の全ての部分集合に対する最短の移動経路を計算
    # これは次のコードを並列化したもの
    # routes=[]
    # for z in product([0,1], repeat=len(K)):
    #   routes.appened(minulate_route(z))
    routes = Parallel(n_jobs=16)(delayed(simulate_route)(z) for z in product([0, 1], repeat=len(K)))

    # 結果が None のもの（自社拠点を通らないもの）を除去
    routes = pd.DataFrame([x for x in routes if x is not None])

    # 結果が Optimal でないもの（ここでは移動時間(h)が長すぎて実行不能となるもの）を削除
    routes = routes[routes.optimal].copy()
    return routes


# 移動経路一覧
routes_df = enumerate_routes()

# %%
"""
移動経路一覧ができたら、これを用いて
各日付について,'移動'と'配送する荷物'の両方を指定したスケジュールを列挙する.

日に依存することなく、移動経路の候補が列挙できているため、実際には
移動経路上で'配送可能な荷物の部分集合'であって、'重量制限を守れるもの'を日ごとに列挙すればよい.
"""


def is_OK(requests: List[int]) -> Tuple[Tuple[Optional[int], Optional[float]], bool]:
    # 指定された荷物の配送が重量制約の元で可能かどうかを確認する
    # 可能である場合は、配送を実行できる最短の移動経路のindex（routes_df におけるもの）とその所要時間を返す.
    # 不可能であれば False
    # requets: R に含まれるリストで、配送する荷物の一覧

    weight = sum([w[r] for r in requests])
    if weight > W:  # 枝刈り 使えない荷物の集合は考慮しない
        return ((None, None), False)

    best_route_index = None
    best_hour = sys.float_info.max
    for route_index, row in routes_df.iterrows():
        # k[r] 荷物r の配送先
        # row.z[k[r]] == 1, 荷物r の配送先k[r] が移動経路の含まれている経路
        all_requests_on_route = all([row.z[k[r]] == 1 for r in requests])
        if all_requests_on_route and row.移動時間 < best_hour:
            best_route_index = route_index
            best_hour = row.移動時間
    if best_route_index is None:
        return ((None, None), False)

    return ((best_route_index, best_hour), True)


def _enumerate_feasible_schedules(
    requests_cands: List[int], current_index_set: List[int], index_to_add: int, res: List[Dict]
):
    # R に含まれるリスト requests_cands を候補として
    # current_index_set で指定される荷物に加えて配送することができる
    # requests_cands[index_to_add:] の部分集合を全て列強する（再帰的に計算する）
    # 配送可能な荷物の集合は、リスト res に追加される
    # 各再帰ごとに、index_to_add を追加する、追加しないの2 通りを試す.
    # requesrts_cands で指数関数的に計算量が増える.

    # index_set_to_check = current_index_set + [index_to_add] で指定される
    # 荷物が配送可能かを確認する
    index_set_to_check = current_index_set + [index_to_add]
    next_index = index_to_add + 1
    is_next_index_valid = next_index < len(requests_cands)
    requests = [requests_cands[i] for i in index_set_to_check]
    (best_route_index, best_hour), is_ok = is_OK(requests=requests)

    if is_ok:
        # index_set_to_check で指定される荷物が配送可能であれば、
        # その配送に用いされるルートの情報を記録する
        res.append(
            {
                "requests": [requests_cands[i] for i in index_set_to_check],
                "route_index": best_route_index,
                "hours": best_hour,
            }
        )

        if is_next_index_valid:
            # さらに荷物を追加できるか確認
            # next_index を一つ進めて、再度is_OK により最短ルートを確保できるか見る
            _enumerate_feasible_schedules(
                requests_cands=requests_cands,
                current_index_set=index_set_to_check,  # index_to_add を追加した新しい set
                index_to_add=next_index,  # ひとつ進める
                res=res,
            )

    if is_next_index_valid:
        # index_to_add をスキップして、next_index 以降の荷物を追加できるか確認する
        _enumerate_feasible_schedules(
            requests_cands=requests_cands,
            current_index_set=current_index_set,  # index_to_add は追加せずに元の set を渡す
            index_to_add=next_index,  # ひとつ進める
            res=res,
        )


def enumerate_feasible_schedules(d: int):
    # _enumerate_feasible_shcedules を用いて、d 日に考慮すべきスケジュールを列挙する

    # 配送日指定に合うものだけを探索
    requests_cands = [r for r in R if d_0[r] <= d <= d_1[r]]

    # res にd 日の時効可能なスケジュールを格納数r
    res = [{"requests": [], "route_index": 0, "hours": 0}]

    _enumerate_feasible_schedules(
        requests_cands=requests_cands, current_index_set=[], index_to_add=0, res=res
    )

    # res を DataFrame にして後処理に必要な値を計算する
    feasible_schedules_df = pd.DataFrame(res)
    feasible_schedules_df["overwork"] = (feasible_schedules_df.hours - H_regular).clip(0)
    feasible_schedules_df["requests_set"] = feasible_schedules_df.requests.apply(set)

    # feasible_schedules_df のうち、不要なスケジュールを削除
    # あるスケジュールの集合が別のスケジュールに対して
    #   配送する荷物の集合の包含関係での比較
    #   残業時間の比較
    # の二つの比較で同時に負けている場合には、そのスケジュールは利用価値がないため破棄

    # 全て
    index_cands = set(feasible_schedules_df.index)

    # 破棄
    index_inferior = set()

    for i in feasible_schedules_df.index:
        for j in feasible_schedules_df.index:
            # 配送する荷物の集合の包含関係で比較
            requests_strict_dominance = (
                feasible_schedules_df.requests_set.loc[j]
                < feasible_schedules_df.requests_set.loc[i]
            )

            # 残業時間の比較
            overwork_weak_dominance = (
                feasible_schedules_df.overwork.loc[j] >= feasible_schedules_df.overwork.loc[i]
            )

            if requests_strict_dominance and overwork_weak_dominance:
                index_inferior.add(j)

    # 残す
    index_superior = index_cands - index_inferior
    superior_feasible_shedules_df = feasible_schedules_df.loc[list(index_superior), :]
    return superior_feasible_shedules_df


_shedules = Parallel(n_jobs=16)([delayed(enumerate_feasible_schedules)(d) for d in D])
feasible_schedules = dict(zip(D, _shedules))

# %%
print("1日の最大スケジュール候補数:", max(len(df) for df in feasible_schedules.values()))  # 939
print("スケジュール候補数の合計:", sum(len(df) for df in feasible_schedules.values()))  # 8430

# %%
"""
列挙したスケジュール候補を用いて、金剛整数計画問題を解く
"""
prob = pulp.LpProblem(sense=pulp.LpMinimize)

# 日ごとにどの配送計画（スケジュール）を採用するか
z = {}
for d in D:
    for q in feasible_schedules[d].index:
        z[d, q] = pulp.LpVariable(f"z_{d}_{q}", cat="Binary")

# 配送を外注するかどうかの補助変数
# yは連続変数だけど、制約のかけ方により必ず0/1 の Binaryとなる.（外部委託するかしないかの制約部分）
y = {r: pulp.LpVariable(f"y_{r}", cat="Continuous", lowBound=0, upBound=1) for r in R}

# 荷物r の配送の回数をy,z の言葉で表しておく
deliv_count = {r: pulp.LpAffineExpression() for r in R}
for d in D:
    for q in feasible_schedules[d].index:
        for r in feasible_schedules[d].loc[q].requests:
            deliv_count[r] += z[d, q]  # スケジュールが採用されれば、荷物r の配達先が配達経路上にある回数が１つ増える

# 日付d の残業時間をy,z の言葉で表しておく
h = {
    d: pulp.lpSum(
        z[d, q] * feasible_schedules[d].overwork.loc[q] for q in feasible_schedules[d].index
    )
    for d in D
}

# 制約
# 一日ひとつのスケジュールを選択
for d in D:
    prob += pulp.lpSum(z[d, q] for q in feasible_schedules[d].index) == 1

# y が外部委託による配送を表すように、荷物r の配達回数が 0なら y >=1になるよう設定（yが1==外部委託によって配達される）
for r in R:
    prob += y[r] >= 1 - deliv_count[r]

# 目的関数
obj_overtime = pulp.lpSum([c * h[d] for d in D])
obj_outsoarcing = pulp.lpSum([f[r] * y[r] for r in R])
obj_total = obj_overtime + obj_outsoarcing
prob += obj_total

prob.solve()
# %%
"""
日毎に可視化
"""


def visualize_route(d: int):
    for q in feasible_schedules[d].index:
        if z[d, q].value() == 1:
            route_summary = feasible_schedules[d].loc[q]
            route_geography = routes_df.loc[route_summary.route_index]
            break

    # 背景
    a = plt.subplot()
    a.scatter(_K[1:, 0], _K[1:, 1], marker="x")
    a.scatter(_K[0, 0], _K[0, 1], marker="o")

    # 移動経路
    # 元々 routes_df で日毎の移動経路を固定して決めているから、それに従ってk->l 間の移動を整理しているだけ.
    # 計算時は、KxK のproduct を使って全点間を考慮しているから、0 の変数が多い（はず）
    moves = [(k_from, k_to) for (k_from, k_to), used in route_geography.route.items() if used == 1]
    # moves
    # [(0, 7), (4, 0), (7, 9), (9, 4)]
    for k_from, k_to in moves:
        p_from = _K[int(k_from)]
        p_to = _K[int(k_to)]
        a.arrow(
            *p_from,
            *(p_to - p_from),
            head_width=3,
            length_includes_head=True,
            overhang=0.5,
            color="gray",
            alpha=0.5,
        )

        # requests は荷物の集合
        requests_at_k_to = [r for r in route_summary.requests if k[r] == k_to]
        a.text(*p_to, "".join([str(r) for r in requests_at_k_to]))
    plt.title(f"Schedule for day: {d}")
    plt.show()


# 0 日目のスケジュール
visualize_route(d=0)
# for d in D:
# visualize_route(d=d)

requests_summary_df = pd.DataFrame(
    [
        {
            "outsourced": y[r].value(),
            "weight": w[r],
            "freight": f[r],  # 外部委託コスト
            "location": k[r],
            "distance_from_o": t[k[r], o],
        }
        for r in R
    ]
)
requests_summary_df.groupby("outsourced")[["weight", "freight", "distance_from_o"]].agg("mean")
# 外注した荷物とそうでない荷物の特徴を確認
#       weight	freight	distance_from_o
# outsourced
# 0.0	1012.845455	48718.181818	106.609091
# 1.0	955.200000	45080.000000	121.700000
requests_summary_df.plot.scatter(x="distance_from_o", y="freight", c="outsourced", cmap="cool")
plt.show()

# %%
