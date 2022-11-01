# %%
# 連立一次方程式を解く

from pprint import pprint

import pulp

problem = pulp.LpProblem(
    name="SLE", sense=pulp.LpMaximize
)  # name はなんでもok, sense は最大化問題を解く、という指定をしている.
# Continuous:= 連続変数であることを表す.
x: pulp.LpVariable = pulp.LpVariable(name="x", cat="Continuous")
y: pulp.LpVariable = pulp.LpVariable(name="y", cat="Continuous")
# 以下は制約式が作られる
# problem.addConstraint(120 * x + 150 * y == 1440) でもok
problem += 120 * x + 150 * y == 1440  # 買い物の合計額
problem += x + y == 10  # りんご、なしの個数の合計10

status = problem.solve()

pprint("Status: {}".format(pulp.LpStatus[status]))
pprint("x={}, y={}".format(x.value(), y.value()))

# %%
# 線形計画問題（領域の最大、最小）
"""
製品p,q、材料m,n がある.
p を1kg 製造するのに、mが1kg,nが2kg
q を1kg 製造するのに、mが3kg,nが1kg 必要である.
m 30kg, n 40kg 在庫がある.
単位量あたりの利得は、p 1万円、q 2万円である.
利益を最大化せよ.
"""
prob: pulp.LpProblem = pulp.LpProblem(name="LP1", sense=pulp.LpMaximize)
p: pulp.LpVariable = pulp.LpVariable(name="p", cat="Continuous")
q: pulp.LpVariable = pulp.LpVariable(name="q", cat="Continuous")
# 条件式
prob += p + 3 * q <= 30  # 材料m
prob += 2 * p + q <= 40  # 材料n
prob += p >= 0
prob += q >= 0
# 目的関数を定義するときは、方程式を立てない
# prob.setObjective(p+2*q) でもok
prob += p + 2 * q

status = prob.solve()

pprint("Status: {}".format(pulp.LpStatus[status]))
# 目的関数を最大化した結果, 利益の最大は 26 になることがわかる.
pprint("p={}, q={}, obj={}".format(p.value(), q.value(), prob.objective.value()))

# %%
