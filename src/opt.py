#%% # noqa
from pprint import pprint

import pulp

problem = pulp.LpProblem(
    name="SLE", sense=pulp.LpMaximize
)  # name はなんでもok, sense は最大か問題を解くという指定をしている.
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
