# %%

import pathlib
import sys

import numpy as np
from matplotlib import pyplot as plt

src = str((pathlib.Path(__file__).parent / "..").resolve())
sys.path.append(src)

from common import EnvLogistic, GreedyAgent, sim

# %%

np.random.seed(0)
n_arms = 4


arms = [[0, 0], [0, 1], [1, 0], [1, 1]]


# ベイズ線形回帰を使ったUCB アルゴリズム
class LinUCBAgent(GreedyAgent):
    def __init__(self) -> None:
        self.sigma = 1  # 報酬が固定の分散に従って生成される
        self.alpha = 1  # LinUBC のUB の定数
        self.phis = np.array([[arm[0], arm[1], 1] for arm in arms]).T  # 3x4
        self.A = np.identity(self.phis.shape[0])  # 3x3
        self.b = np.zeros((self.phis.shape[0], 1))  # 3x1 未知の確率変数分用意

    def get_arm(self):
        # アームを選択する時点では、sample の呼び出しによってt+1 回目の更新は完了してる
        inv_A = np.linalg.inv(self.A)  # 3x3
        mu = inv_A.dot(self.b)  # 各確率変数の期待値
        S = inv_A
        pred_mean = self.phis.T.dot(mu)  # 1x4
        pred_var = self.phis.T.dot(S).dot(self.phis)  # 4x4
        # 1x4 ここでは、全アクション分の次元となる
        ucb = pred_mean.T + self.alpha * np.sqrt(np.diag(pred_var))
        return np.argmax(ucb)

    # 更新式
    def sample(self, arm, reward):
        phi = self.phis[:, [arm]]
        self.b = self.b + phi * reward / (self.sigma**2)
        self.A = self.A + phi.dot(phi.T) / (self.sigma**2)


# 環境の最適値をどれくらい当てられたか割合で評価
arms, _ = sim(agent=LinUCBAgent, env=EnvLogistic, N=500, T=5000)

# %%
acc_eg = np.mean(arms == EnvLogistic.opt(), axis=0)
plt.plot(acc_eg, label=r"LinUCB")
plt.xlabel(r"$t$")
plt.ylabel(r"$\mathbb{E}[x(t) = x^*]$")
plt.legend()
plt.show()

# %%
