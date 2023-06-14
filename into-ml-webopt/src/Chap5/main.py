# %%

import pathlib
import sys

import numpy as np
from matplotlib import pyplot as plt

src = str((pathlib.Path(__file__).parent / "..").resolve())
sys.path.append(src)

from common import EnvBernoulli, GreedyAgent, sim

# %%

np.random.seed(0)
n_arms = 4


class EpsilonGreedyAgent(GreedyAgent):
    def __init__(self, epsilon=0.1) -> None:
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    # 環境におけるアクションを選択
    def get_arm(self):
        if np.random.random() < self.epsilon:
            # 探索
            arm = np.random.randint(n_arms)
        else:
            # 活用
            arm = np.argmax(self.values)

        return arm


# 冷却スケジュール処理追加
# デフォルトのepsilon の値を変更
class AnnealingEpsilonGreedyAgent(GreedyAgent):
    def __init__(self, epsilon=1.0) -> None:
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    # 環境におけるアクションを選択
    def get_arm(self):
        if np.random.random() < self.epsilon:
            # 探索
            arm = np.random.randint(n_arms)
        else:
            # 活用
            arm = np.argmax(self.values)

        self.epsilon *= 0.99

        return arm


class AnnealingSoftmaxAgent(GreedyAgent):
    def __init__(self, tau=1000) -> None:
        self.tau = tau
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def softmax_p(self):
        logit = self.values / self.tau
        logit = logit - np.max(logit)
        p = np.exp(logit) / sum(np.exp(logit))
        return p

    # 環境におけるアクションを選択
    def get_arm(self):
        arm = np.random.choice(n_arms, p=self.softmax_p())
        self.tau *= 0.9
        return arm


# 環境の最適値をどれくらい当てられたか割合で評価
arms_eg, _ = sim(agent=EpsilonGreedyAgent, env=EnvBernoulli)
arms_aeg, _ = sim(AnnealingEpsilonGreedyAgent, EnvBernoulli)
arms_as, _ = sim(AnnealingSoftmaxAgent, EnvBernoulli)
# N 回のシミュレーション結果を行として、行ごとにTrue かどうかをチェック.
# t 回目にどれくらい正解しているか. t 回目の正解の平均を出す（t 回目のアクションの正解率を出す.）
acc_eg = np.mean(arms_eg == EnvBernoulli.opt(), axis=0)
acc_aeg = np.mean(arms_aeg == EnvBernoulli.opt(), axis=0)
acc_as = np.mean(arms_as == EnvBernoulli.opt(), axis=0)
"""
np.mean(np.array([[1,2],[1,2]]), axis=1)
>> array([1.5, 1.5])
np.mean(np.array([[1,2],[1,2]]), axis=0)
>> array([1., 2.])
"""
plt.plot(acc_eg, label=r"$\varepsilon$-greedy")
plt.plot(acc_aeg, label=r"Annealing $\varepsilon$-greedy")
plt.plot(acc_as, label=r"Annealing Softmax")
plt.xlabel(r"$t$")
plt.ylabel(r"$\mathbb{E}[x(t) = x^*]$")
plt.legend()
plt.show()
# %%
