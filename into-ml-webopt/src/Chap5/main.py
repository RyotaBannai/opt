# %%
from abc import ABC, abstractstaticmethod
from typing import Type

import numpy as np
from matplotlib import pyplot as plt

# %%
np.random.seed(0)
n_arms = 4


# エージェントが探索する環境
class Env:
    thetas = [0.1, 0.2, 0.3, 0.4]

    # エージェントの探索に対する環境の反応（どんな報酬を返すか）
    @classmethod
    def react(cls, arm: int):
        if arm >= len(cls.thetas):
            raise Exception(f"アームのindex over flow. {len(cls.thetas)-1} 以下.")
        return 1 if np.random.random() < Env.thetas[arm] else 0

    # 環境における適解（エージェントは知り得ない）
    @classmethod
    def opt(cls):
        return np.argmax(cls.thetas)


class Agent(ABC):
    @abstractstaticmethod
    def get_arm(self):
        pass

    @abstractstaticmethod
    def sample(self, arm: int, reward: int):
        pass


class GreedyAgent(Agent):
    # 環境との相互作用結果をもとに自身の経験を更新
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.values[arm] * (self.counts[arm] - 1)) + reward) / self.counts[arm]


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


# シミュレーション
# N シミュレーションを実行
# 1 回のシミュレーションで実行するアクション回数
# Type[ClassName] でclass を引き受ける.ClassName にするとインスタンスを期待する.
def sim(agent: Type[Agent], N=1000, T=1000, **kwargs):
    selected_arms = [[0 for _ in range(T)] for _ in range(N)]
    earned_rewards = [[0 for _ in range(T)] for _ in range(N)]

    for n in range(N):
        # 1のシミュレーション
        a = agent(**kwargs)
        for t in range(T):
            arm = a.get_arm()
            reward = Env.react(arm)
            a.sample(arm, reward)
            selected_arms[n][t] = arm
            earned_rewards[n][t] = reward
    return np.array(selected_arms), np.array(earned_rewards)


# 環境の最適値をどれくらい当てられたか割合で評価
arms_eg, _ = sim(EpsilonGreedyAgent)
arms_aeg, _ = sim(AnnealingEpsilonGreedyAgent)
arms_as, _ = sim(AnnealingSoftmaxAgent)
# N 回のシミュレーション結果を行として、行ごとにTrue かどうかをチェック.
# t 回目にどれくらい正解しているか. t 回目の正解の平均を出す（t 回目のアクションの正解率を出す.）
acc_eg = np.mean(arms_eg == Env.opt(), axis=0)
acc_aeg = np.mean(arms_aeg == Env.opt(), axis=0)
acc_as = np.mean(arms_as == Env.opt(), axis=0)
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
