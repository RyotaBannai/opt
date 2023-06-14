# %%
from abc import ABC, abstractstaticmethod
from typing import Type

import numpy as np

# %%
n_arms = 4


class Env(ABC):
    @abstractstaticmethod
    def react(self, arm: int):
        pass

    @abstractstaticmethod
    def opt(self):
        pass


# エージェントが探索する環境
class EnvBernoulli(Env):
    thetas = [0.1, 0.2, 0.3, 0.4]

    # エージェントの探索に対する環境の反応（どんな報酬を返すか）
    @classmethod
    def react(cls, arm):
        if arm >= len(cls.thetas):
            raise Exception(f"アームのindex over flow. {len(cls.thetas)-1} 以下.")
        return 1 if np.random.random() < cls.thetas[arm] else 0

    # 環境における適解（エージェントは知り得ない）
    @classmethod
    def opt(cls):
        return np.argmax(cls.thetas)


class EnvLogistic(Env):
    arms = [[0, 0], [0, 1], [1, 0], [1, 1]]

    @classmethod
    def p(cls, arm_index):
        x = cls.arms[arm_index][0] * 0.2 + cls.arms[arm_index][1] * 0.8 - 4
        p = 1 / (1 + np.exp(-x))
        return p

    @classmethod
    def react(cls, arm):
        return 1 if np.random.random() < cls.p(arm) else 0

    @classmethod
    def opt(cls):
        return np.argmax([cls.p(i) for i in range(0, len(cls.arms))])


class Agent(ABC):
    @abstractstaticmethod
    def get_arm(self):
        pass

    @abstractstaticmethod
    def sample(self, arm: int, reward: int):
        pass


class GreedyAgent(Agent):
    # 更新式
    # 環境との相互作用結果をもとに自身の経験を更新
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.values[arm] * (self.counts[arm] - 1)) + reward) / self.counts[arm]


# シミュレーション
# N シミュレーションを実行
# 1 回のシミュレーションで実行するアクション回数
# Type[ClassName] でclass を引き受ける.ClassName にするとインスタンスを期待する.
def sim(agent: Type[Agent], env: Type[Env], N=1000, T=1000, **kwargs):
    selected_arms = [[0 for _ in range(T)] for _ in range(N)]
    earned_rewards = [[0 for _ in range(T)] for _ in range(N)]

    for n in range(N):
        # 1のシミュレーション
        a = agent(**kwargs)
        for t in range(T):
            arm = a.get_arm()
            reward = env.react(arm)
            a.sample(arm, reward)
            selected_arms[n][t] = arm
            earned_rewards[n][t] = reward
    return np.array(selected_arms), np.array(earned_rewards)


# %%
