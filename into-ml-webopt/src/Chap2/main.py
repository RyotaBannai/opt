# %%
import arviz as az
import matplotlib as mlp
import numpy as np
import pymc3 as pm

# %%
Ns = [40, 50]  # アリスのデザインA,B案の表示数
clicks = [2, 4]  # アリスのデザインA,B案 のクリック数
with pm.Model() as model:
    theta = pm.Uniform("theta", lower=0, upper=1, shape=2)
    obs = pm.Binomial("a", p=theta, n=Ns, observed=clicks)
    trace = pm.sample(5000, chains=2, cores=1)  # 事後分布から大量のサンプルを取得する

# %%
az.plot_trace(trace)  # 可視化 3.11.0<=
# pm.traceplot(trace) # <3.11.0
print(az.summary(trace, hdi_prob=0.95))

# %%
az.plot_posterior(trace, hdi_prob=0.95)  # 要約統計量

# %%
