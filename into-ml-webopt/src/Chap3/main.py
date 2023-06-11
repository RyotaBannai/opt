# %%
import arviz as az
import pymc3 as pm

# %%
"""
デザイン案ごとのクリック率θ
"""
Ns = [434, 382, 394, 88]
clicks = [8, 17, 10, 4]
with pm.Model() as model:
    theta = pm.Uniform("theta", lower=0, upper=1, shape=len(Ns))
    obs = pm.Binomial("a", p=theta, n=Ns, observed=clicks)
    trace = pm.sample(5000, chains=2, cores=1)

# %%
az.plot_trace(trace)
print(az.summary(trace, hdi_prob=0.95))  # 要約統計量
az.plot_posterior(trace, hdi_prob=0.95)  # 95%HDI をつけた分布
az.plot_forest(trace, combined=True, hdi_prob=0.95)  # 95%HDI を縦に並べて比較

# %%
"""
部品案ごとのクリック率を組み合わせたθ
"""
img = [0, 0, 1, 1]
btn = [0, 1, 0, 1]
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    gamma = pm.Normal("gamma", mu=0, sigma=10)
    comb = beta[0] * img + beta[1] * btn + alpha + gamma * img * btn
    theta = pm.Deterministic("theta", 1 / (1 + pm.math.exp(-comb)))
    obs = pm.Binomial("obs", p=theta, n=Ns, observed=clicks)
    trace_comb = pm.sample(5000, chains=2, cores=1)

# %%
az.plot_trace(trace_comb)
print(az.summary(trace_comb, hdi_prob=0.95))
az.plot_posterior(trace_comb, hdi_prob=0.95)
# theta のばらつきがalpha やbeta と比べると非常に小さいから、グラフ上は確認できない.
az.plot_forest(trace_comb, combined=True, hdi_prob=0.95)

# %%
"""
比較
部品ごとに確率を設定した時の方がばらつきが小さくなっているのがわかる.
"""

az.plot_forest(
    [trace, trace_comb],
    var_names=["theta"],
    hdi_prob=0.95,
    combined=True,
    model_names=["Individual", "Combined"],
)

# %%
"""
要素の効果
"""
print((trace_comb["beta"][:, 0] > 0).mean())  # 0.7288
print((trace_comb["beta"][:, 1] > 0).mean())  # 0.9834 正の効果があると言える
