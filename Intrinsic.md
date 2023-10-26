---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
cpu_count = 100
import os
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count}'
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, jit, vmap
import jax
cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]
```

```{code-cell} ipython3
n_chains = 10
```

```{code-cell} ipython3
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
import arviz as az
import pandas as pd
import dmeq
```

```{code-cell} ipython3
key = random.PRNGKey(42)
```

```{code-cell} ipython3
def full_solution(params, eir, eta):
    max_age = 99
    p = dmeq.default_parameters()
    for k, v in params.items():
        p[k] = v
    p['EIR'] = eir
    p['eta'] = eta
    s = dmeq.solve(p, dtype=jnp.float64)
    return {
        'pos_M': s[0][:max_age],
        'inc': s[1][:max_age],
        'prob_b': s[2][:max_age],
        'prob_c': s[3][:max_age],
        'prob_d': s[4][:max_age],
        'prop': s[5][:max_age],
    }
```

```{code-cell} ipython3
person_risk_time = 1_000 * 365.
prev_N = 1_000

def prev_stats(solution):
    inc_rates = solution['inc'] * solution['prop']
    return (
        jnp.array([
            solution['pos_M'][3:10].sum() / solution['prop'][3:10].sum(), # Prev 2 - 10
            solution['pos_M'][10:].sum() / solution['prop'][10:].sum(), # Prev 10+
        ]),
        jnp.array([
            solution['inc'][:5].sum() / solution['prop'][:5].sum(), # Inc 0 - 5
            solution['inc'][5:15].sum() / solution['prop'][5:15].sum(), # Inc 5 - 15
            solution['inc'][15:].sum() / solution['prop'][15:].sum() # Inc 15+
        ])
    )
```

```{code-cell} ipython3
prev_stats_multisite = vmap(
    lambda params, eta, eir, impl: prev_stats(impl(params, eta, eir)),
    in_axes=[None, 0, 0, None]
)
```

```{code-cell} ipython3
EIRs = jnp.array([0.05, 3.9, 15., 20., 100., 150., 418.])
key, key_i = random.split(key)
etas = 1. / random.uniform(key_i, shape=(n_sites,), minval=40*365, maxval=100*365, dtype=jnp.float64)
n_sites = len(EIRs)
```

```{code-cell} ipython3
def weak_model(true_EIR=None, prev=None, inc=None, impl=lambda p, e, a: prev_stats_multisite(p, e, a, full_solution)):
    with numpyro.plate('sites', n_sites):
        EIR = numpyro.sample('EIR', dist.Uniform(0., 500.), obs=true_EIR)
    
    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.Uniform(0., 10.))
    ub = numpyro.sample('ub', dist.Uniform(0., 10.))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    IB0 = numpyro.sample('IB0', dist.Uniform(0., 1000.))
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.Uniform(0., 10.))
    uc = numpyro.sample('uc', dist.Uniform(0., 10.))
    phi0 = numpyro.sample('phi0', dist.Beta(1., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 1.))
    IC0 = numpyro.sample('IC0', dist.Uniform(0., 1000.))
    PM = numpyro.sample('PM', dist.Beta(1., 1.))
    dm = numpyro.sample('dm', dist.Uniform(0., 1000.))
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.Uniform(0., 10.))
    ud = numpyro.sample('ud', dist.Uniform(0., 10.))
    d1 = numpyro.sample('d1', dist.Beta(1., 1.))
    ID0 = numpyro.sample('ID0', dist.Uniform(0., 1000.))
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gd = numpyro.sample('gd', dist.Uniform(0., 10.))
    ad0 = numpyro.sample('ad0', dist.Uniform(0., 100.*365.))
    
    ru = numpyro.sample('rU', dist.Uniform(0., 10.))

    x = {
        'kb': kb,
        'ub': ub,
        'b0': b0,
        'IB0': IB0,
        'kc': kc,
        'uc': uc,
        'IC0': IC0,
        'phi0': phi0,
        'phi1': phi1,
        'PM': PM,
        'dm': dm,
        'kd': kd,
        'ud': ud,
        'd1': d1,
        'ID0': ID0,
        'fd0': fd0,
        'gd': gd,
        'ad0': ad0,
        'rU': ru
    }
    
    prev_stats, inc_stats = impl(x, EIR, etas)
    
    numpyro.sample(
        'obs_prev',
        dist.Independent(
            dist.Binomial(total_count=prev_N, probs=prev_stats, validate_args=True),
            1
        ),
        obs=prev
    )

    numpyro.sample(
        'obs_inc',
        dist.Independent(
            dist.Poisson(rate=jnp.maximum(inc_stats * person_risk_time, 1e-12)),
            1
        ),
        obs=inc
    )
```

```{code-cell} ipython3
def model(true_EIR=None, prev=None, inc=None, impl=lambda p, e, a: prev_stats_multisite(p, e, a, full_solution)):
    with numpyro.plate('sites', n_sites):
        EIR = numpyro.sample('EIR', dist.Uniform(0., 500.), obs=true_EIR)
    
    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.LogNormal(0., .1))
    ub = numpyro.sample('ub', dist.LogNormal(0., 1.))
    b0 = numpyro.sample('b0', dist.Beta(5., 1.))
    IB0 = numpyro.sample('IB0', dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.))
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., .1))
    uc = numpyro.sample('uc', dist.LogNormal(0., 1.))
    phi0 = numpyro.sample('phi0', dist.Beta(5., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 2.))
    IC0 = numpyro.sample('IC0',dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.))
    PM = numpyro.sample('PM', dist.Beta(1., 1.))
    dm = numpyro.sample('dm', dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.))
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., .1))
    ud = numpyro.sample('ud', dist.LogNormal(0., 1.))
    d1 = numpyro.sample('d1', dist.Beta(1., 2.))
    ID0 = numpyro.sample('ID0', dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.))
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gd = numpyro.sample('gd', dist.LogNormal(0., .1))
    ad0 = numpyro.sample('ad0', dist.TruncatedDistribution(
            dist.Cauchy(70. * 365., 365.),
            low=40. * 365.,
            high=100. * 365.
        )
    )
    
    ru = numpyro.sample('rU', dist.LogNormal(0., 1.))
    
    x = {
        'kb': kb,
        'ub': ub,
        'b0': b0,
        'IB0': IB0,
        'kc': kc,
        'uc': uc,
        'IC0': IC0,
        'phi0': phi0,
        'phi1': phi1,
        'PM': PM,
        'dm': dm,
        'kd': kd,
        'ud': ud,
        'd1': d1,
        'ID0': ID0,
        'fd0': fd0,
        'gd': gd,
        'ad0': ad0,
        'rU': ru
    }
    
    prev_stats, inc_stats = impl(x, EIR, etas)
    
    numpyro.sample(
        'obs_prev',
        dist.Independent(
            dist.Binomial(total_count=prev_N, probs=prev_stats, validate_args=True),
            2
        ),
        obs=prev
    )

    numpyro.sample(
        'obs_inc',
        dist.Independent(
            dist.Poisson(rate=jnp.maximum(inc_stats * person_risk_time, 1e-12)),
            2
        ),
        obs=inc
    )
```

```{code-cell} ipython3
key, key_i = random.split(key)
true_values = Predictive(model, num_samples=1)(key_i, true_EIR=EIRs)
```

```{code-cell} ipython3
obs_inc, obs_prev = (true_values['obs_inc'], true_values['obs_prev'])
```

```{code-cell} ipython3
print(pd.DataFrame(
    jnp.vstack([EIRs, etas, obs_prev.reshape((len(EIRs), 2)).T, obs_inc.reshape((len(EIRs), 3)).T]).T,
    columns=['EIR', 'eta', 'prev_2_10', 'prev_10+', 'inc_0_5', 'inc_5_15', 'inc_15+']
).to_latex(index=False))
```

```{code-cell} ipython3
def without_obs(params):
    return {k : v for k, v in params.items() if not k in {'obs_inc', 'obs_prev'}}
```

```{code-cell} ipython3
key, key_i = random.split(key)
prior = Predictive(model, num_samples=1000)(key_i)
```

```{code-cell} ipython3
key, key_i = random.split(key)
weak_prior = Predictive(weak_model, num_samples=1000)(key_i)
```

```{code-cell} ipython3
from jax import jacfwd
from jax.tree_util import tree_map
```

```{code-cell} ipython3
from numpyro.infer.util import log_density
```

```{code-cell} ipython3
def densities(p, model):
    ld = log_density(model, [], {'prev': obs_prev, 'inc': obs_inc}, p)
    return (
        ld[0],
        ld[1]['obs_prev']['fn'].base_dist.log_prob(ld[1]['obs_prev']['value']),
        ld[1]['obs_inc']['fn'].base_dist.log_prob(ld[1]['obs_inc']['value'])
    )
```

```{code-cell} ipython3
with jax.default_device(cpu_device):
    sensitivity = vmap(jacfwd(densities), in_axes=[tree_map(lambda _: 0, without_obs(prior)), None])(without_obs(prior), model)
```

```{code-cell} ipython3
with jax.default_device(cpu_device):
    weak_sensitivity = vmap(jacfwd(densities), in_axes=[tree_map(lambda _: 0, without_obs(weak_prior)), None])(without_obs(weak_prior), weak_model)
```

```{code-cell} ipython3
import pandas as pd
age_groups = [
    ['prev_2_10', 'prev_10+'],
    ['inc_0_5', 'inc_5_15', 'inc_15+']
]
sensitivity_df = pd.concat([
    pd.DataFrame({
        'parameter': parameter,
        'gradient': sensitivity[0][parameter]
    })
    for parameter in sensitivity[0].keys()
    if parameter != 'EIR'
])
sensitivity_age_group_df = pd.concat([
    pd.DataFrame({
        'EIR': float(EIRs[j]),
        'eta': float(etas[j]),
        'age_group': age_group,
        'parameter': parameter,
        'gradient': sensitivity[i+1][parameter][:, j, a_i]}
    )
    for j in range(len(EIRs))
    for i, prev in enumerate(['prev', 'inc'])
    for a_i, age_group in enumerate(age_groups[i])
    for parameter in sensitivity[i].keys()
    if parameter != 'EIR'
])
```

```{code-cell} ipython3
weak_sensitivity_df = pd.concat([
    pd.DataFrame({
        'parameter': parameter,
        'gradient': weak_sensitivity[0][parameter]
    })
    for parameter in weak_sensitivity[0].keys()
    if parameter != 'EIR'
])
```

```{code-cell} ipython3
import seaborn as sns
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(19.7, 8.27))
sns.barplot(
    pd.concat([sensitivity_df.assign(priors='tuned'), weak_sensitivity_df.assign(priors='weak')]),
    x='parameter',
    y='gradient',
    hue='priors',
    estimator=lambda x: jnp.mean(jnp.abs(jnp.array(x))),
    errorbar=('ci', 95),
    ax=ax
)
ax.set_xlabel('Intrinsic Parameter')
ax.set_ylabel('Absolute Mean gradient')
ax.set_title('Sensitivity of Priors')
ax.set_yscale('log')
```

```{code-cell} ipython3
norm = plt.Normalize(sensitivity_age_group_df.EIR.min(), sensitivity_age_group_df.EIR.max())
sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
fig, ax = plt.subplots(figsize=(19.7, 8.27))
sns.barplot(
    sensitivity_age_group_df,
    x='parameter',
    y='gradient',
    hue='EIR',
    estimator=lambda x: jnp.mean(jnp.abs(jnp.array(x))),
    errorbar=('ci', 95),
    palette='Reds',
    ax=ax
)
ax.get_legend().remove()
cbar = ax.figure.colorbar(sm, ax=ax)
cbar.ax.set_ylabel('EIR')
ax.set_xlabel('Intrinsic Parameter')
ax.set_ylabel('Absolute Mean Gradient')
ax.set_yscale('log')
ax.set_title('Sensitivity of Tuned Priors Disaggregated by EIR')
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(19.7, 8.27))
sns.barplot(
    sensitivity_age_group_df,
    x='parameter',
    y='gradient',
    hue='age_group',
    estimator=lambda x: jnp.mean(jnp.abs(jnp.array(x))),
    errorbar=('ci', 95),
    ax=ax
)
ax.set_xlabel('Intrinsic Parameter')
ax.set_ylabel('Absolute Mean Gradient')
ax.set_yscale('log')
ax.set_title('Sensitivity of Tuned Priors Disaggregated by Model Output')
```

```{code-cell} ipython3
def get_immunity_curve(params):
    exposures = jnp.arange(100)
    (b0, IB0, kb) = params['b0'], params['IB0'], params['kb']
    (phi0, phi1, IC0, kc) = params['phi0'], params['phi1'], params['IC0'], params['kc']
    (d1, ID0, fd0, gd, ad0, kd) = params['d1'], params['ID0'], params['fd0'], params['gd'], params['ad0'], params['kd']
    b1 = dmeq.default_parameters()['b1']
    a = 5 * 365
    fd = 1-(1-fd0)/(1+(a/ad0)**gd)
    return {
        'prob_b': b0 * ((1 - b1)/(1 + (exposures/IB0)**kb) + b1),
        'prob_c': phi0 * ((1 - phi1)/(1 + (exposures/IC0)**kc) + phi1),
        'prob_d': d1 + (1 - d1)/(1 + fd * (exposures/ID0)**kd)
    }
```

```{code-cell} ipython3
def get_batch_imm_curves(params):
    return vmap(
        get_immunity_curve,
        in_axes=[
            {k: 0 for k in params.keys()}
        ]
    )(params)
```

```{code-cell} ipython3
prior_imm_curves = get_batch_imm_curves(prior)
weak_prior_imm_curves = get_batch_imm_curves(weak_prior)
true_imm_curves = get_batch_imm_curves(without_obs(true_values))
```

```{code-cell} ipython3
n_curves = 500
fig, axs = plt.subplots(1, 3)
imm_labels = ['prob_b', 'prob_c', 'prob_d']

for imm_i, imm in enumerate(imm_labels):
    axs[imm_i].plot(prior_imm_curves[imm][:n_curves, :].T, color='r', alpha=.1)
    axs[imm_i].plot(true_imm_curves[imm][0, :])
    axs[imm_i].set_ylabel(imm)
        
fig.tight_layout()
fig.text(0.5, 0, 'Exposures (number)', ha='center')
fig.text(0.5, 1, 'Prior immunity probability function', ha='center')
```

```{code-cell} ipython3
n_curves = 500
fig, axs = plt.subplots(1, 3)
imm_labels = ['prob_b', 'prob_c', 'prob_d']

for imm_i, imm in enumerate(imm_labels):
    axs[imm_i].plot(weak_prior_imm_curves[imm][:n_curves, :].T, color='r', alpha=.1)
    axs[imm_i].plot(true_imm_curves[imm][0, :])
    axs[imm_i].set_ylabel(imm)
        
fig.tight_layout()
fig.text(0.5, 0, 'Exposures (number)', ha='center')
fig.text(0.5, 1, 'Weak prior immunity probability function', ha='center')
```

```{code-cell} ipython3
def get_curves(params, eirs, etas):
    return vmap(
        vmap(
            full_solution,
            in_axes=[
                {k: 0 for k in params.keys()},
                None,
                None
            ]
        ),
        in_axes=[None, 0, 0]
    )(params, eirs, etas)

prior_curves = get_curves(prior, EIRs, etas)
weak_prior_curves = get_curves(weak_prior, EIRs, etas)
true_curves = get_curves(without_obs(true_values), EIRs, etas)
```

```{code-cell} ipython3
n_curves = 500
fig, axs = plt.subplots(3, len(EIRs), sharey='row', sharex=True)
imm_labels = ['prob_b', 'prob_c', 'prob_d']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm_i, imm in enumerate(imm_labels):
        axs[imm_i, i].plot(prior_curves[imm][i, :n_curves, :].T, color='r', alpha=.01)
        axs[imm_i, i].plot(true_curves[imm][i, 0, :])
        axs[imm_i, 0].set_ylabel(imm)
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Prior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, len(EIRs), sharey='row', sharex=True)
imm_labels = ['prob_b', 'prob_c', 'prob_d']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm_i, imm in enumerate(imm_labels):
        axs[imm_i, i].plot(weak_prior_curves[imm][i, :n_curves, :].T, color='r', alpha=.01)
        axs[imm_i, i].plot(true_curves[imm][i, 0, :])
        axs[imm_i, 0].set_ylabel(imm)
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Weak Prior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey='row', sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for prev_i, prev in enumerate(prev_labels):
        axs[0, i].set_xlabel(
            f'EIR: {EIRs[i]}'
        )
        axs[0, i].xaxis.set_label_position('top')
        axs[prev_i, i].plot(prior_curves[prev][i, :n_curves, :].T / prior_curves['prop'][i, :n_curves, :].T, color='r', alpha=.01)
        axs[prev_i, i].plot(true_curves[prev][i, 0, :] / true_curves['prop'][i, 0, :].T)
        axs[prev_i, 0].set_ylabel(prev)
        #axs[prev_i, 0].set_yscale('log')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Prior pos_M/inc function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey='row', sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for prev_i, prev in enumerate(prev_labels):
        axs[0, i].set_xlabel(
            f'EIR: {EIRs[i]}'
        )
        axs[0, i].xaxis.set_label_position('top')
        axs[prev_i, i].plot(weak_prior_curves[prev][i, :n_curves, :].T / prior_curves['prop'][i, :n_curves, :].T, color='r', alpha=.01)
        axs[prev_i, i].plot(true_curves[prev][i, 0, :] / true_curves['prop'][i, 0, :].T)
        axs[prev_i, 0].set_ylabel(prev)
        #axs[prev_i, 0].set_yscale('log')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Weak Prior pos_M/inc function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(5, len(EIRs), figsize=(50, 80))

pp = jnp.concatenate([prior['obs_prev'], prior['obs_inc']], axis=2)
obs = jnp.concatenate([obs_prev, obs_inc], axis=2)

for i, e in enumerate(EIRs):
    for j, a in enumerate(['prev_2_10', 'prev_10+','inc_0_5', 'inc_5_15', 'inc_15+']):
        axs[0, i].set_xlabel(
            f'EIR: {e}'
        )
        axs[0, i].xaxis.set_label_position('top')
        axs[j, 0].set_ylabel(a)
        axs[j, i].hist(pp[:, i, j])
        axs[j, i].vlines(
            obs[0, i, j],
            0,
            axs[j, i].get_ylim()[1],
            color = 'red',
            linestyle = 'dashed'
        )
```

```{code-cell} ipython3
n_samples = 100
n_warmup = 100

mcmc = MCMC(
    NUTS(model),
    num_samples=n_samples,
    num_warmup=n_warmup,
    num_chains=n_chains,
    chain_method='parallel'
)
mcmc.run(key, None, obs_prev, obs_inc)
mcmc.print_summary(prob=0.7)
```

```{code-cell} ipython3
posterior_samples = mcmc.get_samples()
posterior_predictive = Predictive(
    model,
    posterior_samples
)(key_i)
```

```{code-cell} ipython3
pyro_data = az.from_numpyro(
    mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)
```

```{code-cell} ipython3
az.rcParams["plot.max_subplots"] = 200
keys = list(pyro_data.prior.data_vars.keys())
axs = az.plot_dist_comparison(pyro_data)
for i, key in enumerate(keys):
    if key == 'EIR':
        for j in range(n_sites):
            axs[j, 2].vlines(
                EIRs[j],
                0,
                axs[j, 2].get_ylim()[1],
                color = 'red',
                linestyle = 'dashed'
            )
    else:
        j = i + n_sites - 1
        axs[j, 2].vlines(
            true_values[key][0],
            0,
            axs[j, 2].get_ylim()[1],
            color = 'red',
            linestyle = 'dashed'
        )
```

```{code-cell} ipython3
posterior_curves = get_curves(posterior_samples, EIRs, etas)
```

```{code-cell} ipython3
posterior_imm_curves = get_batch_imm_curves(posterior_samples)
```

```{code-cell} ipython3
n_curves = 500
fig, axs = plt.subplots(1, 3)
imm_labels = ['prob_b', 'prob_c', 'prob_d']

for imm_i, imm in enumerate(imm_labels):
    axs[imm_i].plot(posterior_imm_curves[imm][:n_curves, :].T, color='r', alpha=.1)
    axs[imm_i].plot(true_imm_curves[imm][0, :])
    axs[imm_i].set_ylabel(imm)
        
fig.tight_layout()
fig.text(0.5, 0, 'Exposures (number)', ha='center')
fig.text(0.5, 1, 'Posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, len(EIRs), sharey='row', sharex=True)
imm_labels = ['b', 'c', 'd']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm_i, imm in enumerate([f'prob_{l}' for l in imm_labels]):
        axs[imm_i, i].plot(posterior_curves[imm][i, :, :].T, color='r', alpha=.01)
        axs[imm_i, i].plot(true_curves[imm][i, 0, :].T)
        axs[imm_i, 0].set_ylabel(f'prob. {imm_labels[imm_i]}')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey='row', sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for prev_i, prev in enumerate(prev_labels):
        axs[0, i].set_xlabel(
            f'EIR: {EIRs[i]}'
        )
        axs[0, i].xaxis.set_label_position('top')
        axs[prev_i, i].plot(posterior_curves[prev][i, :, :].T / posterior_curves['prop'][i, :, :].T, color='r', alpha=.01)
        axs[prev_i, i].plot(true_curves[prev][i, 0, :] / true_curves['prop'][i, 0, :])
        axs[prev_i, 0].set_ylabel(prev)
        #axs[prev_i, 0].set_yscale('log')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Posterior pos_M/inc function', ha='center')
```

```{code-cell} ipython3
weak_mcmc = MCMC(
    NUTS(weak_model),
    num_samples=n_samples,
    num_warmup=n_warmup,
    num_chains=n_chains,
    chain_method='parallel'
)
weak_mcmc.run(key, obs_prev, obs_inc)
weak_mcmc.print_summary(prob=0.7)
```

```{code-cell} ipython3
weak_posterior_samples = weak_mcmc.get_samples()
weak_posterior_predictive = Predictive(
    weak_model,
    weak_posterior_samples
)(key, obs_prev, obs_inc)

weak_pyro_data = az.from_numpyro(
    weak_mcmc,
    prior=weak_prior,
    posterior_predictive=weak_posterior_predictive
)
```

```{code-cell} ipython3
az.rcParams["plot.max_subplots"] = 200
keys = list(pyro_data.prior.data_vars.keys())
axs = az.plot_dist_comparison(weak_pyro_data)
for i in range(axs.shape[0]):
    axs[i, 2].vlines(
        true_values[keys[i]][0],
        0,
        axs[i, 2].get_ylim()[1],
        color = 'red',
        linestyle = 'dashed'
    )
```

```{code-cell} ipython3
weak_posterior_curves = get_curves(weak_posterior_samples, EIRs, etas)
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, len(EIRs), sharey='row', sharex=True)
imm_labels = ['b', 'c', 'd']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm_i, imm in enumerate([f'prob_{l}' for l in imm_labels]):
        axs[imm_i, i].plot(weak_posterior_curves[imm][i, :, :].T, color='r', alpha=.01)
        axs[imm_i, i].plot(true_curves[imm][i, 0, :].T)
        axs[imm_i, 0].set_ylabel(f'prob. {imm_labels[imm_i]}')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Weak Posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey='row', sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for prev_i, prev in enumerate(prev_labels):
        axs[0, i].set_xlabel(
            f'EIR: {EIRs[i]}'
        )
        axs[0, i].xaxis.set_label_position('top')
        axs[prev_i, i].plot(weak_posterior_curves[prev][i, :, :].T, color='r', alpha=.01)
        axs[prev_i, i].plot(true_curves[prev][i, 0, :])
        axs[prev_i, 0].set_ylabel(prev)
        #axs[prev_i, 0].set_yscale('log')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Weak Posterior pos_M/inc function', ha='center')
```

```{code-cell} ipython3
posterior_df = pd.concat([
    pd.DataFrame(posterior_samples).assign(priors='tuned'),
    #pd.DataFrame(weak_posterior_samples).assign(priors='weak')
])
```

```{code-cell} ipython3
p = sns.pairplot(posterior_df[['b0', 'ub', 'IB0', 'kb', 'priors']], hue='priors', kind='kde')
p.fig.suptitle('Correlation in the posterior estimates of pre-erythrocytic immunity parameters', y=1.03)
```

```{code-cell} ipython3
p = sns.pairplot(posterior_df[['phi0', 'phi1', 'kc', 'uc', 'IC0', 'PM', 'dm', 'priors']], hue='priors', kind='kde')
p.fig.suptitle('Correlation in the posterior estimates of clinical immunity parameters', y=1.03)
```

```{code-cell} ipython3
p = sns.pairplot(posterior_df[['d1', 'kd', 'ud', 'ID0', 'fd0', 'gd', 'rU', 'priors']], hue='priors', kind='kde')
p.fig.suptitle('Correlation in the posterior estimates of immunity to detection parameters', y=1.08)
```

```{code-cell} ipython3
from numpyro.diagnostics import summary
print(pd.concat([
    pd.DataFrame.from_dict(summary(weak_mcmc.get_samples(group_by_chain=True)), orient='index'),
    pd.DataFrame.from_dict(summary(mcmc.get_samples(group_by_chain=True)), orient='index'),
    pd.DataFrame.from_dict(without_obs(true_values), orient='index', columns=['value']),
], axis=1, keys=['weak', 'tuned', 'latent']).loc[[
        # Pre-erythrocytic immunity
    'kb',
    'ub',
    'b0',
    'IB0',
    
    # Clinical immunity
    'kc',
    'uc',
    'phi0',
    'phi1',
    'IC0',
    'PM',
    'dm',
    
    # Detection immunity
    'kd',
    'ud',
    'd1',
    'ID0',
    'fd0',
    'gd',
    'ad0',
    'rU',
]].to_latex(formatters=[lambda x: '%.3f' % x] * 15))
```

```{code-cell} ipython3
(
    jnp.sum(obs_inc - weak_posterior_predictive['obs_inc'] > 0),
    jnp.sum(obs_prev - weak_posterior_predictive['obs_prev'] > 0)
)
```

```{code-cell} ipython3
fig, axs = plt.subplots(5, len(EIRs), figsize=(50, 80))
n_post = posterior_predictive['obs_prev'].shape[0]

pp = jnp.concatenate([posterior_predictive['obs_prev'], posterior_predictive['obs_inc']], axis=2)
obs = jnp.concatenate([obs_prev, obs_inc], axis=2)

for i, e in enumerate(EIRs):
    for j, a in enumerate(['prev_2_10', 'prev_10+','inc_0_5', 'inc_5_15', 'inc_15+']):
        axs[0, i].set_xlabel(
            f'EIR: {e}'
        )
        axs[0, i].xaxis.set_label_position('top')
        axs[j, 0].set_ylabel(a)
        axs[j, i].hist(pp[:, i, j])
        axs[j, i].vlines(
            obs[0, i, j],
            0,
            axs[j, i].get_ylim()[1],
            color = 'red',
            linestyle = 'dashed'
        )
```

```{code-cell} ipython3
(
    jnp.mean(obs_inc - posterior_predictive['obs_inc']),
    jnp.mean(obs_prev - posterior_predictive['obs_prev'])
)
```

```{code-cell} ipython3

```
