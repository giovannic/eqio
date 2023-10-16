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

# TODO

 * Interpolate full solution
 * Adapt full solution to get prev and inc for each site each year

```{code-cell} ipython3
cpu_count = 50
import os
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count}'
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, jit, vmap
import jax

jax.config.update('jax_platform_name', 'cpu') # for memory purposes
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
prev_data = pd.read_csv('./data/prev.csv')
inc_data = pd.read_csv('./data/inc.csv')
age_data = pd.read_csv('./data/average_age.csv')
```

```{code-cell} ipython3
def mask(xs, ns, age_lower, age_upper):
    age_mask = jnp.arange(xs.shape[0])[jnp.newaxis, :]
    age_mask = (age_mask >= age_lower) & (age_mask <= age_upper)
    return(
        jnp.where(age_mask, xs, 0),
        jnp.where(age_mask, ns, 0)
    )

def est_prev_pos(params, eir, eta, lower, upper, impl=full_solution):
    solution = impl(params, eir, eta)
    pos_M, prop = mask(solution['pos_M'], solution['prop'], lower, upper)
    return jnp.sum(pos_M) / jnp.sum(prop)

def est_inc_rate(params, eir, eta, lower, upper, impl=full_solution):
    solution = impl(params, eir, eta)
    inc, prop = mask(solution['inc'], solution['prop'], lower, upper)
    return jnp.sum(inc) / jnp.sum(prop)
```

```{code-cell} ipython3
sites = pd.concat([
    prev_data[['iso3c', 'name_1']],
    inc_data[['iso3c', 'name_1']]
]).drop_duplicates().reset_index(drop=True).reset_index()
```

```{code-cell} ipython3
inc_index = pd.merge(sites, inc_data)
prev_index = pd.merge(sites, prev_data)
```

```{code-cell} ipython3
mean_age = age_data.groupby('iso3c').agg({'average_age': 'mean'}).reset_index()
eta = pd.merge(sites, mean_age)
inc_i = inc_index['index'].values
prev_i = prev_index['index'].values
inc_eta = 1 / (eta.loc[inc_i].average_age.values * 365)
prev_eta = 1 / (eta.loc[prev_i].average_age.values * 365)
prev_lower = jnp.floor(prev_data.PR_LAR.values)
prev_upper = jnp.ceil(prev_data.PR_UAR.values)
inc_lower = jnp.floor(inc_data.INC_LAR.values)
inc_upper = jnp.ceil(inc_data.INC_UAR.values)
n_sites = len(sites)
```

```{code-cell} ipython3
prev_stats_multisite = vmap(
    est_prev_pos,
    in_axes=[None, 0, 0, 0, 0, None]
)

inc_stats_multisite = vmap(
    est_inc_rate,
    in_axes=[None, 0, 0, 0, 0, None]
)
```

```{code-cell} ipython3
def model(prev_n=None, prev_pos=None, inc_risk_time=None, inc_n=None, impl=full_solution):
    est_EIR = numpyro.sample('EIR', dist.Uniform(jnp.zeros((n_sites,)), jnp.full((n_sites,), 500.)))

    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.LogNormal(0., 1.))
    ub = numpyro.sample('ub', dist.LogNormal(0., 1.))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    IB0 = numpyro.sample('IB0', dist.LeftTruncatedDistribution(dist.Cauchy(50., 10.), low=0.))
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., 1.))
    uc = numpyro.sample('uc', dist.LogNormal(0., 1.))
    phi0 = numpyro.sample('phi0', dist.Beta(5., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 2.))
    IC0 = numpyro.sample('IC0',dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.))
    PM = numpyro.sample('PM', dist.Beta(1., 1.))
    dm = numpyro.sample('dm', dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.))
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., 1.))
    ud = numpyro.sample('ud', dist.LogNormal(0., 1.))
    d1 = numpyro.sample('d1', dist.Beta(1., 2.))
    ID0 = numpyro.sample('ID0', dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.))
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gd = numpyro.sample('gd', dist.LogNormal(0., 1.))
    ad0 = numpyro.sample('ad0', dist.TruncatedDistribution(
            dist.Cauchy(30. * 365., 365.),
            low=20. * 365.,
            high=40. * 365.
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

    prev_stats = prev_stats_multisite(
        x,
        est_EIR[prev_i],
        prev_eta,
        prev_lower,
        prev_upper,
        impl
    )

    inc_stats = inc_stats_multisite(
        x,
        est_EIR[inc_i],
        inc_eta,
        inc_lower,
        inc_upper,
        impl
    )

    numpyro.sample(
        'obs_prev',
        dist.Independent(
            dist.Binomial(total_count=prev_n, probs=prev_stats, validate_args=True),
            1
        ),
        obs=prev_pos
    )

    numpyro.sample(
        'obs_inc',
        dist.Independent(
            dist.Poisson(rate=jnp.maximum(inc_stats, 1e-12)*inc_risk_time, validate_args=True),
            1
        ),
        obs=inc_n
    )
```

```{code-cell} ipython3
def without_obs(params):
    return {k : v for k, v in params.items() if not k in {'obs_inc', 'obs_prev'}}
```

```{code-cell} ipython3
key, key_i = random.split(key)
prior = Predictive(model, num_samples=100)(key_i, prev_n=prev_data.N.values, inc_risk_time=inc_data.PYO.values)
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
    ld = log_density(
        model,
        [],
        {
            'prev_n': prev_data.N.values,
            'prev_pos': prev_data.N_POS.values,
            'inc_risk_time': inc_data.PYO.values,
            'inc_n': inc_data.d.values
        },
        p
    )
    return ld[0]
```

```{code-cell} ipython3
sensitivity = vmap(jacfwd(densities), in_axes=[tree_map(lambda _: 0, without_obs(prior)), None])(without_obs(prior), model)
```

```{code-cell} ipython3
sensitivity_df = pd.concat([
    pd.DataFrame({
        'parameter': parameter,
        'gradient': sensitivity[parameter]
    })
    for parameter in sensitivity.keys()
    if parameter != 'EIR'
])
```

```{code-cell} ipython3
import seaborn as sns
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(19.7, 8.27))
sns.barplot(
    sensitivity_df,
    x='parameter',
    y='gradient',
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
az_prior = az.from_dict(
    observed_data={"obs_prev": prev_data.N_POS.values[None,:], "obs_inc": inc_data.d.values[None,:]},
    prior_predictive={"obs_prev": prior['obs_prev'][None,:,:], "obs_inc": prior['obs_inc'][None,:,:]}
)
```

```{code-cell} ipython3
ax = az.plot_ppc(az_prior, group="prior")
```

```{code-cell} ipython3
plt.hist(prior['obs_inc'].reshape(-1), bins=100)
```

```{code-cell} ipython3
plt.hist(inc_data.d, bins=100)
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
mcmc.run(key, **{
            'prev_n': prev_data.N.values,
            'prev_pos': prev_data.N_POS.values,
            'inc_risk_time': inc_data.PYO.values,
            'inc_n': inc_data.d.values
        })
mcmc.print_summary(prob=0.7)
```

```{code-cell} ipython3
posterior_samples = mcmc.get_samples()
posterior_predictive = Predictive(
    model,
    posterior_samples
)(key, obs_prev, obs_inc)
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
keys = [k for k in pyro_data.prior.data_vars.keys() if key != 'EIR']
axs = az.plot_dist_comparison(pyro_data)
#for i in range(axs.shape[0]):
#    axs[i, 2].vlines(
#        true_values[keys[i]][0],
#        0,
#        axs[i, 2].get_ylim()[1],
#        color = 'red',
#        linestyle = 'dashed'
#    )
```

```{code-cell} ipython3
posterior_curves = get_curves(posterior_samples, EIRs, etas)
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
        axs[prev_i, i].plot(posterior_curves[prev][i, :, :].T, color='r', alpha=.01)
        axs[prev_i, i].plot(true_curves[prev][i, 0, :])
        axs[prev_i, 0].set_ylabel(prev)
        #axs[prev_i, 0].set_yscale('log')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Posterior pos_M/inc function', ha='center')
```
