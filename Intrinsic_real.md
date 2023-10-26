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

 * Priors on EIR? from previous estimate
 * Clean the observational data
    * Filter down incidence by frequency of ACD
    * Filter down incidence by PYO
 * Test IBM surrogate accuracy on prior
 * Apply IBM surrogate
   * Site params
   * Update prior predictive checks to use prior EIR

## Insights

 * Priors really favour zero prev/inc
 * dm breaks at 0
 * prob d priors are too restrictive
 * There is possibly more curvature in immunity at low EIRs (likely because of previous interventions)
 * IB0 and ID0 break at zero
 * d1 is multimodal
 * The second site is mostly ignored on prevalence!

```{code-cell} ipython3
cpu_count = 100
import os
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count}'
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, jit, vmap
import jax
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
prev_data = pd.read_csv('./data/prev.csv')
inc_data = pd.read_csv('./data/inc.csv')
age_data = pd.read_csv('./data/average_age.csv')
```

```{code-cell} ipython3
# Filter the sites down
sites = pd.concat([
    prev_data[['iso3c', 'name_1']],
    inc_data[['iso3c', 'name_1']]
]).drop_duplicates().reset_index(drop=True).reset_index()
sites = sites.loc[[0,12,20]]
prev_data = pd.merge(sites, prev_data).drop('index', axis=1)
inc_data = pd.merge(sites, inc_data).drop('index', axis=1)
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
inc_risk_time = jnp.array(inc_data.PYO.values) * 365.
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
def straight_through(f, x):
  # Create an exactly-zero expression with Sterbenz lemma that has
  # an exactly-one gradient.
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(f(x))
```

```{code-cell} ipython3
def model(prev_n=None, prev_pos=None, inc_risk_time=None, inc_n=None, impl=full_solution):
    with numpyro.plate('sites', n_sites):
        q = numpyro.sample('q', dist.Beta(10., 1.))
        mu = numpyro.sample('mu', dist.Gamma(2., 2.))
        z = numpyro.sample('z', dist.Beta(10., 1.))
        
    #    EIR = numpyro.sample('EIR', dist.Uniform(0., 1000.))
    EIR = numpyro.sample('EIR', dist.TruncatedDistribution(dist.Normal(jnp.array([100., .1, 1.]), jnp.array([10., .1, 1.])), 0., 1000.))

    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.LogNormal(0., 1.))
    ub = numpyro.sample('ub', dist.LogNormal(0., 1.))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    IB0 = numpyro.sample('IB0', dist.TruncatedDistribution(dist.Cauchy(50., 10.), low=0., high=100.))
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., 1.))
    uc = numpyro.sample('uc', dist.LogNormal(0., 1.))
    phi0 = numpyro.sample('phi0', dist.Beta(5., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 2.))
    IC0 = numpyro.sample('IC0',dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=1.))
    PM = numpyro.sample('PM', dist.Beta(1., 1.))
    dm = numpyro.sample('dm', dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.))
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., 1.))
    ud = numpyro.sample('ud', dist.LogNormal(0., 1.))
    d1 = numpyro.sample('d1', dist.Beta(1., 5.))
    ID0 = numpyro.sample('ID0', dist.LeftTruncatedDistribution(dist.Cauchy(25., 10.), low=5.))
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gd = numpyro.sample('gd', dist.LogNormal(0., 1.))
    ad0 = numpyro.sample('ad0', dist.TruncatedDistribution(
            dist.Cauchy(50. * 365., 365.),
            low=20. * 365.,
            high=80. * 365.
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
        EIR[prev_i],
        prev_eta,
        prev_lower,
        prev_upper,
        impl
    )

    inc_stats = inc_stats_multisite(
        x,
        EIR[inc_i],
        inc_eta,
        inc_lower,
        inc_upper,
        impl
    )


    round_n = jax.lax.convert_element_type(straight_through(lambda x: jnp.ceil(x), prev_n * z[prev_i]), jnp.int64)
    
    numpyro.sample(
        'obs_prev',
        dist.Independent(
            dist.Binomial(
                total_count=round_n,
                probs=prev_stats * z[prev_i], validate_args=True),
            1
        ),
        obs=prev_pos
    )

    mean = straight_through(lambda x: jnp.maximum(x, 1e-12), inc_stats * inc_risk_time * mu[inc_i])

    numpyro.sample(
        'obs_inc',
        dist.Independent(
            dist.GammaPoisson(mean * q[inc_i], q[inc_i], validate_args=True),
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
prior = Predictive(model, num_samples=100)(key_i, prev_n=prev_data.N.values, inc_risk_time=inc_risk_time)
```

```{code-cell} ipython3
from jax import jacfwd
from jax.tree_util import tree_map
```

```{code-cell} ipython3
from numpyro.infer.util import log_density
```

```{code-cell} ipython3
def densities(p):
    ld = log_density(
        model,
        [],
        {
            'prev_n': prev_data.N.values,
            'prev_pos': prev_data.N_POS.values,
            'inc_risk_time': inc_risk_time,
            'inc_n': inc_data.d.values
        },
        p
    )
    return ld[0]
```

```{code-cell} ipython3
sensitivity = jax.jit(vmap(jacfwd(densities), in_axes=[tree_map(lambda _: 0, without_obs(prior))]))(without_obs(prior))
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
EIRs = jnp.array([100., .1, 1.])
etas = 1/(eta.average_age.values * 365.)#jnp.full(EIRs.shape, dmeq.default_parameters()['eta'])#eta.average_age.values
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
true_curves = get_curves(tree_map(lambda l: jnp.array([l]), dmeq.default_parameters()), EIRs, etas)

fig, axs = plt.subplots(3, len(EIRs), sharey='row', sharex=True)
imm_labels = ['prob_b', 'prob_c', 'prob_d']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm_i, imm in enumerate(imm_labels):
        axs[imm_i, i].plot(prior_curves[imm][i, :, :].T, color='r', alpha=.1)
        axs[imm_i, i].plot(true_curves[imm][i, 0, :])
        axs[imm_i, 0].set_ylabel(imm)
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Prior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey=False, sharex=True)

prev_labels = ['pos_M', 'inc']
lookup = [prev_i, inc_i]
a = [
    jnp.mean(prev_data[['PR_LAR', 'PR_UAR']].values, axis=1),
    jnp.mean(inc_data[['INC_LAR', 'INC_UAR']].values, axis=1)
]
p = [
    prev_data.N_POS / prev_data.N,
    inc_data.d / inc_risk_time
]
for i in range(len(EIRs)):
    for j, prev in enumerate(prev_labels):
        axs[0, i].set_xlabel(
            f'EIR: {EIRs[i]}'
        )
        site_a = a[j][lookup[j] == i]
        site_p = p[j][lookup[j] == i]
        axs[0, i].xaxis.set_label_position('top')
        axs[j, i].plot(prior_curves[prev][i, :, :].T / prior_curves['prop'][i, :, :].T, color='r', alpha=.1)
        axs[j, i].plot(true_curves[prev][i, 0, :] / true_curves['prop'][i, 0, :].T)
        axs[j, i].scatter(site_a, site_p, marker='x', color='green')
        axs[j, 0].set_ylabel(prev)
        #axs[prev_i, 0].set_yscale('log')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Prior pos_M/inc function', ha='center')
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
            'inc_risk_time': inc_risk_time,
            'inc_n': inc_data.d.values
        })
mcmc.print_summary(prob=0.7)
```

EIR had priors
Noise and overdispersion applied
7 hours of inference

```
                mean       std    median     15.0%     85.0%     n_eff     r_hat
    EIR[0]     99.84      9.70    100.04     89.83    109.63    136.59      1.07
    EIR[1]      0.10      0.06      0.09      0.04      0.13     46.54      1.20
    EIR[2]      1.96      0.80      1.86      1.17      2.78     10.92      1.47
       IB0     47.51     15.54     47.81     35.68     61.69     57.12      1.16
       IC0     86.80     34.20     96.13     81.17    117.13      6.87      1.97
       ID0      9.57      9.84      5.37      0.02     14.61     18.31      1.42
        PM      0.54      0.28      0.59      0.35      0.94     37.28      1.24
       ad0  18135.91   1366.17  18212.90  17683.83  18828.45     87.99      1.10
        b0      0.76      0.16      0.77      0.65      0.97     19.35      1.26
        d1      0.14      0.28      0.05      0.00      0.08      5.09      9.16
        dm    185.86    107.12    198.07    168.28    223.77     14.93      1.32
       fd0      0.04      0.06      0.02      0.00      0.04      5.93      2.92
        gd      2.53      0.90      2.40      1.44      3.03     81.95      1.14
        kb      1.33      1.36      0.92      0.21      1.54     47.42      1.16
        kc      3.78      2.44      3.00      1.26      4.29     61.09      1.16
        kd      1.19      0.59      1.02      0.49      1.45     22.08      1.33
     mu[0]      0.19      0.09      0.16      0.09      0.22     18.60      1.27
     mu[1]      0.88      0.72      0.64      0.29      0.94      6.22      2.69
     mu[2]      1.25      0.92      1.05      0.07      1.56     12.09      1.53
      phi0      0.73      0.17      0.74      0.65      1.00     31.12      1.27
      phi1      0.02      0.02      0.01      0.00      0.03     54.81      1.18
      q[0]      0.88      0.11      0.92      0.84      1.00     11.08      1.47
      q[1]      0.89      0.09      0.90      0.84      1.00     21.48      1.23
      q[2]      0.89      0.09      0.90      0.83      1.00     45.02      1.16
        rU      1.24      1.39      0.82      0.18      1.27     53.31      1.18
        ub      1.46      1.86      0.94      0.06      1.81    108.39      1.09
        uc      1.72      1.58      1.08      0.17      2.06     13.93      1.48
        ud      2.15      2.69      1.42      0.27      2.06     77.17      1.13
      z[0]      0.92      0.11      0.96      0.93      1.00      5.65      3.40
      z[1]      0.48      0.10      0.46      0.37      0.55     63.63      1.21
      z[2]      0.85      0.11      0.84      0.78      1.00      8.99      1.62
```

```{code-cell} ipython3
posterior_samples = mcmc.get_samples()
posterior_predictive = Predictive(
    model,
    posterior_samples
)(key, **{
            'prev_n': prev_data.N.values,
            'prev_pos': prev_data.N_POS.values,
            'inc_risk_time': inc_data.PYO.values,
            'inc_n': inc_data.d.values
        })
```

```{code-cell} ipython3
pyro_data = az.from_numpyro(
    mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)
```

```{code-cell} ipython3
ax = az.plot_ppc(pyro_data, group="posterior")
```

```{code-cell} ipython3
axs = az.plot_trace(pyro_data)
```

```{code-cell} ipython3
az.rcParams["plot.max_subplots"] = 200
keys = [k for k in pyro_data.prior.data_vars.keys()]
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
#EIRs = jnp.array([0.05, 3.9, 15., 20., 100., 150., 418.])
#etas = 1. / random.uniform(key_i, shape=EIRs.shape, minval=20*365, maxval=40*365, dtype=jnp.float64)
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
        axs[imm_i, i].plot(true_curves[imm][i, 0, :])
        axs[imm_i, 0].set_ylabel(f'prob. {imm_labels[imm_i]}')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey=False, sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for j, prev in enumerate(prev_labels):
        axs[0, i].set_xlabel(
            f'EIR: {EIRs[i]}'
        )
        #axs[0, i].xaxis.set_label_position('top')
        #axs[prev_i, i].plot(posterior_curves[prev][i, :, :].T, color='r', alpha=.01)
        #axs[prev_i, 0].set_ylabel(prev)
        #axs[prev_i, 0].set_yscale('log')
        site_a = a[j][lookup[j] == i]
        site_p = p[j][lookup[j] == i]
        axs[0, i].xaxis.set_label_position('top')
        axs[j, i].plot(posterior_curves[prev][i, :, :].T / posterior_curves['prop'][i, :, :].T, color='r', alpha=.01)
        axs[j, i].plot(true_curves[prev][i, 0, :] / true_curves['prop'][i, 0, :].T)
        axs[j, i].scatter(site_a, site_p, marker='x', color='green')
        axs[j, 0].set_ylabel(prev)
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Posterior pos_M/inc function', ha='center')
```

```{code-cell} ipython3

```
