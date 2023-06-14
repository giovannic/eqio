---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# TODO

 * mox development
   * ~make python package~
   * ~write tests~
   * ~build documentation~
   * ~make homepage~
   * ~translate this notebook to jupytext~
   * ~use mox primitives?~
 * ~investigate du -> rU~
 * ~investigate cu/cd -> cU/cD~
 * investigate feature learning and context decoders
   * or the alternative loss function where you aggregate the output?
 * investigate robust training
   * regularisation (dropout/ l2)
   * gradient towards high loss
 * investigate history matching

```{code-cell} ipython3
cpu_count = 100
import os
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count}'
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, jit, vmap
```

```{code-cell} ipython3
import dmeq
from mox.sampling import LHSStrategy
```

```{code-cell} ipython3
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
import arviz as az
import pandas as pd
```

```{code-cell} ipython3
n_chains = 10
```

```{code-cell} ipython3
key = random.PRNGKey(42)
```

```{code-cell} ipython3
def full_solution(params, eir, eta):
    p = dmeq.default_parameters()
    for k, v in params.items():
        p[k] = v
    p['EIR'] = eir
    p['eta'] = eta
    s = dmeq.solve(p, dtype=jnp.float64)
    return {
        'pos_M': s[0],
        'inc': s[1],
        'prob_b': s[3],
        'prob_c': s[4],
        'prob_d': s[5],
        'prop': s[2],
    }
```

```{code-cell} ipython3
population = 1_000_000
prev_N = 1_000

def prev_stats(solution):
    inc_rates = solution['inc'] * solution['prop'] * population
    return (
        jnp.array([
            solution['pos_M'][3:10].sum() / solution['prop'][3:10].sum(), # Prev 2 - 10
            solution['pos_M'][10:].sum() / solution['prop'][10:].sum(), # Prev 10+
        ]),
        jnp.array([
            inc_rates[:5].sum(), # Inc 0 - 5
            inc_rates[5:15].sum(), # Inc 5 - 15
            inc_rates[15:].sum() # Inc 15+
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
etas = 1. / random.uniform(key_i, shape=EIRs.shape, minval=20*365, maxval=40*365, dtype=jnp.float64)
```

```{code-cell} ipython3
def model(prev=None, inc=None, impl=lambda p, e, a: prev_stats_multisite(p, e, a, full_solution)):
    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.LogNormal(0., .25))
    ub = numpyro.sample('ub', dist.LogNormal(0., .25))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    IB0 = numpyro.sample('IB0', dist.LeftTruncatedDistribution(dist.Normal(50., 10.), low=0.))
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., .25))
    uc = numpyro.sample('uc', dist.LogNormal(0., .25))
    phi0 = numpyro.sample('phi0', dist.Beta(5., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 2.))
    IC0 = numpyro.sample('IC0', dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.))
    PM = numpyro.sample('PM', dist.Beta(1., 1.))
    dm = numpyro.sample('dm', dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.))
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., .25))
    ud = numpyro.sample('ud', dist.LogNormal(0., .25))
    d1 = numpyro.sample('d1', dist.Beta(1., 2.))
    ID0 = numpyro.sample('ID0', dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.))
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gd = numpyro.sample('gd', dist.LogNormal(0., .25))
    ad0 = numpyro.sample('ad0', dist.TruncatedDistribution(
        dist.Cauchy(30. * 365., 365.),
        low=20. * 365.,
        high=40. * 365.
    ))
    
    ru = numpyro.sample('rU', dist.LogNormal(0., .25))
    
    # FOIM
    cd = numpyro.sample('cD', dist.Beta(1., 2.))
    cu = numpyro.sample('cU', dist.Beta(1., 5.))
    g_inf = numpyro.sample('g_inf', dist.Gamma(3., 1.))
    
    prev_stats, inc_stats = impl({
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
        'rU': ru,
        'cD': cd,
        'cU': cu,
        'g_inf': g_inf
    }, EIRs, etas)
    
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
            dist.Poisson(rate=inc_stats, validate_args=True),
            1
        ),
        obs=inc
    )
```

```{code-cell} ipython3
key, key_i = random.split(key)
true_values = Predictive(model, num_samples=1)(key_i)
```

```{code-cell} ipython3
true_values
```

```{code-cell} ipython3
obs_inc, obs_prev = (true_values['obs_inc'], true_values['obs_prev'])
```

```{code-cell} ipython3
print(pd.DataFrame(
    jnp.vstack([EIRs, etas, obs_prev[0].T, obs_inc[0].T]).T,
    columns=['EIR', 'eta', 'prev_2_10', 'prev_10+', 'inc_0_5', 'inc_5_15', 'inc_15+']
).to_latex(index=False))
```

```{code-cell} ipython3
def without_obs(params):
    return {k : v for k, v in params.items() if not k in {'obs_inc', 'obs_prev'}}
```

```{code-cell} ipython3
key, key_i = random.split(key)
prior = Predictive(model, num_samples=600)(key)
```

```{code-cell} ipython3
from jax import pmap, tree_map
import jax
import pandas as pd
from scipy.stats.qmc import LatinHypercube

train_samples = int(1e4)
device_count = len(jax.devices())
```

```{code-cell} ipython3
# Create the X_lhs dataset
intrinsic_bounds = pd.DataFrame.from_records([
    ('kb', 0, 10),
    ('ub', 0, 10),
    ('b0', 0, 1),
    ('IB0', 0, 100),
    ('kc', 0, 10),
    ('uc', 0, 10),
    ('IC0', 0, 200),
    ('phi0', 0, 1),
    ('phi1', 0, 1),
    ('PM', 0, 1),
    ('dm', 0, 500),
    ('kd', .01, 10),
    ('ud', 0, 10),
    ('d1', 0, 1),
    ('ID0', 0, 100),
    ('fd0', 0, 1),
    ('gd', 0, 10),
    ('ad0', 20 * 365, 40 * 365),
    ('rU', 0, 1/100),
    ('cD', 0, 1),
    ('cU', 0, 1),
    ('g_inf', 0, 10)
], columns=['name', 'lower', 'upper'])

lhs_parameter_space = [
    {
        name: LHSStrategy(lower, upper)
        for name, lower, upper in intrinsic_bounds.itertuples(index=False)
    },
    LHSStrategy(0., 500.),
    LHSStrategy(1/(40 * 365), 1/(20 * 365))
]
```

```{code-cell} ipython3
print(pd.concat([intrinsic_bounds]).to_latex(index=False, float_format="{:0.0f}".format))
```

```{code-cell} ipython3
max_val = jnp.finfo(jnp.float32).max
min_val = jnp.finfo(jnp.float32).min
```

```{code-cell} ipython3
from mox.sampling import sample
from mox.surrogates import make_surrogate
from mox.training import train_surrogate
from mox.loss import mse
```

```{code-cell} ipython3
key_i, key = random.split(key)
X_lhs_full = sample(lhs_parameter_space, train_samples, key_i)
y_lhs_full = vmap(full_solution, in_axes=[{n: 0 for n in intrinsic_bounds.name}, 0, 0])(*X_lhs_full)

y_min_full = {
    'pos_M': jnp.full((100,), 0.),
    'inc': jnp.full((100,), 0.),
    'prob_b': jnp.full((100,), 0.),
    'prob_c': jnp.full((100,), 0.),
    'prob_d': jnp.full((100,), 0.),
    'prop': jnp.full((100,), min_val)
}

y_max_full = {
    'pos_M': jnp.full((100,), 1.),
    'inc': jnp.full((100,), max_val),
    'prob_b': jnp.full((100,), 1.),
    'prob_c': jnp.full((100,), 1.),
    'prob_d': jnp.full((100,), 1.),
    'prop': jnp.full((100,), 1.)
}

surrogate_lhs_full = make_surrogate(
    X_lhs_full,
    y_lhs_full,
    y_min=y_min_full,
    y_max=y_max_full
)
key_i, key = random.split(key)
params_lhs_full = train_surrogate(
    X_lhs_full,
    y_lhs_full,
    surrogate,
    mse,
    key_i
)
```

```{code-cell} ipython3
key_i, key = random.split(key)
X_lhs_fixed = sample(lhs_parameter_space[0:1], train_samples, key_i)
y_lhs_fixed = vmap(lambda p: prev_stats_multisite(p, EIRs, etas, full_solution), in_axes=[{n: 0 for n in intrinsic_bounds.name}])(*X_lhs_fixed)

y_min_fixed = (0., 1e-12)
y_max_fixed = (1., max_val)

surrogate_lhs_fixed = make_surrogate(
    X_lhs_fixed,
    y_lhs_fixed,
    y_min=y_min_fixed,
    y_max=y_max_fixed
)
key_i, key = random.split(key)
params_lhs_fixed = train_surrogate(
    X_lhs_fixed,
    y_lhs_fixed,
    surrogate_lhs_fixed,
    mse,
    key_i
)
```

# TODO:

* convergence
* ~approximation error~
  * ~test sets: prior, space filling, surrogate posterior~
  * ~observation PE plots~
  * ~Relative Error for each output~
* why negative on full??

```{code-cell} ipython3
from flax.linen.module import _freeze_attr
```

```{code-cell} ipython3
# Write function for validation set generation
def full_solution_surrogate(surrogate, surrogate_params, params, eir, eta):
    surrogate_input = _freeze_attr([params, eir, eta])
    return surrogate.apply(surrogate_params, surrogate_input)

def fixed_surrogate(surrogate, surrogate_params, params):
    surrogate_input = _freeze_attr([params])
    return surrogate.apply(surrogate_params, surrogate_input)

def prev_stats_surrogate(*args):
    return prev_stats(full_solution_surrogate(*args))

def prev_stats_surrogate_batch(surrogate, surrogate_params, params):
    f = lambda p, e, a: full_solution_surrogate(surrogate, surrogate_params, p, e, a)
    return vmap(
        lambda p, e, a: prev_stats_multisite(p, e, a, f),
        in_axes=[{k: 0 for k in params.keys()}, None, None]
    )(params, EIRs, etas)

def prev_stats_fixed_surrogate_batch(surrogate, surrogate_params, params):
    return vmap(
        lambda p: fixed_surrogate(surrogate, surrogate_params, p),
        in_axes=[{k: 0 for k in params.keys()}]
    )(params)

def sort_dict(d):
    return {k: d[k] for k in intrinsic_bounds.name}

def prev_stats_batch(params):
    return vmap(
        lambda p, e, a: prev_stats_multisite(p, e, a, full_solution),
        in_axes=[{k: 0 for k in params.keys()}, None, None]
    )(params, EIRs, etas)
```

```{code-cell} ipython3
val_size = int(1e4)
y_val_prior = prev_stats_batch(without_obs(prior))
y_val_prior_full_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full, sort_dict(without_obs(prior)))
y_val_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed, sort_dict(without_obs(prior)))

key, key_i = random.split(key)
X_val_lhs = sample(lhs_parameter_space[0], val_size, key_i)
y_val_lhs = prev_stats_batch(X_val_lhs)
y_val_lhs_full_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full, X_val_lhs)
y_val_lhs_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed, X_val_lhs)
```

```{code-cell} ipython3
def surrogate_posterior(surrogate, params):
    def surrogate_impl(p, e, a):
        return prev_stats_multisite(p, e, a, lambda p_, e_, a_: full_solution_surrogate(surrogate, params, sort_dict(p_), e_, a_))
    n_samples = 100
    n_warmup = 100

    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized' #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
    )
    mcmc.run(key, obs_prev, obs_inc, surrogate_impl)
    return mcmc.get_samples()

def surrogate_posterior_fixed(surrogate, params):
    def surrogate_impl(p, e, a):
        return fixed_surrogate(surrogate, params, p)
    
    n_samples = 100
    n_warmup = 100

    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized' #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
    )
    mcmc.run(key, obs_prev, obs_inc, surrogate_impl)
    return mcmc.get_samples()
```

```{code-cell} ipython3
X_post_lhs_full = surrogate_posterior(surrogate_lhs_full, params_lhs_full)
X_post_lhs_fixed = surrogate_posterior_fixed(surrogate_lhs_fixed, params_lhs_fixed)
```

```{code-cell} ipython3
y_post_lhs_full = prev_stats_batch(X_post_lhs_full)
y_post_lhs_full_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full, X_post_lhs_full)
y_post_lhs_fixed = prev_stats_batch(X_post_lhs_fixed)
y_post_lhs_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed, X_post_lhs_fixed)
```

```{code-cell} ipython3
def plot_predictive_error(y, y_hat):
    fig, axs = plt.subplots(5, len(EIRs), figsize=(10, 8))
    y_labels = ['prev2-10', 'prev10+', 'inc0-5', 'inc5-15', 'inc15+']
    y = jnp.concatenate(y, axis=2)
    y_hat = jnp.concatenate(y_hat, axis=2)
    
    for i in range(5):
        axs[i, 0].set_ylabel(y_labels[i])
        for j in range(len(EIRs)):
            axs[0, j].set_xlabel(
                f'EIR: {EIRs[j]}'
            )
            axs[0, j].xaxis.set_label_position('top')
            axs[i, j].plot(
                y[:,j,i],
                y_hat[:,j,i],
                linestyle='',
                marker='o',
                markersize=0.7
            )
            lim = min(axs[i,j].get_ylim()[1], axs[i,j].get_xlim()[1])
            guide = jnp.linspace(0, lim)
            axs[i, j].plot(guide, guide, c='r')

    fig.tight_layout()

    fig.text(0.5, 0, 'Predictive error', ha='center')
```

```{code-cell} ipython3
plot_predictive_error(y_val_prior, y_val_prior_full_hat)
```

```{code-cell} ipython3
plot_predictive_error(y_val_prior, y_val_prior_fixed_hat)
```

```{code-cell} ipython3
plot_predictive_error(y_val_lhs, y_val_lhs_fixed_hat)
```

```{code-cell} ipython3
plot_predictive_error(y_val_lhs, y_val_lhs_full_hat)
```

```{code-cell} ipython3
plot_predictive_error(y_post_lhs_full, y_post_lhs_full_hat)
```

```{code-cell} ipython3
plot_predictive_error(y_post_lhs_fixed, y_post_lhs_fixed_hat)
```

```{code-cell} ipython3
def approximation_error(exps, labels, ys, y_hats):
    y_labels = ['prev2-10', 'prev10+', 'inc0-5', 'inc5-15', 'inc15+']
    ys = [jnp.concatenate(y, axis=2) for y in ys]
    y_hats = [jnp.concatenate(y_hat, axis=2) for y_hat in y_hats]
    return pd.concat([
        pd.DataFrame({
            'L1': jnp.abs(y - y_hat)[i, :, j],
            'RE': jnp.abs(y - y_hat)[i, :, j] / y[i, :, j],
            'EIR': float(EIRs[i]),
            'output': y_labels[j],
            'test_set': label,
            'experiment': exp
        })
        for i in range(len(EIRs))
        for j in range(len(y_labels))
        for exp, label, y, y_hat in zip(exps, labels, ys, y_hats)
    ])
```

```{code-cell} ipython3
approximation_error(
    ['lhs_full'] * 3 + ['lhs_fixed'] * 3,
    ['prior', 'lhs', 'posterior'] * 2,
    [y_val_prior, y_val_lhs, y_post_lhs_full, y_val_prior, y_val_lhs, y_post_lhs_fixed],
    [y_val_prior_full_hat, y_val_lhs_full_hat, y_post_lhs_full_hat, y_val_prior_fixed_hat, y_val_lhs_fixed_hat, y_post_lhs_fixed_hat]
).groupby(['experiment', 'test_set', 'output']).agg({'L1': 'mean', 'RE': 'mean'})#, 'EIR', 'output']).mean()
```

# TODO

 * ~Unnormalised prev loss (for survey stats) for easy comparison~
 * ~Is X_site included in training set??~
 * ~Why is unnormalised loss 0?~
 * ~Kernel resarting for surrogate~
 * ~Validation set in training~
 * ~Are the bounds sensible?? No, for EIR, for etas?~
 * Fix props?
 * Good posterior found with low error. However, weird convergence statistics
   * ~constraining neural network?~
   * ~regularisation~
   * fourier net?
   * The density of the true values is much lower in the surrogate!
     * high loss at low EIRs
 * Exchange replica with random walk

```{code-cell} ipython3
n_samples = 100
n_warmup = 100

mcmc = MCMC(
    NUTS(model),
    num_samples=n_samples,
    num_warmup=n_warmup,
    num_chains=n_chains,
    chain_method='vectorized'
)
mcmc.run(key, obs_prev, obs_inc)
mcmc.print_summary(prob=0.7)
```

```{code-cell} ipython3
surrogate_impl = lambda p, e, a: prev_stats_multisite(p, e, a, full_solution)
```

```{code-cell} ipython3
def surrogate_impl(p, e, a):
    return prev_stats(full_solution_surrogate(surrogate, params, p, e, a))
```

```{code-cell} ipython3
# n_samples = 1000
# n_warmup = 1000

# from numpyro.contrib.tfp.mcmc import RandomWalkMetropolis
# import tensorflow_probability as tfp

# kernel = RandomWalkMetropolis(
#     model,
#     new_state_fn=tfp.substrates.jax.mcmc.random_walk_normal_fn(scale=.04)
# )

n_samples = 100
n_warmup = 100

kernel = NUTS(model)

mcmc_surrogate = MCMC(
    kernel,
    num_samples=n_samples,
    num_warmup=n_warmup,
    num_chains=n_chains,
    chain_method='vectorized' #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
)
mcmc_surrogate.run(key, obs_prev, obs_inc, surrogate_impl)
mcmc_surrogate.print_summary(prob=0.7)
```

```{code-cell} ipython3
posterior_samples = mcmc.get_samples()
posterior_predictive = Predictive(
    model,
    posterior_samples
)(key, obs_prev, obs_inc)
```

```{code-cell} ipython3
posterior_samples_surrogate = mcmc_surrogate.get_samples()
posterior_predictive_surrogate = Predictive(
    model,
    posterior_samples_surrogate
)(key, obs_prev, obs_inc)
```

```{code-cell} ipython3
from scipy.stats import ks_2samp
sample_keys = list(posterior_samples.keys())
ks_tests = pd.DataFrame([
    {'statistic': ks_2samp(posterior_samples[k], posterior_samples_surrogate[k]).statistic, 'p-value': ks_2samp(posterior_samples[k], posterior_samples_surrogate[k]).pvalue}
    for k in sample_keys
], sample_keys)
```

```{code-cell} ipython3
from numpyro.diagnostics import summary
#d = pd.DataFrame(summary(mcmc.get_samples(group_by_chain=True), prob=0.7)).transpose()
d = pd.concat([
    pd.DataFrame(summary(mcmc.get_samples(group_by_chain=True), prob=0.7)).transpose()[['mean', 'std', 'n_eff', 'r_hat']],
    pd.DataFrame(summary(mcmc_surrogate.get_samples(group_by_chain=True), prob=0.7)).transpose()[['mean', 'std', 'n_eff', 'r_hat']],
    ks_tests
], axis=1, keys=['equilibrium solution', 'surrogate', 'KS'])
d['true_value'] = pd.Series(without_obs(true_values)).apply(lambda x: x[0]).astype(float)
print(d.to_latex(float_format="{:0.2f}".format))
```

```{code-cell} ipython3
pyro_data = az.from_numpyro(
    mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)
pyro_data_surrogate = az.from_numpyro(
    mcmc_surrogate,
    prior=prior,
    posterior_predictive=posterior_predictive_surrogate
)
```

```{code-cell} ipython3
az.rcParams["plot.max_subplots"] = 200
keys = list(pyro_data.prior.data_vars.keys())
axs = az.plot_dist_comparison(pyro_data)
axs = az.plot_dist_comparison(
    pyro_data_surrogate,
    ax=axs
)
for i in range(axs.shape[0]):
    axs[i, 2].vlines(
        true_values[keys[i]][0],
        0,
        axs[i, 2].get_ylim()[1],
        color = 'red',
        linestyle = 'dashed'
    )
    s_prior_lines = [axs[i, 0].get_lines()[1], axs[i, 2].get_lines()[2]]
    s_posterior_lines = [axs[i, 1].get_lines()[1], axs[i, 2].get_lines()[3]]
    for s_prior_line in s_prior_lines:
        s_prior_line.set_color('C3')
        s_prior_line.set_label('surrogate prior')
    for s_posterior_line in s_posterior_lines:
        s_posterior_line.set_color('C4')
        s_posterior_line.set_label('surrogate posterior')
    
    for j in range(3):
        axs[i, j].legend()
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
```

```{code-cell} ipython3
posterior_curves_surrogate = get_curves(posterior_samples_surrogate, EIRs, etas)
posterior_curves = get_curves(posterior_samples, EIRs, etas)
prior_curves = get_curves(prior, EIRs, etas)
true_curves = get_curves(without_obs(true_values), EIRs, etas)
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, len(EIRs), sharey=True, sharex=True)
imm_labels = ['b', 'c', 'd']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm in range(3):
        axs[imm, i].plot(posterior_curves[i, :, 2+imm, :].T, color='r', alpha=.01)
        axs[imm, i].plot(true_curves[i, 0, 2+imm, :])
        axs[imm, 0].set_ylabel(f'prob. {imm_labels[imm]}')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey='row', sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for prev in range(2):
        axs[0, i].set_xlabel(
            f'EIR: {EIRs[i]}'
        )
        axs[0, i].xaxis.set_label_position('top')
        axs[prev, i].plot(posterior_curves[i, :, prev, :].T, color='r', alpha=.01)
        axs[prev, i].plot(true_curves[i, 0, prev, :])
        axs[prev, 0].set_ylabel(prev_labels[prev])
        #axs[prev, 0].set_yscale('log')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Posterior pos_M/inc function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, len(EIRs), sharey=True, sharex=True)
imm_labels = ['b', 'c', 'd']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm in range(3):
        axs[imm, i].plot(posterior_curves_surrogate[i, :, 2+imm, :].T, color='r', alpha=.01)
        axs[imm, i].plot(true_curves[i, 0, 2+imm, :])
        axs[imm, 0].set_ylabel(f'prob. {imm_labels[imm]}')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Surrogate posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey='row', sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for prev in range(2):
        axs[0, i].set_xlabel(f'EIR: {EIRs[i]}')
        axs[0, i].xaxis.set_label_position('top')
        axs[prev, i].plot(posterior_curves_surrogate[i, :, prev, :].T, color='r', alpha=.01)
        axs[prev, i].plot(true_curves[i, 0, prev, :])
        axs[prev, 0].set_ylabel(prev_labels[prev])
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Surrogate posterior pos_M/inc function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, len(EIRs), sharey=True, sharex=True)
imm_labels = ['b', 'c', 'd']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm in range(3):
        axs[imm, i].plot(prior_curves[i, :, 2+imm, :].T, color='r', alpha=.01)
        axs[imm, i].plot(true_curves[i, 0, 2+imm, :])
        axs[imm, 0].set_ylabel(f'prob. {imm_labels[imm]}')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Prior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey='row', sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for prev in range(2):
        axs[0, i].set_xlabel(
            f'EIR: {EIRs[i]}',
        )
        axs[0, i].xaxis.set_label_position('top')
        axs[prev, i].plot(prior_curves[i, :, prev, :].T, color='r', alpha=.01)
        axs[prev, i].plot(true_curves[i, 0, prev, :])
        axs[prev, 0].set_ylabel(prev_labels[prev])
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Prior pos_M/inc function', ha='center')
```

```{code-cell} ipython3
posterior_predictive_surrogate_s = Predictive(
    model,
    posterior_samples_surrogate
)(key, obs_prev, obs_inc, surrogate_impl)

posterior_predictive_s = Predictive(
    model,
    posterior_samples
)(key, obs_prev, obs_inc, surrogate_impl)
```

```{code-cell} ipython3
p_s_p = jnp.concatenate([
    posterior_predictive_surrogate['obs_prev'],
    posterior_predictive_surrogate['obs_inc']
], axis=3)
p_s_p_s = jnp.concatenate([
    posterior_predictive_surrogate_s['obs_prev'],
    posterior_predictive_surrogate_s['obs_inc']
], axis=3)
pp = jnp.concatenate([
    posterior_predictive['obs_prev'],
    posterior_predictive['obs_inc']
], axis=3)
pp_s = jnp.concatenate([
    posterior_predictive_s['obs_prev'],
    posterior_predictive_s['obs_inc']
], axis=3)
```

```{code-cell} ipython3
jnp.mean(pp - pp_s, axis=0)
```

```{code-cell} ipython3
jnp.mean(p_s_p - p_s_p_s, axis=0)
```

```{code-cell} ipython3

```
