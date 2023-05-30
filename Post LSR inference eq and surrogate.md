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
   * write tests
   * ~build documentation~
   * make homepage
   * ~translate this notebook to jupytext~
   * use mox primitives
 * investigate du -> rU
 * investigate cu/cd -> cU/cD
 * investigate feature learning and context decoders
 * investigate robust training
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
    return dmeq.solve(p, dtype=jnp.float64)
```

```{code-cell} ipython3
population = 1_000_000
prev_N = 1_000

def prev_stats(params, eir, eta, impl=full_solution):
    solution = impl(params, eir, eta)
    inc_rates = solution[1] * solution[-1] * population
    return (
        jnp.array([
            solution[0, 3:10].sum() / solution[-1, 3:10].sum(), # Prev 2 - 10
            solution[0, 10:].sum() / solution[-1, 10:].sum(), # Prev 10+
        ]),
        jnp.array([
            inc_rates[:5].sum(), # Inc 0 - 5
            inc_rates[5:15].sum(), # Inc 5 - 15
            inc_rates[15:].sum() # Inc 15+
        ])
    )
```

```{code-cell} ipython3
prev_stats_multisite = vmap(prev_stats, in_axes=[None, 0, 0, None])
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
    
    du = numpyro.sample('du', dist.LeftTruncatedDistribution(dist.Cauchy(50., 1.), low=0.))
    
    # FOIM
    cd = numpyro.sample('cd', dist.Beta(1., 2.))
    cu = numpyro.sample('cu', dist.Beta(1., 5.))
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
        'rU': 1. / du,
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
            dist.Poisson(rate=inc_stats),
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

train_samples = int(1e5)
device_count = len(jax.devices())
```

```{code-cell} ipython3
# Create the X_prior dataset

def pmap_prior(k):
    return Predictive(model, num_samples=train_samples // cpu_count)(k)

key, *keys = random.split(key, num=cpu_count + 1)
train_prior = pmap(pmap_prior, in_axes=0, devices=jax.devices('cpu'))(jnp.stack(keys))
train_prior = jax.tree_map(lambda x: jnp.reshape(x, (train_samples, -1)), train_prior)
X_prior, x_def = jax.tree_util.tree_flatten(without_obs(train_prior))
X_prior = jnp.concatenate(X_prior, axis=1)
```

```{code-cell} ipython3
# Create the X_lhs dataset
bounds = pd.DataFrame.from_records([
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
    ('du', 0, 100),
    ('cd', 0, 1),
    ('cu', 0, 1),
    ('g_inf', 0, 10)
], columns=['name', 'lower', 'upper'])

sampler = LatinHypercube(d=len(bounds), seed=42)
samples = sampler.random(train_samples)
ordered_bounds = bounds.set_index('name').loc[pd.Series(without_obs(train_prior).keys())]

X_lhs = samples * (
    ordered_bounds.upper - ordered_bounds.lower
).to_numpy() + ordered_bounds.lower.to_numpy()
```

```{code-cell} ipython3
# Sample site parameters

site_bounds = pd.DataFrame.from_records([
    ('EIR', 0., 500.),
    ('eta', 1/(40 * 365), 1/(20 * 365))
], columns=['name', 'lower', 'upper'])
sampler = LatinHypercube(d=len(site_bounds), seed=42)
X_site = sampler.random(train_samples)
X_site = X_site * (site_bounds.upper - site_bounds.lower).to_numpy() + site_bounds.lower.to_numpy()
```

```{code-cell} ipython3
print(pd.concat([site_bounds, bounds]).to_latex(index=False, float_format="{:0.0f}".format))
```

```{code-cell} ipython3
key_i, key = random.split(key)
order = random.choice(key_i, EIRs.shape[0], (train_samples,), replace=True)
X_site_fixed = jnp.concatenate([
    EIRs[order, jnp.newaxis],
    etas[order, jnp.newaxis]
], axis=1)
```

```{code-cell} ipython3
# Sample full y
def sample_full_y(x, x_site):
    params = jax.tree_util.tree_unflatten(x_def, x)
    return full_solution(params, x_site[0], x_site[1])

def sample_y_from_x(x, x_site, impl):
    y = pmap(vmap(impl), devices=jax.devices('cpu'))(
        jnp.reshape(x, (cpu_count, x.shape[0] // cpu_count,) + x.shape[1:]),
        jnp.reshape(x_site, (cpu_count, x_site.shape[0] // cpu_count,) + x_site.shape[1:])
    )
    y_shape = y.shape[2:]
    return (jnp.reshape(y, (train_samples,) + y_shape), y_shape)

y_prior_full, y_shape = sample_y_from_x(X_prior, X_site, sample_full_y)
y_lhs_full, y_shape = sample_y_from_x(X_lhs, X_site, sample_full_y)
y_lhs_full_fixed_site, y_shape = sample_y_from_x(X_lhs, X_site_fixed, sample_full_y)

# sample fixed y
def sample_fixed_y(x, x_site):
    params = jax.tree_util.tree_unflatten(x_def, x)
    return jnp.concatenate(prev_stats_multisite(params, EIRs, etas, full_solution), axis=1)

y_prior_fixed , y_shape_fixed = sample_y_from_x(X_lhs, X_site, sample_fixed_y)
y_lhs_fixed , y_shape_fixed = sample_y_from_x(X_lhs, X_site, sample_fixed_y)
```

```{code-cell} ipython3
def split(x, split):
    return (x[:split], x[split:])

val_split = int(.8 * train_samples)

def standardise(x, mean, std):
    return (x - mean) / std

def inverse_standardise(x, mean, std):
    return x * std + mean

def adapt(x, x_val, axis):
    mean, std = jnp.mean(x, axis=axis, keepdims=True), jnp.std(x, axis=axis, keepdims=True)
    return (standardise(x, mean, std), standardise(x_val, mean, std), mean, std)

datasets = {
    'X_prior': (X_prior, 0),
    'X_lhs': (X_lhs, 0),
    'X_site': (X_site, 0),
    'X_site_fixed': (X_site_fixed, 0),
    'y_prior_full': (y_prior_full, (0, 2)),
    'y_lhs_full': (y_lhs_full, (0, 2)),
    'y_lhs_full_fixed_site': (y_lhs_full_fixed_site, (0, 2)),
    # 'y_prior_full': (y_prior_full, 0),
    # 'y_lhs_full': (y_lhs_full, 0),
    'y_prior_fixed': (y_prior_fixed, 0),
    'y_lhs_fixed': (y_lhs_fixed, 0)
}

def process_dataset(x, axis):
    x, x_val = split(x, val_split)
    return adapt(x, x_val, axis)

datasets = { k: process_dataset(*v) for k, v in datasets.items()}
```

```{code-cell} ipython3
from flax import linen as nn
from jax.nn import softplus
import optax
from jax import value_and_grad
from jaxtyping import Array
```

```{code-cell} ipython3
def maskedminmaxrelu(x, min_x, max_x, idx):
    '''relu with a min and max, however max_x is only set at idx'''
    filtered_min = jnp.maximum(x, min_x)
    filtered_max = jnp.minimum(filtered_min, max_x)
    return filtered_min.at[idx].set(filtered_max[idx])

class MaskedMinMaxSurrogate(nn.Module):
    units: int
    n_hidden: int
    n_output: int
    y_min: Array
    y_max: Array
    idx_max: Array

    @nn.compact
    def __call__(self, x):
        layers = [nn.Dense(self.units) for _ in range(self.n_hidden)]
        for i, lyr in enumerate(layers):
            x = lyr(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_output)(x)
        return maskedminmaxrelu(x, self.y_min, self.y_max, self.idx_max)

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()

def log_cosh(model, params, x_batched, y_batched):
    # Define a numerically stable log cosh for a single pair (x,y)
    def error(x, y):
        pred = model.apply(params, x)
        diff = y - pred
        return jnp.mean(diff + softplus(-2 * diff) - jnp.log(2.))

    # Regularisation loss
    reg_loss = sum(
        l2_loss(w, alpha=0.001)
        for w in jax.tree_util.tree_leaves(params)
    )

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(vmap(error)(x_batched, y_batched), axis=0) + reg_loss
```

```{code-cell} ipython3
def make_surrogate(key, X, X_val, y, y_shape, idx_max):
    y, y_val, y_mean, y_std = y
    surrogate_model = MaskedMinMaxSurrogate(
        units=288,
        n_hidden=3,
        n_output=jnp.product(jnp.array(y_shape)),
        y_min=standardise(jnp.zeros(y_shape), y_mean, y_std)[0].reshape(-1),
        y_max=standardise(jnp.ones(y_shape), y_mean, y_std)[0].reshape(-1),
        idx_max=idx_max
    )
    surrogate_params = surrogate_model.init(key, X)

    tx = optax.adam(learning_rate=.001)
    opt_state = tx.init(surrogate_params)
    loss_grad_fn = value_and_grad(jit(lambda p, x, y: log_cosh(surrogate_model, p, x, y)))

    batch_size = 100

    n_batches = X.shape[0] // batch_size
    X_batched = jnp.reshape(X, (n_batches, batch_size, -1))
    y_batched = jnp.reshape(y, (n_batches, batch_size, -1))

    epochs = 100
    
    for i in range(epochs):
        key, key_i = random.split(key)
        
        for b in random.permutation(key_i, n_batches, independent=True):
            loss_val, grads = loss_grad_fn(
                surrogate_params,
                X_batched[b],
                y_batched[b]
            )
            updates, opt_state = tx.update(grads, opt_state)
            surrogate_params = optax.apply_updates(surrogate_params, updates)

    return (surrogate_model, surrogate_params)
```

```{code-cell} ipython3
def make_full_surrogate(key, x, x_site, y):
    x, x_val = x[:2]
    x_site, x_site_val = x_site[:2]

    X = jnp.concatenate([x_site, x], axis=1)
    X_val = jnp.concatenate([x_site_val, x_val], axis=1)
    idx_max = jnp.arange(jnp.product(jnp.array(y_shape)))
    return make_surrogate(
        key,
        X,
        X_val,
        y,
        y_shape,
        idx_max
    )

def make_fixed_surrogate(key, x, y):
    x, x_val = x[:2]
    idx_max = jnp.arange(jnp.product(jnp.array(y_shape_fixed)))
    idx_max = jnp.reshape(idx_max.reshape(y_shape_fixed)[:,:2], -1)
    return make_surrogate(
        key,
        x,
        x_val,
        y,
        y_shape_fixed,
        idx_max
    )
```

```{code-cell} ipython3
n_sample_points = 2

def subsample(d, sample_size):
    return {k: tuple([a[:sample_size] for a in v]) for k, v in datasets.items()}

def surrogates_for_sample_size(key_i, sample_size):
    print('sample_size:', sample_size)
    d = subsample(datasets, sample_size)
    
    return {
        'prior_full': make_full_surrogate(
            key_i[0],
            d['X_prior'],
            d['X_site'],
            d['y_prior_full']
        ),
        'lhs_full': make_full_surrogate(
            key_i[1],
            d['X_lhs'],
            d['X_site'],
            d['y_lhs_full']
        ),
        'prior_fixed': make_fixed_surrogate(
            key_i[2],
            d['X_prior'],
            d['y_prior_fixed']
        ),
        'lhs_fixed': make_fixed_surrogate(
            key_i[3],
            d['X_lhs'],
            d['y_lhs_fixed']
        ),
        'lhs_full_fixed_site': make_full_surrogate(
            key_i[3],
            d['X_lhs'],
            d['X_site_fixed'],
            d['y_lhs_full_fixed_site']
        ),
    }
```

```{code-cell} ipython3
key, *key_i = random.split(key, num=4*n_sample_points + 1)
sample_points = jnp.linspace(1000, train_samples, num=n_sample_points, dtype=jnp.int64)
surrogates = [
    surrogates_for_sample_size(keys, sample_size)
    for keys, sample_size in
    list(zip(
        jnp.reshape(jnp.stack(key_i), (len(sample_points), 4, -1)),
        [1000]#sample_points
    ))
]
```

```{code-cell} ipython3
sample_map = {k: 0 for k in without_obs(prior).keys()}

def full_solution_surrogate(model, params, model_params, x, x_site, y, eir, eta):
    X_site_mean, X_site_std = x_site[2:]
    X_mean, X_std = x[2:]
    y_mean, y_std = y[2:]
    p = jnp.concatenate([
        jnp.array([eir, eta]),
        jax.flatten_util.ravel_pytree(model_params)[0]
    ])
    mean = jnp.concatenate([X_site_mean, X_mean], axis=1)
    std = jnp.concatenate([X_site_std, X_std], axis=1)
    y = model.apply(params, standardise(p, mean, std)[0])
    return inverse_standardise(
        jnp.reshape(y, y_shape),
        y_mean,
        y_std
    )[0]

pred_y_full = vmap(
    lambda x, f: jnp.concatenate(prev_stats_multisite(x, EIRs, etas, f), axis=1),
    [sample_map, None]
)

def fixed_solution_surrogate(model, params, model_params, x, y):
    X_mean, X_std = x[2:]
    y_mean, y_std = y[2:]
    p = jax.flatten_util.ravel_pytree(model_params)[0]
    y = model.apply(params, standardise(p, X_mean, X_std)[0])
    return inverse_standardise(
        jnp.reshape(y, y_shape_fixed),
        y_mean,
        y_std
    )[0]

pred_y_fixed = vmap(fixed_solution_surrogate, [None, None, sample_map, None, None])
```

```{code-cell} ipython3
# calculate the loss in prev/incidence
y = pred_y_full(without_obs(prior), full_solution)
```

```{code-cell} ipython3
y_hat_prior_full = [pred_y_full(
    without_obs(prior),
    lambda p, e, a: full_solution_surrogate(
        *s['prior_full'],
        p,
        datasets['X_prior'],
        datasets['X_site'],
        datasets['y_prior_full'],
        e,
        a,
    )
) for s in surrogates]

y_hat_lhs_full = [pred_y_full(
    without_obs(prior),
    lambda p, e, a: full_solution_surrogate(
        *s['lhs_full'],
        p,
        datasets['X_lhs'],
        datasets['X_site'],
        datasets['y_lhs_full'],
        e,
        a,
    )
) for s in surrogates]

y_hat_lhs_full_fixed_site = [pred_y_full(
    without_obs(prior),
    lambda p, e, a: full_solution_surrogate(
        *s['lhs_full_fixed_site'],
        p,
        datasets['X_lhs'],
        datasets['X_site_fixed'],
        datasets['y_lhs_full_fixed_site'],
        e,
        a,
    )
) for s in surrogates]
```

```{code-cell} ipython3
y_hat_prior_fixed = [pred_y_fixed(
    *s['prior_fixed'],
    without_obs(prior),
    datasets['X_prior'],
    datasets['y_prior_fixed']
) for s in surrogates]

y_hat_lhs_fixed = [pred_y_fixed(
    *s['lhs_fixed'],
    without_obs(prior),
    datasets['X_lhs'],
    datasets['y_lhs_fixed']
) for s in surrogates]
```

```{code-cell} ipython3
import pandas as pd
import seaborn as sns
y_labels = ['prev2-10', 'prev10+', 'inc0-5', 'inc5-15', 'inc15+']
```

```{code-cell} ipython3
y_hats = {
    'prior_full': y_hat_prior_full,
    'lhs_full': y_hat_lhs_full,
    'prior_fixed': y_hat_prior_fixed,
    'lhs_fixed': y_hat_lhs_fixed,
    'lhs_full_fixed_site': y_hat_lhs_full_fixed_site
}
```

```{code-cell} ipython3
df = pd.concat([
    pd.DataFrame({
        'samples': sample_points[u],
        'RE': jnp.abs(y - y_hat[u])[:, i, j] / y[:, i, j],
        'EIR': EIRs[i],
        'output': y_labels[j],
        'surrogate': s_label
    })
    for i in range(len(EIRs))
    for j in range(len(y_labels))
    for s_label, y_hat in y_hats.items()
    for u in range(len(sample_points))
])
```

```{code-cell} ipython3
df.EIR = df.EIR.astype(int)
```

```{code-cell} ipython3
df.samples = df.samples.astype(int)
```

```{code-cell} ipython3
df[df.samples == 100000].groupby('surrogate').agg({'RE': ['mean', 'std']})
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
---
g = sns.FacetGrid(df, row="output", col='surrogate', hue='EIR', sharey=False)
g.map(sns.lineplot, "samples", "RE")
g.add_legend()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
---
g = sns.FacetGrid(df[df.samples == sample_points[-1]], col='output', row='surrogate', sharex=False)
g.map(sns.boxplot, "RE")
g.add_legend()
```

```{code-cell} ipython3
df[(df.samples == sample_points[-1]) & (df.EIR == 0)].groupby(
    ['EIR', 'output', 'surrogate'], as_index=False
).agg({'RE': ['mean', 'std']}).sort_values(('RE', 'mean'))
```

```{code-cell} ipython3
def plot_surrogate_predictive_error(y_hat):
    fig, axs = plt.subplots(5, len(EIRs), figsize=(10, 8))
    
    for i in range(5):
        axs[i, 0].set_ylabel(y_labels[i])
        for j in range(len(EIRs)):
            axs[0, j].set_xlabel(
                f'EIR: {EIRs[j]}'
            )
            axs[0, j].xaxis.set_label_position('top')
            axs[i, j].plot(
                y[:,j, i],
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
plot_surrogate_predictive_error(y_hat_lhs_full[-1])
```

```{code-cell} ipython3
plot_surrogate_predictive_error(y_hat_lhs_fixed[-1])
```

```{code-cell} ipython3
plot_surrogate_predictive_error(y_hat_lhs_full_fixed_site[-1])
```

```{code-cell} ipython3
plt.plot(full_solution(true_values, EIRs[0], etas[0])[1])
plt.plot(full_solution_surrogate(
    *surrogates[-1]['lhs_full'],
    without_obs(true_values),
    datasets['X_lhs'],
    datasets['X_site'],
    datasets['y_lhs_full'], EIRs[0], etas[0])[1]
)
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
    r = fixed_solution_surrogate(*surrogates[-1]['lhs_fixed'], p, datasets['X_lhs'], datasets['y_lhs_fixed'])
    return r[:,:2], r[:,2:]
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
