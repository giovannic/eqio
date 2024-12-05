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
jax.config.update('jax_enable_x64', True)

cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]
```

```{code-cell} ipython3
key = random.PRNGKey(42)
```

```{code-cell} ipython3
from fastms.density.transformer import load as load_transformer
from fastms.density.rnn import load as load_rnn
from fastms.sample.save import load_samples, load_pytree
import pickle
```

```{code-cell} ipython3
validation_sample = load_samples(
    [
        f'/mnt/gc1610/home/fastms/data/samples/ibm_prior_low_v3/ibm_priors_{x}.npz'
        for x in range(9800, 10000)
    ],
    '/mnt/gc1610/home/fastms/data/samples/ibm_prior_low_v3/def_ibm_prior',
    dtype=jnp.float64
)
```

```{code-cell} ipython3
from jax.tree_util import tree_map
(x, x_seq, x_t), y = validation_sample
x, x_seq, y = tree_map(lambda x: x[:1], (x, x_seq, y))
first_sample = ((x, x_seq, x_t), y)
```

```{code-cell} ipython3
surrogate, net, params = load_rnn('../fastms/rnn_6_annealed_5/', first_sample)
```

```{code-cell} ipython3
(x, x_seq, x_t), y = first_sample
```

```{code-cell} ipython3
from mox.seq2seq.rnn import apply_surrogate as apply_rnn
#with jax.default_device(cpu_device):
(x, x_seq, x_t), y = validation_sample
mu, log_sigma = apply_rnn(surrogate, net, params, (x, x_seq))
```

```{code-cell} ipython3
import scipy as sp
from jax import scipy as jsp
import numpy as np
from fastms.density.train import _standardise
```

```{code-cell} ipython3
# Calibration error
@jax.jit
def cdf(mu, log_sigma, y):
    sigma = jnp.exp(log_sigma)
    return jsp.stats.truncnorm.cdf(
        y,
        _standardise(jnp.zeros_like(mu), mu, sigma),
        np.inf,
        mu,
        sigma
    )

p = jnp.linspace(0.001, 1, num=50, endpoint=False)
#with jax.default_device(cpu_device):
inc_cdf = cdf(
    mu['n_inc_clinical'],
    log_sigma['n_inc_clinical'],
    y['n_inc_clinical']
)
detect_cdf = cdf(
    mu['n_detect'],
    log_sigma['n_detect'],
    y['n_detect']
)
n_cdf = cdf(
    mu['n'],
    log_sigma['n'],
    y['n']
)
```

```{code-cell} ipython3
#with jax.default_device(cpu_device):
p_hat = jax.jit(vmap(lambda x, emp: jnp.sum(emp < x) / emp.size, in_axes=[0, None]))
p_inc_hat = p_hat(p, inc_cdf)
p_detect_hat = p_hat(p, detect_cdf)
p_n_hat = p_hat(p, n_cdf)
```

```{code-cell} ipython3
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

plots = [
    {
        'x': p_inc_hat,
        'title': 'Calibration of incidence'
    },
    {
        'x': p_detect_hat,
        'title': 'Calibration of prevalence'
    },
    {
        'x': p_n_hat,
        'title': 'Calibration of pop counts'
    }
]

for i in range(3):
    ax = axs[i]
    ax.plot(plots[i]['x'], p, linestyle = '-', marker = 'o')
    ax.plot(p, p, linestyle = '-', marker = '', alpha=0.5)
    if i == 0:
        ax.set_ylabel('Empirical cumulative density')
    ax.set_xlabel('Target cumulative density')
    ax.set_title(plots[i]['title'])
```

```{code-cell} ipython3
@jax.jit
def log_likelihood(mu, log_sigma, y):
    sigma = jnp.exp(log_sigma)
    return jsp.stats.truncnorm.logpdf(
        y,
        _standardise(jnp.zeros_like(mu), mu, sigma),
        np.inf,
        mu,
        sigma
    )

@jax.jit
def squared_error(mu, y):
    return jnp.square(mu - y)

@jax.jit
def stand_squared_error(mu, y):
    y_mean, y_std = jnp.mean(y, axis=0), jnp.std(y, axis=0)
    return jnp.square((mu - y_mean) / y_std - (y - y_mean) / y_std)
```

```{code-cell} ipython3
#with jax.default_device(cpu_device):
n_inc_se = squared_error(
    mu['n_inc_clinical'],
    y['n_inc_clinical']
)
n_detect_se = squared_error(
    mu['n_detect'],
    y['n_detect']
)
n_se = squared_error(
    mu['n'],
    y['n']
)
```

```{code-cell} ipython3
#with jax.default_device(cpu_device):
n_inc_ll = log_likelihood(
    mu['n_inc_clinical'],
    log_sigma['n_inc_clinical'],
    y['n_inc_clinical']
)
n_detect_ll = log_likelihood(
    mu['n_detect'],
    log_sigma['n_detect'],
    y['n_detect']
)
n_ll = log_likelihood(
    mu['n'],
    log_sigma['n'],
    y['n']
)
```

```{code-cell} ipython3
import seaborn as sns

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

plots = [
    {
        'x': n_inc_ll,
        'title': 'Likelihood of true incidence'
    },
    {
        'x': n_detect_ll,
        'title': 'Likelihood of true prevalence'
    },
    {
        'x': n_ll,
        'title': 'Likelihood of true pop counts'
    }
]

for i in range(3):
    ax = axs[i]
    sns.heatmap(jnp.mean(plots[i]['x'], axis=0).T, ax=ax)
    if i == 0:
        ax.set_ylabel('age group (year)')
    ax.set_xlabel('timestep (month)')
    ax.set_title(plots[i]['title'])
```

```{code-cell} ipython3
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

plots = [
    {
        'x': n_inc_se,
        'title': 'Approx. error of incidence'
    },
    {
        'x': n_detect_se,
        'title': 'Approx. error of prevalence'
    },
    {
        'x': n_se,
        'title': 'Approx. error of pop counts'
    }
]

for i in range(3):
    ax = axs[i]
    sns.heatmap(jnp.mean(plots[i]['x'], axis=0).T, ax=ax)
    if i == 0:
        ax.set_ylabel('age group (year)')
    ax.set_xlabel('timestep (month)')
    ax.set_title(plots[i]['title'])
```

```{code-cell} ipython3
n_inc_mse = jnp.mean(n_inc_se, axis=(1, 2))
n_detect_mse = jnp.mean(n_detect_se, axis=(1, 2))
n_mse = jnp.mean(n_se, axis=(1, 2))
```

```{code-cell} ipython3
n_inc_sll = jnp.sum(n_inc_ll, axis=(1, 2))
n_detect_sll = jnp.sum(n_detect_ll, axis=(1, 2))
n_sll = jnp.sum(n_ll, axis=(1, 2))
```

```{code-cell} ipython3
ranking = jnp.argsort(n_inc_sll)
worst_inc = ranking[:16]
best_inc = ranking[-16:]
ranking = jnp.argsort(n_detect_sll)
worst_detect = ranking[:16]
best_detect = ranking[-16:]
ranking = jnp.argsort(n_sll)
worst_n = ranking[:16]
best_n = ranking[-16:]
```

```{code-cell} ipython3
def plot_predictions(y, mu, log_sigma, mse, title):
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15,15))
    sigma = np.exp(log_sigma)
    
    for i in range(len(mse)):
        lower, upper = sp.stats.truncnorm.interval(
            .95,
            _standardise(np.zeros_like(mu[i]), mu[i], sigma[i]),
            np.inf,
            mu[i],
            sigma[i]
        )
        ax = axs[i // 4, i % 4]
        ax.plot(np.arange(len(y[i])), y[i])
        ax.plot(np.arange(len(mu[i])), mu[i])
        ax.fill_between(jnp.arange(len(mu[i])), lower, upper, color='orange', alpha=.5)
        ax.set_title(f'log likelihood:{mse[i]:.2f}')
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

    fig.tight_layout()
    fig.text(0.5, -0.05, title, ha='center', fontsize=20)
```

```{code-cell} ipython3
plot_predictions(
    y['n_inc_clinical'][best_inc, ..., 0],
    mu['n_inc_clinical'][best_inc, ..., 0],
    log_sigma['n_inc_clinical'][best_inc, ..., 0],
    n_inc_sll[best_inc],
    'Best inc (0 year olds)'
)
```

```{code-cell} ipython3
plot_predictions(
    y['n_inc_clinical'][worst_inc, ..., 0],
    mu['n_inc_clinical'][worst_inc, ..., 0],
    log_sigma['n_inc_clinical'][worst_inc, ..., 0],
    n_inc_sll[worst_inc],
    'Worst inc (0 year olds)'
)
```

```{code-cell} ipython3
plot_predictions(
    y['n_detect'][best_detect, ..., 0],
    mu['n_detect'][best_detect, ..., 0],
    log_sigma['n_detect'][best_detect, ..., 0],
    n_detect_sll[best_detect],
    'Best prevalence (0 year olds)'
)
```

```{code-cell} ipython3
plot_predictions(
    y['n_detect'][worst_detect, ..., 0],
    mu['n_detect'][worst_detect, ..., 0],
    log_sigma['n_detect'][worst_detect, ..., 0],
    n_detect_sll[worst_detect],
    'Worst prevalence (0 year olds)'
)
```

```{code-cell} ipython3
plot_predictions(
    y['n'][best_n, ..., 0],
    mu['n'][best_n, ..., 0],
    log_sigma['n'][best_n, ..., 0],
    n_sll[best_n],
    'Best pop count (0 year olds)'
)
```

```{code-cell} ipython3
plot_predictions(
    y['n'][worst_n, ..., 0],
    mu['n'][worst_n, ..., 0],
    log_sigma['n'][worst_n, ..., 0],
    n_sll[worst_n],
    'Worst pop count (0 year olds)'
)
```

```{code-cell} ipython3
#Surrogate posterior approximation error

#with jax.default_device(cpu_device):
defpath = '/mnt/gc1610/home/fastms/data/samples/bnaf_rnn_6_annealed_5/def_bnaf_rnn_6'
paths = [
    f'/mnt/gc1610/home/fastms/data/samples/bnaf_rnn_6_annealed_5/bnaf_rnn_6_{x}.npz'
    for x in range(0, 1000)
]
with open(defpath, 'rb') as f:
    treedef = pickle.load(f)
p_pickles = [
    load_pytree(treedef, path, jnp.float64)
    for path in paths
]

from jax.tree_util import tree_map
posterior_samples = tree_map(lambda *leaves: jnp.concatenate(leaves), *p_pickles)
```

```{code-cell} ipython3
(post_x, post_x_seq, x_t), post_y = posterior_samples[0]
post_mu, post_log_sigma = apply_rnn(surrogate, net, params, (post_x, post_x_seq))
```

```{code-cell} ipython3
post_n_inc_ll = log_likelihood(
    post_mu['n_inc_clinical'],
    post_log_sigma['n_inc_clinical'],
    post_y['n_inc_clinical']
)
post_n_detect_ll = log_likelihood(
    post_mu['n_detect'],
    post_log_sigma['n_detect'],
    post_y['n_detect']
)
post_n_ll = log_likelihood(
    post_mu['n'],
    post_log_sigma['n'],
    post_y['n']
)
```

```{code-cell} ipython3
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
plots = [
    {
        'x': n_inc_ll,
        'post': post_n_inc_ll,
        'title': 'Likelihood of true incidence'
    },
    {
        'x': n_detect_ll,
        'post': post_n_detect_ll,
        'title': 'Likelihood of true prevalence'
    },
    {
        'x': n_ll,
        'post': post_n_ll,
        'title': 'Likelihood of true pop counts'
    }
]
x_labels=['prior', 'posterior']

for i in range(3):
    ax = axs[i]
    ax.boxplot(
        jnp.stack([
            plots[i]['x'].reshape(-1),
            plots[i]['post'].reshape(-1),
        ]),
        showfliers=False,
        labels=x_labels
    )
    if i == 0:
        ax.set_ylabel('Surrogate log likelihood')
    ax.set_xlabel('Validation set')
    ax.set_title(plots[i]['title'])
```

```{code-cell} ipython3
post_n_inc_se = squared_error(
    post_mu['n_inc_clinical'],
    post_y['n_inc_clinical']
)
post_n_detect_se = squared_error(
    post_mu['n_detect'],
    post_y['n_detect']
)
post_n_se = squared_error(
    post_mu['n'],
    post_y['n']
)
```

```{code-cell} ipython3
jnp.mean(n_inc_se, axis=(1,2)).shape
```

```{code-cell} ipython3
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
def agg(x):
    return jnp.mean(x, axis=(1,2))

plots = [
    {
        'x': agg(n_inc_se),
        'post': agg(post_n_inc_se),
        'title': 'Likelihood of true incidence'
    },
    {
        'x': agg(n_detect_se),
        'post': agg(post_n_detect_se),
        'title': 'Likelihood of true prevalence'
    },
    {
        'x': agg(n_se),
        'post': agg(post_n_se),
        'title': 'Likelihood of true pop counts'
    }
]
x_labels=['prior', 'posterior']

for i in range(3):
    ax = axs[i]
    ax.boxplot(
        jnp.stack([
            plots[i]['x'].reshape(-1),
            plots[i]['post'].reshape(-1),
        ]),
        showfliers=False,
        labels=x_labels
    )
    if i == 0:
        ax.set_ylabel('Surrogate log likelihood')
    ax.set_xlabel('Validation set')
    ax.set_title(plots[i]['title'])
```

```{code-cell} ipython3
import jax
jax.config.update('jax_enable_x64', True)

import pickle
from jax.tree_util import tree_map
from fastms.sample.save import save_compressed_pytree, load_pytree
from jax import numpy as jnp
cpu_device = jax.devices('cpu')[0]


defpath = '/mnt/gc1610/home/fastms/data/samples/bnaf_rnn_6_annealed_4/def_bnaf_rnn_6'
paths = [
    f'/mnt/gc1610/home/fastms/data/samples/bnaf_rnn_6_annealed_4/bnaf_rnn_6_{x}.npz'
    for x in range(10000)
]
with open(defpath, 'rb') as f:
    treedef = pickle.load(f)
    
with jax.default_device(cpu_device):
    p_pickles = [
        load_pytree(treedef, path, jnp.float64)
        for path in paths
    ]
```

```{code-cell} ipython3
with jax.default_device(cpu_device):
    posterior_samples = tree_map(lambda *leaves: jnp.concatenate(leaves), *p_pickles)

save_compressed_pytree(
    posterior_samples[0],
    '/mnt/gc1610/home/fastms/data/samples/bnaf_6_annealed_4_train.npz'
)
```

```{code-cell} ipython3

```
