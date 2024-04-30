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

 * MMD
 * Change pipeline to sample train size for rounds
 * Add annealing

```{code-cell} ipython3
cpu_count = 100
import os
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count}'
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, jit, vmap
import jax
jax.config.update('jax_enable_x64', True)

#jax.config.update('jax_platform_name', 'cpu') # for memory purposes
cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]
```

```{code-cell} ipython3
import dmeq
from mox.sampling import LHSStrategy, DistStrategy
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
n_sites = EIRs.shape[0]
key, key_i = random.split(key)
etas = 1. / random.uniform(key_i, shape=(n_sites,), minval=40*365, maxval=100*365, dtype=jnp.float64)
```

```{code-cell} ipython3
def model(
    true_EIRs=None,
    prev=None,
    inc=None,
    impl=lambda p, e, a: prev_stats_multisite(p, e, a, full_solution),
    alpha=1.
    ):
    with numpyro.plate('sites', n_sites):
        EIR = numpyro.sample('EIR', dist.Uniform(0., 500.), obs=true_EIRs)
    
    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.LogNormal(0., .25))
    ub = numpyro.sample('ub', dist.LogNormal(0., 1.))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    IB0 = numpyro.sample(
        'IB0',
        dist.TruncatedDistribution(dist.Normal(50., 20.), low=25., high=75.)
    )
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., .25))
    uc = numpyro.sample('uc', dist.LogNormal(0., 1.))
    phi0 = numpyro.sample('phi0', dist.Beta(2., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 5.))
    IC0 = numpyro.sample(
        'IC0',
        dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.)
    )
    PM = numpyro.sample('PM', dist.Beta(1., 1.))
    dm = numpyro.sample(
        'dm',
        dist.TruncatedDistribution(dist.Normal(50., 20.), low=5., high=100.)
    )
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., .25))
    ud = numpyro.sample('ud', dist.LogNormal(0., 1.))
    d1 = numpyro.sample('d1', dist.Beta(1., 5.))
    ID0 = numpyro.sample(
        'ID0',
        dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.)
    )
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gd = numpyro.sample('gd', dist.LogNormal(0., 2.))
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
    
    with numpyro.handlers.scale(scale=alpha):
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
true_values = Predictive(model, num_samples=1)(key_i, true_EIRs=EIRs)
```

```{code-cell} ipython3
key, key_i = random.split(key)
prior = Predictive(model, num_samples=600)(key_i)
```

```{code-cell} ipython3
prior_space = [
    {
        'kb': DistStrategy(dist.LogNormal(0., .25)),
        'ub': DistStrategy(dist.LogNormal(0., 1.)),
        'b0': DistStrategy(dist.Beta(1., 1.)),
        'IB0': DistStrategy(dist.TruncatedDistribution(dist.Normal(50., 20.), low=25., high=75.)),
        'kc': DistStrategy(dist.LogNormal(0., .25)),
        'uc': DistStrategy(dist.LogNormal(0., 1.)),
        'IC0': DistStrategy(dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.)),
        'phi0': DistStrategy(dist.Beta(2., 1.)),
        'phi1': DistStrategy(dist.Beta(1., 5.)),
        'PM': DistStrategy(dist.Beta(1., 1.)),
        'dm': DistStrategy(dist.TruncatedDistribution(dist.Normal(50., 20.), low=5., high=100.)),
        'kd': DistStrategy(dist.LogNormal(0., .25)),
        'ud': DistStrategy(dist.LogNormal(0., 1.)),
        'd1': DistStrategy(dist.Beta(1., 5.)),
        'ID0': DistStrategy(dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.)),
        'fd0': DistStrategy(dist.Beta(1., 1.)),
        'gd': DistStrategy(dist.LogNormal(0., 2.)),
        'ad0': DistStrategy(dist.TruncatedDistribution(
            dist.Normal(70. * 365., 365.),
            low=40. * 365.,
            high=100. * 365.
        )),
        'rU': DistStrategy(dist.LogNormal(0., 1.))
    },
    DistStrategy(dist.Uniform(0., 500.)), # EIR
    DistStrategy(dist.Uniform(1/(100 * 365), 1/(40 * 365))) # eta
]
```

```{code-cell} ipython3
obs_inc, obs_prev = (true_values['obs_inc'], true_values['obs_prev'])
```

```{code-cell} ipython3
def without_obs(params):
    return {k : v for k, v in params.items() if not k in {'obs_inc', 'obs_prev'}}
```

```{code-cell} ipython3
from jax import pmap, tree_map
import jax
import pandas as pd

train_samples = int(5_000)
device_count = len(jax.devices())
n_epochs = 1_000
```

```{code-cell} ipython3
max_val = jnp.finfo(jnp.float64).max
min_val = jnp.finfo(jnp.float64).smallest_normal
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
    ('IC0', 0, 50),
    ('phi0', 0, 1),
    ('phi1', 0, 1),
    ('PM', 0, 1),
    ('dm', 0, 100),
    ('kd', min_val, 10),
    ('ud', 0, 10),
    ('d1', 0, 1),
    ('ID0', 0, 50),
    ('fd0', 0, 1),
    ('gd', 0, 10),
    ('ad0', 40 * 365, 100 * 365),
    ('rU', 0, 1),
], columns=['name', 'lower', 'upper'])

lhs_train_space = [
    {
        name: LHSStrategy(lower, upper)
        for name, lower, upper in intrinsic_bounds.itertuples(index=False)
    },
    LHSStrategy(0., 500.),
    LHSStrategy(1/(100 * 365), 1/(40 * 365))
]

lhs_test_space = [
    {
        name: LHSStrategy(lower, upper)
        for name, lower, upper in intrinsic_bounds.itertuples(index=False)
    },
    LHSStrategy(0., 500.),
    LHSStrategy(1/(100 * 365), 1/(40 * 365))
]

x_min = [{
    name: lower
    for name, lower, _ in intrinsic_bounds.itertuples(index=False)
}, 0., 1/(100 * 365)]

x_max = [{
    name: upper
    for name, _, upper in intrinsic_bounds.itertuples(index=False)
}, 500., 1/(40 * 365)]
```

```{code-cell} ipython3
print(pd.concat([intrinsic_bounds]).to_latex(index=False, float_format="{:0.0f}".format))
```

```{code-cell} ipython3
from mox.sampling import sample
from mox.surrogates import (
    make_surrogate,
    init_surrogate,
    apply_surrogate,
    MLP
)
from mox.training import train_surrogate
from mox.loss import mse
from mox.utils import tree_leading_axes as tla
```

```{code-cell} ipython3
max_age = 99
y_min_full = {
    'pos_M': jnp.full((max_age,), 0., dtype=jnp.float64),
    'inc': jnp.full((max_age,), 0., dtype=jnp.float64),
    'prob_b': jnp.full((max_age,), 0., dtype=jnp.float64),
    'prob_c': jnp.full((max_age,), 0., dtype=jnp.float64),
    'prob_d': jnp.full((max_age,), 0., dtype=jnp.float64),
    'prop': jnp.full((max_age,), 1e-12, dtype=jnp.float64)
}

y_max_full = {
    'pos_M': jnp.full((max_age,), 1., dtype=jnp.float64),
    'inc': jnp.full((max_age,), max_val, dtype=jnp.float64),
    'prob_b': jnp.full((max_age,), 1., dtype=jnp.float64),
    'prob_c': jnp.full((max_age,), 1., dtype=jnp.float64),
    'prob_d': jnp.full((max_age,), 1., dtype=jnp.float64),
    'prop': jnp.full((max_age,), 1., dtype=jnp.float64)
}
```

```{code-cell} ipython3
key_i, key = random.split(key)
with jax.default_device(cpu_device):
    X_lhs_full = sample(lhs_train_space, train_samples, key_i)
    y_lhs_full = vmap(
        full_solution,
        in_axes=[{n: 0 for n in intrinsic_bounds.name}, 0, 0]
    )(*X_lhs_full)
```

```{code-cell} ipython3
def make_net(surrogate, y):
    y0 = tree_map(lambda x: x[0], y)
    y0_vec = surrogate.vectorise_output(y0)
    return MLP(
        units=256,
        n_hidden=2,
        n_output=jnp.size(y0_vec),
        dropout_rate=.2,
        batch_norm=True
    )

surrogate_lhs_full = make_surrogate(
    X_lhs_full,
    y_lhs_full,
    y_min=y_min_full,
    y_max=y_max_full
)

net_full = make_net(surrogate_lhs_full, y_lhs_full)

key_i, key = random.split(key)
params_lhs_full = init_surrogate(key_i, surrogate_lhs_full, net_full, X_lhs_full)
train_state_lhs_full = train_surrogate(
    X_lhs_full,
    y_lhs_full,
    surrogate_lhs_full,
    net_full,
    mse,
    key_i,
    params_lhs_full,
    epochs=n_epochs
)
```

```{code-cell} ipython3
key_i, key = random.split(key)
with jax.default_device(cpu_device):
    X_prior_full = sample(prior_space, train_samples, key_i)
    y_prior_full = vmap(full_solution, in_axes=tree_map(lambda x: 0, X_prior_full))(*X_prior_full)
```

```{code-cell} ipython3
def make_net(surrogate, y):
    y0 = tree_map(lambda x: x[0], y)
    y0_vec = surrogate.vectorise_output(y0)
    return MLP(
        units=256,
        n_hidden=2,
        n_output=jnp.size(y0_vec),
        dropout_rate=.2,
        batch_norm=True,
        
    )

surrogate_prior_full = make_surrogate(
    X_prior_full,
    y_prior_full,
    y_min=y_min_full,
    y_max=y_max_full
)

key_i, key = random.split(key)
net_full = make_net(surrogate_prior_full, y_prior_full)

params_prior_full = init_surrogate(
    key_i,
    surrogate_prior_full,
    net_full,
    X_prior_full
)
train_state_prior_full = train_surrogate(
    X_prior_full,
    y_prior_full,
    surrogate_prior_full,
    net_full,
    mse,
    key_i,
    params_prior_full,
    epochs=n_epochs
)
```

```{code-cell} ipython3
key_i, key = random.split(key)
X_lhs_fixed = X_lhs_full
y_lhs_fixed = vmap(prev_stats, in_axes=[tla(y_lhs_full)])(y_lhs_full)

y_min_fixed = (0., min_val)
y_max_fixed = (1., max_val)

surrogate_lhs_fixed = make_surrogate(
    X_lhs_fixed,
    y_lhs_fixed,
    y_min=y_min_fixed,
    y_max=y_max_fixed
)

net_fixed = make_net(surrogate_lhs_fixed, y_lhs_fixed)

key_i, key = random.split(key)
params_lhs_fixed = init_surrogate(
    key_i,
    surrogate_lhs_fixed,
    net_fixed,
    X_lhs_fixed
)
train_state_lhs_fixed = train_surrogate(
    X_lhs_fixed,
    y_lhs_fixed,
    surrogate_lhs_fixed,
    net_fixed,
    mse,
    key_i,
    params_lhs_fixed,
    epochs=n_epochs
)
```

```{code-cell} ipython3
key_i, key = random.split(key)
X_prior_fixed = X_prior_full
y_prior_fixed = vmap(prev_stats, in_axes=[tla(y_prior_full)])(y_prior_full)

surrogate_prior_fixed = make_surrogate(
    X_prior_fixed,
    y_prior_fixed,
    y_min=y_min_fixed,
    y_max=y_max_fixed
)
key_i, key = random.split(key)
params_prior_fixed = init_surrogate(key_i, surrogate_prior_fixed, net_fixed, X_prior_fixed)
train_state_prior_fixed = train_surrogate(
    X_prior_fixed,
    y_prior_fixed,
    surrogate_prior_fixed,
    net_fixed,
    mse,
    key_i,
    params_prior_fixed,
    epochs=n_epochs
)
```

```{code-cell} ipython3
# Write function for validation set generation
def apply_dmeq_surrogate(surrogate, net, state, params, eir, eta):
    return tree_map(
        lambda leaf: leaf[0, 0],
        apply_surrogate(
            surrogate,
            net,
            {'params': state.params, 'batch_stats': state.batch_stats},
            tree_map(jnp.atleast_1d, [params, eir, eta])
        )
    )

def prev_stats_surrogate(*args):
    return prev_stats(apply_dmeq_surrogate(*args))

def posterior_EIR(params):
    return params['EIR'], {k: v for k, v in params.items() if k != 'EIR'}

def prev_stats_full_surrogate_posterior(surrogate, net, net_params, params):
    eir, params = posterior_EIR(params)
    f = lambda p, e, a: apply_dmeq_surrogate(surrogate, net, net_params, p, e, a)
    return vmap(
        lambda p, e, a: vmap(lambda _e, _a: prev_stats(f(p, _e, _a)))(e, a),
        in_axes=[{k: 0 for k in params.keys()}, 0, None]
    )(params, eir, etas)

def prev_stats_fixed_surrogate_posterior(surrogate, net, net_params, params):
    eir, params = posterior_EIR(params)
    return vmap(
        lambda p, e, a: vmap(
            apply_dmeq_surrogate,
            in_axes=[None, None, None, None, 0, 0]
        )(surrogate, net, net_params, p, e, a),
        in_axes=[{k: 0 for k in params.keys()}, 0, None]
    )(params, eir, etas)

def sort_dict(d):
    return {k: d[k] for k in intrinsic_bounds.name}

def prev_stats_posterior(params):
    eir, params = posterior_EIR(params)
    return vmap(
        lambda p, e, a: vmap(lambda _e, _a: prev_stats(full_solution(p, _e, _a)))(e, a),
        in_axes=[{k: 0 for k in params.keys()}, 0, None]
    )(params, eir, etas)
```

```{code-cell} ipython3
def surrogate_impl_full(surrogate, net, net_params):
    return lambda p, e, a: prev_stats_multisite(
        p,
        e,
        a,
        lambda p_, e_, a_: apply_dmeq_surrogate(
            surrogate,
            net,
            net_params,
            sort_dict(p_),
            e_,
            a_
        )
    )

def surrogate_impl_fixed(surrogate, net, net_params):
    return lambda p, e, a: vmap(
        apply_dmeq_surrogate,
        in_axes=[None, None, None, None, 0, 0]
    )(
        surrogate,
        net,
        net_params,
        sort_dict(p),
        e,
        a
    )
```

```{code-cell} ipython3
from numpyro.infer.util import log_density

def densities(p, model, impl):
    ld = log_density(model, [], {'prev': obs_prev, 'inc': obs_inc, 'impl': impl}, p)
    return ld[0]
```

```{code-cell} ipython3
from jax import jacfwd

def get_sensitivity(impl):
    sensitivity = vmap(
        jacfwd(densities),
        in_axes=[tree_map(lambda _: 0, without_obs(prior)), None, None]
    )(without_obs(prior), model, impl)
    return pd.concat([
        pd.DataFrame({
            'parameter': parameter,
            'gradient': sensitivity[parameter]
        })
        for parameter in sensitivity.keys()
        if parameter != 'EIR'
    ])
```

```{code-cell} ipython3
sensitivity_prior_full = get_sensitivity(
    surrogate_impl_full(surrogate_prior_full, net_full, train_state_prior_full)
)
```

```{code-cell} ipython3
sensitivity_prior_fixed = get_sensitivity(
    surrogate_impl_fixed(surrogate_prior_fixed, net_fixed, train_state_prior_fixed)
)
sensitivity_lhs_fixed = get_sensitivity(
    surrogate_impl_fixed(surrogate_lhs_fixed, net_fixed, train_state_lhs_fixed)
)
```

```{code-cell} ipython3
sensitivity_lhs_full = get_sensitivity(
    surrogate_impl_full(surrogate_lhs_full, net_full, train_state_lhs_full)
)
```

```{code-cell} ipython3
with jax.default_device(cpu_device):
    sensitivity_underlying = get_sensitivity(
        lambda p, e, a: prev_stats_multisite(p, e, a, full_solution)
    )
```

```{code-cell} ipython3
import seaborn as sns
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(19.7, 8.27))
sns.barplot(
    pd.concat([
        sensitivity_underlying.assign(model='underlying'),
        sensitivity_prior_full.assign(model='prior_full'),
        sensitivity_lhs_full.assign(model='lhs_full'),
        sensitivity_prior_fixed.assign(model='prior_fixed'),
        sensitivity_lhs_fixed.assign(model='lhs_fixed')
    ]),
    x='parameter',
    y='gradient',
    hue='model',
    estimator=lambda x: jnp.mean(jnp.abs(jnp.array(x))),
    errorbar=('ci', 95),
    ax=ax
)
ax.set_xlabel('Intrinsic Parameter')
ax.set_ylabel('Absolute Mean gradient')
ax.set_title('Sensitivity of Surrogate Models')
ax.set_yscale('log')
```

```{code-cell} ipython3
def get_curves(params, eirs, etas, impl=full_solution):
    return vmap(
        vmap(
            impl,
            in_axes=[
                {k: 0 for k in params.keys()},
                None,
                None
            ]
        ),
        in_axes=[None, 0, 0]
    )(params, eirs, etas)

prior_curves = get_curves(prior, EIRs, etas)
true_curves = get_curves(without_obs(true_values), EIRs, etas)
```

```{code-cell} ipython3
prior_full_prior_curves = get_curves(
    prior,
    EIRs,
    etas,
    impl=lambda p, e, a: tree_map(
        jnp.squeeze,
        apply_dmeq_surrogate(
            surrogate_prior_full,
            net_full,
            train_state_prior_full,
            sort_dict(p),
            e,
            a
        )
    )
)
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
        axs[imm_i, i].plot(prior_full_prior_curves[imm][i, :n_curves, :].T, color='r', alpha=.01)
        axs[imm_i, i].plot(true_curves[imm][i, 0, :])
        axs[imm_i, 0].set_ylabel(imm)
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Prior full prior immunity probability function', ha='center')
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
        axs[prev_i, i].plot(
            prior_full_prior_curves[prev][i, :n_curves, :].T / prior_full_prior_curves['prop'][i, :n_curves, :].T,
            color='r',
            alpha=.01
        )
        axs[prev_i, i].plot(true_curves[prev][i, 0, :] / true_curves['prop'][i, 0, :].T)
        axs[prev_i, 0].set_ylabel(prev)
        #axs[prev_i, 0].set_yscale('log')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Prior full prior pos_M/inc function', ha='center')
```

```{code-cell} ipython3
import numpyro

def surrogate_posterior(key, impl):
    n_samples = 500
    n_warmup = 500

    # Reverse mode has lead to initialisation errors
    kernel = NUTS(model, forward_mode_differentiation=True)

    #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized'
    )
    mcmc.run(key, None, obs_prev, obs_inc, impl)
    return mcmc

def surrogate_posterior_full(surrogate, net, params, key):
    return surrogate_posterior(key, surrogate_impl_full(surrogate, net, params))

def surrogate_posterior_fixed(surrogate, net, params, key):
    return surrogate_posterior(key, surrogate_impl_fixed(surrogate, net, params))
```

```{code-cell} ipython3
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, RenyiELBO
from numpyro.infer.autoguide import AutoBNAFNormal

def surrogate_posterior_svi(key, impl):
    n_samples = 500
    n_train_samples = 50_000
    
    guide = AutoBNAFNormal(model, num_flows=5)
    svi = SVI(
        model,
        guide,
        optim.ClippedAdam(1e-4),
        loss=Trace_ELBO(num_particles=8),
        true_EIRs=None,
        prev=obs_prev,
        inc=obs_inc,
        impl=impl
    )

    # train SVI
    sample_key, key = random.split(key, 2)
    svi_result = svi.run(sample_key, n_train_samples, stable_update=True)
    svi_params = svi_result.params

    # sample posterior
    post_key, key = random.split(key, 2)
    posterior_samples = Predictive(
        guide,
        params=svi_params,
        num_samples=n_samples
    )(post_key)

    return posterior_samples

def surrogate_posterior_full_svi(surrogate, net, params, key):
    return surrogate_posterior_svi(key, surrogate_impl_full(surrogate, net, params))
```

```{code-cell} ipython3
def posterior_EIR(params):
    return params['EIR'], {k: v for k, v in params.items() if k != 'EIR'}

def sample_full_from_posterior(params):
    eir, params = posterior_EIR(params)
    n = eir.shape[0]
    X = [
        tree_map(lambda leaf: jnp.repeat(leaf, n_sites), params),
        jnp.reshape(eir, -1),
        jnp.tile(etas, n)
    ]
    y = vmap(
        lambda p, e, a: vmap(lambda _e, _a: full_solution(p, _e, _a))(e, a),
        in_axes=[{k: 0 for k in params.keys()}, 0, None]
    )(params, eir, etas)
    y = tree_map(lambda leaf: leaf.reshape((-1, leaf.shape[-1])), y)
    return X, y

def surrogate_posterior_annealed_svi(surrogate, net, params, key):
    rounds = 10
    n_samples = train_samples
    n_train_samples = 50_000

    T = 1. / (0.45 ** jnp.arange(rounds))[::-1] #jnp.linspace(1000, 1, num=rounds)
    
    guide = AutoBNAFNormal(model, num_flows=5)

    svi_state = None

    training_samples = (X_prior_full, y_prior_full)

    sample_sets = list()

    #step_sizes = jnp.linspace(1e-4, 1e-5, num=rounds)
    
    #def optim_size(i):
        #return step_sizes[i // n_train_samples]
    
    svi = SVI(
        model,
        guide,
        optim.ClippedAdam(step_size=1e-4),
        loss=Trace_ELBO(num_particles=8)
    )
    
    for t in T:
        impl = surrogate_impl_full(surrogate, net, params)
    
        # train SVI
        sample_key, key = random.split(key, 2)
        svi_result = svi.run(
            sample_key,
            n_train_samples,
            stable_update=True,
            init_state=svi_state,
            true_EIRs=None,
            prev=obs_prev,
            inc=obs_inc,
            impl=impl,
            alpha=1./t
        )
        svi_params = svi_result.params
        svi_state = svi_result.state
    
        # sample posterior
        with jax.default_device(cpu_device):
            post_key, key = random.split(key, 2)
            posterior_samples = wo_latent(Predictive(
                guide,
                params=svi_params,
                num_samples=n_samples
            )(post_key))
    
            sample_sets.append(posterior_samples)

        # retrain surrogate
        if t != T[-1]:
            with jax.default_device(cpu_device):
                new_samples = sample_full_from_posterior(posterior_samples)
                X, y = training_samples
                X_new, y_new = new_samples
                training_samples = (
                    tree_map(lambda *x: jnp.concatenate(x), X, X_new),
                    tree_map(lambda *y: jnp.concatenate(y), y, y_new),
                )
            train_key, key = random.split(key, 2)
            params = train_surrogate(
                training_samples[0],
                training_samples[1],
                surrogate,
                net,
                mse,
                train_key,
                params_prior_full,
                epochs=n_epochs
            )
        

    return sample_sets, params
```

```{code-cell} ipython3
def surrogate_posterior_seq_svi(surrogate, net, params, key):
    rounds = 10
    n_samples = train_samples
    n_train_samples = 50_000

    training_samples = (X_prior_full, y_prior_full)

    sample_sets = list()

    guide = AutoBNAFNormal(model, num_flows=5)
    
    svi = SVI(
        model,
        guide,
        optim.ClippedAdam(step_size=1e-4),
        loss=Trace_ELBO(num_particles=8)
    )

    svi_state = None
    
    for i in range(rounds):
        impl = surrogate_impl_full(surrogate, net, params)
    
        # train SVI
        sample_key, key = random.split(key, 2)
        svi_result = svi.run(
            sample_key,
            n_train_samples,
            stable_update=True,
            init_state=svi_state,
            true_EIRs=None,
            prev=obs_prev,
            inc=obs_inc,
            impl=impl
        )
        svi_params = svi_result.params
        svi_state = svi_result.state
    
        # sample posterior
        with jax.default_device(cpu_device):
            post_key, key = random.split(key, 2)
            posterior_samples = wo_latent(Predictive(
                guide,
                params=svi_params,
                num_samples=n_samples
            )(post_key))
    
            sample_sets.append(posterior_samples)

        # retrain surrogate
        if i != (rounds - 1):
            with jax.default_device(cpu_device):
                new_samples = sample_full_from_posterior(posterior_samples)
                X, y = training_samples
                X_new, y_new = new_samples
                training_samples = (
                    tree_map(lambda *x: jnp.concatenate(x), X, X_new),
                    tree_map(lambda *y: jnp.concatenate(y), y, y_new),
                )
            train_key, key = random.split(key, 2)
            params = train_surrogate(
                training_samples[0],
                training_samples[1],
                surrogate,
                net,
                mse,
                train_key,
                params_prior_full,
                epochs=n_epochs
            )
        

    return sample_sets, params
```

```{code-cell} ipython3
def wo_latent(X):
    return {k: v for k, v in X.items() if k != '_auto_latent'}
```

```{code-cell} ipython3
prior_full_mcmc = surrogate_posterior_full(
    surrogate_prior_full,
    net_full,
    train_state_prior_full,
    key
)
X_post_prior_full = prior_full_mcmc.get_samples()
y_post_prior_full = prev_stats_posterior(X_post_prior_full)
```

```{code-cell} ipython3
prior_full_mcmc.print_summary()
```

```{code-cell} ipython3
y_post_prior_full_hat = prev_stats_full_surrogate_posterior(
    surrogate_prior_full,
    net_full,
    train_state_prior_full,
    X_post_prior_full
)
```

```{code-cell} ipython3
X_post_prior_full_svi = surrogate_posterior_full_svi(
    surrogate_prior_full,
    net_full,
    train_state_prior_full,
    key
)
```

```{code-cell} ipython3
X_post_prior_full_seq_svi, train_state_prior_full_seq = surrogate_posterior_seq_svi(
    surrogate_prior_full,
    net_full,
    train_state_prior_full,
    key
)
```

```{code-cell} ipython3
X_post_prior_full_annealed_svi, train_state_prior_full_annealed = surrogate_posterior_annealed_svi(
    surrogate_prior_full,
    net_full,
    train_state_prior_full,
    key
)
```

```{code-cell} ipython3
X_post_prior_full_annealed_svi_v2 = surrogate_posterior_annealed_svi(
    surrogate_prior_full,
    net_full,
    train_state_prior_full,
    key
)
```

```{code-cell} ipython3
y_val_prior = prev_stats_posterior(without_obs(prior))
```

```{code-cell} ipython3
y_val_prior_prior_full_hat = prev_stats_full_surrogate_posterior(
    surrogate_prior_full,
    net_full,
    train_state_prior_full,
    without_obs(prior)
)
```

```{code-cell} ipython3
import arviz as az

def _to_arviz_dict(samples):
    return {
        k: v[None, ...]
        for k, v in samples.items()
    }

prior_full_svi_pp = Predictive(model, X_post_prior_full_svi)(key_i)
prior_full_svi_idata = az.from_dict(
    posterior=_to_arviz_dict(X_post_prior_full_svi),
    posterior_predictive=_to_arviz_dict(prior_full_svi_pp),
    observed_data={
        'obs_prev': obs_prev,
        'obs_inc': obs_inc
    }
)

prior_full_pp = Predictive(model, X_post_prior_full)(key_i)
prior_full_idata = az.from_dict(
    posterior=_to_arviz_dict(X_post_prior_full),
    posterior_predictive=_to_arviz_dict(prior_full_pp),
    observed_data={
        'obs_prev': obs_prev,
        'obs_inc': obs_inc
    }
)
```

```{code-cell} ipython3
def sq_euclidean_dist(x, y):
    if len(x.shape) == 1:
        x = x.reshape(x.shape[0], 1)
    if len(y.shape) == 1:
        y = y.reshape(y.shape[0], 1)

    assert x.shape[-1] == y.shape[-1]

    dist = jnp.sum(
            jnp.square(x), axis=-1
            )[..., None] + jnp.sum(
                    jnp.square(y), axis=-1
                    )[..., None].T - 2 * jnp.dot(x, y.T)
    return dist

class SquaredExponential:
    """
    Squared exponential kernel.
    K(x1, x2) = var * exp(-0.5 * ||x1 - x2||^2/l**2)
    """

    def __init__(self, lengthscale=1., variance=1.):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, x1, x2):
        assert x1.shape[-1] == x2.shape[-1]
        dist = sq_euclidean_dist(x1/self.lengthscale, x2/self.lengthscale)
        k = self.variance * jnp.exp(-0.5 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


def squared_mmd(X, Y, kernel=SquaredExponential()): 
    X = jnp.array(X)
    Y = jnp.array(Y)
    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    n = K_XX.shape[0]
    m = K_YY.shape[0]

    mmd_squared = (
            jnp.sum(K_XX) - jnp.trace(K_XX)
            ) / (
                    (n * (n - 1)) +
                    (jnp.sum(K_YY) - jnp.trace(K_YY)) / (m * (m - 1)) -
                    2 * jnp.sum(K_XY) / (m * n)
                    )

    return mmd_squared
```

```{code-cell} ipython3
from jax.tree_util import tree_leaves

def vectorise_posterior(X):
    return jnp.concatenate(
        [
            l if l.ndim == 2 else l[...,None]
            for l in tree_leaves(X)
        ],
        axis=1
    )

X_vec_svi = vectorise_posterior(wo_latent(X_post_prior_full_svi))
X_vec_mcmc = vectorise_posterior(X_post_prior_full)
```

```{code-cell} ipython3
squared_mmd(X_vec_svi, X_vec_mcmc, kernel=SquaredExponential(10., 1.))
```

```{code-cell} ipython3
squared_mmd(X_vec_svi, X_vec_mcmc, kernel=SquaredExponential(10., 1.))
```

```{code-cell} ipython3
squared_mmd(X_vec_svi, X_vec_svi, kernel=SquaredExponential(10., 1.))
```

```{code-cell} ipython3
az.plot_bpv(prior_full_idata, kind='p_value')
```

```{code-cell} ipython3
az.plot_bpv(prior_full_svi_idata, kind='p_value')
```

```{code-cell} ipython3
def plot_predictive_error(y, y_hat):
    fig, axs = plt.subplots(5, len(EIRs), figsize=(50, 40))
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
            upper_lim = max(axs[i,j].get_ylim()[1], axs[i,j].get_xlim()[1])
            lower_lim = min(axs[i,j].get_ylim()[0], axs[i,j].get_xlim()[0])
            guide = jnp.linspace(lower_lim, upper_lim)
            axs[i, j].plot(guide, guide, c='r')

    fig.tight_layout()

    fig.text(0.5, 0, 'Predictive error', ha='center')
```

```{code-cell} ipython3
plot_predictive_error(y_val_prior, y_val_lhs_full_hat)
#plt.savefig('pe.png')
```

```{code-cell} ipython3
plot_predictive_error(y_post_prior_full, y_post_prior_full_hat)
```

```{code-cell} ipython3
prior_full_samples = prior_full_mcmc.get_samples()
prior_full_predictive = Predictive(
    model,
    prior_full_samples
)(key, obs_prev, obs_inc)
```

```{code-cell} ipython3
with jax.default_device(cpu_device):
    #NUTS_posterior_curves = get_curves(prior_full_samples, EIRs, etas)
    SVI_posterior_curves = get_curves(X_post_prior_full_annealed_svi[8], EIRs, etas)
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, len(EIRs), sharey=True, sharex=True)
imm_labels = ['b', 'c', 'd']
for i in range(len(EIRs)):
    axs[0, i].set_xlabel(
        f'EIR: {EIRs[i]}'
    )
    axs[0, i].xaxis.set_label_position('top')
    for imm_i, imm in enumerate([f'prob_{l}' for l in imm_labels]):
        #axs[imm_i, i].plot(NUTS_posterior_curves[imm][i, :n_curves, :].T, color='r', alpha=.01)
        axs[imm_i, i].plot(SVI_posterior_curves[imm][i, :n_curves, :].T, color='g', alpha=.01)
        axs[imm_i, i].plot(true_curves[imm][i, 0, :].T)
        axs[imm_i, 0].set_ylabel(f'prob. {imm_labels[imm_i]}')
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Surrogate posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, len(EIRs), sharey='row', sharex=True)

prev_labels = ['pos_M', 'inc']
for i in range(len(EIRs)):
    for prev_i, prev in enumerate(prev_labels):
        axs[0, i].set_xlabel(f'EIR: {EIRs[i]}')
        axs[0, i].xaxis.set_label_position('top')
        #axs[prev_i, i].plot(NUTS_posterior_curves[prev][i, :n_curves, :].T / NUTS_posterior_curves['prop'][i, :n_curves, :].T, color='r', alpha=.01)
        axs[prev_i, i].plot(SVI_posterior_curves[prev][i, :n_curves, :].T / SVI_posterior_curves['prop'][i, :n_curves, :].T, color='g', alpha=.01)
        axs[prev_i, i].plot(true_curves[prev][i, 0, :] / true_curves['prop'][i, 0, :])
        axs[prev_i, 0].set_ylabel(prev)
        
fig.tight_layout()
fig.text(0.5, 0, 'Age (years)', ha='center')
fig.text(0.5, 1, 'Surrogate posterior pos_M/inc function', ha='center')
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

def get_batch_imm_curves(params):
    return vmap(
        get_immunity_curve,
        in_axes=[
            {k: 0 for k in params.keys()}
        ]
    )(params)
```

```{code-cell} ipython3
true_imm_curves = get_batch_imm_curves(without_obs(true_values))
```

```{code-cell} ipython3
posterior_imm_curves = get_batch_imm_curves(X_post_prior_full)
```

```{code-cell} ipython3
posterior_imm_curves_svi = get_batch_imm_curves(X_post_prior_full_svi)
```

```{code-cell} ipython3
posterior_imm_curves_svi = get_batch_imm_curves(X_post_prior_full_annealed_svi[8])
```

```{code-cell} ipython3
posterior_imm_curves_svi = get_batch_imm_curves(X_post_prior_full_seq_svi[5])
```

```{code-cell} ipython3
n_curves = 500
fig, axs = plt.subplots(1, 3)
imm_labels = ['prob_b', 'prob_c', 'prob_d']

for imm_i, imm in enumerate(imm_labels):
    #axs[imm_i].plot(posterior_imm_curves[imm][:n_curves, :].T, color='r', alpha=.1)
    axs[imm_i].plot(posterior_imm_curves_svi[imm][:n_curves, :].T, color='g', alpha=.1)
    axs[imm_i].plot(true_imm_curves[imm][0, :])
    axs[imm_i].set_ylabel(imm)
        
fig.tight_layout()
fig.text(0.5, 0, 'Exposures (number)', ha='center')
fig.text(0.5, 1, 'Surrogate posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```{code-cell} ipython3
samples = sorted(
    list(
        {int(path.split('_')[0]) for path in glob('*_round_*_ll.csv')}
    )
)
rounds = list(range(5))
```

```{code-cell} ipython3
ks_error = pd.concat([
    pd.read_csv(f'{s}_round_{r}_svi_ks_error.csv').assign(samples=s, round=r, sampler='svi')
    for s in samples
    for r in rounds
] + [
    pd.read_csv(f'{s}_round_{r}_ks_error.csv').assign(samples=s, round=r, sampler='mcmc')
    for s in samples
    for r in rounds
])
```

```{code-cell} ipython3
g = sns.FacetGrid(
    ks_error,
    col="samples",
    row='sampler',
    hue='experiment',
    margin_titles=True
)
g.map(sns.lineplot, "round", "statistic")
g.add_legend()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(19.7, 8.27))
sns.barplot(
    ks_error[
        (ks_error.samples == 500000) &
        (ks_error['round'] == max(ks_error['round'])) &
        (ks_error['sampler'] == 'mcmc')
        #ks_error.variable.isin(['b0', 'phi0', 'phi1'])
    ],
    x='variable',
    y='statistic',
    hue='experiment',
    #estimator=lambda x: jnp.mean(jnp.abs(jnp.array(x))),
    #errorbar=('ci', 95),
    ax=ax
)
ax.set_xlabel('Model Parameter')
ax.set_ylabel('KS error with original method')
ax.set_title('Posterior agreement of surrogates with original model')
```

```{code-cell} ipython3
approx_error = pd.concat([
    pd.read_csv(f'{s}_round_{r}_svi_approx_error.csv').assign(samples=s, round=r, sampler='svi')
    for s in samples
    for r in rounds
] + [
    pd.read_csv(f'{s}_round_{r}_approx_error.csv').assign(samples=s, round=r, sampler='mcmc')
    for s in samples
    for r in rounds
])

stand_approx_error = pd.concat([
    pd.read_csv(f'{s}_round_{r}_svi_stand_approx_error.csv').assign(samples=s, round=r, sampler='svi')
    for s in samples
    for r in rounds
] + [
    pd.read_csv(f'{s}_round_{r}_stand_approx_error.csv').assign(samples=s, round=r, sampler='mcmc')
    for s in samples
    for r in rounds
])
```

```{code-cell} ipython3
g = sns.FacetGrid(
    stand_approx_error,
    row="test_set",
    col="samples",
    hue="experiment",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "round", "mse")
for ax in g.axes_dict.values():
    ax.set_yscale('log')
g.add_legend()
```

```{code-cell} ipython3
g = sns.FacetGrid(
    approx_error[approx_error['round'] == 4],
    row="test_set",
    col="output",
    hue="experiment",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "samples", "mse")
for ax in g.axes_dict.values():
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_yscale('log')
g.add_legend()
```

```{code-cell} ipython3
timings = pd.concat([
    pd.read_csv(f'{s}_round_{r}_timings.csv').assign(samples=s, round=r)
    for s in samples
    for r in rounds
])
```

```{code-cell} ipython3
timings = pd.melt(
    timings,
    id_vars=['experiment', 'samples', 'round'],
    value_vars=['sampling', 'training', 'mcmc', 'svi'],
    var_name='task',
    value_name='time'
)
```

```{code-cell} ipython3
sns.lineplot(
    timings[(timings.task.isin(['sampling', 'training'])) & (timings['round'] == 0)],
    x="samples",
    y="time",
    hue='task'
)
```

```{code-cell} ipython3
sns.violinplot(timings[timings.task.isin(['svi', 'mcmc'])], x='task', y='time')
```

```{code-cell} ipython3
ll = pd.concat([
    pd.read_csv(
        f'{s}_round_{r}_ll.csv'
    ).assign(samples=s, round=r, sampler='mcmc')
    for s in samples
    for r in rounds
] + [
    pd.read_csv(
        f'{s}_round_{r}_svi_ll.csv'
    ).assign(samples=s, round=r, sampler='svi')
    for s in samples
    for r in rounds
])
```

```{code-cell} ipython3
u_ll = pd.read_csv('underlying_ll.csv')
```

```{code-cell} ipython3
u_ll = pd.melt(
    u_ll,
    id_vars=['experiments'],
    value_vars=['obs_inc', 'obs_prev'],
    var_name='obs',
    value_name='log_likelihood'
)
```

```{code-cell} ipython3
ll = pd.melt(
    ll,
    id_vars=['experiments', 'samples', 'round', 'sampler'],
    value_vars=['obs_inc', 'obs_prev'],
    var_name='obs',
    value_name='log_likelihood'
)
```

```{code-cell} ipython3
u_ll.log_likelihood
```

```{code-cell} ipython3
g = sns.FacetGrid(
    ll[ll.obs == 'obs_inc'],
    col='samples',
    row="sampler",
    hue="experiments",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "round", "log_likelihood")
for (e, s), ax in g.axes_dict.items():
    #ax.set_yscale('log')
    y = u_ll[u_ll.obs == 'obs_inc'].log_likelihood.iloc[0]
    ax.axhline(y, 0, 4, ls='--')
    #ax.fill_between(list(range(5)), y.min(), y.max(), color='black', alpha=.1)
g.add_legend()
```

```{code-cell} ipython3
pp_ll = pd.concat([
    pd.read_csv(
        f'{s}_round_{r}_pp_ll.csv'
    ).assign(samples=s, round=r, sampler='mcmc')
    for s in samples
    for r in rounds
] + [
    pd.read_csv(
        f'{s}_round_{r}_svi_pp_ll.csv'
    ).assign(samples=s, round=r, sampler='svi')
    for s in samples
    for r in rounds
])
```

```{code-cell} ipython3
pp_ll = pd.melt(
    pp_ll,
    id_vars=['experiment', 'samples', 'round', 'sampler', 'EIR'],
    value_vars=['prev_2_10', 'prev_10+', 'inc_0_5', 'inc_5_15', 'inc_15+'],
    var_name='output',
    value_name='log_likelihood'
)
```

```{code-cell} ipython3
u_pp_ll = pd.read_csv('underlying_pp_ll.csv')
```

```{code-cell} ipython3
u_pp_ll = pd.melt(
    u_pp_ll,
    id_vars=['experiment', 'EIR'],
    value_vars=['prev_2_10', 'prev_10+', 'inc_0_5', 'inc_5_15', 'inc_15+'],
    var_name='output',
    value_name='log_likelihood'
)
```

```{code-cell} ipython3
g = sns.FacetGrid(
    pp_ll,
    col='samples',
    row="sampler",
    hue="experiment",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "round", "log_likelihood")
for (e, s), ax in g.axes_dict.items():
    #ax.set_yscale('log')
    y = u_pp_ll.log_likelihood
    ax.axhline(y.mean(), 0, 4, ls='--')
    ax.fill_between(list(range(5)), y.min(), y.max(), color='black', alpha=.1)
g.add_legend()
```

```{code-cell} ipython3
summaries = pd.concat([
    pd.read_csv(f'{s}_round_{r}_mcmc_summary.csv').assign(samples=s, round=r)
    for s in samples
    for r in rounds
])
summaries = summaries[summaries['index'] != 'EIR']
summaries.r_hat = pd.to_numeric(summaries.r_hat)
summaries.n_eff = pd.to_numeric(summaries.n_eff)
```

```{code-cell} ipython3
g = sns.FacetGrid(
    summaries,
    col='samples',
    hue="experiment",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "round", "r_hat")
g.add_legend()
```

```{code-cell} ipython3
g = sns.FacetGrid(
    summaries,
    col='samples',
    hue="experiment",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "round", "n_eff")
g.add_legend()
```

```{code-cell} ipython3

```
