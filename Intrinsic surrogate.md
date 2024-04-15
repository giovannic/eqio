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
 * History matching

# Insights

 * longer chain convergence for surrogate

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
from mox.sampling import LHSStrategy
```

```{code-cell} ipython3
n_chains = 4
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
from mox.sampling import DistStrategy
```

```{code-cell} ipython3
# TODO: take this from the model
# noise = 10.
# est_EIR = dist.TruncatedNormal(EIRs, noise, low=0., high=500.).sample(key)

prior_train_space = [
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
        'd1': DistStrategy(dist.Beta(1., 1.)),
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
    #DistStrategy(SiteDistribution({'EIR': est_EIR, 'etas': etas}, noise))
    DistStrategy(dist.Uniform(0., 500.)), # EIR
    DistStrategy(dist.Uniform(1/(100 * 365), 1/(40 * 365))) # eta
]

prior_test_space = [
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
        'd1': DistStrategy(dist.Beta(1., 1.)),
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
def model(true_EIRs=None, prev=None, inc=None, impl=lambda p, e, a: prev_stats_multisite(p, e, a, full_solution)):
    with numpyro.plate('sites', n_sites):
        EIR = numpyro.sample('EIR', dist.Uniform(0., 500.), obs=true_EIRs)
    
    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.LogNormal(0., .25))
    ub = numpyro.sample('ub', dist.LogNormal(0., 1.))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    IB0 = numpyro.sample('IB0', dist.TruncatedDistribution(dist.Normal(50., 20.), low=25., high=75.))
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., .25))
    uc = numpyro.sample('uc', dist.LogNormal(0., 1.))
    phi0 = numpyro.sample('phi0', dist.Beta(2., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 5.))
    IC0 = numpyro.sample('IC0',dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.))
    PM = numpyro.sample('PM', dist.Beta(1., 1.))
    dm = numpyro.sample('dm', dist.TruncatedDistribution(dist.Normal(50., 20.), low=5., high=100.))
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., .25))
    ud = numpyro.sample('ud', dist.LogNormal(0., 1.))
    d1 = numpyro.sample('d1', dist.Beta(1., 1.))
    ID0 = numpyro.sample('ID0', dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.))
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
obs_inc, obs_prev = (true_values['obs_inc'], true_values['obs_prev'])
```

```{code-cell} ipython3
def without_obs(params):
    return {k : v for k, v in params.items() if not k in {'obs_inc', 'obs_prev'}}
```

```{code-cell} ipython3
key, key_i = random.split(key)
prior = Predictive(model, num_samples=1000)(key)
```

```{code-cell} ipython3
from jax import pmap, tree_map
import jax
import pandas as pd

train_samples = int(1e5)
device_count = len(jax.devices())
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
from flax import linen as nn

class MLP(nn.Module):
    """MLP. A multi layer perceptron
    """

    units: int
    n_hidden: int
    n_output: int
    dropout_rate: float
    batch_norm: bool
    dtype = jnp.float64

    @nn.compact
    def __call__(self, x, training: bool):
        denses = [nn.Dense(self.units, param_dtype=self.dtype) for _ in range(self.n_hidden)]
        dropouts = [
            nn.Dropout(rate=self.dropout_rate, deterministic=not training)
            for _ in range(self.n_hidden)
        ]
        if self.batch_norm:
            norms = [
                nn.BatchNorm(use_running_average=not training)
                for _ in range(self.n_hidden)
            ]
            layers = zip(denses, dropouts, norms)
            for dense, dropout, norm in layers:
                x = dense(x)
                x = norm(x)
                x = dropout(x)
                x = nn.relu(x)
        else:
            layers = zip(denses, dropouts)
            for dense, dropout in layers:
                x = dense(x)
                x = dropout(x)
                x = nn.relu(x)

        return nn.Dense(self.n_output)(x)
```

```{code-cell} ipython3
def make_net(surrogate, y):
    y0 = tree_map(lambda x: x[0], y)
    y0_vec = surrogate.vectorise_output(y0)
    return MLP(
        units=265,
        n_hidden=2,
        n_output=jnp.size(y0_vec),
        dropout_rate=.2,
        batch_norm=False
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
    params_lhs_full
)
```

```{code-cell} ipython3
key_i, key = random.split(key)
with jax.default_device(cpu_device):
    X_prior_full = sample(prior_train_space, train_samples, key_i)
    y_prior_full = vmap(full_solution, in_axes=tree_map(lambda x: 0, X_prior_full))(*X_prior_full)
```

```{code-cell} ipython3
surrogate_prior_full = make_surrogate(
    X_prior_full,
    y_prior_full,
    y_min=y_min_full,
    y_max=y_max_full
)
key_i, key = random.split(key)

params_prior_full = init_surrogate(key_i, surrogate_prior_full, net_full, X_prior_full)
train_state_prior_full = train_surrogate(
    X_prior_full,
    y_prior_full,
    surrogate_prior_full,
    net_full,
    mse,
    key_i,
    params_prior_full
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
params_lhs_fixed = init_surrogate(key_i, surrogate_lhs_fixed, net_fixed, X_lhs_fixed)
train_state_lhs_fixed = train_surrogate(
    X_lhs_fixed,
    y_lhs_fixed,
    surrogate_lhs_fixed,
    net_fixed,
    mse,
    key_i,
    params_lhs_fixed
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
    params_prior_fixed
)
```

```{code-cell} ipython3
# Write function for validation set generation
def apply_dmeq_surrogate(surrogate, net, net_params, params, eir, eta):
    return tree_map(
        lambda leaf: leaf[0, 0],
        apply_surrogate(
            surrogate,
            net,
            {'params': net_params},
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
    sensitivity = vmap(jacfwd(densities), in_axes=[tree_map(lambda _: 0, without_obs(prior)), None, None])(without_obs(prior), model, impl)
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
    surrogate_impl_full(surrogate_prior_full, net_full, train_state_prior_full.params)
)
```

```{code-cell} ipython3
sensitivity_prior_fixed = get_sensitivity(
    surrogate_impl_fixed(surrogate_prior_fixed, net_fixed, train_state_prior_fixed.params)
)
sensitivity_lhs_fixed = get_sensitivity(
    surrogate_impl_fixed(surrogate_lhs_fixed, net_fixed, train_state_lhs_fixed.params)
)
```

```{code-cell} ipython3
sensitivity_lhs_full = get_sensitivity(
    surrogate_impl_full(surrogate_lhs_full, net_full, train_state_lhs_full.params)
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
            train_state_prior_full.params,
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
lhs_full_mcmc = surrogate_posterior_full(
    surrogate_lhs_full,
    net_full,
    train_state_lhs_full.params,
    key
)
X_post_lhs_full = lhs_full_mcmc.get_samples()
```

```{code-cell} ipython3
y_post_lhs_full = prev_stats_posterior(X_post_lhs_full)
y_post_lhs_full_hat = prev_stats_full_surrogate_posterior(
    surrogate_lhs_full,
    net_full,
    train_state_lhs_full.params,
    X_post_lhs_full
)
```

```{code-cell} ipython3
lhs_fixed_mcmc = surrogate_posterior_fixed(
    surrogate_lhs_fixed,
    net_fixed,
    train_state_lhs_fixed.params,
    key
)
X_post_lhs_fixed = lhs_fixed_mcmc.get_samples()
y_post_lhs_fixed = prev_stats_posterior(X_post_lhs_fixed)
y_post_lhs_fixed_hat = prev_stats_fixed_surrogate_posterior(
    surrogate_lhs_fixed,
    net_fixed,
    train_state_lhs_fixed.params,
    X_post_lhs_fixed
)
```

```{code-cell} ipython3
prior_full_mcmc = surrogate_posterior_full(
    surrogate_prior_full,
    net_full,
    train_state_prior_full.params,
    key
)
X_post_prior_full = prior_full_mcmc.get_samples()
y_post_prior_full = prev_stats_posterior(X_post_prior_full)
y_post_prior_full_hat = prev_stats_full_surrogate_posterior(
    surrogate_prior_full,
    net_full,
    train_state_prior_full.params,
    X_post_prior_full
)
```

```{code-cell} ipython3
prior_fixed_mcmc = surrogate_posterior_fixed(
    surrogate_prior_fixed,
    net_fixed,
    train_state_prior_fixed.params,
    key
)
X_post_prior_fixed = prior_fixed_mcmc.get_samples()
y_post_prior_fixed = prev_stats_posterior(X_post_prior_fixed)
y_post_prior_fixed_hat = prev_stats_fixed_surrogate_posterior(
    surrogate_prior_fixed,
    net_fixed,
    train_state_prior_fixed.params,
    X_post_prior_fixed
)
```

```{code-cell} ipython3
val_size = int(1e4)
y_val_prior = prev_stats_posterior(without_obs(prior))
```

```{code-cell} ipython3
y_val_lhs_full_hat = prev_stats_full_surrogate_posterior(
    surrogate_lhs_full,
    net_full,
    train_state_lhs_full.params,
    without_obs(prior)
)
y_val_prior_fixed_hat = prev_stats_fixed_surrogate_posterior(
    surrogate_prior_fixed,
    net_fixed,
    train_state_prior_fixed.params,
    without_obs(prior)
)
```

```{code-cell} ipython3
y_val_prior_prior_full_hat = prev_stats_surrogate_batch(
    surrogate_prior_full,
    prior_full_state,
    sort_dict(without_obs(prior))
)
```

```{code-cell} ipython3
y_val_prior_comb_full_hat = prev_stats_surrogate_batch(surrogate_comb_full, comb_full_state, sort_dict(without_obs(prior)))
```

```{code-cell} ipython3
y_val_prior_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_prior_fixed, params_prior_fixed, sort_dict(without_obs(prior)))

y_val_lhs_prior_full_hat = prev_stats_surrogate_batch(surrogate_prior_full, params_prior_full, X_val_lhs)
y_val_lhs_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_prior_fixed, params_prior_fixed, X_val_lhs)
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
plot_predictive_error(y_post_prior_fixed, y_post_prior_fixed_hat)
```

```{code-cell} ipython3
plot_predictive_error(y_post_lhs_fixed, y_post_lhs_fixed_hat)
```

```{code-cell} ipython3
def approximation_error(exps, labels, ys, y_hats):
    y_labels = ['prev2-10', 'prev10+', 'inc0-5', 'inc5-15', 'inc15+']
    ys = [jnp.concatenate(y, axis=2) for y in ys]
    y_hats = [jnp.concatenate(y_hat, axis=2) for y_hat in y_hats]
    return pd.DataFrame([
        {
            'mse': jnp.mean(jnp.square(y - y_hat)[:, i, j]),
            'RE': jnp.mean(jnp.abs(y - y_hat)[:, i, j] / y[:, i, j]),
            'EIR': float(EIRs[i]),
            'output': y_labels[j],
            'test_set': label,
            'experiment': exp
        }
        for i in range(len(EIRs))
        for j in range(len(y_labels))
        for exp, label, y, y_hat in zip(exps, labels, ys, y_hats)
    ])
```

```{code-cell} ipython3
approximation_error(
    ['lhs_full', 'prior_fixed'],
    ['prior', 'prior'],
    [y_val_prior, y_val_prior],
    [y_val_lhs_full_hat, y_val_prior_fixed_hat]
)
```

```{code-cell} ipython3
# TODO: why are these so far?
```

```{code-cell} ipython3
labels, posteriors = ['lhs_full'], [X_post_lhs_full]
prev_columns = ['prev_2_10', 'prev_10+']
inc_columns = ['inc_0_5', 'inc_5_15', 'inc_15+']
keys = random.split(key, len(posteriors))
posterior_predictives = [
    Predictive(model, p)(key_i)
    for key_i, p in zip(keys, posteriors)
]
label, pp = list(zip(labels, posterior_predictives))[0]
pd.concat([
    pd.DataFrame(
        jnp.mean(jnp.square(obs_inc - pp['obs_inc']), axis=0),
        columns=inc_columns
    ).assign(EIR=EIRs),
    pd.DataFrame(
        jnp.mean(jnp.square(obs_prev - pp['obs_prev']), axis=0),
        columns=prev_columns
    ).assign(EIR=EIRs),
], axis=1).assign(experiment=label)
```

```{code-cell} ipython3
def mspe(key, labels, posteriors):
    prev_columns = ['prev_2_10', 'prev_10+']
    inc_columns = ['inc_0_5', 'inc_5_15', 'inc_15+']
    keys = random.split(key, len(posteriors))
    posterior_predictives = [
        Predictive(model, p)(key_i)
        for key_i, p in zip(keys, posteriors)
    ]
    return pd.concat([
        pd.concat([
            pd.DataFrame(
                jnp.mean(jnp.square(obs_inc - pp['obs_inc']), axis=0),
                columns=inc_columns
            ),
            pd.DataFrame(
                jnp.mean(jnp.square(obs_prev - pp['obs_prev']), axis=0),
                columns=prev_columns
            ),
        ], axis=1).assign(experiment=label)
        for label, pp in zip(labels, posterior_predictives)
    ])

mspe(key, ['lhs_full'], [X_post_lhs_full])
```

```{code-cell} ipython3
print(approximation_error(
    ['lhs_full'] * 3 + ['lhs_fixed'] * 3 + ['prior_full'] * 3 + ['prior_fixed'] * 3,
    ['prior', 'lhs', 'posterior'] * 4,
    [y_val_prior, y_val_lhs, y_post_lhs_full, y_val_prior, y_val_lhs, y_post_lhs_fixed, y_val_prior, y_val_lhs, y_post_prior_full, y_val_prior, y_val_lhs, y_post_prior_fixed],
    [y_val_prior_full_hat, y_val_lhs_full_hat, y_post_lhs_full_hat, y_val_prior_fixed_hat, y_val_lhs_fixed_hat, y_post_lhs_fixed_hat,
     y_val_prior_prior_full_hat, y_val_lhs_prior_full_hat, y_post_prior_full_hat, y_val_prior_prior_fixed_hat, y_val_lhs_prior_fixed_hat, y_post_prior_fixed_hat]
).groupby(['experiment', 'test_set'], as_index=False).agg({'RE': 'mean'}).pivot(index='experiment', columns='test_set', values='RE').to_latex(float_format="{:0.2f}".format))#, 'EIR', 'output']).mean()
```

```{code-cell} ipython3
jnp.mean(jnp.abs(jnp.concatenate(y_val_prior, axis=2) - jnp.concatenate(y_val_prior_comb_full_hat, axis=2)) / jnp.concatenate(y_val_prior, axis=2))
```

```{code-cell} ipython3
jnp.mean(jnp.abs(jnp.concatenate(y_post_comb_full, axis=2) - jnp.concatenate(y_post_comb_full_hat, axis=2)) / jnp.concatenate(y_post_comb_full, axis=2))
```

```{code-cell} ipython3
jnp.mean(jnp.abs(jnp.concatenate(y_post_prior_full, axis=2) - jnp.concatenate(y_post_prior_full_hat, axis=2)) / jnp.concatenate(y_post_prior_full, axis=2))
```

```{code-cell} ipython3
approximation_error(
    ['prior_full'],
    ['prior'],
    [y_val_prior],
    [y_val_prior_prior_full_hat]
).groupby(['output', 'EIR']).agg({'RE': 'mean', 'L1': 'mean'})
```

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
mcmc.run(key, None, obs_prev, obs_inc)
mcmc.print_summary(prob=0.7)
```

```{code-cell} ipython3
import pickle
with open('chapter_2_mcmc_underlying.pkl', 'rb') as f:
    mcmc = pickle.load(f)
```

```{code-cell} ipython3
posterior_samples = mcmc.get_samples()
posterior_predictive = Predictive(
    model,
    posterior_samples
)(key, obs_prev, obs_inc)
```

```{code-cell} ipython3
prior_full_samples = prior_full_mcmc.get_samples()
prior_full_predictive = Predictive(
    model,
    prior_full_samples
)(key, obs_prev, obs_inc)
```

```{code-cell} ipython3
lhs_fixed_samples = lhs_fixed_mcmc.get_samples()
lhs_fixed_predictive = Predictive(
    model,
    lhs_fixed_samples
)(key, obs_prev, obs_inc)
```

```{code-cell} ipython3
from scipy.stats import ks_2samp
sample_keys = list(posterior_samples.keys())
ks_data = pd.concat([
    pd.DataFrame([
        {
            'experiment': name,
            'variable': k,
            'statistic': ks_2samp(posterior_samples[k], posterior[k]).statistic,
            #'p-value': ks_2samp(posterior_samples[k], posterior[k]).pvalue,
            'mean': float(jnp.mean(posterior[k])),
            'true_mean': float(jnp.mean(posterior_samples[k])),
        }
        for k in sample_keys
        for name, posterior in [
            #('prior_fixed', prior_fixed_mcmc.get_samples()),
            ('prior_full', prior_full_mcmc.get_samples()),
            ('lhs_fixed', lhs_fixed_mcmc.get_samples()),
            #('lhs_full', lhs_full_mcmc.get_samples()),
            #('comb_full', comb_full_mcmc.get_samples())
        ]
        if k != 'EIR'
    ]),
    pd.DataFrame([
        {
            'experiment': name,
            'variable': f'EIR_{i}',
            'statistic': ks_2samp(posterior_samples['EIR'][:,i], posterior['EIR'][:,i]).statistic,
            #'p-value': ks_2samp(posterior_samples['EIR'][:,i], posterior['EIR'][:,i]).pvalue,
            'mean': float(jnp.mean(posterior['EIR'][:,i])),
            'true_mean': float(jnp.mean(posterior_samples['EIR'][:,i]))
        }
        for name, posterior in [
            #('prior_fixed', prior_fixed_mcmc.get_samples()),
            ('prior_full', prior_full_mcmc.get_samples()),
            #('comb_full', comb_full_mcmc.get_samples())
            ('lhs_fixed', lhs_fixed_mcmc.get_samples()),
            #('lhs_full', lhs_full_mcmc.get_samples())
            #('comb_full', comb_full_mcmc.get_samples())
        ]
        for i in range(posterior['EIR'].shape[1])
    ])
])
```

```{code-cell} ipython3
ks_data.pivot(
    columns='experiment',
    index='variable'
).swaplevel(axis='columns').sort_index(axis=1, level=0).loc[
    [f'EIR_{i}' for i in range(n_sites)] +
    [
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
    ]
]
```

```{code-cell} ipython3
print(ks_data.pivot(columns='experiment', index='variable').swaplevel(axis='columns').sort_index(axis=1, level=0).loc[[
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
]].style.format(precision=2).to_latex())
```

```{code-cell} ipython3
from flax.training import orbax_utils
import orbax.checkpoint

def save_model(name, surrogate, params):
    ckpt = {
        'surrogate': surrogate,
        'params': params
    }
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(f'orbax/{name}', ckpt, force=True, save_args=save_args)
    
save_model(f'prior_fixed_{train_samples}', surrogate_prior_fixed, params_prior_fixed)
```

```{code-cell} ipython3
pyro_data = az.from_numpyro(
    mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)
pyro_data_surrogate = az.from_numpyro(
    prior_full_mcmc,
    prior=prior,
    posterior_predictive=comb_full_predictive
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
        
for i in range(axs.shape[0]):
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
posterior_curves_surrogate = get_curves(prior_full_samples, EIRs, etas)
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
        axs[imm_i, i].plot(posterior_curves_surrogate[imm][i, :n_curves, :].T, color='r', alpha=.01)
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
        axs[prev_i, i].plot(posterior_curves_surrogate[prev][i, :n_curves, :].T / posterior_curves_surrogate['prop'][i, :n_curves, :].T, color='r', alpha=.01)
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
posterior_imm_curves = get_batch_imm_curves(prior_full_samples)
true_imm_curves = get_batch_imm_curves(without_obs(true_values))
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
        {int(path.split('_')[0]) for path in glob('*_round_*_approx_error.csv')}
    )
)
rounds = list(range(5))
```

```{code-cell} ipython3
ks_error = pd.concat([
    pd.read_csv(f'{s}_round_{r}_ks_error.csv').assign(samples=s, round=r)
    for s in samples
    for r in rounds
])
```

```{code-cell} ipython3
g = sns.FacetGrid(
    ks_error,
    col="samples",
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
        (ks_error['round'] == max(ks_error['round']))
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
    pd.read_csv(f'{s}_round_{r}_approx_error.csv').assign(samples=s, round=r)
    for s in samples
    for r in rounds
])
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
g = sns.FacetGrid(
    approx_error,
    row="test_set",
    col="samples",
    hue="experiment",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "round", "mse")
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
    value_vars=['sampling', 'training', 'mcmc'],
    var_name='task',
    value_name='time'
)
```

```{code-cell} ipython3
sns.lineplot(
    timings[(timings.task == 'sampling') & (timings['round'] == 0)],
    x="samples",
    y="time"
)
```

```{code-cell} ipython3
sns.boxplot(timings[(timings.task == 'mcmc')][['time']])
```

```{code-cell} ipython3
sns.boxplot(timings[(timings.task == 'training')][['time']])
```

```{code-cell} ipython3
mspe = pd.concat([
    pd.read_csv(f'{s}_round_{r}_mspe.csv').assign(samples=s, round=r)
    for s in samples
    for r in rounds
])
u_mspe = pd.read_csv('underlying_mspe.csv')
```

```{code-cell} ipython3
mspe = pd.melt(
    mspe,
    id_vars=['experiment', 'samples', 'EIR', 'round'],
    value_vars=['inc_0_5', 'inc_5_15', 'inc_15+', 'prev_2_10', 'prev_10+'],
    var_name='output',
    value_name='mspe'
)
u_mspe = pd.melt(
    u_mspe,
    id_vars=['experiment', 'EIR'],
    value_vars=['inc_0_5', 'inc_5_15', 'inc_15+', 'prev_2_10', 'prev_10+'],
    var_name='output',
    value_name='mspe'
)
```

```{code-cell} ipython3
g = sns.FacetGrid(
    mspe,
    col='samples',
    row="EIR",
    hue="experiment",
    margin_titles=True
)
g.map(sns.lineplot, "round", "mspe")
for (e, s), ax in g.axes_dict.items():
    ax.set_yscale('log')
    y = u_mspe[u_mspe.EIR == e].mspe.mean()
    ax.axhline(y, 0, 4, ls='--')
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
```

```{code-cell} ipython3
g = sns.FacetGrid(
    summaries[summaries['index'] != 'EIR'],
    col='samples',
    hue="experiment",
    margin_titles=True
)
g.map(sns.lineplot, "round", "r_hat")
g.add_legend()
```

```{code-cell} ipython3
summaries[
    (summaries['index'] != 'EIR') &
    (summaries.samples == 500000) &
    (summaries['round'] == 0) &
    (summaries['experiment'] == 'lhs_fixed')
][['index', 'samples', 'round', 'r_hat', 'experiment']]
```

```{code-cell} ipython3

```
