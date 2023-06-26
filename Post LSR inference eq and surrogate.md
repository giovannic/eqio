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
 * convergence
 * ~investigate feature learning~
   * ~Refactor module setup out of Surrogate~
   * ~Create standardiser from full surrogate~
   * ~Take vectoriser from full surrogate~
   * ~Refactor MLP module into features and output~
   * ~Create TransferredNN from full MLP~
   * ~Make a Surrogate class from new components~
   * ~Refactor train_surrogate to take an optimiser~
   * ~vmap the Transferred learner~
 * investigate robust training
   * regularisation (dropout/ l2)
   * gradient towards high loss (in progress)
 * ~investigate history matching~
 * ~try removing the limiter~
 * ~Increase sample size~
 * Consolidate prior parameter space and numpyro model
 * Train models separately for loading and evaluation

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
        jnp.maximum(
            jnp.array([
                inc_rates[:5].sum(), # Inc 0 - 5
                inc_rates[5:15].sum(), # Inc 5 - 15
                inc_rates[15:].sum() # Inc 15+
            ]),
            1e-12
        )
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
from mox.sampling import DistStrategy
```

```{code-cell} ipython3
prior_parameter_space = [
    {
        'kb': DistStrategy(dist.LogNormal(0., .25)),
        'ub': DistStrategy(dist.LogNormal(0., .25)),
        'b0': DistStrategy(dist.Beta(1., 1.)),
        'IB0': DistStrategy(dist.LeftTruncatedDistribution(dist.Normal(50., 10.), low=0.)),
        'kc': DistStrategy(dist.LogNormal(0., .25)),
        'uc': DistStrategy(dist.LogNormal(0., .25)),
        'IC0': DistStrategy(dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)),
        'phi0': DistStrategy(dist.Beta(5., 1.)),
        'phi1': DistStrategy(dist.Beta(1., 2.)),
        'PM': DistStrategy(dist.Beta(1., 1.)),
        'dm': DistStrategy(dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.)),
        'kd': DistStrategy(dist.LogNormal(0., .25)),
        'ud': DistStrategy(dist.LogNormal(0., .25)),
        'd1': DistStrategy(dist.Beta(1., 2.)),
        'ID0': DistStrategy(dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.)),
        'fd0': DistStrategy(dist.Beta(1., 1.)),
        'gd': DistStrategy(dist.LogNormal(0., .25)),
        'ad0': DistStrategy(dist.TruncatedDistribution(
            dist.Cauchy(30. * 365., 365.),
            low=20. * 365.,
            high=40. * 365.
        )),
        'rU': DistStrategy(dist.LogNormal(0., .25)),
        'cD': DistStrategy(dist.Beta(1., 2.)),
        'cU': DistStrategy(dist.Beta(1., 5.)),
        'g_inf': DistStrategy(dist.LogNormal(0., .25))
    },
    DistStrategy(dist.Uniform(0., 500.)), # EIR
    DistStrategy(dist.Uniform(1/(40 * 365), 1/(20 * 365))) # eta
]
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
    IC0 = numpyro.sample('IC0',dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.))
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
    g_inf = numpyro.sample('g_inf', dist.LogNormal(0., .25))
    
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
        'rU': ru,
        'cD': cd,
        'cU': cu,
        'g_inf': g_inf
    }
    
    prev_stats, inc_stats = impl(x, EIRs, etas)
    
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
            dist.Poisson(rate=jnp.maximum(inc_stats, 1e-12), validate_args=True),
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

x_min = [{
    name: lower
    for name, lower, _ in intrinsic_bounds.itertuples(index=False)
}]

x_max = [{
    name: upper
    for name, _, upper in intrinsic_bounds.itertuples(index=False)
}]
```

```{code-cell} ipython3
print(pd.concat([intrinsic_bounds]).to_latex(index=False, float_format="{:0.0f}".format))
```

```{code-cell} ipython3
max_val = jnp.finfo(jnp.float32).max
min_val = jnp.finfo(jnp.float32).smallest_normal
```

```{code-cell} ipython3
from mox.sampling import sample
from mox.surrogates import make_surrogate, pytree_init
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
    surrogate_lhs_full,
    mse,
    key_i
)
```

```{code-cell} ipython3
key_i, key = random.split(key)
X_prior_full = sample(prior_parameter_space, train_samples, key_i)
y_prior_full = vmap(full_solution, in_axes=tree_map(lambda x: 0, X_prior_full))(*X_prior_full)

surrogate_prior_full = make_surrogate(
    X_prior_full,
    y_prior_full,
    y_min=y_min_full,
    y_max=y_max_full
)
key_i, key = random.split(key)
params_prior_full = train_surrogate(
    X_prior_full,
    y_prior_full,
    surrogate_prior_full,
    mse,
    key_i
)
```

```{code-cell} ipython3
key_i, key = random.split(key)
X_lhs_fixed = sample(lhs_parameter_space[0:1], train_samples, key_i)
y_lhs_fixed = vmap(lambda p: prev_stats_multisite(p, EIRs, etas, full_solution), in_axes=[{n: 0 for n in intrinsic_bounds.name}])(*X_lhs_fixed)

y_min_fixed = (0., min_val)
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

```{code-cell} ipython3
key_i, key = random.split(key)
X_prior_fixed = sample(prior_parameter_space[0:1], train_samples, key_i)
y_prior_fixed = vmap(lambda p: prev_stats_multisite(p, EIRs, etas, full_solution), in_axes=tree_map(lambda x: 0, X_prior_fixed))(*X_prior_fixed)

surrogate_prior_fixed = make_surrogate(
    X_prior_fixed,
    y_prior_fixed,
    y_min=y_min_fixed,
    y_max=y_max_fixed
)
key_i, key = random.split(key)
params_prior_fixed = train_surrogate(
    X_prior_fixed,
    y_prior_fixed,
    surrogate_prior_fixed,
    mse,
    key_i
)
```

```{code-cell} ipython3
from flax.linen.module import _freeze_attr
```

```{code-cell} ipython3
# TODO: This is broken

from mox.active import active_training

key, key_i = random.split(key)

f = lambda p: prev_stats_multisite(p, EIRs, etas, full_solution)

def neg_loss(x, surrogate, params):
    return -jnp.sum(vmap(lambda x: mse(surrogate.apply(params, x), f(x[0])), in_axes=[tree_map(lambda _: 0, x)])(x))

params_lhs_fixed_active = active_training(
    X_lhs_fixed,
    y_lhs_fixed,
    surrogate_lhs_fixed,
    mse,
    key_i,
    f,
    neg_loss,
    lhs_parameter_space[0:1],
    x_min=_freeze_attr(x_min),
    x_max=_freeze_attr(x_max)
)
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
# Transferred model
from flax import linen as nn
from typing import List, Tuple
from jaxtyping import Array, PyTree
from mox.surrogates import summary, Recover, Limiter, InverseStandardiser, _standardise, MLP, Standardiser, Vectoriser
from jax.tree_util import tree_leaves, tree_structure

class TransferredModel(nn.Module):
    x_mean: PyTree
    x_std: PyTree
    y_shapes: List[Tuple]
    y_boundaries: Tuple
    y_mean: PyTree
    y_std: PyTree
    y_min: PyTree
    y_max: PyTree
    units: int
    n_take: int
    n_hidden: int
    eirs: Array
    etas: Array
    
    def setup(self):
        self.rec = Recover(
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries
        )
        self.limiter = Limiter(self.y_min, self.y_max)
        self.inv_std = InverseStandardiser(self.y_mean, self.y_std)
        self.full_features = MLP(self.units, self.n_take, self.units)
        self.output_layers = MLP(self.units, self.n_hidden, self.y_boundaries[-1])
        self.std = Standardiser(self.x_mean, self.x_std)
        self.vec = Vectoriser()
        
    def __call__(self, x):
        return self.inv_std(self.limiter(self.unstandardised(x)))
    
    def unstandardised(self, x):
        # full_features = [ #TODO: vmap the feature learning
        #     self.full_features(self.vec(self.std(x + (eir, eta))))
        #     for eir, eta in zip(self.eirs, self.etas)
        # ]
        # x = jnp.concatenate(full_features)
        full_features = vmap(
            lambda eir, eta: self.full_features(self.vec(self.std(x + (eir, eta))))
        )(self.eirs, self.etas)
        x = full_features.reshape(-1)
        y = self.output_layers(x)
        y = self.rec(y)
        return y
    
def create_transferred_model(
        base,
        x,
        y,
        y_std_axis = None,
        y_min = None,
        y_max = None,
        units = 256,
        n_take = 2,
        n_hidden = 1
    ) -> nn.Module:
    y_mean, y_std = summary(y, y_std_axis)
    y_shapes = [leaf.shape[1:] for leaf in tree_leaves(y)]
    y_boundaries = tuple([
        int(i) for i in
        jnp.cumsum(jnp.array([jnp.prod(jnp.array(s)) for s in y_shapes]))
    ])

    y_min_std = tree_map(
        _standardise,
        y_min,
        y_mean,
        y_std
    )
    y_max_std = tree_map(
        _standardise,
        y_max,
        y_mean,
        y_std
    )

    return TransferredModel(
        base.x_mean,
        base.x_std,
        y_shapes,
        y_boundaries,
        y_mean,
        y_std,
        y_min_std,
        y_max_std,
        units,
        n_take,
        n_hidden,
        EIRs,
        etas
    )
```

```{code-cell} ipython3
surrogate_trans = create_transferred_model(
    surrogate_lhs_full,
    X_lhs_fixed,
    y_lhs_fixed,
    y_min=y_min_fixed,
    y_max=y_max_fixed
)
```

```{code-cell} ipython3
params_trans = pytree_init(key, surrogate_trans, _freeze_attr(X_lhs_fixed))
```

```{code-cell} ipython3
from flax.core.frozen_dict import freeze

def transfer_params(base_params, new_params):
    new_params = new_params.unfreeze()
    for layer in range(3):
        new_params['params']['full_features'][f'Dense_{layer}'] = base_params['params']['nn'][f'Dense_{layer}']
    return freeze(new_params)
```

```{code-cell} ipython3
params_trans = transfer_params(params_lhs_full, params_trans)
```

```{code-cell} ipython3
from flax import traverse_util
import optax

partition_optimizers = {'trainable': optax.adam(5e-3), 'frozen': optax.set_to_zero()}
param_partitions = freeze(traverse_util.path_aware_map(
  lambda path, v: 'frozen' if ('full_features' in path) else 'trainable', params_trans))
tx = optax.multi_transform(partition_optimizers, param_partitions)
```

```{code-cell} ipython3
params_trans = train_surrogate(
    X_lhs_fixed,
    y_lhs_fixed,
    surrogate_trans,
    mse,
    key_i,
    optimiser=tx,
    params=params_trans
)
```

```{code-cell} ipython3
from numpyro.infer import init_to_sample

def surrogate_posterior(surrogate, params, key):
    def surrogate_impl(p, e, a):
        return prev_stats_multisite(p, e, a, lambda p_, e_, a_: full_solution_surrogate(surrogate, params, sort_dict(p_), e_, a_))
    n_samples = 100
    n_warmup = 100

    kernel = NUTS(model, init_strategy=init_to_sample())

    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized' #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
    )
    mcmc.run(key, obs_prev, obs_inc, surrogate_impl)
    return mcmc.get_samples()

def surrogate_posterior_fixed(surrogate, params, key):
    def surrogate_impl(p, e, a):
        return fixed_surrogate(surrogate, params, p)
    
    n_samples = 100
    n_warmup = 100

    kernel = NUTS(model, init_strategy=init_to_sample())

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
X_post_lhs_full = surrogate_posterior(surrogate_lhs_full, params_lhs_full, key)
y_post_lhs_full = prev_stats_batch(X_post_lhs_full)
y_post_lhs_full_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full, X_post_lhs_full)

X_post_lhs_fixed = surrogate_posterior_fixed(surrogate_lhs_fixed, params_lhs_fixed, key)
y_post_lhs_fixed = prev_stats_batch(X_post_lhs_fixed)
y_post_lhs_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed, X_post_lhs_fixed)
```

```{code-cell} ipython3
X_post_prior_full = surrogate_posterior(surrogate_prior_full, params_prior_full, key)
y_post_prior_full = prev_stats_batch(X_post_prior_full)
y_post_prior_full_hat = prev_stats_surrogate_batch(surrogate_prior_full, params_prior_full, X_post_prior_full)

X_post_prior_fixed = surrogate_posterior_fixed(surrogate_prior_fixed, params_prior_fixed, key)
y_post_prior_fixed = prev_stats_batch(X_post_prior_fixed)
y_post_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_prior_fixed, params_prior_fixed, X_post_prior_fixed)
```

```{code-cell} ipython3
X_post_lhs_trans = surrogate_posterior_fixed(surrogate_trans, params_trans, key)
y_post_lhs_trans = prev_stats_batch(X_post_lhs_trans)
y_post_lhs_trans_hat = prev_stats_fixed_surrogate_batch(surrogate_trans, params_trans, X_post_lhs_trans)
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
y_val_prior_prior_full_hat = prev_stats_surrogate_batch(surrogate_prior_full, params_prior_full, sort_dict(without_obs(prior)))
y_val_prior_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_prior_fixed, params_prior_fixed, sort_dict(without_obs(prior)))

y_val_lhs_prior_full_hat = prev_stats_surrogate_batch(surrogate_prior_full, params_prior_full, X_val_lhs)
y_val_lhs_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_prior_fixed, params_prior_fixed, X_val_lhs)
```

```{code-cell} ipython3
y_val_prior_trans_hat = prev_stats_fixed_surrogate_batch(surrogate_trans, params_trans, sort_dict(without_obs(prior)))
y_val_lhs_trans_hat = prev_stats_fixed_surrogate_batch(surrogate_trans, params_trans, X_val_lhs)
```

```{code-cell} ipython3
def _concat(a, b):
    return jnp.concatenate([a, b])

def iterative_training(surrogate, params, f, f_posterior, x, y, key, n_iter=5):
    for i in range(n_iter):
        key_i, key = random.split(key)
        x_post = f_posterior(surrogate, params, key_i)
        y_post = f(x_post)
        x = tree_map(_concat, x, x_post)
        y = tree_map(_concat, y, y_post)
        params = train_surrogate(
            x,
            y,
            surrogate,
            mse,
            key_i,
            optimiser=tx,
            params=params
        )
    return params

params_trans_iter = iterative_training(
    surrogate_trans,
    params_trans,
    lambda x: prev_stats_batch(x[0]),
    lambda m, p, k: [surrogate_posterior_fixed(m, p, k)],
    X_lhs_fixed,
    y_lhs_fixed,
    key
)
```

```{code-cell} ipython3
X_post_trans_iter = surrogate_posterior_fixed(surrogate_trans, params_trans_iter, key)
y_post_trans_iter = prev_stats_batch(X_post_trans_iter)
y_post_trans_iter_hat = prev_stats_fixed_surrogate_batch(surrogate_trans, params_trans_iter, X_post_trans_iter)
```

```{code-cell} ipython3
X_post_lhs_fixed_active = surrogate_posterior_fixed(surrogate_lhs_fixed, params_lhs_fixed_active)
y_post_lhs_fixed_active = prev_stats_batch(X_post_lhs_fixed_active)
y_post_lhs_fixed_active_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed_active, X_post_lhs_fixed_active)
```

```{code-cell} ipython3
y_val_lhs_fixed_active_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed_active, X_val_lhs)
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
            upper_lim = max(axs[i,j].get_ylim()[1], axs[i,j].get_xlim()[1])
            lower_lim = min(axs[i,j].get_ylim()[0], axs[i,j].get_xlim()[0])
            guide = jnp.linspace(lower_lim, upper_lim)
            axs[i, j].plot(guide, guide, c='r')

    fig.tight_layout()

    fig.text(0.5, 0, 'Predictive error', ha='center')
```

```{code-cell} ipython3
plot_predictive_error(y_val_lhs, y_val_lhs_full_hat)
```

```{code-cell} ipython3
plot_predictive_error(y_post_lhs_fixed, y_post_lhs_fixed_hat)
```

```{code-cell} ipython3
plot_predictive_error(y_post_lhs_full, y_post_lhs_full_hat)
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
    ['lhs_full'] * 3 + ['lhs_fixed'] * 3 + ['prior_full'] * 3 + ['prior_fixed'] * 3,
    ['prior', 'lhs', 'posterior'] * 4,
    [y_val_prior, y_val_lhs, y_post_lhs_full, y_val_prior, y_val_lhs, y_post_lhs_fixed, y_val_prior, y_val_lhs, y_post_prior_full, y_val_prior, y_val_lhs, y_post_prior_fixed],
    [y_val_prior_full_hat, y_val_lhs_full_hat, y_post_lhs_full_hat, y_val_prior_fixed_hat, y_val_lhs_fixed_hat, y_post_lhs_fixed_hat,
     y_val_prior_prior_full_hat, y_val_lhs_prior_full_hat, y_post_prior_full_hat, y_val_prior_prior_fixed_hat, y_val_lhs_prior_fixed_hat, y_post_prior_fixed_hat]
).groupby(['experiment', 'test_set']).agg({'RE': 'mean'})#, 'EIR', 'output']).mean()
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
mcmc.run(key, obs_prev, obs_inc)
mcmc.print_summary(prob=0.7)
```

```{code-cell} ipython3
def surrogate_impl(p, e, a):
    return fixed_surrogate(surrogate_prior_fixed, params_prior_fixed, sort_dict(p))
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
