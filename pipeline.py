#!/usr/bin/env python
# coding: utf-8

cpu_count = 100
import os
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count}'
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, jit, vmap
import dmeq
from mox.sampling import LHSStrategy
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from numpyro.infer.util import log_density
import arviz as az
import pandas as pd


key = random.PRNGKey(42)
n_chains = 10

max_age = 99
def full_solution(params, eir, eta):
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


# In[8]:


prev_stats_multisite = vmap(
    lambda params, eta, eir, impl: prev_stats(impl(params, eta, eir)),
    in_axes=[None, 0, 0, None]
)


# In[9]:


EIRs = jnp.array([0.05, 3.9, 15., 20., 100., 150., 418.])
key, key_i = random.split(key)
etas = 1. / random.uniform(key_i, shape=EIRs.shape, minval=20*365, maxval=40*365, dtype=jnp.float64)


# In[10]:


from mox.sampling import DistStrategy


# In[11]:


# TODO: take this from the model
prior_parameter_space = [
    {
        'kb': DistStrategy(dist.LogNormal(0., 1.)),
        'ub': DistStrategy(dist.LogNormal(0., 1.)),
        'b0': DistStrategy(dist.Beta(1., 1.)),
        'IB0': DistStrategy(dist.LeftTruncatedDistribution(dist.Cauchy(50., 10.), low=0.)),
        'kc': DistStrategy(dist.LogNormal(0., 1.)),
        'uc': DistStrategy(dist.LogNormal(0., 1.)),
        'IC0': DistStrategy(dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)),
        'phi0': DistStrategy(dist.Beta(5., 1.)),
        'phi1': DistStrategy(dist.Beta(1., 2.)),
        'PM': DistStrategy(dist.Beta(1., 1.)),
        'dm': DistStrategy(dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.)),
        'kd': DistStrategy(dist.LogNormal(0., 1.)),
        'ud': DistStrategy(dist.LogNormal(0., 1.)),
        'd1': DistStrategy(dist.Beta(1., 2.)),
        'ID0': DistStrategy(dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.)),
        'fd0': DistStrategy(dist.Beta(1., 1.)),
        'gd': DistStrategy(dist.LogNormal(0., 1.)),
        'ad0': DistStrategy(dist.TruncatedDistribution(
            dist.Cauchy(30. * 365., 365.),
            low=20. * 365.,
            high=40. * 365.
        )),
        'rU': DistStrategy(dist.LogNormal(0., 1.))
    },
    DistStrategy(dist.Uniform(0., 500.)), # EIR
    DistStrategy(dist.Uniform(1/(40 * 365), 1/(20 * 365))) # eta
]


# In[12]:


def model(prev=None, inc=None, impl=lambda p, e, a: prev_stats_multisite(p, e, a, full_solution)):
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
    
    prev_stats, inc_stats = impl(x, EIRs, etas)
    
    numpyro.sample(
        'obs_prev',
        dist.Independent(
            dist.Binomial(total_count=prev_N, probs=prev_stats.reshape(-1), validate_args=True),
            1
        ),
        obs=prev
    )

    numpyro.sample(
        'obs_inc',
        dist.Independent(
            dist.Poisson(rate=jnp.maximum(inc_stats.reshape(-1), 1e-12)),
            1
        ),
        obs=inc
    )

key, key_i = random.split(key)
true_values = Predictive(model, num_samples=1)(key_i)


obs_inc, obs_prev = (true_values['obs_inc'], true_values['obs_prev'])


def without_obs(params):
    return {k : v for k, v in params.items() if not k in {'obs_inc', 'obs_prev'}}


key, key_i = random.split(key)
prior = Predictive(model, num_samples=1000)(key)


from jax import pmap, tree_map
import jax
import pandas as pd

device_count = len(jax.devices())

max_val = jnp.finfo(jnp.float32).max
min_val = jnp.finfo(jnp.float32).smallest_normal

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
    ('kd', min_val, 10),
    ('ud', 0, 10),
    ('d1', 0, 1),
    ('ID0', 0, 100),
    ('fd0', 0, 1),
    ('gd', 0, 10),
    ('ad0', 20 * 365, 40 * 365),
    ('rU', 0, 10)
], columns=['name', 'lower', 'upper'])

lhs_train_space = [
    {
        name: LHSStrategy(lower, upper)
        for name, lower, upper in intrinsic_bounds.itertuples(index=False)
    },
    DistStrategy(dist.MixtureGeneral(
        dist.Categorical(probs=jnp.array([.25, .75])),
        [
            dist.TruncatedDistribution(dist.Normal(0., .05), low=0., high=500.),
            dist.Uniform(0., 500.)
        ]
    )),
    LHSStrategy(1/(40 * 365), 1/(20 * 365))
]

lhs_test_space = [
    {
        name: LHSStrategy(lower, upper)
        for name, lower, upper in intrinsic_bounds.itertuples(index=False)
    },
    LHSStrategy(0., 500.),
    LHSStrategy(1/(40 * 365), 1/(20 * 365))
]


# In[20]:


print(pd.concat([intrinsic_bounds]).to_latex(index=False, float_format="{:0.0f}".format))


# In[22]:


from mox.sampling import sample
from mox.surrogates import make_surrogate, pytree_init
from mox.training import train_surrogate
from mox.loss import make_regularised_predictive_loss, mse

loss = make_regularised_predictive_loss(mse, 1e-4)

y_min_full = {
    'pos_M': jnp.full((max_age,), 0.),
    'inc': jnp.full((max_age,), 0.),
    'prob_b': jnp.full((max_age,), 0.),
    'prob_c': jnp.full((max_age,), 0.),
    'prob_d': jnp.full((max_age,), 0.),
    'prop': jnp.full((max_age,), min_val)
}

y_max_full = {
    'pos_M': jnp.full((max_age,), 1.),
    'inc': jnp.full((max_age,), 1.),
    'prob_b': jnp.full((max_age,), 1.),
    'prob_c': jnp.full((max_age,), 1.),
    'prob_d': jnp.full((max_age,), 1.),
    'prop': jnp.full((max_age,), 1.)
}

y_min_fixed = (0., min_val)
y_max_fixed = (1., max_val)

from flax.linen.module import _freeze_attr


# In[28]:


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

def surrogate_impl_full(surrogate, params):
    return lambda p, e, a: prev_stats_multisite(p, e, a, lambda p_, e_, a_: full_solution_surrogate(surrogate, params, sort_dict(p_), e_, a_))

def surrogate_impl_fixed(surrogate, params):
    return lambda p, e, a: fixed_surrogate(surrogate, params, sort_dict(p))

def surrogate_posterior_full(surrogate, params, key):
    return surrogate_posterior(surrogate, params, key, surrogate_impl_full(surrogate, params))

def surrogate_posterior_fixed(surrogate, params, key):
    return surrogate_posterior(surrogate, params, key, surrogate_impl_fixed(surrogate, params))

import numpyro

def surrogate_posterior(surrogate, params, key, impl):
    n_samples = 100
    n_warmup = 100

    kernel = NUTS(model, forward_mode_differentiation=True) # Reverse mode has lead to initialisation errors

    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized' #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
    )
    mcmc.run(key, obs_prev, obs_inc, impl)
    return mcmc


val_size = int(1e5)
y_val_prior = prev_stats_batch(without_obs(prior))
key, key_i = random.split(key)
X_val_lhs = sample(lhs_test_space[0], val_size, key_i)
y_val_lhs = prev_stats_batch(X_val_lhs)


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

posterior_samples = mcmc.get_samples()

def run_pipeline(train_samples, key):

    key_i, key = random.split(key)
    X_lhs_full = sample(lhs_test_space, train_samples, key_i)
    X_lhs_full_tuned = sample(lhs_train_space, train_samples, key_i)
    y_lhs_full = vmap(full_solution, in_axes=[{n: 0 for n in intrinsic_bounds.name}, 0, 0])(*X_lhs_full)
    y_lhs_full_tuned = vmap(full_solution, in_axes=[{n: 0 for n in intrinsic_bounds.name}, 0, 0])(*X_lhs_full_tuned)

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
        loss,
        key_i
    )
    params_lhs_full_tuned = train_surrogate(
        X_lhs_full_tuned,
        y_lhs_full_tuned,
        surrogate_lhs_full,
        loss,
        key_i
    )


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
        loss,
        key_i
    )

    key_i, key = random.split(key)
    X_lhs_fixed = sample(lhs_test_space[0:1], train_samples, key_i)
    y_lhs_fixed = vmap(lambda p: prev_stats_multisite(p, EIRs, etas, full_solution), in_axes=[{n: 0 for n in intrinsic_bounds.name}])(*X_lhs_fixed)

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
        loss,
        key_i
    )


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
        loss,
        key_i
    )
    
    save_model(f'prior_fixed_{train_samples}', surrogate_prior_fixed, params_prior_fixed)
    save_model(f'prior_full_{train_samples}', surrogate_prior_full, params_prior_full)
    save_model(f'lhs_fixed_{train_samples}', surrogate_lhs_fixed, params_lhs_fixed)
    save_model(f'lhs_full_{train_samples}', surrogate_lhs_full, params_lhs_full)
    save_model(f'lhs_full_tuned_{train_samples}', surrogate_lhs_full, params_lhs_full_tuned)

    lhs_full_mcmc = surrogate_posterior_full(surrogate_lhs_full, params_lhs_full, key)
    X_post_lhs_full = lhs_full_mcmc.get_samples()
    y_post_lhs_full = prev_stats_batch(X_post_lhs_full)
    y_post_lhs_full_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full, X_post_lhs_full)
    
    lhs_full_tuned_mcmc = surrogate_posterior_full(surrogate_lhs_full, params_lhs_full_tuned, key)
    X_post_lhs_full_tuned = lhs_full_tuned_mcmc.get_samples()
    y_post_lhs_full_tuned = prev_stats_batch(X_post_lhs_full_tuned)
    y_post_lhs_full_tuned_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full_tuned, X_post_lhs_full_tuned)

    lhs_fixed_mcmc = surrogate_posterior_fixed(surrogate_lhs_fixed, params_lhs_fixed, key)
    X_post_lhs_fixed = lhs_fixed_mcmc.get_samples()
    y_post_lhs_fixed = prev_stats_batch(X_post_lhs_fixed)
    y_post_lhs_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed, X_post_lhs_fixed)


    prior_full_mcmc = surrogate_posterior_full(surrogate_prior_full, params_prior_full, key)
    X_post_prior_full = prior_full_mcmc.get_samples()
    y_post_prior_full = prev_stats_batch(X_post_prior_full)
    y_post_prior_full_hat = prev_stats_surrogate_batch(surrogate_prior_full, params_prior_full, X_post_prior_full)

    prior_fixed_mcmc = surrogate_posterior_fixed(surrogate_prior_fixed, params_prior_fixed, key)
    X_post_prior_fixed = prior_fixed_mcmc.get_samples()
    y_post_prior_fixed = prev_stats_batch(X_post_prior_fixed)
    y_post_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_prior_fixed, params_prior_fixed, X_post_prior_fixed)

    y_val_prior_full_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full, sort_dict(without_obs(prior)))
    y_val_prior_full_tuned_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full_tuned, sort_dict(without_obs(prior)))
    y_val_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed, sort_dict(without_obs(prior)))

    y_val_lhs_full_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full, X_val_lhs)
    y_val_lhs_full_tuned_hat = prev_stats_surrogate_batch(surrogate_lhs_full, params_lhs_full_tuned, X_val_lhs)
    y_val_lhs_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_lhs_fixed, params_lhs_fixed, X_val_lhs)

    y_val_prior_prior_full_hat = prev_stats_surrogate_batch(surrogate_prior_full, params_prior_full, sort_dict(without_obs(prior)))
    y_val_prior_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_prior_fixed, params_prior_fixed, sort_dict(without_obs(prior)))

    y_val_lhs_prior_full_hat = prev_stats_surrogate_batch(surrogate_prior_full, params_prior_full, X_val_lhs)
    y_val_lhs_prior_fixed_hat = prev_stats_fixed_surrogate_batch(surrogate_prior_fixed, params_prior_fixed, X_val_lhs)

    approximation_error(
        ['lhs_full'] * 3 + ['lhs_full_tuned'] * 3 + ['lhs_fixed'] * 3 + ['prior_full'] * 3 + ['prior_fixed'] * 3,
        ['prior', 'lhs', 'posterior'] * 5,
        [y_val_prior, y_val_lhs, y_post_lhs_full, y_val_prior, y_val_lhs, y_post_lhs_full_tuned, y_val_prior, y_val_lhs, y_post_lhs_fixed, y_val_prior, y_val_lhs, y_post_prior_full, y_val_prior, y_val_lhs, y_post_prior_fixed],
        [y_val_prior_full_hat, y_val_lhs_full_hat, y_post_lhs_full_hat, y_val_prior_full_tuned_hat, y_val_lhs_full_tuned_hat, y_post_lhs_full_tuned_hat, y_val_prior_fixed_hat, y_val_lhs_fixed_hat, y_post_lhs_fixed_hat,
         y_val_prior_prior_full_hat, y_val_lhs_prior_full_hat, y_post_prior_full_hat, y_val_prior_prior_fixed_hat, y_val_lhs_prior_fixed_hat, y_post_prior_fixed_hat]
    ).to_csv(f'{train_samples}_approx_error.csv', index=False)
    
    from scipy.stats import ks_2samp
    sample_keys = list(posterior_samples.keys())
    ks_data = pd.DataFrame([
        {'experiment': name, 'variable': k, 'statistic': ks_2samp(posterior_samples[k], posterior[k]).statistic, 'p-value': ks_2samp(posterior_samples[k], posterior[k]).pvalue}
        for k in sample_keys
        for name, posterior in [
            ('prior_fixed', prior_fixed_mcmc.get_samples()),
            ('prior_full', prior_full_mcmc.get_samples()),
            ('lhs_fixed', lhs_fixed_mcmc.get_samples()),
            ('lhs_full', lhs_full_mcmc.get_samples()),
            ('lhs_full_tuned', lhs_full_tuned_mcmc.get_samples())
        ]
    ]).to_csv(f'{train_samples}_ks_error.csv', index=False)
    
for n_batches in jnp.linspace(int(10), int(2e3), num=10, dtype=jnp.int64):
    train_samples = int(n_batches * 100)
    try:
        run_pipeline(train_samples, key)
    except Exception as e:
        import traceback
        print(train_samples)
        print(e)
        print(traceback.format_exc())
        pass