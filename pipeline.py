#!/usr/bin/env python
# coding: utf-8

cpu_count = 100
import os
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count}'
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import random, jit, vmap
import dmeq
from mox.sampling import LHSStrategy
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood
from numpyro.infer.util import log_density 
from numpyro.diagnostics import summary
from numpyro import handlers
import numpyro.distributions as dist
from scipy.stats import ks_2samp
import arviz as az
import pandas as pd
import pickle
from time import time
import logging
import os

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

out_dir = 'outputs/v1'

def timing(f):
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return te-ts, result
    return wrap

key = random.PRNGKey(42)
cpu_device = jax.devices('cpu')[0]
n_chains = 10
epochs = 1000
n_rounds = 5

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

# In[8]:


prev_stats_multisite = vmap(
    lambda params, eta, eir, impl: prev_stats(impl(params, eta, eir)),
    in_axes=[None, 0, 0, None]
)


# In[9]:


EIRs = jnp.array([0.05, 3.9, 15., 20., 100., 150., 418.])
n_sites = EIRs.shape[0]
key, key_i = random.split(key)
etas = 1. / random.uniform(key_i, shape=(n_sites,), minval=40*365, maxval=100*365, dtype=jnp.float64)


# In[10]:


from mox.sampling import DistStrategy


# In[11]:


# TODO: take this from the model
prior_parameter_space = [
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
    #DistStrategy(SiteDistribution({'EIR': est_EIR, 'etas': etas}, noise))
    DistStrategy(dist.Uniform(0., 500.)), # EIR
    DistStrategy(dist.Uniform(1/(100 * 365), 1/(40 * 365))) # eta
]


# In[12]:


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
    d1 = numpyro.sample('d1', dist.Beta(1., 5.))
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


logger.info('Making truth')
key, key_i = random.split(key)
true_values = Predictive(model, num_samples=1)(key_i, true_EIRs=EIRs)


obs_inc, obs_prev = (true_values['obs_inc'], true_values['obs_prev'])


print(pd.DataFrame(
    jnp.vstack([EIRs, etas, obs_prev[0].T, obs_inc[0].T]).T,
    columns=['EIR', 'eta', 'prev_2_10', 'prev_10+', 'inc_0_5', 'inc_5_15', 'inc_15+']
).to_latex(index=False))


def without_obs(params):
    return {k : v for k, v in params.items() if not k in {'obs_inc', 'obs_prev'}}


logger.info('Sampling prior')
key, key_i = random.split(key)
with jax.default_device(cpu_device):
    prior = Predictive(model, num_samples=600)(key_i)
logger.info('done')


from jax import pmap
from jax.tree_util import tree_map, tree_leaves
import pandas as pd

device_count = len(jax.devices())

max_val = jnp.finfo(jnp.float64).max
min_val = jnp.finfo(jnp.float64).smallest_normal

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


print(pd.concat([intrinsic_bounds]).to_latex(index=False, float_format="{:0.0f}".format))


from mox.sampling import sample
from mox.surrogates import (
    make_surrogate,
    init_surrogate,
    apply_surrogate,
    #MLP
)
from mox.training import train_surrogate
from mox.loss import mse
from mox.utils import tree_leading_axes as tla


max_age = 99
y_min_full = {
    'pos_M': jnp.full((max_age,), 0.),
    'inc': jnp.full((max_age,), 0.),
    'prob_b': jnp.full((max_age,), 0.),
    'prob_c': jnp.full((max_age,), 0.),
    'prob_d': jnp.full((max_age,), 0.),
    'prop': jnp.full((max_age,), 1e-12)
}

y_max_full = {
    'pos_M': jnp.full((max_age,), 1.),
    'inc': jnp.full((max_age,), max_val),
    'prob_b': jnp.full((max_age,), 1.),
    'prob_c': jnp.full((max_age,), 1.),
    'prob_d': jnp.full((max_age,), 1.),
    'prop': jnp.full((max_age,), 1.)
}

y_min_fixed = (0., min_val)
y_max_fixed = (1., max_val)

from flax import linen as nn

# TODO: update mox MLP
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

# In[28]:

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

def sample_fixed_from_posterior(params):
    eir, params = posterior_EIR(params)
    n = eir.shape[0]
    X = [
        tree_map(lambda leaf: jnp.repeat(leaf, n_sites), params),
        jnp.reshape(eir, -1),
        jnp.tile(etas, n)
    ]
    y = vmap(
        lambda p, e, a: vmap(lambda _e, _a: prev_stats(full_solution(p, _e, _a)))(e, a),
        in_axes=[{k: 0 for k in params.keys()}, 0, None]
    )(params, eir, etas)
    y = tree_map(lambda leaf: leaf.reshape((-1, leaf.shape[-1])), y)
    return X, y

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

def surrogate_posterior(key, impl, init_params, n_samples):
    # Initialise MCMC
    kernel = NUTS(model, forward_mode_differentiation=True) # Reverse mode has lead to initialisation errors
    n_warmup = 500
    mcmc = MCMC(
        kernel,
        num_samples=n_samples // n_chains,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized' #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
    )

    mcmc.run(
        key,
        None,
        obs_prev,
        obs_inc,
        impl,
        #init_params=init_params
    )
    return mcmc

from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoBNAFNormal

def wo_latent(X):
    return {
        k: v
        for k, v in X.items()
        if k != '_auto_latent'
    }

def surrogate_posterior_svi(key, svi, guide, svi_state, impl, n_samples):
    n_train_samples = 50_000

    # train SVI
    sample_key, key = random.split(key, 2)
    svi_result = svi.run(
        sample_key,
        n_train_samples,
        init_state=svi_state,
        stable_update=True,
        true_EIRs=None,
        prev=obs_prev,
        inc=obs_inc,
        impl=impl
    )
    svi_params = svi_result.params

    # sample posterior
    post_key, key = random.split(key, 2)
    posterior_samples = Predictive(
        guide,
        params=svi_params,
        num_samples=n_samples
    )(post_key)

    posterior_samples = wo_latent(posterior_samples) 

    return posterior_samples, svi_result.state, svi_result.losses

logger.info('Making validation set')
y_val_prior = prev_stats_posterior(without_obs(prior))
logger.info('done')
key, key_i = random.split(key)

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

def stand_approximation_error(exps, labels, ys, y_hats, std_surrogate):
    ys = [
        vmap(std_surrogate.vectorise_output, in_axes=[tla(y)])(y)
        for y in ys
    ]
    y_hats = [
        vmap(std_surrogate.vectorise_output, in_axes=[tla(y_hat)])(y_hat)
        for y_hat in y_hats
    ]
    
    return pd.DataFrame([
        {
            'mse': jnp.mean(jnp.square(y - y_hat)),
            'test_set': label,
            'experiment': exp
        }
        for exp, label, y, y_hat in zip(exps, labels, ys, y_hats)
    ])

def save_mcmc(name, mcmc):
    with open(os.path.join(out_dir, f'chapter_2_mcmc_{name}.pkl'), 'wb') as f:
        pickle.dump(mcmc, f)

def pp_ll(key, p):
    pp = Predictive(model, p)(key)
    _t = {
        k: v[0] if k == 'EIR' else v
        for k, v in without_obs(true_values).items()
    }
    ld = log_density(handlers.seed(model, key), [], {}, _t)
    return pd.DataFrame(
        jnp.concatenate(
            [
                jnp.mean(ld[1]['obs_prev']['fn'].base_dist.log_prob(pp['obs_prev']), axis=0),
                jnp.mean(ld[1]['obs_inc']['fn'].base_dist.log_prob(pp['obs_inc']), axis=0)
            ],
            axis=1
        ),
        columns=['prev_2_10', 'prev_10+', 'inc_0_5', 'inc_5_15', 'inc_15+']
    ).assign(EIR=EIRs)

def pp_ll_df(key, ps, labels):
    return pd.concat([
        pp_ll(key, p).assign(experiment=l)
        for p, l in zip(ps, labels)
    ])

def ll_df(X_post, labels):
    return pd.DataFrame([
        tree_map(jnp.mean, log_likelihood(model, X, prev=obs_prev, inc=obs_inc))
        for X in X_post
    ]).assign(experiments=labels)

n_samples = 100
n_warmup = 100

if os.path.exists(os.path.join(out_dir, 'chapter_2_mcmc_underlying.pkl')):
    logger.info('Loading posterior')
    with open(os.path.join(out_dir, 'chapter_2_mcmc_underlying.pkl'), 'rb') as f:
        mcmc = pickle.load(f)
    logger.info('done')
else:
    mcmc = MCMC(
        NUTS(model),
        num_samples=n_samples,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized'
    )
    logger.info('Sampling posterior')
    mcmc.run(key_i, None, obs_prev, obs_inc)
    logger.info('done')
    save_mcmc('underlying', mcmc)

mcmc.print_summary(prob=0.7)
posterior_samples = mcmc.get_samples()
logger.info('Writing underlying ll')

pp_ll_df(
    key_i,
    [posterior_samples],
    ['underlying']
).to_csv(os.path.join(out_dir, f'underlying_pp_ll.csv'), index=False)

ll_df(
    [posterior_samples],
    ['underlying']
).to_csv(os.path.join(out_dir, f'underlying_ll.csv'), index=False)
logger.info('done')

def make_surrogate_objects(samples, y_min, y_max):
    X, y = samples
    surrogate = make_surrogate(
        X,
        y,
        y_min=y_min,
        y_max=y_max
    )
    net = make_net(surrogate, y)
    return surrogate, net

def train_surrogate_objects(key, surrogate_obj, samples):
    surrogate, net = surrogate_obj
    X, y = samples
    params = init_surrogate(key, surrogate, net, X)
    return timing(train_surrogate)(
        X,
        y,
        surrogate,
        net,
        mse,
        key,
        params,
        epochs=epochs
    )

def perform_mcmc(
    key,
    impl,
    val_helper,
    obj,
    train_state,
    init_params,
    n_samples
    ):
    surrogate, net = obj
    t, mcmc = timing(surrogate_posterior)(
        key,
        impl(
            surrogate,
            net,
            train_state.params
        ),
        init_params,
        n_samples
    )
    X_post = mcmc.get_samples()
    y_post = prev_stats_posterior(X_post)
    y_post_hat = val_helper(
        surrogate,
        net,
        train_state.params,
        X_post
    )
    summary_df = pd.DataFrame.from_dict(
        summary(mcmc.get_samples(group_by_chain=True)),
        orient='index'
    ).reset_index()
    init_sample = tree_map(
        lambda leaf: leaf[:,-1],
        mcmc.get_samples(True)
    )
    return (
        t,
        summary_df,
        X_post,
        y_post,
        y_post_hat,
        init_sample
    )

def perform_svi(
    key,
    impl,
    val_helper,
    obj,
    train_state,
    svi,
    guide,
    svi_state,
    n_samples
    ):
    surrogate, net = obj
    t, (X_post, svi_state, losses) = timing(surrogate_posterior_svi)(
        key,
        svi,
        guide,
        svi_state,
        impl(
            surrogate,
            net,
            train_state.params
        ),
        n_samples
    )
    y_post = prev_stats_posterior(X_post)
    y_post_hat = val_helper(
        surrogate,
        net,
        train_state.params,
        X_post
    )
    return (
        t,
        X_post,
        y_post,
        y_post_hat,
        svi_state,
        losses
    )

def vectorise_posterior(X):
    return jnp.concatenate(
        [
            l if l.ndim == 2 else l[...,None]
            for l in tree_leaves(X)
        ],
        axis=1
    )

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


def squared_mmd(X, Y, kernel=SquaredExponential(10., 1.)): 
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

def run_pipeline(train_samples, key):
    experiments = [
        'lhs_full',
        'prior_full',
        'lhs_fixed',
        'prior_fixed'
    ]

    methods = ['nuts', 'svi']

    logger.info('Sampling LHS')
    key_i, key = random.split(key)
    with jax.default_device(cpu_device):
        X_lhs_full = sample(lhs_parameter_space, train_samples, key_i)
        t_sample_lhs, y_lhs_full = timing(vmap(full_solution, in_axes=[{n: 0 for n in intrinsic_bounds.name}, 0, 0]))(*X_lhs_full)
        
    X_lhs_fixed = X_lhs_full
    y_lhs_fixed = vmap(prev_stats, in_axes=[tla(y_lhs_full)])(y_lhs_full)

    logger.info('Sampling Prior')
    key_i, key = random.split(key)
    with jax.default_device(cpu_device):
        X_prior_full = sample(prior_parameter_space, train_samples, key_i)
        t_sample_prior, y_prior_full = timing(vmap(full_solution, in_axes=tree_map(lambda x: 0, X_prior_full)))(*X_prior_full)

    X_prior_fixed = X_prior_full
    y_prior_fixed = vmap(prev_stats, in_axes=[tla(y_prior_full)])(y_prior_full)

    samples = [
        (X_lhs_full, y_lhs_full),
        (X_prior_full, y_prior_full),
        (X_lhs_fixed, y_lhs_fixed),
        (X_prior_fixed, y_prior_fixed)
    ]

    samples_mcmc = samples_svi = samples

    logger.info('Making surrogates')

    surrogates = [
        make_surrogate_objects(s, y_min_full, y_max_full)
        for s in samples[:2]
    ] + [
        make_surrogate_objects(s, y_min_fixed, y_max_fixed)
        for s in samples[2:]
    ]


    # Initialise SVI
    guide = AutoBNAFNormal(model, num_flows=5)
    svi = SVI(
        model,
        guide,
        optim.ClippedAdam(1e-4),
        loss=Trace_ELBO(num_particles=8),
    )

    init_mcmc = [None] * len(experiments)
    svi_states = [None] * len(experiments)

    for r in range(n_rounds):
        logger.info(f'Training surrogates: round {r}')
        if train_samples > 5_000 and r > 0:
            continue

        key, *keys = random.split(key, len(experiments) + 1)

        train_times, train_states_mcmc = zip(
            *[
                train_surrogate_objects(key, obj, s)
                for key, obj, s
                in zip(
                    keys,
                    surrogates,
                    samples_mcmc
                )
            ]
        )

        if r == 0:
            train_states_svi = train_states_mcmc
        else:
            _, train_states_svi = zip(
                *[
                    train_surrogate_objects(key, obj, s)
                    for key, obj, s
                    in zip(
                        keys,
                        surrogates,
                        samples_svi
                    )
                ]
            )

        logger.info('Performing MCMC')
        key, *keys = random.split(key, len(experiments) + 1)
        val_helpers = [prev_stats_full_surrogate_posterior] * 2 + [prev_stats_fixed_surrogate_posterior] * 2
        (
            mcmc_times,
            summaries,
            X_post,
            y_post,
            y_post_hat,
            init_mcmc
        ) = zip(
            *[
                perform_mcmc(
                    key,
                    impl,
                    val_helper,
                    obj,
                    state,
                    init_sample,
                    train_samples
                )
                for key, impl, val_helper, obj, state, init_sample
                in zip(
                    keys,
                    [surrogate_impl_full] * 2 + [surrogate_impl_fixed] * 2,
                    val_helpers,
                    surrogates,
                    train_states_mcmc,
                    init_mcmc
                )
            ]
        )

        logger.info('Performing SVI')
        key, *keys = random.split(key, len(experiments) + 1)
        (
            svi_times,
            X_post_svi,
            y_post_svi,
            y_post_svi_hat,
            svi_states,
            svi_losses
        ) = zip(
            *[
                perform_svi(
                    key,
                    impl,
                    val_helper,
                    obj,
                    state,
                    svi,
                    guide,
                    svi_state,
                    train_samples
                )
                for key, impl, val_helper, obj, state, svi_state
                in zip(
                    keys,
                    [surrogate_impl_full] * 2 + [surrogate_impl_fixed] * 2,
                    val_helpers,
                    surrogates,
                    train_states_svi,
                    svi_states
                )
            ]
        )

        logger.info('Calculating validation data')
        y_val_mcmc = [
            helper(
                surrogate,
                net,
                state.params,
                without_obs(prior)
            )
            for helper, (surrogate, net), state
            in zip(val_helpers, surrogates, train_states_mcmc)
        ]

        y_val_svi = [
            helper(
                surrogate,
                net,
                state.params,
                without_obs(prior)
            )
            for helper, (surrogate, net), state
            in zip(val_helpers, surrogates, train_states_svi)
        ]

        logger.info('Writing results')
        posts = [
            (X_post, y_post, y_post_hat),
            (X_post_svi, y_post_svi, y_post_svi_hat)
        ]

        vals = [
            y_val_mcmc,
            y_val_svi
        ]

        pd.concat([
            approximation_error(
                [exp for exp in experiments for _ in range(2)],
                ['prior', 'posterior'] * len(experiments),
                [truth for pair in zip([y_val_prior] * len(experiments), y_post_out) for truth in pair],
                [hat for pair in zip(y_val_out, y_post_hat_out) for hat in pair]
            ).assign(method=method)
            for method, y_val_out, (_, y_post_out, y_post_hat_out) in zip(methods, vals, posts)
        ]).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_approx_error.csv'), index=False)

        pd.concat([
            stand_approximation_error(
                [exp for exp in experiments for _ in range(2)],
                ['prior', 'posterior'] * len(experiments),
                [truth for pair in zip([y_val_prior] * len(experiments), y_post_out) for truth in pair],
                [hat for pair in zip(y_val_out, y_post_hat_out) for hat in pair],
                surrogates[2][0] # lhs_fixed surrogate for standardising
            ).assign(method=method)
            for method, y_val_out, (_, y_post_out, y_post_hat_out) in zip(methods, vals, posts)
        ]).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_stand_approx_error.csv'), index=False)

        pd.DataFrame({
            'experiment': experiments,
            'sampling': [t_sample_prior, t_sample_prior, t_sample_lhs, t_sample_lhs],
            'training': train_times,
            'mcmc': mcmc_times,
            'svi': svi_times 
        }).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_timings.csv'), index=False)

        logger.info('Writing likelihoods')
        pd.concat([
            pp_ll_df(
                key_i,
                X,
                experiments
            ).assign(method=method)
            for method, (X, _, _) in zip(methods, posts)
        ]).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_pp_ll.csv'), index=False)

        pd.concat([
            ll_df(
                X,
                experiments
            ).assign(method=method)
            for method, (X, _, _) in zip(methods, posts)
        ]).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_ll.csv'), index=False)

        logger.info('Writing KS Error')
        sample_keys = list(posterior_samples.keys())

        pd.concat([
            pd.DataFrame([
                {
                    'experiment': name,
                    'variable': k,
                    'statistic': ks_2samp(jnp.reshape(posterior_samples[k], -1), jnp.reshape(posterior[k], -1)).statistic,
                    'p-value': ks_2samp(jnp.reshape(posterior_samples[k], -1), jnp.reshape(posterior[k], -1)).pvalue
                }
                for k in sample_keys
                for name, posterior in zip(experiments, X)
            ]).assign(method=method)
            for method, (X, _, _) in zip(methods, posts)
        ]).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_ks_error.csv'), index=False)

        logger.info('Writing MMD')
        pd.concat([
            pd.DataFrame([
                {
                    'experiment': name,
                    'MMD': squared_mmd(
                        vectorise_posterior(posterior_samples),
                        vectorise_posterior(posterior)
                    )
                }
                for name, posterior in zip(experiments, X)
            ]).assign(method=method)
            for method, (X, _, _) in zip(methods, posts)
        ]).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_mmd.csv'), index=False)

        logger.info('Writing SVI losses')
        pd.concat([
            pd.DataFrame({
                'loss': losses[::1000],
                'step': jnp.arange(0, len(losses), 1000)
            }).assign(experiment=name)
            for name, losses in zip(experiments, svi_losses)
        ]).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_svi_losses.csv'), index=False)

        logger.info('MCMC stats')
        pd.concat([
            s.assign(experiment=name)
            for name, s in zip(experiments, summaries)
        ]).to_csv(os.path.join(out_dir, f'{train_samples}_round_{r}_mcmc_summary.csv'), index=False)

        if r != n_rounds - 1:
            logger.info('Concatting samples')
            new_samples_mcmc = [
                sample_full_from_posterior(X)
                for X in X_post[:2]
            ] + [
                sample_fixed_from_posterior(X)
                for X in X_post[2:]
            ]
            samples_mcmc = [
                (
                    tree_map(lambda *x: jnp.concatenate(x), X, X_new),
                    tree_map(lambda *y: jnp.concatenate(y), y, y_new),
                )
                for ((X, y), (X_new, y_new))
                in zip(samples_mcmc, new_samples_mcmc)
            ]

            new_samples_svi = [
                sample_full_from_posterior(X)
                for X in X_post_svi[:2]
            ] + [
                sample_fixed_from_posterior(X)
                for X in X_post_svi[2:]
            ]
            samples_svi = [
                (
                    tree_map(lambda *x: jnp.concatenate(x), X, X_new),
                    tree_map(lambda *y: jnp.concatenate(y), y, y_new),
                )
                for ((X, y), (X_new, y_new))
                in zip(samples_svi, new_samples_svi)
            ]

runs = jnp.concatenate([
    jnp.linspace(int(10), int(50), num=5, dtype=jnp.int64),
    jnp.linspace(int(100), int(5e3), num=5, dtype=jnp.int64)
])

for n_batches in runs:
    train_samples = n_batches * 100
    logger.info(f'{train_samples} samples')
    run_pipeline(train_samples, key)
