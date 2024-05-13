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

#jax.config.update('jax_platform_name', 'cpu') # for memory purposes
cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]
```

```{code-cell} ipython3
import dmeq
from mox.sampling import DistStrategy
```

```{code-cell} ipython3
import numpyro
from numpyro.infer import Predictive
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
EIRs = jnp.array([3.9, 15., 20., 100., 150., 418.]) #jnp.array([0.05, 3.9, 15., 20., 100., 150., 418.])
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
def wo_EIR(p):
    return {k:v for k, v in p.items() if k != 'EIR'}
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

train_samples = int(10_000)
device_count = len(jax.devices())
n_epochs = 1_000
```

```{code-cell} ipython3
max_val = jnp.finfo(jnp.float64).max
min_val = jnp.finfo(jnp.float64).smallest_normal
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
        batch_norm=False,
        
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
            p_,
            e_,
            a_
        )
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
        sensitivity_prior_full.assign(model='prior_full')
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
    without_obs(wo_EIR(prior)),
    EIRs,
    etas,
    impl=lambda p, e, a: tree_map(
        jnp.squeeze,
        apply_dmeq_surrogate(
            surrogate_prior_full,
            net_full,
            train_state_prior_full,
            p,
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
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, RenyiELBO
from numpyro.infer.autoguide import AutoBNAFNormal
import tqdm
from numpyro.infer.svi import SVIRunResult
```

```{code-cell} ipython3
def posterior_EIR(params):
    return params['EIR'], wo_EIR(params)

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
```

```{code-cell} ipython3
def surrogate_posterior_annealed_svi(surrogate, net, params, key):
    rounds = 5
    n_samples = train_samples
    n_train_samples = 50_000
    
    guide = AutoBNAFNormal(model, num_flows=5)

    svi_state = None

    training_samples = (X_prior_full, y_prior_full)

    sample_sets = list()
    
    svi = SVI(
        model,
        guide,
        optim.ClippedAdam(step_size=1e-4),
        loss=Trace_ELBO(num_particles=8)
    )

    T = 1. / (0.45 ** jnp.arange(rounds))[::-1]
    
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
def wo_latent(X):
    return {k: v for k, v in X.items() if k != '_auto_latent'}
```

```{code-cell} ipython3
X_post_prior_full_seq_svi, train_state_prior_full_seq = surrogate_posterior_annealed_svi(
    surrogate_prior_full,
    net_full,
    train_state_prior_full,
    key
)
```

```{code-cell} ipython3
y_post_prior_full = prev_stats_posterior(X_post_prior_full_seq_svi[-1])
y_post_prior_full_hat = prev_stats_full_surrogate_posterior(
    surrogate_prior_full,
    net_full,
    train_state_prior_full_seq,
    wo_latent(X_post_prior_full_seq_svi[-1])
)
```

```{code-cell} ipython3
y_val_prior = prev_stats_posterior(without_obs(prior))
```

```{code-cell} ipython3
y_val_prior_prior_full_hat = prev_stats_full_surrogate_posterior(
    surrogate_prior_full,
    net_full,
    train_state_prior_full_seq,
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

prior_full_svi_pp = Predictive(model, X_post_prior_full_seq_svi[-1])(key_i)
prior_full_svi_idata = az.from_dict(
    posterior=_to_arviz_dict(X_post_prior_full_seq_svi[-1]),
    posterior_predictive=_to_arviz_dict(prior_full_svi_pp),
    observed_data={
        'obs_prev': obs_prev,
        'obs_inc': obs_inc
    }
)
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
plot_predictive_error(y_val_prior, y_val_prior_prior_full_hat)
#plt.savefig('pe.png')
```

```{code-cell} ipython3
plot_predictive_error(y_post_prior_full, y_post_prior_full_hat)
```

```{code-cell} ipython3
with jax.default_device(cpu_device):
    #NUTS_posterior_curves = get_curves(prior_full_samples, EIRs, etas)
    #SVI_posterior_curves = get_curves(X_post_prior_full_svi, EIRs, etas)
    SVI_posterior_curves = get_curves(X_post_prior_full_seq_svi[-1], EIRs, etas)
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
posterior_imm_curves_svi = get_batch_imm_curves(X_post_prior_full_seq_svi[-1])
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
