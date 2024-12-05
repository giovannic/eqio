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
from fastms.density.rnn import load as load_rnn
from fastms.sample.save import load_samples, load_pytree
import pickle
```

```{code-cell} ipython3
sample = load_samples(
    [
        f'/mnt/gc1610/home/fastms/data/samples/ibm_prior_low_v3/ibm_priors_7001.npz'
    ],
    '/mnt/gc1610/home/fastms/data/samples/ibm_prior_low_v3/def_ibm_prior',
    dtype=jnp.float64
)
```

```{code-cell} ipython3
surrogate, net, params = load_rnn('../fastms/rnn_6_annealed_4/', sample)
```

```{code-cell} ipython3
from mox.seq2seq.rnn import apply_surrogate as apply_rnn
```

```{code-cell} ipython3
import scipy as sp
from jax import scipy as jsp
import numpy as np
from fastms.density.train import _standardise
```

```{code-cell} ipython3
import arviz as az
import pandas as pd
```

```{code-cell} ipython3
pe_imm = [
    'kb',
    'ub',
    'b0',
    'ib0'
]
clin_imm = [
    'kc',
    'uc',
    'ic0',
    'phi0',
    'phi1',
    'pcm',
    'rm'
]

det_imm = [
    'kd',
    'ud',
    'd1',
    'id0',
    'fd0',
    'gammad',
    'ad',
    'ru'
]

foim = [
    'cd',
    'cu',
    'gamma1'
]

intrinsic = pe_imm + clin_imm + det_imm + foim
az.rcParams["plot.max_subplots"] = 200
```

```{code-cell} ipython3
inf_fake = az.from_netcdf('../fastms/bnaf_rnn_6_fake')
```

```{code-cell} ipython3
az.plot_bpv(
    inf_fake,
    kind='p_value',
    group='prior'
)
```

```{code-cell} ipython3
az.plot_bpv(
    inf_fake,
    kind='p_value',
    mse=True
)
```

```{code-cell} ipython3
import pickle
with open('../fastms/bnaf_rnn_6_fake_truth.pkl', 'rb') as f:
    truth = pickle.load(f)
```

```{code-cell} ipython3
import numpy as np
truth = {k: np.array(v) for k, v in truth.items()}
```

```{code-cell} ipython3
def make_ref_val(var_names, true_params):
    """Make a reference value for the posterior."""
    return {k: [{'ref_val': true_params[k][0]}] for k in var_names}

plot_vars = intrinsic #[n for n in pe_imm if n != 'ub']
az.plot_posterior(
    inf_fake,
    var_names=plot_vars,
    ref_val=make_ref_val(plot_vars, truth),
    kind='hist'
)
```

```{code-cell} ipython3
az.plot_posterior(
    inf_fake,
    var_names=['eir'],
    ref_val=list(truth['eir'][0]),
    kind='hist',
)
```

```{code-cell} ipython3
from jax import vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
def get_immunity_curve(params):
    exposures = jnp.arange(100)
    (b0, IB0, kb) = params['b0'], params['ib0'], params['kb'] + 1
    (phi0, phi1, IC0, kc) = params['phi0'], params['phi1'], params['ic0'], params['kc'] + 1
    (d1, ID0, fd0, gd, ad0, kd) = (
        params['d1'], params['id0'], params['fd0'], params['gammad'], params['ad'], params['kd']
    )
    #test: remove!
    #kb = truth['kb']
    #b0 = truth['b0']
    #d1 = .9
    
    b1 = .5
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

n_draws = 100
pe_imm + clin_imm + det_imm
site_draws = az.extract(
    inf_fake,
    var_names=intrinsic,
    num_samples=n_draws
)
prior_site_draws = az.extract(
    inf_fake,
    var_names=intrinsic,
    num_samples=n_draws,
    group='prior'
)
x_intrinsic = {
    key: np.array(site_draws[key])
    for key in pe_imm + clin_imm + det_imm #intrinsic
}
x_intrinsic_prior = {
    key: np.array(prior_site_draws[key])
    for key in pe_imm + clin_imm + det_imm #intrinsic
}
prior_imm_curves = get_batch_imm_curves(x_intrinsic_prior)
posterior_imm_curves = get_batch_imm_curves(x_intrinsic)
true_imm_curves = get_batch_imm_curves(truth)
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 3)
imm_labels = ['prob_b', 'prob_c', 'prob_d']

for imm_i, imm in enumerate(imm_labels):
    axs[imm_i].plot(prior_imm_curves[imm].T, color='g', alpha=.1)
    axs[imm_i].plot(posterior_imm_curves[imm].T, color='r', alpha=.1)
    axs[imm_i].plot(true_imm_curves[imm][0, :])
    axs[imm_i].set_ylabel(imm)

fig.tight_layout()
```

```{code-cell} ipython3
inf = az.from_netcdf('../fastms/bnaf_rnn_6_annealed_4')
```

```{code-cell} ipython3
az.plot_bpv(
    inf,
    kind='p_value',
    mse=True,
    group='prior'
)
```

```{code-cell} ipython3
az.plot_bpv(
    inf,
    kind='p_value',
    mse=True
)
```

```{code-cell} ipython3
from fastms.sites import make_site_inference_data
from jax.tree_util import tree_map
from jax import numpy as jnp, vmap
```

```{code-cell} ipython3
site_data = make_site_inference_data('../fastms/data/', 1985, 2018)
```

```{code-cell} ipython3
n_draws = 100
site_draws = az.extract(
    inf,
    var_names=intrinsic + ['eir'],
    num_samples=n_draws
)
prior_site_draws = az.extract(
    inf,
    var_names=intrinsic + ['eir'],
    num_samples=n_draws,
    group='prior'
)
```

```{code-cell} ipython3
x_intrinsic = {
    key: np.array(site_draws[key])
    for key in intrinsic
}
x_intrinsic_prior = {
    key: np.array(prior_site_draws[key])
    for key in intrinsic
}
```

```{code-cell} ipython3
#TODO Add variance to these plots
```

```{code-cell} ipython3
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from fastms.density.train import _standardise
selected = [1, 5, 18, 29, 30]#list(range(site_data.n_sites))
fig, axs = plt.subplots(nrows=len(selected), ncols=2, figsize=(20, 20))#figsize=(20,100))

for ax_i, site_i in enumerate(selected):
    prev_i = np.where(site_data.prev_index == site_i)[0][0]
    site_name = site_data.site_index.name_1.iloc[site_i]
    site_iso = site_data.site_index.iso3c.iloc[site_i]
    z = 1.96
    x_sites = tree_map(
        lambda x: jnp.repeat(x[site_i:site_i+1], n_draws, axis=0),
        site_data.x_sites
    )
    
    x_eir = jnp.array(site_draws['eir'][site_i])
    x = {
        'intrinsic': x_intrinsic,
        'init_EIR': x_eir,
        'seasonality': x_sites['seasonality'],
        'vector_composition': x_sites['vectors']
    }
    x_seq = {
        'interventions': x_sites['interventions'],
        'demography': x_sites['demography']
    }
    x_in = (x, x_seq)
    
    mu, _ = apply_rnn(
        surrogate,
        net,
        params,
        x_in
    )

    x_eir_prior = jnp.array(prior_site_draws['eir'][site_i])
    x = {
        'intrinsic': x_intrinsic_prior,
        'init_EIR': x_eir_prior,
        'seasonality': x_sites['seasonality'],
        'vector_composition': x_sites['vectors']
    }
    mu_prior, _ = apply_rnn(
        surrogate,
        net,
        params,
        (x, x_seq)
    )
    
    ax = axs[ax_i, 0]
    lar, uar = (
        site_data.prev_lar[prev_i],
        site_data.prev_uar[prev_i]
    )
    lt, ut = (
        site_data.prev_start_time[prev_i],
        site_data.prev_end_time[prev_i]
    ) 
    prev = jnp.sum(
        mu['n_detect'][..., lar:uar + 1], axis=2
    ) / jnp.sum(mu['n'][..., lar:uar + 1], axis=2)
    prev_q = sp.stats.mstats.mquantiles(prev, axis=0)
    ax.plot(jnp.arange(len(prev_q[1])), prev_q[1], color='red')
    ax.fill_between(jnp.arange(len(prev_q[1])), prev_q[0], prev_q[2], color='red', alpha=.1)
    prev = jnp.sum(
        mu_prior['n_detect'][..., lar:uar + 1], axis=2
    ) / jnp.sum(mu_prior['n'][..., lar:uar + 1], axis=2)
    prev_q = sp.stats.mstats.mquantiles(prev, axis=0)
    ax.plot(jnp.arange(len(prev_q[1])), prev_q[1], color='green')
    ax.fill_between(jnp.arange(len(prev_q[1])), prev_q[0], prev_q[2], color='green', alpha=.1)
    p = site_data.prev[prev_i] / site_data.n_prev[prev_i]
    p_ci = np.sqrt(p * (1 - p) / site_data.n_prev[prev_i])
    t = (lt + ut) / 2
    terr = ut - t
    ax.errorbar([t], [p], yerr=[p_ci], xerr=[terr], fmt='bo')
    ax.set_title(f'Prevalence in {site_name} ({site_iso}) between {lar} and {uar} years')
    
    ax = axs[ax_i, 1]
    inc_i = np.where(site_data.inc_index == site_i)[0][0]
    lar, uar = (
        site_data.inc_lar[inc_i],
        site_data.inc_uar[inc_i]
    )
    lt, ut = (
        site_data.inc_start_time[inc_i],
        site_data.inc_end_time[inc_i]
    )
    inc = jnp.sum(
        mu['n_inc_clinical'][..., lar:uar + 1], axis=2
    ) / jnp.sum(
        mu['n'][..., lar:uar + 1], axis=2
    )
    inc_q = sp.stats.mstats.mquantiles(inc, axis=0)
    ax.plot(jnp.arange(len(inc_q[1])), inc_q[1], color='red')
    ax.fill_between(jnp.arange(len(inc_q[1])), inc_q[0], inc_q[2], color='red', alpha=.1)
    inc = jnp.sum(
        mu_prior['n_inc_clinical'][..., lar:uar + 1], axis=2
    ) / jnp.sum(
        mu_prior['n'][..., lar:uar + 1], axis=2
    )
    inc_q = sp.stats.mstats.mquantiles(inc, axis=0)
    ax.plot(jnp.arange(len(inc_q[1])), inc_q[1], color='green')
    ax.fill_between(jnp.arange(len(inc_q[1])), inc_q[0], inc_q[2], color='green', alpha=.1)
    inc_risk_time = site_data.inc_pop[inc_i] * (ut - lt)
    lam = site_data.inc[inc_i] / inc_risk_time
    lam_ci = z*np.sqrt(lam / inc_risk_time)
    t = (lt + ut) / 2
    terr = ut - t
    ax.errorbar([t], [lam], yerr=lam_ci, xerr=terr, fmt='bo')
    ax.set_title(f'Incidence in {site_name} ({site_iso}) between {lar} and {uar} years')
fig.tight_layout()
```

```{code-cell} ipython3
posterior_imm_curves = get_batch_imm_curves(x_intrinsic)
```

```{code-cell} ipython3
from matplotlib import pyplot as plt
```

```{code-cell} ipython3
jamie = pd.read_csv('/mnt/gc1610/home/fastms/data/jamie_draws.csv')
```

```{code-cell} ipython3
j_params = jamie.replace({
    'IB0': 'ib0',
    'bmin': 'b0',
    'IC0': 'ic0',
    'P_IC_M': 'pcm',
    'dmin': 'd1',
    'ID0': 'id0',
    'ad0': 'ad',
    'gamma_inf': 'gamma1'
})
j_params = j_params[j_params.parameter.isin(intrinsic)]
j_params = {
    k: jnp.array(v)
    for k, v
    in j_params.groupby('parameter').value.apply(list).reset_index().itertuples(index=False)
}
```

```{code-cell} ipython3
jamie_imm_curves = get_batch_imm_curves(j_params)
```

```{code-cell} ipython3
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1.1, 1.05))

fig, axs = plt.subplots(1, 3)
imm_labels = ['prob_b', 'prob_c', 'prob_d']

for imm_i, imm in enumerate(imm_labels):
    axs[imm_i].plot(prior_imm_curves[imm].T, color='g', alpha=.1, label='prior')
    axs[imm_i].plot(posterior_imm_curves[imm].T, color='r', alpha=.1, label='current')
    axs[imm_i].plot(jamie_imm_curves[imm].T, color='b', alpha=.1, label='previous')
    axs[imm_i].set_ylabel(imm)
        
fig.tight_layout()
legend_without_duplicate_labels(axs[-1])
fig.text(0.5, 0, 'Exposures (number)', ha='center')
fig.text(0.5, 1, 'Estimated posterior immunity probability function', ha='center')
```

```{code-cell} ipython3
print(
    az.summary(inf, kind='stats').reset_index().to_latex(
        index=False, float_format="{:0.2f}".format)
)
```

```{code-cell} ipython3
print(
    az.summary(inf, var_names=intrinsic, kind='stats').reset_index().to_latex(
        index=False, float_format="{:0.2f}".format)
)
```

```{code-cell} ipython3
site_summ = site_data.site_index.copy()
```

```{code-cell} ipython3
site_summ['prev_counts'] = [jnp.sum(site_data.prev_index == i) for i in range(len(site_summ))]
site_summ['inc_counts'] = [jnp.sum(site_data.inc_index == i) for i in range(len(site_summ))]
```

```{code-cell} ipython3
print(site_summ.groupby('iso3c').agg(
    {'name_1': len, 'prev_counts': 'sum', 'inc_counts': 'sum'}
).reset_index().rename(
    {'name_1': 'n_sites', 'prev_counts': 'prev_points', 'inc_counts': 'inc_points'},
    axis=1
).to_latex(index=False, float_format="{:0.0f}".format))
```

```{code-cell} ipython3
site_data.site_index
```

```{code-cell} ipython3

```
