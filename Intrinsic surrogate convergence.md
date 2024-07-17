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
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import numpy as np
```

```{code-cell} ipython3
base_dir = 'outputs/v4/'
```

```{code-cell} ipython3
samples = sorted(
    list(
        {
            int(path.replace(base_dir, '').split('_')[0])
            for path in glob(f'{base_dir}*_round_*_ll.csv')
        }
    )
)
rounds = list(range(5))
```

```{code-cell} ipython3
samplers = ['nuts', 'svi', 'svi_annealed']
props = ['lhs_full', 'lhs_fixed', 'prior_full', 'prior_fixed']
```

```{code-cell} ipython3
def get_desc(samples, prop, sampler, round):
    return f'{samples}_{prop}_{sampler}_round_{round}'
```

```{code-cell} ipython3
ks_error = pd.concat([
    pd.read_csv(
        f'{base_dir}{get_desc(s, prop, m, r)}_ks_error.csv'
    ).assign(samples=s, round=r, sampler=m, proposal=prop)
    for s in samples
    for r in rounds
    for m in samplers
    for prop in props
] + [
    pd.read_csv(
        f'outputs/v7/{get_desc(100_000, "prior_full", m, r)}_ks_error.csv'
    ).assign(samples=100_000, round=r, sampler=m, proposal="prior_full")
    for r in rounds
    for m in ['svi', 'svi_annealed'] 
])
ks_error = ks_error.assign(null=np.where(ks_error['p-value'] < 0.1, 'reject', 'accept'))
```

```{code-cell} ipython3
g = sns.FacetGrid(
    ks_error[ks_error['round'] == ks_error['round'].max()],
    col="sampler",
    hue='proposal',
    margin_titles=True
)
g.map(sns.lineplot, "samples", "statistic")
g.add_legend()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(19.7, 8.27))
sns.barplot(
    ks_error[
        (ks_error.samples == max(ks_error['samples'])) &
        (ks_error['round'] == max(rounds)) &
        (ks_error['sampler'] == 'svi_annealed') &
        (ks_error['experiment'] == 'prior_full')
        #ks_error.variable.isin(['b0', 'phi0', 'phi1'])
    ],
    x='variable',
    y='statistic',
    hue='null',
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
    pd.read_csv(
        f'{base_dir}{get_desc(s, prop, m, r)}_approx_error.csv'
    ).assign(samples=s, round=r, sampler=m, proposal=prop)
    for s in samples
    for r in rounds
    for m in samplers
    for prop in props
])

stand_approx_error = pd.concat([
    pd.read_csv(
        f'{base_dir}{get_desc(s, prop, m, r)}_stand_approx_error.csv'
    ).assign(samples=s, round=r, sampler=m, proposal=prop)
    for s in samples
    for r in rounds
    for m in samplers
    for prop in props
])
```

```{code-cell} ipython3
ax = sns.lineplot(
    stand_approx_error[
        (stand_approx_error.test_set == 'prior') &
        (stand_approx_error['round'] == 0) & 
        (stand_approx_error.mse < 1e10)
    ],
    x = 'samples',
    y = 'mse',
    hue = 'proposal'
)
```

```{code-cell} ipython3
g = sns.FacetGrid(
    approx_error[
        (approx_error.test_set == 'prior') &
        (approx_error['round'] == 0) & 
        (approx_error.mse < 1e10)
    ],
    col="output",
    hue="proposal",
    margin_titles=True,
    sharey=True
)
g.map(sns.lineplot, "samples", "mse")
for ax in g.axes_dict.values():
    ax.set_yscale('log')
g.add_legend()
```

```{code-cell} ipython3
ax = sns.lineplot(
    stand_approx_error[
        (stand_approx_error.test_set == 'prior') &
        (stand_approx_error['samples'] == stand_approx_error['samples'].max()) &
        (stand_approx_error.mse < 1e10)
    ],
    x = 'round',
    y = 'mse',
    hue = 'proposal'
)
ax.set_xticks(range(5))
```

```{code-cell} ipython3
g = sns.FacetGrid(
    stand_approx_error[
        (stand_approx_error.test_set == 'posterior') &
        (stand_approx_error['round'] == 4)
    ],
    col="sampler",
    hue="proposal",
    margin_titles=True,
    sharey=True
)
g.map(sns.lineplot, "samples", "mse")
g.add_legend()
```

```{code-cell} ipython3
g = sns.FacetGrid(
    approx_error[
        (approx_error.test_set == 'posterior') &
        (approx_error['round'] == 4) & 
        (approx_error.mse < 1e10)
    ],
    col="output",
    row='sampler',
    hue="proposal",
    margin_titles=True,
    sharey=True
)
g.map(sns.lineplot, "samples", "mse")
for ax in g.axes_dict.values():
    ax.set_yscale('log')
g.add_legend()
```

```{code-cell} ipython3
g = sns.FacetGrid(
    stand_approx_error[
        (stand_approx_error.test_set == 'posterior') &
        (stand_approx_error['samples'] == stand_approx_error['samples'].max())
    ],
    col="sampler",
    #row='sampler',
    hue="proposal",
    margin_titles=True,
    sharey=True
)
g.map(sns.lineplot, "round", "mse")
g.add_legend()
```

```{code-cell} ipython3
timings = pd.concat([
    pd.read_csv(
        f'{base_dir}{get_desc(s, prop, m, r)}_timings.csv'
    ).assign(samples=s, round=r, sampler=m, proposal=prop)
    for s in samples
    for r in rounds
    for m in samplers
    for prop in props
])
```

```{code-cell} ipython3
timings = pd.melt(
    timings,
    id_vars=['proposal', 'samples', 'round', 'sampler'],
    value_vars=['sampling', 'training', 'inference'],
    var_name='task',
    value_name='time'
)
```

```{code-cell} ipython3
g = sns.FacetGrid(
    timings[(timings.task.isin(['sampling', 'training'])) & (timings['round'] == 0)],
    col="task",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "samples", "time")

#sns.lineplot(
#    timings[(timings.task.isin(['sampling'])) & (timings['round'] == 0)],
#    x="samples",
#    y="time",
#    hue='task'
#)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2)
for i, s in enumerate(['nuts', 'svi']):
    sns.boxplot(
        timings[
            (timings.task == 'inference') & (timings.sampler == s)
        ],
        x='sampler',
        y='time',
        ax=axes[i]
    )
axes[1].set_ylabel('')
fig.tight_layout()
```

```{code-cell} ipython3
ll = pd.concat([
    pd.read_csv(
        f'{base_dir}{get_desc(s, prop, m, r)}_ll.csv'
    ).assign(samples=s, round=r, sampler=m, proposal=prop)
    for s in samples
    for r in rounds
    for m in samplers
    for prop in props
]+ [
    pd.read_csv(
        f'outputs/v7/{get_desc(100_000, "prior_full", m, r)}_ll.csv'
    ).assign(samples=100_000, round=r, sampler=m, proposal="prior_full")
    for r in rounds
    for m in ['svi', 'svi_annealed'] 
])
```

```{code-cell} ipython3
u_ll = pd.read_csv(f'{base_dir}underlying_ll.csv')
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
    id_vars=['proposal', 'samples', 'round', 'sampler'],
    value_vars=['obs_inc', 'obs_prev'],
    var_name='obs',
    value_name='log_likelihood'
)
```

```{code-cell} ipython3
g = sns.FacetGrid(
    ll[
        #(ll.obs == 'obs_inc') &
        #ll.proposal.isin(['prior_full', 'prior_fixed'])
        #(ll.samples == ll.samples.max()) &
        (ll['round'] == ll['round'].max()) &
        (ll.proposal == 'prior_full')
    ],
    col='sampler',
    #row="sampler",
    #hue="proposal",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "samples", "log_likelihood")
for _, ax in g.axes_dict.items():
    #ax.set_yscale('log')
    y = u_ll.log_likelihood
    ax.axhline(y.mean(), 0, 4, ls='--')
    ax.fill_between(list(range(5)), y.min(), y.max(), color='black', alpha=.1)
g.add_legend()
```

```{code-cell} ipython3
pp_ll = pd.concat([
    pd.read_csv(
        f'{base_dir}{get_desc(s, prop, m, r)}_pp_ll.csv'
    ).assign(samples=s, round=r, sampler=m, proposal=prop)
    for s in samples
    for r in rounds
    for m in samplers
    for prop in props
]+ [
    pd.read_csv(
        f'outputs/v7/{get_desc(100_000, "prior_full", m, r)}_pp_ll.csv'
    ).assign(samples=100_000, round=r, sampler=m, proposal="prior_full")
    for r in rounds
    for m in ['svi', 'svi_annealed'] 
])
```

```{code-cell} ipython3
pp_ll = pd.melt(
    pp_ll,
    id_vars=['proposal', 'samples', 'round', 'sampler', 'EIR'],
    value_vars=['prev_2_10', 'prev_10+', 'inc_0_5', 'inc_5_15', 'inc_15+'],
    var_name='output',
    value_name='log_likelihood'
)
```

```{code-cell} ipython3
u_pp_ll = pd.read_csv(f'{base_dir}underlying_pp_ll.csv')
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
    pp_ll[
        (pp_ll['round']== pp_ll['round'].max()) &
        (pp_ll.proposal == 'prior_full')
    ],
    col='sampler',
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "samples", "log_likelihood")
for _, ax in g.axes_dict.items():
    #ax.set_yscale('log')
    y = u_pp_ll.log_likelihood
    ax.axhline(y.mean(), 0, 4, ls='--')
    ax.fill_between(list(range(5)), y.min(), y.max(), color='black', alpha=.1)
g.add_legend()
```

```{code-cell} ipython3
summaries = pd.concat([
    pd.read_csv(
        f'{base_dir}{get_desc(s, prop, m, r)}_mcmc_summary.csv'
    ).assign(samples=s, round=r, sampler=m, proposal=prop)
    for s in samples
    for r in rounds
    for m in ['nuts']
    for prop in props
])
summaries = summaries[summaries['index'] != 'EIR']
summaries.r_hat = pd.to_numeric(summaries.r_hat)
summaries.n_eff = pd.to_numeric(summaries.n_eff)
```

```{code-cell} ipython3
g = sns.FacetGrid(
    summaries,
    col='samples',
    #hue="proposal",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "round", "r_hat")
g.add_legend()
```

```{code-cell} ipython3
losses = pd.concat([
    pd.read_csv(
        f'{base_dir}{get_desc(s, prop, m, r)}_svi_losses.csv'
    ).assign(samples=s, round=r, sampler=m, proposal=prop)
    for s in samples
    for r in rounds
    for m in ['svi', 'svi_annealed']
    for prop in props
])
```

```{code-cell} ipython3
g = sns.FacetGrid(
    losses[
        (losses['round'] == max(rounds)) &
        (losses['step'] > 0)
    ],
    col='samples',
    row='sampler',
    hue="proposal",
    margin_titles=True,
    sharey=False
)
g.map(sns.lineplot, "step", "loss")
#for ax in g.axes_dict.values():
#    ax.set_yscale('log')
g.add_legend()
```

```{code-cell} ipython3

```
