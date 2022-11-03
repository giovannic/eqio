args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
n <- as.numeric(args[2])
seed <- as.numeric(args[3])
out_dir <- args[4]

p <- malariaEquilibrium::load_parameter_set()

set.seed(seed)

basic_params <- data.frame(do.call(
  'rbind',
  list(
    list(name='EIR', min=0, max=100),
    list(name='eta', min=1/(40 * 365), max=1/(20 * 365)),
    list(name='Q0', min=0, max=1),
    list(name='s2', min=1, max=3),
    list(name='rU', min=1/1000, max=1/30),
    list(name='cD', min=0, max=1),
    list(name='g_inf', min=0.01, max=10),
    list(name='cU', min=0, max=1),
    list(name='kb', min=0.01, max=10),
    list(name='ub', min=1, max=10),
    list(name='uc', min=1, max=10),
    list(name='ud', min=1, max=10),
    list(name='kc', min=.01, max=10),
    list(name='b0', min=.01, max=.99),
    list(name='b1', min=.01, max=.99),
    list(name='IB0', min=1, max=100),
    list(name='IC0', min=1, max=100),
    list(name='tau', min=1, max=20),
    list(name='phi0', min=0, max=1),
    list(name='phi1', min=0, max=1),
    list(name='mu', min=0, max=1),
    list(name='f', min=0, max=1),
    list(name='fd0', min=0, max=1),
    list(name='ad0', min=20 * 365, max=40 * 365),
    list(name='gd', min=1, max=10)
  )
))

sample_params <- function(n, paramset) {
  r <- lhs::randomLHS(n, nrow(paramset))
  cols <- lapply(
    seq(nrow(paramset)),
    function(i) {
      qunif(r[,i], min=paramset[[i, 'min']], max=paramset[[i, 'max']])
    }
  )

  names(cols) <- paramset[,'name']
  data.frame(cols)
}

samples <- sample_params(n, basic_params)


prev <- function(row) {
  sample_p <- p
  for (name in names(row)) {
    if (name == 'EIR') {
      next
    } else {
      sample_p[name] <- row[name]
    }
  }
  eq_ages <- 0:100
  g2l10 <- (eq_ages > 2) & (eq_ages < 10)
  eq <- malariaEquilibrium::human_equilibrium(
    EIR = row['EIR'],
    ft = 0,
    sample_p,
    age=eq_ages
  )
  sum(eq$states[g2l10,'pos_M']) / sum(eq$states[g2l10,'prop'])
}

samples[,'pfpr2_10'] <- apply(samples, 1, prev)

write.csv(
  samples,
  file.path(out_dir, paste0('realisations_', node,'.csv')),
  row.names = FALSE
)

basic_params[,'name'] <- as.character(basic_params[,'name'])
basic_params[,'min'] <- as.numeric(basic_params[,'min'])
basic_params[,'max'] <- as.numeric(basic_params[,'max'])
write.csv(
  basic_params,
  file.path(out_dir, 'bounds.csv'),
  row.names = FALSE
)

