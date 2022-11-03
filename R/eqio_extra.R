args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
n <- as.numeric(args[2])
seed <- as.numeric(args[3])
out_dir <- args[4]

p <- malariaEquilibrium::load_parameter_set()

set.seed(seed)

basic_params <- list(
  EIR = list(min=0, max=1000),
  eta = list(min=1/(40 * 365), max=1/(20 * 365)),
  Q0 = list(min=0, max=1)
)

all_params <- c(
  basic_params,
  list(
    s2 = list(min=1, max=3),
    rU = list(min=1/100, max=1/30),
    cT = list(min=0, max=1),
    cD = list(min=0, max=1),
    g_inf = list(min=0.01, max=10),
    cU = list(min=0, max=1),
    kb = list(min=0.01, max=10),
    ub = list(min=1, max=10),
    uc = list(min=1, max=10),
    ud = list(min=1, max=10),
    kc = list(min=0.01, max=10),
    b0 = list(min=0.01, max=0.99),
    b1_prop = list(min=0, max=1),
    IB0 = list(min=1, max=100),
    IC0 = list(min=1, max=100)
  )
)

sample_params <- function(n, paramset) {
  r <- lhs::randomLHS(n, length(paramset))
  cols <- lapply(
    seq_along(paramset),
    function(i) {
      qunif(r[,i], min=paramset[[i]]$min, max=paramset[[i]]$max)
    }
  )
  names(cols) <- names(paramset)
  data.frame(cols)
}

samples <- sample_params(n, all_params)
# samples <- sample_params(n, basic_params)

run <- function(row) {
  sample_p <- p
  for (name in names(row)) {
    if (name == 'b1_prop') {
      sample_p['b1'] <- row['b0'] * row['b1_prop']
    } else if (name == 'EIR') {
      next
    } else {
      sample_p[name] <- row[name]
    }
  }
  ages = 0:100
  eq <- malariaEquilibrium::human_equilibrium(
    EIR = row['EIR'],
    ft = 0,
    sample_p,
    age=ages
  )
  output_columns = c(
    'pos_M',
    'prop',
    'ICA',
    'ICM',
    'ID',
    'IB',
    'inc'
  )

  output = NULL
  output_names = NULL
  for (c in output_columns) {
    output <- c(output, eq$states[,c])
    output_names = c(output_names, paste0(c, '_', ages))
  }
  names(output) <- output_names
  return(output)
}

output <- apply(samples, 1, run)
output <- cbind(samples, t(output))

write.csv(
  output,
  file.path(out_dir, paste0('realisations_', node,'.csv')),
  row.names = FALSE
)

