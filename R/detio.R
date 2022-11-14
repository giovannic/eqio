args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
n_batches <- as.numeric(args[2])
batch_size <- as.numeric(args[3])
seed <- as.numeric(args[4])
out_dir <- args[5]

n <- n_batches * batch_size

set.seed(seed)

basic_params <- data.frame(do.call(
  'rbind',
  list(
    list(name='init_EIR', min=0, max=100),
    list(name='eta', min=1/(40 * 365), max=1/(20 * 365)),
    list(name='Q0', min=0, max=1),
    list(name='sigma2', min=1, max=3),
    list(name='rU', min=1/1000, max=1/30),
    list(name='cD', min=0, max=1),
    list(name='gamma1', min=0.01, max=10),
    list(name='cU', min=0, max=1),
    list(name='kB', min=0.01, max=10),
    list(name='uB', min=1, max=10),
    list(name='uCA', min=1, max=10),
    list(name='uD', min=1, max=10),
    list(name='kC', min=.01, max=10),
    list(name='b0', min=.01, max=.99),
    list(name='b1', min=.01, max=.99),
    list(name='IB0', min=1, max=100),
    list(name='IC0', min=1, max=100),
    list(name='delayMos', min=1, max=20),
    list(name='phi0', min=0, max=1),
    list(name='phi1', min=0, max=1),
    list(name='mu0', min=0, max=1),
    list(name='fD0', min=0, max=1),
    list(name='aD', min=20 * 365, max=40 * 365),
    list(name='gammaD', min=1, max=10),
    list(name='ssa0', min=-10, max=10),
    list(name='ssa1', min=-10, max=10),
    list(name='ssa2', min=-10, max=10),
    list(name='ssa3', min=-10, max=10),
    list(name='ssb1', min=-10, max=10),
    list(name='ssb2', min=-10, max=10),
    list(name='ssb3', min=-10, max=10)
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

process <- function(row) {
  p <- row
  p['max_t'] <- 500 * 365
  p['tolerance'] <- 1e-2
  tryCatch({
      model_output <- do.call(ICDMM:::run_model_until_stable, as.list(p))
      model_output <- tail(model_output, 365)
      outputs <- c(model_output$prev, mean(model_output$EIR_out))
      names(outputs) <- c(paste0('prev_', seq(365)), 'EIR')
      return(outputs)
    },
    error = function(msg) {
      print('model run error:')
      print(msg)
      outputs <- rep(-1, 366)
      names(outputs) <- c(paste0('prev_', seq(365)), 'EIR')
      return(outputs)
    }
  )
}

samples <- sample_params(n, basic_params)

batches <- split(
  seq(nrow(samples)),
  (seq(nrow(samples))-1) %/% batch_size
)

for (i in seq_along(batches)) {
  outpath <- file.path(
    out_dir,
    paste0('realisations_', node,'_batch_',i,'.csv')
  )
  if (!file.exists(outpath)) {
    sample_batch = samples[batches[[i]],]

    output <- apply(sample_batch, 1, process)
    output <- cbind(sample_batch, t(output))

    write.csv(output, outpath, row.names = FALSE)
  }
}
