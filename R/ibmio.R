args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
n_batches <- as.numeric(args[2])
batch_size <- as.numeric(args[3])
seed <- as.numeric(args[4])
tolerance <- as.numeric(args[5])
max_t <- as.numeric(args[6])
n_pop <- as.numeric(args[7])
out_dir <- args[8]

n <- n_batches * batch_size

set.seed(seed)

basic_params <- data.frame(do.call(
  'rbind',
  list(
    list(name='init_EIR', min=0, max=100),
    list(name='average_age', min=(20* 365), max=(40 * 365)),
    list(name='Q0', min=0, max=1),
    list(name='sigma_squared', min=1, max=3),
    list(name='du', min=30, max=1000),
    list(name='cd', min=0, max=1),
    list(name='gamma1', min=0.01, max=10),
    list(name='cu', min=0, max=1),
    list(name='kb', min=0.01, max=10),
    list(name='ub', min=1, max=10),
    list(name='uc', min=1, max=10),
    list(name='ud', min=1, max=10),
    list(name='kc', min=.01, max=10),
    list(name='b0', min=.01, max=.99),
    list(name='b1', min=.01, max=.99),
    list(name='ib0', min=1, max=100),
    list(name='ic0', min=1, max=100),
    list(name='dem', min=1, max=20),
    list(name='phi0', min=0, max=1),
    list(name='phi1', min=0, max=1),
    list(name='mum', min=0, max=1),
    list(name='fd0', min=0, max=1),
    list(name='ad', min=20 * 365, max=40 * 365),
    list(name='gammad', min=1, max=10),
    list(name='g0', min=-10, max=10),
    list(name='g1', min=-10, max=10),
    list(name='g2', min=-10, max=10),
    list(name='g3', min=-10, max=10),
    list(name='h1', min=-10, max=10),
    list(name='h2', min=-10, max=10),
    list(name='h3', min=-10, max=10)
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
  p <- as.list(
    row[!(names(row) %in% c('init_EIR', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3'))]
  )
  p$g <- as.numeric(row[c('g1', 'g2', 'g3')])
  p$h <- as.numeric(row[c('h1', 'h2', 'h3')])
  p$human_population <- n_pop
  params <- malariasimulation::get_parameters(p)
  params$individual_mosquitoes <- FALSE
  params$model_seasonality <- TRUE
  params <- malariasimulation::set_equilibrium(params, row[['init_EIR']])
  tryCatch({
      output <- malariasimulation:::run_simulation_until_stable(
        params,
        tolerance = tolerance,
        max_t = max_t,
        post_t = 365,
        post_parameters = function(t) params
      )
      baseline <- mean(output$post$EIR_All) / n_pop * 365
      prev <- as.numeric(output$post[,'n_detect_730_3650'] / output$post[,'n_730_3650'])
      outputs <- c(prev, baseline)
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
