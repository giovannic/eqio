args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
n_batches <- as.numeric(args[2])
batch_size <- as.numeric(args[3])
seed <- as.numeric(args[4])
seasonality_file <- args[5]
out_dir <- args[6]

n <- n_batches * batch_size

set.seed(seed)

all_params <- list(
  init_EIR = list(min=0, max=100),
  eta = list(min=1/(40 * 365), max=1/(20 * 365)),
  Q0 = list(min=0, max=1),
  sigma2 = list(min=1, max=3),
  rU = list(min=1/100, max=1/30),
  cT = list(min=0, max=1),
  cD = list(min=0, max=1),
  gamma1 = list(min=0.01, max=10),
  cU = list(min=0, max=1),
  kB = list(min=0.01, max=10),
  uB = list(min=1, max=10),
  uCA = list(min=1, max=10),
  uD = list(min=1, max=10),
  kC = list(min=0.01, max=10),
  b0 = list(min=0.01, max=0.99),
  b1_prop = list(min=0, max=1),
  IB0 = list(min=1, max=100),
  IC0 = list(min=1, max=100)
)

sample_df <- function(df, n) {
  df[sample(nrow(df), n, replace = TRUE), ]
}

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

process <- function(row) {
  p <- row[names(row) != 'b1_prop']
  p['b1'] <- row['b0'] * row['b1_prop']
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

samples <- sample_params(n, all_params)
seasonality <- read.csv(seasonality_file)
names(seasonality) <- c('ssa0', 'ssa1', 'ssa2', 'ssa3', 'ssb1', 'ssb2', 'ssb3')
samples <- cbind(samples, sample_df(seasonality, n))

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
