args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
n_batches <- as.numeric(args[2])
batch_size <- as.numeric(args[3])
seed <- as.numeric(args[4])
tolerance <- as.numeric(args[5])
max_t <- as.numeric(args[6])
human_population <- as.numeric(args[7])
n_pop <- as.numeric(args[8])
out_dir <- args[9]

n <- n_batches * batch_size

set.seed(seed)

extrinsic_params <- function() {
  p <- data.frame(do.call(
    'rbind',
    list(
      list(name='init_EIR', min=0, max=100),
      list(name='average_age', min=20 * 365, max=40 * 365),
      list(name='g0', min=-10, max=10),
      list(name='g1', min=-10, max=10),
      list(name='g2', min=-10, max=10),
      list(name='g3', min=-10, max=10),
      list(name='h1', min=-10, max=10),
      list(name='h2', min=-10, max=10),
      list(name='h3', min=-10, max=10)
    )
  ))
  do.call(
    'rbind',
    lapply(
      seq(n_pop),
      function (i) {
        pop_p <- p
        pop_p[,'name'] <- paste0(pop_p[,'name'], '_', i)
        pop_p
      }
    )
  )
}

usage_params <- function(len) {
  itn_indices <- expand.grid(t = seq(len), p = seq(n_pop))
  data.frame(
    do.call(
      'rbind',
      lapply(
        seq_len(nrow(itn_indices)),
        function(i) {
          list(
            name=paste0('itn_usage_', itn_indices[i, 'p'], '_', itn_indices[i, 't']),
            min=0,
            max=.8
          )
        }
      )
    )
  )
}

p_cap_params <- function() {
  data.frame(
    do.call(
      'rbind',
      lapply(
        seq_len(nrow(non_diagonals)),
        function(i) {
          list(
            name = paste0(
              'p_captured_',
              non_diagonals[i, 'row'],
              '_',
              non_diagonals[i, 'col']
            ),
            min = 0,
            max = 1
          )
        }
      )
    )
  )
}

basic_params <- function() {
  data.frame(do.call(
    'rbind',
    list(
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
      list(name='gammad', min=1, max=10)
    )
  ))
}

sample_lhs_params <- function(r, paramset) {
  params <- lapply(
    seq(nrow(paramset)),
    function(i) {
      qunif(r[,i], min=paramset[[i, 'min']], max=paramset[[i, 'max']])
    }
  )
  names(params) <- paramset[,'name']
  data.frame(params)
}

# method from Geomblog http://blog.geomblog.org/2005/10/sampling-from-simplex.html
sample_row_simplex <- function(r, prefix) {
  col_names <- lapply(
    seq(nrow(matrix_indices)),
    function(i) {
      paste0(prefix, matrix_indices[i, 'row'], '_', matrix_indices[i, 'col'])
    }
  )
  simplexes <- t(vapply(
    seq(nrow(r)),
    function(i) {
      m <- log(matrix(r[i,], nrow=n_pop, ncol=n_pop))
      m <- m / rowSums(m)
    },
    numeric(n_pop^2)
  ))

  s <- data.frame(simplexes)
  names(s) <- col_names
  s
}

sample_params <- function(n, paramset, usage, extrinsic, p_cap_set) {
  r <- lhs::randomLHS(
    n,
    nrow(paramset) + nrow(usage) + nrow(extrinsic) + 2 * n_pop ^ 2 - n_pop
  )

  start_index <- 0
  param_samples <- sample_lhs_params(r[,seq(nrow(paramset)),drop=FALSE], paramset)
  start_index <- start_index + nrow(paramset)
  usage_samples <- sample_lhs_params(r[,seq(nrow(usage)) + start_index,drop=FALSE], usage)
  start_index <- start_index + nrow(usage)
  extrinsic_samples <- sample_lhs_params(r[,seq(nrow(extrinsic)) + start_index,drop=FALSE], extrinsic)
  start_index <- start_index + nrow(extrinsic)
  mixing_samples <- sample_row_simplex(
    r[,seq(n_pop^2) + start_index,drop=FALSE],
    'mixing_'
  )
  start_index <- start_index + n_pop ^ 2
  p_captured <- sample_lhs_params(
    r[,seq(n_pop^2 - n_pop) + start_index,drop=FALSE],
    p_cap_set
  )

  do.call(
    'cbind',
    list(param_samples, extrinsic_samples, usage_samples, mixing_samples, p_captured)
  )
}

process <- function(row) {
  params <- lapply(
    seq(n_pop),
    function(i) {
      p <- as.list(row[as.character(t(basic[,'name']))])
      p$g0 <- row[[paste0('g0_', i)]]
      p$g <- as.numeric(row[paste0(c('g1', 'g2', 'g3'), '_', i)])
      p$h <- as.numeric(row[paste0(c('h1', 'h2', 'h3'), '_', i)])
      p$average_age <- row[[paste0('average_age_', i)]]
      p$human_population <- human_population
      params <- malariasimulation::get_parameters(p)
      params$individual_mosquitoes <- FALSE
      params$model_seasonality <- TRUE
      params <- malariasimulation::set_equilibrium(
        params,
        row[[paste0('init_EIR_', i)]]
      )
    }
  )
  post_warmup_parameters <- function(warmup) {
    lapply(
      seq_along(params),
      function(i) {
        malariasimulation::set_bednets(
          params[[i]],
          timesteps = warmup + seq(NET_LENGTH) * 365,
          coverages = row[paste0('itn_usage_', i, '_', seq(NET_LENGTH))],
          retention = 5 * 365,
          dn0 = matrix(rep(.533, NET_LENGTH), nrow=NET_LENGTH, ncol=1),
          rn = matrix(rep(.56, NET_LENGTH), nrow=NET_LENGTH, ncol=1),
          rnm = matrix(rep(.24, NET_LENGTH), nrow=NET_LENGTH, ncol=1),
          gamman = rep(2.64 * 365, NET_LENGTH)
        )
      }
    )
  }
  tryCatch({
    mixing <- matrix(
      row[paste0(
        'mixing_',
        matrix_indices[, 'row'],
        '_',
        matrix_indices[, 'col']
      )],
      nrow = n_pop,
      ncol = n_pop
    )
    p_captured <- matrix(
      1,
      nrow = n_pop,
      ncol = n_pop
    )
    p_captured[as.matrix(non_diagonals)] <- row[
      as.character(t(p_cap_set[,'name']))
    ]

    output <- malariasimulation:::run_metapop_simulation_until_stable(
      params,
      mixing_tt = 1,
      mixing = list(mixing),
      p_captured_tt = 1,
      p_captured = list(p_captured),
      p_success = .9,
      tolerance = tolerance,
      max_t = max_t,
      post_t = NET_LENGTH * 365,
      post_parameters = post_warmup_parameters
    )
    baseline <- vapply(
      seq_along(params),
      function(i) {
        mean(output$post[[i]]$EIR_All) / human_population * 365
      },
      numeric(1)
    )
    prev <- unlist(
      lapply(
        seq_along(params),
        function(i) {
          as.numeric(
            output$post[[i]][,'n_detect_730_3650'] / output$post[[i]][,'n_730_3650']
          )
        }
      )
    )
    prev_n <- NET_LENGTH * 365
    outputs <- c(prev, baseline)
    names(outputs) <- c(
      unlist(
        lapply(
          seq_along(params),
          function(i) {
            paste0('prev_', i, '_', seq(prev_n))
          }
        )
      ),
      paste0('EIR_', seq_along(params))
    )
    return(outputs)
  },
  error = function(msg) {
    print('model run error:')
    print(msg)
    outputs <- rep(-1, 366)
    names(outputs) <- c(paste0('prev_', seq(365)), 'EIR')
    return(outputs)
  })
}

NET_LENGTH <- 15
matrix_indices <- expand.grid(row = seq(n_pop), col = seq(n_pop))
non_diagonals <- matrix_indices[matrix_indices[,'row'] != matrix_indices[,'col'] ,]
basic <- basic_params()
usage <- usage_params(NET_LENGTH)
extrinsic <- extrinsic_params()
p_cap_set <- p_cap_params()
samples <- sample_params(n, basic, usage, extrinsic, p_cap_set)

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
