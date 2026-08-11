# Confidence levels used in the simulation study
levels <- c(0.50, 0.90, 0.95)

prepare_data <- function(data) {
  # Make group columns categorical
  data$l3 <- factor(data$l3)
  data$l2 <- factor(data$l2)
  
  data
}

fit_model <- function(data) {
  data <- prepare_data(data)
  
  # Y_{ijk} = \mu + u_i + v_{ij} + \epsilon_{ijk}
  model.lme <- suppressMessages(
    suppressWarnings(
      lme4::lmer(
        formula = y ~ 1 + (1 | l3) + (1 | l3:l2),
        data = data,
        # Might use this so that methods are all comparable with eachother
        # REML = FALSE,
        # Ignore the singular fit warnings
        control = lme4::lmerControl(
          check.conv.singular = "ignore"
        )
      )
    )
  )
  
  model.lme
}

extract_model_components <- function(model) {
  # Extract the std. dev components
  vc <- lme4::VarCorr(model)
  
  c(
    # Global mean
    mu = unname(lme4::fixef(model)[1]),
    # std. dev of u_i
    sd_l3 = unname(attr(vc[["l3"]], "stddev")[1]),
    # std. dev of v_{ij}
    sd_l2 = unname(attr(vc[["l3:l2"]], "stddev")[1])
  )
}

# Used for the nonparametric cases bootstrap CI
statistic <- function(data) {
  model.lme <- fit_model(data)
  
  extract_model_components(model.lme)
}

# Util to avoid repeating the same concatenation
concat_results <- function(intervals, estimates) {
  # Concatenate the results together
  result <- cbind(intervals[[1]], intervals[[2]], intervals[[3]])
  result <- result[, c(5, 3, 1, 2, 4, 6)]
  
  colnames(result) <- c("0.025", "0.05", "0.25", "0.75", "0.95", "0.975")
  rownames(result) <- c("mu", "sd_l3", "sd_l2")
  
  cbind(result, estimate = estimates)
}

# Used for the profile-likelihood CI
# Compute them for multiple levels at the same time
profile_cis <- function(data, n_cpus = 1) {
  model.lme <- fit_model(data)
  
  # Get the estimates from the model
  estimates <- extract_model_components(model.lme)
  
  # Parameters we are interested in
  parameters = c("(Intercept)", "sd_(Intercept)|l3", "sd_(Intercept)|l3:l2") 
  
  # Compute the profile once
  p <- suppressMessages(
    suppressWarnings(
      profile(
        model.lme,
        which = parameters,
        parallel = "multicore",
        ncpus = n_cpus,
        signames = FALSE
      )
    )
  )
  
  # Compute two sided CI at different levels (we can use the bounds of 
  # two sided ones to construct one sided ones)
  intervals <- lapply(
    levels,
    function(level) {
      ci <- suppressMessages(
        suppressWarnings(confint(p, level = level))
      )
      ci[parameters,]
    }
  )
  
  concat_results(intervals, estimates)
}

# Used for parametric percentile bootstrap CI
# Compute them for multiple levels at the same time
parametric_boot_cis <- function(data, B = 1000, seed = NULL, n_cpus = 1) {
  model.lme <- fit_model(data)
  
  # Get the estimates from the model
  estimates <- extract_model_components(model.lme)
  
  # Compute bootstrap results
  boot <- lme4::bootMer(
    model.lme,
    FUN = extract_model_components,
    seed = seed,
    type = "parametric",
    # Make sure random effects are resampled from the assumed normal distribution
    use.u = FALSE,
    parallel = "multicore",
    ncpus = n_cpus,
    nsim = B
  )
  
  # Compute two sided CI at different levels (we can use the bounds of 
  # two sided ones to construct one sided ones)
  intervals <- lapply(
    levels,
    function(level) {
      ci <- suppressMessages(
        suppressWarnings(confint(boot, level = level, type = "perc"))
      )
      
      ci
    }
  )
  
  concat_results(intervals, estimates)
}
