# Confidence levels used in the simulation study
levels <- c(0.50, 0.90, 0.95)

# Parameters we are interested in
parameters = c("(Intercept)", "sd_(Intercept)|l3", "sd_(Intercept)|l3:l2") 

prepare_data <- function(data) {
  # Make group columns categorical
  data$l3 <- factor(data$l3)
  data$l2 <- factor(data$l2)
  
  data
}

fit_model <- function(data) {
  data <- prepare_data(data)
  
  # Y_{ijk} = \mu + u_i + v_{ij} + \epsilon_{ijk}
  model.lme <- tryCatch(
    suppressMessages(
      suppressWarnings(
        lme4::lmer(
          formula = y ~ 1 + (1 | l3) + (1 | l3:l2),
          data = data,
          # Use ML estimation since profile likelihood refits to ML
          # and comparison is therefore cleaner
          REML = FALSE,
          # Ignore the singular fit warnings
          control = lme4::lmerControl(
            check.conv.singular = "ignore"
          )
        )
      )
    ), error = function(e) NULL
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

# Used for the nonparametric cases bootstrap CI -> return NAs if fit didn't succeed
statistic <- function(data) {
  model.lme <- fit_model(data)
  
  if (is.null(model.lme)) {
    return(c(mu = NA_real_, sd_l3 = NA_real_, sd_l2 = NA_real_))
  }
  
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

# Util to return an empty result if model failed to fit to data
null_result <- function() {
    out <- matrix(
      NA_real_,
      nrow = length(parameters),
      ncol = 2 * length(levels),
      dimnames = list(
        c("mu", "sd_l3", "sd_l2"),
        c("0.025", "0.05", "0.25", "0.75", "0.95", "0.975")
      )
    )
    
    out
}

# Used for the profile-likelihood CI
# Compute them for multiple levels at the same time
profile_cis <- function(data, n_cpus = 1) {
  model.lme <- fit_model(data)
  
  
  # Failed to fit a model to the data -> return a matrix of NAs
  if (is.null(model.lme)) {
    null <- null_result()
    # Add estimate
    result <- cbind(null, estimate = NA_real_)
    
    return(result)
  }
  
  # Get the estimates from the model
  estimates <- extract_model_components(model.lme)
  
  # Compute the profile once - profiling might fail when the parameter estimate
  # is near the boundary of the parameter space. Compute separate profiles for
  # different parameters
  p_list <- setNames(
    lapply(
      parameters,
      FUN = function(param) {
        p <- tryCatch(
          suppressMessages(
            suppressWarnings(
              profile(
                model.lme,
                which = param,
                parallel = "multicore",
                ncpus = n_cpus,
                signames = FALSE
              )
            )
          ),
          error = function(e) NULL
        )
        
        p
      }
    ),
    parameters
  )
  
  # Compute two sided CI at different levels (we can use the bounds of 
  # two sided ones to construct one sided ones).
  intervals <- lapply(
    levels,
    FUN = function(level) {
      ci_out <- matrix(
        data = NA_real_,
        nrow = length(parameters),
        ncol = 2,
      )
      rownames(ci_out) <- parameters
      
      # Compute CIs for parameters separately
      for (parameter in parameters) {
        # Extract the profile for the parameter
        p <- p_list[[parameter]]
        
        # If the profile couldn't be calculated, skip it -> leave as NA
        if (is.null(p)) {
          next
        }
        
        # CI construction can sometimes also fail
        ci <- tryCatch(
          suppressMessages(
            suppressWarnings(confint(p, level = level))
          ), error = function(e) NULL
        ) 
        
        # Confidence interval calculation failed
        if (is.null(ci)) {
          next
        }
        
        # Sanity check
        if (parameter %in% rownames(ci)) {
          ci_out[parameter,] <- ci[parameter,]
        }
      }
      
      ci_out
    }
  )
  
  concat_results(intervals, estimates)
}

# Used for parametric percentile bootstrap CI
# Compute them for multiple levels at the same time
parametric_boot_cis <- function(data, B = 1000, seed = NULL, n_cpus = 1) {
  model.lme <- fit_model(data)
  
  # Failed to fit a model to the data -> return a matrix of NAs
  if (is.null(model.lme)) {
    null <- null_result()
    # Add estimate
    result <- cbind(null, estimate = NA_real_)
    
    return(result)
  }
  
  # Get the estimates from the model
  estimates <- extract_model_components(model.lme)
  
  # Compute bootstrap results
  boot <- tryCatch(lme4::bootMer(
      model.lme,
      FUN = extract_model_components,
      seed = seed,
      type = "parametric",
      # Make sure random effects are resampled from the assumed normal distribution
      use.u = FALSE,
      parallel = "multicore",
      ncpus = n_cpus,
      nsim = B
    ), error = function(e) NULL
  )
  
  if (is.null(boot)) {
    null <- null_result()
    # Add estimate
    result <- cbind(null, estimate = estimates)
    
    return(result)
  }
  
  # Compute two sided CI at different levels (we can use the bounds of 
  # two sided ones to construct one sided ones)
  boot_params <- c("mu", "sd_l3", "sd_l2")
  intervals <- lapply(
    levels,
     FUN = function(level) {
       ci_out <- matrix(
         data = NA_real_,
         nrow = length(parameters),
         ncol = 2,
       )
       
      rownames(ci_out) <- boot_params
      
      ci <- tryCatch(
        suppressMessages(
          suppressWarnings(confint(boot, level = level, type = "perc"))
        ), error = function(e) NULL
      )
      
      # If CIs failed to be computed
      if (is.null(ci)) {
        return(ci_out)
      }
      
      available <- intersect(boot_params, rownames(ci))
      ci_out[available,] <- ci[available,]
      
      ci_out
    }
  )
  
  concat_results(intervals, estimates)
}
