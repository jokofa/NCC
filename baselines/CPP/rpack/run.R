#!/usr/bin/env Rscript
suppressMessages(library(rpack))
suppressMessages(library(tools))
suppressMessages(library(optparse))
suppressMessages(library(assertthat))
suppressMessages(library(tidyverse))

# command line flags
option_list <- list(
  make_option(c("-f", "--file"), type="character",
              help="file path to file to load", metavar="character"),
  make_option("--path", type="character", default="outputs/rpack/",
              help="output file directory [default= %default]", metavar="character"),
  make_option(c("-l", "--num_init"), type="integer", default=10,
              help="Number of restarts [default= %default]", metavar="integer"),
  make_option(c("-k", "--nc"), type="integer", default=NULL,
              help="Number of clusters [default= %default]"),
  make_option("--plot", action="store_true", default=FALSE,
              help="flag if should plot example of loaded data [default= %default]"),
  make_option("--plot_solved",action="store_true", default=FALSE,
              help="flag if should plot example of solved instance [default= %default]"),
  make_option(c("-t", "--time"),action="store_true", default=FALSE,
              help="flag if should time execution per instance [default= %default]"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="flag if should print additional info [default= %default]"),
  make_option(c("-s", "--size"), type="integer", default=NULL,
              help="Number of instances to solve [default= %default]"),
  make_option("--offset", type="integer", default=0,
              help="Number of instances to skip [default= %default]"),
  make_option("--seed", type="integer", default=123,
              help="rng seed [default= %default]"),
  make_option(c("-c", "--cores"), type="integer", default=1,
              help="number of cores/threads, 0 for 'all' [default= %default]"),
  make_option("--timeout", type="integer", default=180,
              help="number of seconds until total timeout [default= %default]"),
  make_option("--gurobi_timeout", type="integer", default=60,
              help="number of seconds until timeout for gurobi [default= %default]")
)

# parse
args <- parse_args(OptionParser(option_list=option_list))
print("Running with arguments:")
print(paste(unlist(args), collapse=", "))

set.seed(args$seed)

if(args$plot | args$plot_solved){
  plt_pth <- "rpack/Rplots.pdf"
  pdf(plt_pth)
}
if(args$verbose){
  print_output <- "steps" #"progress"
} else {
  print_output <- "NULL"
}


# https://github.com/terolahderanta/rpack/blob/master/R/capacitated_LA.R
capacitated_LA <- function(coords,
                           weights,
                           k,
                           ranges,
                           params = NULL,
                           capacity_weights = weights,
                           d = euc_dist2,
                           center_init = NULL,
                           fixed_centers = NULL ,
                           lambda = NULL,
                           place_to_point = TRUE,
                           frac_memb = FALSE,
                           gurobi_params = NULL,
                           dist_mat = NULL,
                           multip_centers = rep(1, nrow(coords)),
                           print_output = NULL,
                           lambda_fixed = NULL,
                           timeout = NULL,  ###
                           t_start = NULL   ###
){

  cat(paste("provided timeout: ", timeout, " and t_start: ", t_start, "\n"))

  # Number of objects in data
  n <- nrow(coords)

  # Number of fixed centers
  n_fixed <- ifelse(is.null(fixed_centers), 0, nrow(fixed_centers))

  # Cluster centers with fixed centers first
  if(n_fixed > 0){

    if(place_to_point){

      fixed_center_ids <- prodlim::row.match(fixed_centers %>% as.data.frame(), coords %>% as.data.frame())
      #fixed_center_ids <- which(
      #  ((coords %>% pull(1)) %in% (fixed_centers %>% pull(1))) &
      #    ((coords %>% pull(2)) %in% (fixed_centers %>% pull(2)))
      #)

    }
  } else {
    fixed_center_ids <- NULL
  }

  # Initialize centers
  switch(center_init,
         # Pick initial centers randomly
         random = {

           if(place_to_point){
             # Don't sample the points in fixed_centers
             sample_points <- which(!(1:n %in% fixed_center_ids))
             center_ids <- c(fixed_center_ids, sample(sample_points, k - n_fixed))
             centers <- coords[center_ids,]
           } else {
             center_ids <- NULL
             centers <- coords[sample(1:n, k),]
           }
         },

         # Pick initial centers with k-means++
         kmpp = {

           # Replicate data points according to their weight
           weighted_coords <- apply(coords, 2, function(x) rep(x, round(weights)))

           # Run k-means++
           init_kmpp <- kmpp(weighted_coords, k)

           if(place_to_point){

             # Select closest points to the kmpp-result as the centers
             center_ids <- apply(X = init_kmpp$centers, MARGIN = 1, FUN = function(x){
               temp_dist_kmpp <-
                 apply(X = coords, MARGIN = 1, FUN = function(y){d(x,y)})
               which.min(temp_dist_kmpp)
             })

             # Do not choose the same point twice
             ifelse(duplicated(center_ids),
                    sample((1:n)[-center_ids],size = 1),
                    center_ids)

             # Select the centers according to ids
             centers <- coords[center_ids,]

           } else {
             centers <- init_kmpp$centers
             center_ids <- NULL
           }
         },

         stop("No such choice for center initialization! (rpack)")
  )

  # Maximum number of laps
  max_laps <- 100

  for (iter in 1:max_laps) {
    # Old mu is saved to check for convergence
    old_centers <- centers

    # Print detailed steps
    if(print_output == "steps"){
      cat("A-step... ")
      temp_time <- Sys.time()
    }

    # Clusters in equally weighted data (Allocation-step)
    valid <- TRUE
    temp_alloc <- tryCatch(
        allocation_step(
        coords = coords,
        weights = weights,
        params = params,
        k = k,
        centers = centers,
        ranges = ranges,
        center_ids = center_ids,
        capacity_weights = capacity_weights,
        lambda = lambda,
        d = d,
        frac_memb = frac_memb,
        gurobi_params = gurobi_params,
        dist_mat = dist_mat,
        multip_centers = multip_centers
      ),
        error = function(err) {
          valid <- FALSE
        }
    )
    # if timed out during allocation, do not do another location step
    if (
      (!valid) |
        ((!is.null(timeout)) & (!is.null(t_start)) &
        (difftime(Sys.time(), t_start, units = "secs")[[1]] >= timeout-3))
    ){
      cat("LA timeout! \n")
      break
    }

    # Print detailed steps
    if(print_output == "steps"){
      cat(paste("Done! (", format(round(Sys.time() - temp_time, digits = 3), nsmall = 3), ")\n", sep = ""))
    }

    # Save the value of the objective function
    obj_min <- temp_alloc$obj_min

    # Print detailed steps
    if(print_output == "steps"){
      cat("L-step... ")
      temp_time <- Sys.time()
    }

    # Updating cluster centers (Parameter-step)
    temp_loc <- location_step(
      coords = coords,
      weights = weights,
      k = k,
      params = params,
      assign_frac = temp_alloc$assign_frac,
      fixed_centers = fixed_centers,
      d = d,
      place_to_point = place_to_point,
      dist_mat = dist_mat,
      lambda_fixed = lambda_fixed
    )

    # Print detailed steps
    if(print_output == "steps"){
      cat(paste("Done! (", format(round(Sys.time() - temp_time, digits = 3), nsmall = 3), ")\n", sep = ""))
    }

    centers <- temp_loc$centers

    center_ids <- temp_loc$center_ids

    # Print a message showing that max number of iterations was reached
    if(iter == max_laps & print_output == "steps"){
      cat(paste("WARNING! Reached maximum number of LA-iterations! Returning the clustering from last lap...\n",sep = ""))
    }

    # If nothing is changing, stop
    if(all(old_centers == centers)) break

  }

  # Save the assignments
  assign_frac <- temp_alloc$assign_frac

  # Hard clusters from assign_frac
  clusters <- apply(assign_frac, 1, which.max)

  # Cluster 99 is the outgroup
  clusters <- ifelse(clusters == (k+1), 99, clusters)

  # Return cluster allocation, cluster center and the current value of the objective function
  return(list(clusters = clusters, centers = centers, obj = obj_min, assign_frac = assign_frac))
}

#' Update cluster allocations by minimizing the objective function.
#'
#' @param coords A matrix or data.frame containing the coordinates.
#' @param weights A vector of weights for each data point.
#' @param k The number of clusters.
#' @param centers The parameters (locations) that define the k distributions.
#' @param ranges Lower and upper limits for the clustering
#' @param center_ids Ids for the data points that are selected as centers.
#' @param capacity_weights Different weights for capacity limits.
#' @param lambda Outlier-parameter
#' @param d Distance function.
#' @param frac_memb If TRUE memberships are fractional.
#' @param gurobi_params A list of parameters for gurobi function e.g. time limit, number of threads.
#' @param dist_mat Distance matrix for all the points.
#' @param multip_centers Vector (n-length) defining how many centers a point is allocated to.
#' @return New cluster allocations for each object in data and the maximum of the objective function.
#' @keywords internal
allocation_step <- function(coords,
                            weights,
                            k,
                            centers,
                            ranges,
                            params = NULL,
                            center_ids = NULL,
                            capacity_weights = weights,
                            lambda = NULL,
                            d = euc_dist2,
                            frac_memb = FALSE,
                            gurobi_params = NULL,
                            dist_mat = NULL,
                            multip_centers = rep(1, nrow(coords))) {

  # Number of objects in data
  n <- nrow(coords)

  # Is there an outgroup cluster
  is_outgroup <- !is.null(lambda)

  # Number of range groups
  if(is.vector(ranges)){
    ranges <- matrix(data = ranges, nrow = 1, ncol = 2)
    g <- 1
  } else {
    g <- nrow(ranges)
  }

  # Number of decision variables
  n_decision <- n * k +
    # More than one range groups
    ifelse(g > 1, k * g, 0) +
    # Outliers
    ifelse(is_outgroup, n, 0)

  # Calculate the distances to centers (matrix C)
  if(is.null(dist_mat) | length(center_ids) == 0){


    C <- matrix(0, ncol = k, nrow = n)
    if (is.null(params)) {
      for (i in 1:k) {
        C[, i] <- apply(coords,
                        MARGIN = 1,
                        FUN = d,
                        x2 = centers[i, ])
      }
    } else {

      for (i in 1:k) {
        centers_i <- centers[i, ]
        params_i <- params[which((centers_i %>% pull(1)) == (coords %>% pull(1)) &
                                   (centers_i %>% pull(2)) == (coords %>% pull(2))),]

        C[, i] <- apply(cbind(coords, params),
                        MARGIN = 1,
                        FUN = d,
                        x2 = c(centers_i, params_i))
      }
    }
  } else {
    # Read distances from distance matrix
    C <- dist_mat[,center_ids]
  }

  # Use weighted distances
  C <- C * weights

  # Gurobi model
  model <- list()

  if(g == 1){

    # Constraints for the upper and lower limit
    const_LU <- rbind(
      Matrix::spMatrix(
        nrow = k,
        ncol = n_decision,
        i = rep(1:k, each = n),
        j = 1:(n * k),
        x = rep(capacity_weights, k)
      ),
      Matrix::spMatrix(
        nrow = k,
        ncol = n_decision,
        i = rep(1:k, each = n),
        j = 1:(n * k),
        x = rep(capacity_weights, k)
      )
    )

    # Add the constraints to the model
    model$A <- rbind(Matrix::spMatrix(
      nrow = n,
      ncol = n_decision,
      i = rep(1:n, times = ifelse(is_outgroup, k + 1, k)),
      j = rep(1:n_decision),
      x = rep(1, n_decision)
    ),
    const_LU)

    # Right hand side values
    model$rhs <- c(multip_centers,
                   rep(ranges[1, 2], k),
                   rep(ranges[1, 1], k))

    # Model sense
    model$sense      <- c(rep('=', n),
                          rep('<', k),
                          rep('>', k))


  } else {
    # Large number
    M <- 1000

    # In constraint matrices: first rangegroups, then clusters

    # Constraints for the first lower and upper limit
    const_LU <- rbind(
      Matrix::spMatrix(
        nrow = k,
        ncol = n_decision,
        i = c(rep(1:k, each = n), 1:k),
        j = c(1:(n * k), (n * k) + 1:k),
        x = c(rep(capacity_weights, k), rep(M, k))
      ),
      Matrix::spMatrix(
        nrow = k,
        ncol = n_decision,
        i = c(rep(1:k, each = n), 1:k),
        j = c(1:(n * k), (n * k) + 1:k),
        x = c(rep(capacity_weights, k), rep(-M, k))
      )
    )

    # Right hand side values for the first capacity constraints
    rhs_LU <- c(rep(ranges[1, 2] + M, k),
                rep(ranges[1, 1] - M, k))

    # Model sense for the first capacity constraints
    sense_LU     <- c(rep('<', k),
                      rep('>', k))

    # Constraints, rhs and sense for the rest of the lower and upper limits
    for(i in 2:g){
      const_LU <- rbind(
        const_LU,
        Matrix::spMatrix(
          nrow = k,
          ncol = n_decision,
          i = c(rep(1:k, each = n), 1:k),
          j = c(1:(n * k), (n * k + k * (i - 1)) + 1:k),
          x = c(rep(capacity_weights, k), rep(M, k))
        ),
        Matrix::spMatrix(
          nrow = k,
          ncol = n_decision,
          i = c(rep(1:k, each = n), 1:k),
          j = c(1:(n * k), (n * k + k * (i - 1)) + 1:k),
          x = c(rep(capacity_weights, k), rep(-M, k))
        )
      )

      rhs_LU <- c(rhs_LU,
                  rep(ranges[i, 2] + M, k),
                  rep(ranges[i, 1] - M, k))

      sense_LU <- c(sense_LU,
                    rep('<', k),
                    rep('>', k))

    }

    # Constraints for the cluster size group
    const_group <- Matrix::spMatrix(
      nrow = k,
      ncol = n_decision,
      i = rep(1:k, each = g),
      j = c((n * k) + 1:(k * g)),
      x = rep(1, k * g)
    )

    # Add all constraints to the model
    model$A <- rbind(
      Matrix::spMatrix(
        nrow = n,
        ncol = n_decision,
        i = rep(1:n, times = ifelse(is_outgroup, k + 1, k)),
        j = rep(1:(n * k)),
        x = rep(1, n * k)
      ),
      const_LU,
      const_group
    )

    # Right hand side values (multiple membership, upper and lower limits, cluster groups)
    model$rhs <- c(multip_centers,
                   rhs_LU,
                   rep(1, k))

    # Model sense
    model$sense <- c(rep('=', n),
                     sense_LU,
                     rep('=', k))
  }


  # Objective function
  obj_fn <- c(c(C),
              switch(g > 1, rep(0, k * g), NULL),
              switch(is_outgroup, lambda * weights, NULL))

  model$obj <- obj_fn

  # Minimization task
  model$modelsense <- 'min'

  # B = Binary, C = Continuous
  model$vtype <- ifelse(frac_memb, 'C', 'B')

  # Using timelimit-parameter to stop the optimization if time exceeds 10 minutes
  # and disabling the print output from gurobi.
  if(is.null(gurobi_params)){
    gurobi_params <- list()
    gurobi_params$TimeLimit <- 600
    gurobi_params$OutputFlag <- 0
  }

  # Solving the linear program
  result <- gurobi::gurobi(model, params = gurobi_params)

  # Send error message if the model was infeasible
  if(result$status == "INFEASIBLE") {stop("Model was infeasible! (rpack)")}

  # Returns the assignments
  assign_frac <- Matrix::Matrix(matrix((result$x)[1:(ifelse(is_outgroup, n * k + n, n * k))],
                                       ncol = ifelse(is_outgroup, k + 1, k)), sparse = TRUE)

  # Returns the value of the objective function
  obj_min <- round(result$objval, digits = 5)

  # Clear space
  rm(model, result)

  return(list(assign_frac = assign_frac,
              obj_min = obj_min))
}


#' Updates the parameters (centers) for each cluster.
#'
#' @param coords A matrix or data.frame containing the coordinates.
#' @param weights The weights of the data points.
#' @param k The number of clusters.
#' @param assign_frac A vector of cluster assignments for each data point.
#' @param fixed_centers Predetermined center locations.
#' @param d The distance function.
#' @param place_to_point if TRUE, cluster centers will be placed to a point.
#' @param dist_mat Distance matrix for all the points.
#' @return New cluster centers.
#' @keywords internal
location_step <- function(coords,
                          weights,
                          k,
                          assign_frac,
                          params = NULL,
                          fixed_centers = NULL,
                          d = euc_dist2,
                          place_to_point = TRUE,
                          dist_mat = NULL,
                          lambda_fixed = NULL) {

  # Number of fixed centers
  n_fixed <- ifelse(is.null(fixed_centers), 0, nrow(fixed_centers))

  # Initialization of cluster center matrix
  centers <- matrix(0, nrow = k, ncol = ncol(coords))

  # Add fixed centers first
  if (n_fixed > 0) {
    centers[1:n_fixed, ] <- fixed_centers
  }

  if (place_to_point) {
    # Initialization of cluster id vector
    center_ids <- rep(0, k)

    # Add fixed centers first
    if (n_fixed > 0) {

      center_ids <- prodlim::row.match(fixed_centers %>% as.data.frame(), coords %>% as.data.frame())
      #center_ids <- c(which(((coords %>% pull(1)) %in% (fixed_centers %>% pull(1))
      #) &
      #  ((coords %>% pull(2)) %in% (fixed_centers %>% pull(2))
      #  )), rep(0, k - n_fixed))
    }
  }

  # Update center for each cluster
  if (place_to_point) {
    # If fixed servers are allowed to be released
    if (!is.null(lambda_fixed)) {
      # Potential new centers
      potential_center_ids <- rep(0, n_fixed)

      # Ids for the fixed servers
      fixed_center_ids <- which(((coords %>% pull(1)) %in% (fixed_centers %>% pull(1))) &
                                  ((coords %>% pull(2)) %in% (fixed_centers %>% pull(2))))

      # Most optimal center location for the fixed center clusters
      for (i in 1:n_fixed) {
        # Compute medoids only with points that are relevant in the cluster i
        relevant_cl <- assign_frac[, i] > 0.001

        relevant_ids <- which(relevant_cl)

        # Computing medoid ids for cluster i
        potential_center_ids[i] <-
          medoid_dist_mat(dist_mat = dist_mat,
                          ids = relevant_ids,
                          w = weights)

        # Combined distance to the potential center
        wdist_pot_center <- sum(dist_mat[relevant_ids, potential_center_ids[i]] *
                                  weights[relevant_ids])

        # Combined distance to the fixed center
        wdist_fixed_center <- sum(dist_mat[relevant_ids, fixed_center_ids[i]] *
                                    weights[relevant_ids])

        # TODO: add lambda_fixed to here
        if (wdist_fixed_center > wdist_pot_center + lambda_fixed) {
          center_ids[i] <- potential_center_ids[i]
        }

      }

      # Decide the rest of the center locations
      for (i in (n_fixed + 1):k) {
        # Compute medoids only with points that are relevant in the cluster i
        relevant_cl <- assign_frac[, i] > 0.001

        # Computing medoid ids for cluster i
        center_ids[i] <- medoid_dist_mat(dist_mat = dist_mat,
                                         ids = which(relevant_cl),
                                         w = weights)
      }

      # Decide centers from the ids
      centers <- coords[center_ids, ]

    } else {

      if(n_fixed < k){
        for (i in (n_fixed + 1):k) {
          # Compute medoids only with points that are relevant in the cluster i
          relevant_cl <- assign_frac[, i] > 0.001

          # Computing medoid ids for cluster i
          center_ids[i] <- medoid_dist_mat(dist_mat = dist_mat,
                                           ids = which(relevant_cl),
                                           w = weights)
        }
      }

      # Decide centers from the ids
      centers <- coords[center_ids, ]
    }
  } else {
    if(is.null(params)){

      for (i in (ifelse(n_fixed > 0, n_fixed + 1, 1)):k) {
        # Check whether euc_dist or euc_dist2 is used
        if (d(0, 2) == 2) {
          w_assign <- weights * assign_frac[, i]

          # If only one point belongs to the cluster, the Weiszfeld algorithm won't work
          if (sum(w_assign > 0) == 1) {
            centers[i,] <- coords %>%
              slice(which(w_assign > 0)) %>%
              unlist(., use.names = FALSE)

          } else {
            # Weighted median
            weiszfeld <-
              Gmedian::Weiszfeld(coords, weights = w_assign)$median

          }

        } else if (d(0, 2) == 4) {
          # Weighted mean
          centers[i,] <-
            colSums(coords * weights * assign_frac[, i]) / sum(assign_frac[, i] * weights)
        }
      }
    } else {
      # TODO: Tähän lokaatio parametreilla



    }
    center_ids <- NULL
  }


  return(list(centers = centers, center_ids = center_ids))
}

# https://github.com/terolahderanta/rpack/blob/master/R/alt_alg.R
# adapted with timeout
alt_alg2 <- function(coords,
                     weights,
                     k,
                     params = NULL,
                     N = 10,
                     range = c(min(weights)/2, sum(weights)),
                     capacity_weights = weights,
                     d = euc_dist2,
                     center_init = "random",
                     lambda = NULL,
                     frac_memb = FALSE,
                     place_to_point = TRUE,
                     fixed_centers = NULL,
                     gurobi_params = NULL,
                     multip_centers = rep(1, nrow(coords)),
                     dist_mat = NULL,
                     print_output = "progress",
                     normalization = TRUE,
                     lambda_fixed = NULL,
                     lambda_params = NULL,
                     timeout = NULL
){

  t_start <- Sys.time()

  # Check arguments
  assertthat::assert_that(is.matrix(coords) || is.data.frame(coords), msg = "coords must be a matrix or a data.frame!")

  assertthat::assert_that(nrow(coords) >= k, msg = "must have at least k coords points!")
  assertthat::assert_that(is.numeric(weights), msg = "weights must be an numeric vector!")
  assertthat::assert_that(is.numeric(capacity_weights), msg = "capacity weights must be an numeric vector!")
  assertthat::assert_that(length(weights) == nrow(coords), msg = "coords and weight must have the same number of rows!")
  assertthat::assert_that(length(capacity_weights) == nrow(coords), msg = "coords and capacity weights must have the same number of rows!")
  assertthat::assert_that(is.numeric(k), msg = "k must be a numeric scalar!")
  assertthat::assert_that(length(k) == 1, msg = "k must be a numeric scalar!")

  assertthat::assert_that(is.numeric(range))

  if(!purrr::is_null(lambda)) assertthat::is.number(lambda)
  if(!purrr::is_null(lambda_fixed)) assertthat::is.number(lambda_fixed)

  assertthat::assert_that(is.logical(normalization), msg = "normalization must be TRUE or FALSE!")
  assertthat::assert_that(is.logical(frac_memb), msg = "frac_memb must be TRUE or FALSE!")
  assertthat::assert_that(is.logical(place_to_point), msg = "place_to_point must be TRUE or FALSE!")

  # Calculate distance matrix
  if(is.null(dist_mat) & place_to_point){

    # Print information about the distance matrix
    n <- nrow(coords)
    cat(paste("Creating ", n, "x", n ," distance matrix... ", sep = ""))
    temp_mat_time <- Sys.time()

    if(is.null(params)) {
      # Calculate distances with distance metric d
      dist_mat <- apply(
        X = coords,
        MARGIN = 1,
        FUN = function(x) {
          apply(
            X = coords,
            MARGIN = 1,
            FUN = d,
            x2 = x
          )
        }
      )
    } else {
      # Calculate distances with distance metric d
      dist_mat <- apply(
        X = cbind(coords, params),
        MARGIN = 1,
        FUN = function(x) {
          apply(
            X = cbind(coords, params),
            MARGIN = 1,
            FUN = d,
            x2 = x,
            lambda = lambda_params
          )
        }
      )
    }

    cat(paste("Matrix created! (", format(round(Sys.time() - temp_mat_time)) ,")\n\n", sep = ""))

    # Normalizing distances
    if(normalization){
      dist_mat <- dist_mat / max(dist_mat)
    }

  } else if(place_to_point){

    # Normalizing distances
    if(normalization){
      dist_mat <- dist_mat / max(dist_mat)
    }

  } else {
    # If no distance matrix is used
    dist_mat <- NULL
  }

  if(normalization) {
    # Normalization for the capacity weights
    max_cap_w <- max(capacity_weights)
    capacity_weights <- capacity_weights / max_cap_w
    range <- range / max_cap_w

    # Normalization for the weights
    weights <- weights / max(weights)
  }

  # Print the information about run
  if(print_output == "progress"){
    cat(paste("Progress (N = ", N,"):\n", sep = ""))
    cat(paste("______________________________\n"))
    progress_bar <- 0
  }

  # Total iteration time
  temp_total_time <- Sys.time()

  for (i in 1:N) {
    ###
    if (
      (!is.null(timeout)) &
      (difftime(Sys.time(), t_start, units = "secs")[[1]] >= timeout-1)
    ){
      cat("alt_alg timeout!\n")
      break
    }

    if(print_output == "steps"){
      cat(paste("\nIteration ", i, "/", N, "\n---------------------------\n", sep = ""))
      temp_iter_time <- Sys.time()
    }

    # One clustering
    temp_clust <- capacitated_LA(coords = coords,
                                 weights = weights,
                                 k = k,
                                 params = params,
                                 ranges = range,
                                 capacity_weights = capacity_weights,
                                 lambda = lambda,
                                 d = d,
                                 dist_mat = dist_mat,
                                 center_init = center_init,
                                 place_to_point = place_to_point,
                                 frac_memb = frac_memb,
                                 fixed_centers = fixed_centers,
                                 gurobi_params = gurobi_params,
                                 multip_centers = multip_centers,
                                 print_output = print_output,
                                 lambda_fixed = lambda_fixed,
                                 timeout = timeout,   ###
                                 t_start = t_start)   ###

    # Save the first iteration as the best one
    if(i == 1){
      min_obj <- temp_clust$obj
      best_clust <- temp_clust
    }

    # Print the number of completed laps
    if(print_output == "progress") {
      if((floor((i / N) * 30) > progress_bar)) {
        cat(paste0(rep("#", floor((
          i / N
        ) * 30) - progress_bar), collapse = ""))
        progress_bar <- floor((i / N) * 30)
      }
    } else if(print_output == "steps"){
      cat(paste("Iteration time: ", format(round(Sys.time() - temp_iter_time)), "\n", sep = ""))
    }

    # Save the iteration with the lowest value of objective function
    if(temp_clust$obj < min_obj){
      min_obj <- temp_clust$obj
      best_clust <-  temp_clust
    }
  }

  cat("\n\n")

  # Print total iteration time
  cat(paste("Total iteration time: ", format(round(Sys.time() - temp_total_time)),"\n", sep = ""))


  return(best_clust)
}


plt <- function (dat){
  #id <-  1:nrow(dat)
  fig <- ggplot(data = dat, aes(x = x_coord, y = y_coord, size = weight, label = label)) +
      geom_point() +
      # Scale objects sizes
      scale_size(range = c(2, 6)) +
      # Point size in legend
      guides(
        color = guide_legend(
          override.aes = list(size=5)
        )
      ) +
      labs(x = "x", y = "y", title = "Unclustered data") +
      # Legend position and removing ticks from axis
      theme(
        legend.position = "right",
        axis.text.x = ggplot2::element_blank(),
        axis.text.y = ggplot2::element_blank(),
        axis.ticks = ggplot2::element_blank()
      )
  print(fig)
}


# load data
if(args$verbose){
print(paste0("loading file from: ", args$file))
}
Data <- read.csv(file = args$file, header = TRUE)
if(args$verbose){
print(Data[1:5, ])
}

num_rows <- nrow(Data)
SIZE <- length(unique(Data$batch_id))
if(args$verbose){
print(paste0("Data has ", num_rows, " rows of ", SIZE, " instances."))
}

# select number of instances if specified
if(is.integer(args$size)){
  n <- floor(num_rows / SIZE)
  SIZE <- args$size
  idx_select <- SIZE * n
  print(paste0("selecting only first ", SIZE, " instances." ))
  Data <- Data[1:idx_select, ]
  sstr <- paste0("ss", SIZE, "_")
} else {
  sstr <- ""
}


# split into list of tibbles by batch_id
df_list <- Data %>% group_split(batch_id)
if(args$verbose){
print(df_list[[1]])
}

if(args$plot){
    print(paste0("plotting data to: ", plt_pth))
    plt(df_list[[1]])
  }

print("Start solving...")
time_table <- vector("list", SIZE)
NewData <- vector("list", SIZE)
for(i in 1:SIZE){
  start_time <- Sys.time()

  dat <- df_list[[i]]
  if(is.integer(args$nc)){
    k <- args$nc
  } else {
    k <- length(unique(dat$label))
  }

  # Lower und upper limit for cluster weight
  # [0, 1] since weights are normalized w.r.t. max capacity
  L <- 0
  U <- 1

  # TimeLimit of 600 is rpack default
  gurobi_params <- list()
  gurobi_params$TimeLimit <- max(min(as.integer(args$gurobi_timeout), 600), 5)
  gurobi_params$OutputFlag <- 0
  gurobi_params$Threads <- args$cores

  # run alternaring algorithm to compute cluster assignment
  clusters <- alt_alg2(
    coords = dat %>% select(x_coord, y_coord),
    k = k,
    N = args$num_init,
    weights = dat %>% pull(weight),
    range = c(L, U),
    place_to_point = FALSE,  # Clusters heads can be located anywhere
    print_output = print_output,
    gurobi_params=gurobi_params,
    timeout=args$timeout
  )

  centers <- clusters$centers
  cluster_labels <- clusters$clusters

  time_ellapsed <- difftime(Sys.time(), start_time, units = "secs")[[1]]
  time_table[[i]] <- time_ellapsed

  if(args$plot_solved & i == 1){
    print(paste0("plotting clusters to: ", plt_pth))
    plot_cl <- plot_clusters(
      coords = dat %>% select(x_coord, y_coord),
      weights = dat %>% pull(weight),
      clusters = cluster_labels,
      centers = centers,
      title = paste0(
        "Capacitated clustering, k = ",
        k,
        ", squread Euclidean distance"
      ),
      subtitle = paste0("Uniform prior in [", L, ", ", U, "] on cluster sizes")
    )
    print(plot_cl)
  }

  # add cluster labels to data
  dat$rpack_label <- cluster_labels - 1

  NewData[[i]] <- dat
}

if(args$verbose){
print(NewData[[1]])
}
if(args$plot | args$plot_solved){
  dev.off()
}

SaveData <- bind_rows(NewData)
if(args$verbose){
print(SaveData[1:5, ])
}
# file and path naming
# fname <- basename(file_path_sans_ext(args$file))
# fname <- paste0(fname, sstr, "l", args$num_inits, "_rpack.csv")
fname <- "rpack_results.csv"
pth <- file.path(args$path, fname)
print(paste0("saving generated data as: ", pth))
write.csv(
  SaveData,
  file = pth,
  row.names = FALSE
)

if(args$time){
  time_table <- cbind(time_table)

  # fname <- basename(file_path_sans_ext(pth))
  # fname <- paste0(fname, "_timing.csv")
  fname <- "rpack_runtime.csv"
  pth <- file.path(args$path, fname)
  print(paste0("saving execution times as: ", pth))
  write.csv(
    time_table,
    file = pth,
    row.names = FALSE
  )
}
print("rpack finished.")
