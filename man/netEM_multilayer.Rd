% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{netEM_multilayer}
\alias{netEM_multilayer}
\title{Returns the EM object for multilayer network}
\usage{
netEM_multilayer(
  y,
  K,
  p,
  eps,
  num_rand_start,
  num_run_smallEM,
  max_itr_smallEM,
  burn,
  MCMC_itr,
  sigma_mult,
  psi_mult,
  n,
  alpha
)
}
\arguments{
\item{y}{multiple network}

\item{K}{number of clusters}

\item{p}{number of layers}

\item{eps}{epsilon for convergence}

\item{num_rand_start}{number of random starts}

\item{num_run_smallEM}{number of runs for small EM}

\item{max_itr_smallEM}{maximum number of runs for small EM}

\item{burn}{number of runs for burn for Metropolis Hastings}

\item{MCMC_itr}{number of runs for Metropolis Hastings iterations}

\item{sigma_mult}{scaling multiplier for Sigma matrix}

\item{psi_mult}{scaling multiplier for Psi matrix}

\item{n}{number of nodes of the network}

\item{alpha}{seed provided by the user}
}
\value{
EM object
}
\description{
Returns the EM object for multilayer network
}
