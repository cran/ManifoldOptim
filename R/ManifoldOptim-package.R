#' @title
#' Problem definition
#'
#' @description
#' Define a problem for ManifoldOptim to solve.
#'
#' @details
#' A problem definition contains an objective function \eqn{f} and a gradient
#' function \eqn{g}. The gradient \eqn{g} is computed as if \eqn{f} is defined
#' on a Euclidean space. If \eqn{g} is not specified it will be computed
#' numerically, which is potentially much slower.
#'
#' The easiest way to define a problem is completely in \code{R}. Example 1
#' below illustrates how to construct a problem using a given \eqn{f} and
#' \eqn{g}. Example 2 constructs the same problem without providing \eqn{g}.
#' The \code{Rcpp Module} framework (Eddelbuettel, 2013) creates underlying
#' \code{C++} objects necessary to invoke the \code{ROPTLIB} library.
#'
#' The performance of solving an \code{RProblem} may be too slow for some
#' applications; here, the \code{C++} optimizer calls \code{R} functions,
#' which requires some overhead. A faster alternative is to code your problem
#' in \code{C++} directly, and allow it to be manipulated in \code{R}. An
#' example is provided in this package, under
#' \code{tests/brockett/cpp_standalone/}. Example 3 below shows how to
#' instantiate this problem.
#'
#' Package authors may want to use \code{ManifoldOptim} within a package to solve
#' a problem written in \code{C++}. In this case, the author would probably
#' not want to use \code{sourceCpp}, but instead have the problem compiled
#' when the package was installed. An example is provided within this package;
#' \code{tests/brockett/cpp_pkg/driver.R} instantiates the problem defined in:
#'
#' \code{src/ManifoldOptim/BrockettProblem.cpp}.
#'
#' @examples
#' \dontrun{
#' # --- Example 1: Define a problem in R ---
#' f <- function(x) { ... }
#' g <- function(x) { ... }
#' mod <- Module("ManifoldOptim_module", PACKAGE = "ManifoldOptim")
#' prob <- new(mod$RProblem, f, g)
#'
#' # --- Example 2: Define a problem in R without specifying gradient ---
#' f <- function(x) { ... }
#' mod <- Module("ManifoldOptim_module", PACKAGE = "ManifoldOptim")
#' prob <- new(mod$RProblem, f)
#'
#' # --- Example 3: Instantiate a problem written in C++ ---
#' p <- 5; n <- 150
#' B <- matrix(rnorm(n*n), nrow=n)
#' B <- B + t(B) # force symmetric
#' D <- diag(p:1, p)
#' Rcpp::sourceCpp("brockett_problem.cpp")
#' prob <- new(BrockettProblem, B, D)
#' }
#'
#' @name Problem definition
#'
#' @references
#' Dirk Eddelbuettel. Seamless R and C++ Integration with Rcpp,
#'   Chapter 7: Modules, pages 83-102. Springer New York, New York, NY, 2013.
#'
#' Wen Huang, P.A. Absil, K.A. Gallivan, Paul Hand (2016a). "ROPTLIB: an
#' object-oriented C++ library for optimization on Riemannian manifolds."
#' Technical Report FSU16-14, Florida State University.
NULL
