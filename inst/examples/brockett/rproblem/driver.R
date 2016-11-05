library(ManifoldOptim)
source("../../test_util.R")

set.seed(1234)

p <- 5
n <- 150
B <- matrix(rnorm(n*n), nrow=n)
B <- B + t(B) # force symmetric
D <- diag(p:1, p)

# Variables within the optimizer are treated as a single vector.
# tx is a helper function that transforms the optimization variable
# into something we can directly use.
tx <- function(x) {
    matrix(x, n, p)
}
f <- function(x) {
	X <- tx(x)
	Trace( t(X) %*% B %*% X %*% D )
}
g <- function(x) {
	X <- tx(x)
	2 * B %*% X %*% D
}

# Take the problem above (defined in R) and make a Problem object for it.
# The Problem can be passed to the optimizer in C++
mod <- Module("ManifoldOptim_module", PACKAGE = "ManifoldOptim")
prob <- new(mod$RProblem, f, g)

X0 <- orthonorm(matrix(rnorm(n*p), nrow=n, ncol=p))
x0 <- as.numeric(X0)
prob$objFun(x0)			# Test the obj fn
head(prob$gradFun(x0))	# Test the grad fn

mani.params <- get.manifold.params(IsCheckParams = TRUE)
solver.params <- get.solver.params(isconvex = TRUE, DEBUG = 0, Tolerance = 1e-4,
	Max_Iteration = 1000, IsCheckParams = TRUE, IsCheckGradHess = FALSE)
mani.defn <- get.stiefel.defn(n, p)

res <- manifold.optim(prob, mani.defn, method = "RTRSR1",
	mani.params = mani.params, solver.params = solver.params, x0 = x0)
print(res)
head(tx(res$xopt))
