library(ManifoldOptim)
library(mvtnorm)

set.seed(1234)

# Try to estimate jointly: Sigma in SPD manifold and mu in sphere manifold
n <- 400
p <- 3
mu.true <- rep(1/sqrt(p), p)
Sigma.true <- diag(2,p) + 0.1
y <- rmvnorm(n, mean = mu.true, sigma = Sigma.true)

tx <- function(x) {
	idx.mu <- 1:p
	idx.S <- 1:p^2 + p
	mu <- x[idx.mu]
	S <- matrix(x[idx.S], p, p)
	# S[lower.tri(S)] <- t(S)[lower.tri(S)]
	list(mu = mu, Sigma = S)
}
f <- function(x) {
	par <- tx(x)
	-sum(dmvnorm(y, mean = par$mu, sigma = par$Sigma, log = TRUE))
}

# Take the problem above (defined in R) and make a Problem object for it.
# The Problem can be passed to the optimizer in C++
mod <- Module("ManifoldOptim_module", PACKAGE = "ManifoldOptim")
prob <- new(mod$RProblem, f)

mu0 <- diag(1, p)[,1]
Sigma0 <- diag(1, p)
x0 <- c(mu0, as.numeric(Sigma0))

# Test the obj and grad fn
prob$objFun(x0)
# head(prob$gradFun(x0))

mani.defn <- get.product.defn(get.sphere.defn(p), get.spd.defn(p))
mani.params <- get.manifold.params(IsCheckParams = TRUE)
solver.params <- get.solver.params(isconvex = TRUE, DEBUG = 0, Tolerance = 1e-4,
	Max_Iteration = 1000, IsCheckParams = TRUE, IsCheckGradHess = FALSE)

res <- manifold.optim(prob, mani.defn, method = "LRBFGS", 
	mani.params = mani.params, solver.params = solver.params, x0 = x0)
print(res)
print(tx(res$xopt))

