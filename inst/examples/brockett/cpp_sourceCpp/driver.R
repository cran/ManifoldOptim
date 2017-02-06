library(ManifoldOptim)

set.seed(1234)

p <- 5
n <- 150
B <- matrix(rnorm(n*n), nrow=n)
B <- B + t(B) # force symmetric
D <- diag(p:1, p)

tx <- function(x) { matrix(x, n, p) }

# The Problem class is written in C++. Get a handle to it and set it up from R
Rcpp::sourceCpp(code = '
//[[Rcpp::depends(RcppArmadillo,ManifoldOptim)]]
#include <RcppArmadillo.h>
#include <ManifoldOptim.h>

using namespace Rcpp;
using namespace arma;

class BrockettProblem : public MatrixManifoldOptimProblem
{
public:
	BrockettProblem(const arma::mat& B, const arma::mat& D)
	: MatrixManifoldOptimProblem(false, false), m_B(B), m_D(D)
	{
	}

	virtual ~BrockettProblem() { }

	double objFun(const arma::mat& X) const
	{
		return arma::trace(X.t() * m_B * X * m_D);
	}

	arma::mat gradFun(const arma::mat& X) const
	{
		return 2 * m_B * X * m_D;
	}

	arma::vec hessEtaFun(const arma::mat& X, const arma::vec& eta) const
	{
		return 2 * arma::kron(_D, m_B) * eta;
	}

	const arma::mat& GetB() const
	{
		return m_B;
	}

	const arma::mat& GetD() const
	{
		return m_D;
	}

private:
	arma::mat m_B;
	arma::mat m_D;
};

RCPP_MODULE(Brockett_module) {
	class_<BrockettProblem>("BrockettProblem")
	.constructor<mat,mat>()
	.method("objFun", &BrockettProblem::objFun)
	.method("gradFun", &BrockettProblem::gradFun)
	.method("hessEtaFun", &BrockettProblem::hessEtaFun)
	.method("GetB", &BrockettProblem::GetB)
	.method("GetD", &BrockettProblem::GetD)
	;
}
')

prob <- new(BrockettProblem, B, D)

X0 <- orthonorm(matrix(rnorm(n*p), nrow=n, ncol=p))
x0 <- as.numeric(X0)
prob$objFun(X0)			# Test the obj fn
head(prob$gradFun(X0))	# Test the grad fn

# ----- Run manifold.optim -----
mani.params <- get.manifold.params(IsCheckParams = TRUE)
solver.params <- get.solver.params(DEBUG = 0, Tolerance = 1e-4,
	Max_Iteration = 1000, IsCheckParams = TRUE, IsCheckGradHess = FALSE)
mani.defn <- get.stiefel.defn(n, p)

res <- manifold.optim(prob, mani.defn, method = "RTRSR1",
	mani.params = mani.params, solver.params = solver.params, x0 = x0)
print(res)
head(tx(res$xopt))
