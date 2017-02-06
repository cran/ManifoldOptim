//[[Rcpp::depends(RcppArmadillo,ManifoldOptim)]]
#include <RcppArmadillo.h>
#include <ManifoldOptim.h>

using namespace Rcpp;
using namespace arma;

class BrockettProblem : public MatrixManifoldOptimProblem
{
public:
	BrockettProblem(const arma::mat& B, const arma::mat& D)
	: m_B(B), m_D(D), MatrixManifoldOptimProblem(false)
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
	.method("GetB", &BrockettProblem::GetB)
	.method("GetD", &BrockettProblem::GetD)
	;
}

