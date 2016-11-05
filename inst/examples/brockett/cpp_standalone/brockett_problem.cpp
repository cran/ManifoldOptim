//[[Rcpp::depends(RcppArmadillo,ManifoldOptim)]]
#include <RcppArmadillo.h>
#include <ManifoldOptim.h>

using namespace Rcpp;
using namespace arma;

class BrockettProblem : public MatrixManifoldOptimProblem
{
public:
	BrockettProblem(const arma::mat& B, const arma::mat& D)
	: _B(B), _D(D), MatrixManifoldOptimProblem(false)
	{
	}

	virtual ~BrockettProblem() { }

	double objFun(const arma::mat& X) const
	{
		return arma::trace(X.t() * _B * X * _D);
	}

	arma::mat gradFun(const arma::mat& X) const
	{
		return 2 * _B * X * _D;
	}

	const arma::mat& GetB() const
	{
		return _B;
	}

	const arma::mat& GetD() const
	{
		return _D;
	}

private:
	arma::mat _B;
	arma::mat _D;
};

RCPP_MODULE(Brockett_module) {
	class_<BrockettProblem>("BrockettProblem")
	.constructor<mat,mat>()
	.method("objFun", &BrockettProblem::objFun)
	.method("gradFun", &BrockettProblem::gradFun)
	.method("GetB", &BrockettProblem::GetB)
	.method("GetD", &BrockettProblem::GetD)
	//.method("EucHessianEta", &BrockettProblem::EucHessianEta)
	;
}

