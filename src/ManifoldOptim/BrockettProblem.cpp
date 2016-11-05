#include <RcppArmadillo.h>
#include "MatrixManifoldOptimProblem.h"
#include "ManifoldOptimException.h"

using namespace Rcpp;
using namespace arma;

class BrockettProblem : public MatrixManifoldOptimProblem
{
public:
	BrockettProblem(const arma::mat& B, const arma::mat& D)
	: MatrixManifoldOptimProblem(false, true), _B(B), _D(D)
	{
	}

	virtual ~BrockettProblem() { }

	virtual double objFun(const arma::mat& X) const
	{
		return arma::trace(X.t() * _B * X * _D);
	}

	virtual arma::mat gradFun(const arma::mat& X) const
	{
		return 2 * _B * X * _D;
	}

	virtual arma::vec hessEtaFun(const arma::mat& X, const arma::vec& eta) const
	{
		throw ManifoldOptimException("This function is not implemented");
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
