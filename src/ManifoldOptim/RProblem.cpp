#include "RProblem.h"

// This constructor uses numerical differentiation rather than a given _gradFun
// or _hessEtaFun. For example, if the user attempts to call _gradFun, it will call
// out to default.grad in R.
// This function informs the R user that they can't access gradFun from there.
// Is this better than asking for the user to set up numDeriv themselves?
RProblem::RProblem(const Rcpp::Function& objFun)
: VectorManifoldOptimProblem(true, true), _objFun(objFun), 
  _gradFun(Environment::namespace_env("ManifoldOptim")["default.grad"]),
  _hessEtaFun(Environment::namespace_env("ManifoldOptim")["default.hessEta"])
{
}

RProblem::RProblem(const Rcpp::Function& objFun, const Rcpp::Function& gradFun)
: VectorManifoldOptimProblem(false, true), _objFun(objFun), _gradFun(gradFun),
  _hessEtaFun(Environment::namespace_env("ManifoldOptim")["default.hessEta"])
{
}

RProblem::RProblem(const Rcpp::Function& objFun, const Rcpp::Function& gradFun,
	const Rcpp::Function& hessEtaFun)
: VectorManifoldOptimProblem(false, false), _objFun(objFun), _gradFun(gradFun),
	_hessEtaFun(hessEtaFun)
{
}

double RProblem::objFun(const arma::vec &X) const
{
	return Rcpp::as<double>(_objFun(X));
}

arma::mat RProblem::gradFun(const arma::vec &X) const
{
	return Rcpp::as<arma::mat>(_gradFun(X));
}

arma::vec RProblem::hessEtaFun(const arma::vec &X, const arma::vec &eta) const
{
	return Rcpp::as<arma::vec>(_hessEtaFun(X, eta));
}
