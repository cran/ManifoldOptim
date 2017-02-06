#include "RProblem.h"

// This constructor uses numerical differentiation rather than a given m_gradFun
// or _hessEtaFun. For example, if the user attempts to call m_gradFun, it will call
// out to default.grad in R.
// This function informs the R user that they can't access gradFun from there.
// Is this better than asking for the user to set up numDeriv themselves?
RProblem::RProblem(const Rcpp::Function& objFun)
: VectorManifoldOptimProblem(true, true), m_objFun(objFun), 
  m_gradFun(Environment::namespace_env("ManifoldOptim")["default.grad"]),
  m_hessEtaFun(Environment::namespace_env("ManifoldOptim")["default.hessEta"])
{
}

RProblem::RProblem(const Rcpp::Function& objFun, const Rcpp::Function& gradFun)
: VectorManifoldOptimProblem(false, true), m_objFun(objFun), m_gradFun(gradFun),
  m_hessEtaFun(Environment::namespace_env("ManifoldOptim")["default.hessEta"])
{
}

RProblem::RProblem(const Rcpp::Function& objFun, const Rcpp::Function& gradFun,
	const Rcpp::Function& hessEtaFun)
: VectorManifoldOptimProblem(false, false), m_objFun(objFun), m_gradFun(gradFun),
  m_hessEtaFun(hessEtaFun)
{
}

double RProblem::objFun(const arma::vec &X) const
{
	return Rcpp::as<double>(m_objFun(X));
}

arma::mat RProblem::gradFun(const arma::vec &X) const
{
	return Rcpp::as<arma::mat>(m_gradFun(X));
}

arma::vec RProblem::hessEtaFun(const arma::vec &X, const arma::vec &eta) const
{
	return Rcpp::as<arma::vec>(m_hessEtaFun(X, eta));
}
