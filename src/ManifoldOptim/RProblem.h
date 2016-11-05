#ifndef R_PROBLEM_H
#define R_PROBLEM_H

#include <RcppArmadillo.h>
#include "VectorManifoldOptimProblem.h"
#include <cstring>
#include <def.h>
#include <assert.h>
#include <iostream>

using namespace Rcpp;

class RProblem : public VectorManifoldOptimProblem
{

public:
	RProblem(const Rcpp::Function& objFun);
	RProblem(const Rcpp::Function& objFun, const Rcpp::Function& gradFun);
	RProblem(const Rcpp::Function& objFun, const Rcpp::Function& gradFun,
		const Rcpp::Function& hessEtaFun);

	double objFun(const arma::vec& X) const;
	arma::mat gradFun(const arma::vec& X) const;
	arma::vec hessEtaFun(const arma::vec& X, const arma::vec& eta) const;

	Rcpp::Function _objFun;
	Rcpp::Function _gradFun;
	Rcpp::Function _hessEtaFun;
};

#endif

