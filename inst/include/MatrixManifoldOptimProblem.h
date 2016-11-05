#ifndef MATRIX_MANIFOLD_OPTIM_PROBLEM_H
#define MATRIX_MANIFOLD_OPTIM_PROBLEM_H

#include <RcppArmadillo.h>
#include <cstring>
#include <assert.h>
#include <iostream>
#include "ManifoldOptimProblem.h"
#include "ManifoldOptimException.h"

class MatrixManifoldOptimProblem : public ManifoldOptimProblem
{

public:
	MatrixManifoldOptimProblem(bool useNumericalGrad, bool useNumericalHessEta)
	: ManifoldOptimProblem(useNumericalGrad, useNumericalHessEta)
	{
	}

	virtual ~MatrixManifoldOptimProblem() {}

	virtual double objFun(const arma::mat &X) const = 0;

	virtual arma::mat gradFun(const arma::mat& X) const
	{
		throw ManifoldOptimException("gradFun is not implemented");
	}

	virtual arma::vec hessEtaFun(const arma::mat& X, const arma::vec& eta) const
	{
		throw ManifoldOptimException("hessEtaFun is not implemented");
	}
};

#endif

