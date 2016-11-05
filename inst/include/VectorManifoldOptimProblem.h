#ifndef VECTOR_MANIFOLD_OPTIM_PROBLEM_H
#define VECTOR_MANIFOLD_OPTIM_PROBLEM_H

#include <RcppArmadillo.h>
#include <cstring>
#include <assert.h>
#include <iostream>
#include "ManifoldOptimProblem.h"
#include "ManifoldOptimException.h"

class VectorManifoldOptimProblem : public ManifoldOptimProblem
{
public:
	VectorManifoldOptimProblem(bool useNumericalGrad, bool useNumericalHessEta)
	: ManifoldOptimProblem(useNumericalGrad, useNumericalHessEta)
	{
	}
	virtual ~VectorManifoldOptimProblem() {}

	virtual double objFun(const arma::vec& X) const = 0;

	virtual arma::mat gradFun(const arma::vec& X) const
	{
		throw ManifoldOptimException("gradFun is not implemented");
	}

	virtual arma::vec hessEtaFun(const arma::vec& X, const arma::vec& eta) const
	{
		throw ManifoldOptimException("hessEtaFun is not implemented");
	}
};

#endif

