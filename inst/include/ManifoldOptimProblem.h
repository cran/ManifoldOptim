#ifndef MANIFOLD_OPTIM_PROBLEM_H
#define MANIFOLD_OPTIM_PROBLEM_H

#include <RcppArmadillo.h>
#include <cstring>
#include <assert.h>
#include <iostream>

class ManifoldOptimProblem
{

public:
	ManifoldOptimProblem(bool useNumericalGrad, bool useNumericalHessEta)
	: _useNumericalGrad(useNumericalGrad), _useNumericalHessEta(useNumericalHessEta)
	{
	}
	virtual ~ManifoldOptimProblem() {};
	bool UseNumericalGrad() const { return _useNumericalGrad; }
	bool UseNumericalHessEta() const { return _useNumericalHessEta; }

protected:
	bool _useNumericalGrad;
	bool _useNumericalHessEta;
};

#endif

