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
	: m_useNumericalGrad(useNumericalGrad), m_useNumericalHessEta(useNumericalHessEta)
	{
	}
	virtual ~ManifoldOptimProblem() {};
	bool UseNumericalGrad() const { return m_useNumericalGrad; }
	bool UseNumericalHessEta() const { return m_useNumericalHessEta; }

protected:
	bool m_useNumericalGrad;
	bool m_useNumericalHessEta;
};

#endif

