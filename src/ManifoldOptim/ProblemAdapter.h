#ifndef PROBLEM_ADAPTER_H
#define PROBLEM_ADAPTER_H

#include <RcppArmadillo.h>
#include <iostream>
#include "Problem.h"
#include "Util.h"
#include "ManifoldOptimProblem.h"
#include "VectorManifoldOptimProblem.h"
#include "MatrixManifoldOptimProblem.h"

class ProblemAdapter : public Problem
{
public:
	ProblemAdapter(VectorManifoldOptimProblem* up);
	ProblemAdapter(MatrixManifoldOptimProblem* up);
	virtual ~ProblemAdapter();

	double f(Variable* x) const;
	void EucGrad(Variable* x, Vector* egf) const;
	void EucHessianEta(Variable *x, Vector *etax, Vector *exix) const;
	bool UseNumericalGrad() const;
	bool UseNumericalHessEta() const;

private:
	void NumericalEucGrad(Variable* x, Vector* egf) const;
	void NumericalEucHessianEta(Variable *x, Vector *etax, Vector *exix) const;

	mutable VectorManifoldOptimProblem* _upVec;
	mutable MatrixManifoldOptimProblem* _upMat;
	bool _useMatrix;

	// When using ManifoldOptim, x is an element from a product manifold.
	// For this problem, it only has a single element.
	//void ToArmaMat(Variable* x) const;
};

#endif
