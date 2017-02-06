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

	mutable VectorManifoldOptimProblem* m_upVec;
	mutable MatrixManifoldOptimProblem* m_upMat;
	bool m_useMatrix;
};

#endif
