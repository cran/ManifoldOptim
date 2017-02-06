#include "ProblemAdapter.h"

using namespace Rcpp;
using namespace ROPTLIB;
using namespace arma;

ProblemAdapter::ProblemAdapter(VectorManifoldOptimProblem* up)
: m_upVec(up), m_upMat(NULL), m_useMatrix(false)
{
}
 
ProblemAdapter::ProblemAdapter(MatrixManifoldOptimProblem* up)
: m_upVec(NULL), m_upMat(up), m_useMatrix(true)
{
} 

ProblemAdapter::~ProblemAdapter(){}

bool ProblemAdapter::UseNumericalGrad() const
{
	if (m_useMatrix) {
		return m_upMat->UseNumericalGrad();
	} else {
		return m_upVec->UseNumericalGrad();
	}
}

bool ProblemAdapter::UseNumericalHessEta() const
{
	if (m_useMatrix) {
		return m_upMat->UseNumericalHessEta();
	} else {
		return m_upVec->UseNumericalHessEta();
	}
}


double ProblemAdapter::f(Variable* x) const
{
	if (m_useMatrix) {
		const arma::mat& X = ToArmaMat(x);
		return m_upMat->objFun(X);
	} else {
		const arma::vec& X = ToArmaVec(x);
		return m_upVec->objFun(X);
	}
}

void ProblemAdapter::EucGrad(Variable* x, Vector* egf) const
{
	if (m_useMatrix) {
		if (m_upMat->UseNumericalGrad()) {
			NumericalEucGrad(x, egf);
		} else {
			arma::mat X = ToArmaMat(x);
			const arma::mat& G = m_upMat->gradFun(X);
			CopyFrom(egf, G);
		}
	} else {
		if (m_upVec->UseNumericalGrad()) {
			NumericalEucGrad(x, egf);
		} else {
			arma::vec X = ToArmaVec(x);
			const arma::mat& G = m_upVec->gradFun(X);
			CopyFrom(egf, G);
		}
	}
}

void ProblemAdapter::EucHessianEta(Variable* x, Vector* etax, Vector* exix) const
{
	if (m_useMatrix) {
		if (m_upMat->UseNumericalHessEta()) {
			NumericalEucHessianEta(x, etax, exix);
		} else {
			arma::mat X = ToArmaMat(x);
			arma::vec eta = ToArmaVec(etax);
			const arma::vec& hessEta = m_upMat->hessEtaFun(X, eta);
			CopyFrom(exix, hessEta);
		}
	} else {
		if (m_upVec->UseNumericalHessEta()) {
			NumericalEucHessianEta(x, etax, exix);
		} else {
			arma::vec X = ToArmaVec(x);
			arma::vec eta = ToArmaVec(etax);
			const arma::vec& hessEta = m_upVec->hessEtaFun(X, eta);
			CopyFrom(exix, hessEta);
		}
	}
}

// Calculate a numerical approximation to the Jacobian (Euclidean Gradient).
// Reference:
// www.maths.lth.se/na/courses/FMN081/FMN081-06/lecture7.pdf
//
// function [J]=jacobian(func,x)
//	 % computes the Jacobian of a function
//	 n=length(x);
// fx=feval(func,x);
// eps=1.e-8; % could be made better
//		 xperturb=x;
// for i=1:n
//				xperturb(i)=xperturb(i)+eps;
// J(:,i)=(feval(func,xperturb)-fx)/eps;
// xperturb(i)=x(i);
// end;

void ProblemAdapter::NumericalEucGrad(Variable* x, Vector* egf) const
{
	double eps = 1e-12;

	double fx = f(x);
	size_t nn = x->Getlength();
	Variable* x_eps = x->ConstructEmpty();

	const double* x_ptr = x->ObtainReadData();
	double* x_eps_ptr = x_eps->ObtainWriteEntireData();
	double* egf_ptr = egf->ObtainWriteEntireData();

	for (size_t i = 0; i < nn; ++i) {
		x_eps_ptr[i] = x_ptr[i];
	}

	for (size_t i = 0; i < nn; ++i) {
		x_eps_ptr[i] += eps;
		double fp = f(x_eps);
		egf_ptr[i] = (fp - fx) / eps;
		x_eps_ptr[i] = x_ptr[i];
	}

	delete x_eps;
}

// Calculate a numerical approximation to the Hessian. Reference:
// http://objectmix.com/fortran/730003-how-calculate-hessian-matrix.html
void ProblemAdapter::NumericalEucHessianEta(Variable* x, Vector* etax, Vector* exix) const
{
	double eps = 1e-4;
	integer nn = x->Getlength();
	const double* x_ptr = x->ObtainReadData();
	const double* etax_ptr = etax->ObtainReadData();
	double* exix_ptr = exix->ObtainWriteEntireData();

	Variable* x_00 = x->ConstructEmpty();
	Variable* x_01 = x->ConstructEmpty();
	Variable* x_10 = x->ConstructEmpty();
	Variable* x_11 = x->ConstructEmpty();
	double* x_00_ptr = x_00->ObtainWriteEntireData();
	double* x_01_ptr = x_01->ObtainWriteEntireData();
	double* x_10_ptr = x_10->ObtainWriteEntireData();
	double* x_11_ptr = x_11->ObtainWriteEntireData();

	for (size_t i = 0; i < nn; ++i) {
		x_00_ptr[i] = x_ptr[i];
		x_01_ptr[i] = x_ptr[i];
		x_10_ptr[i] = x_ptr[i];
		x_11_ptr[i] = x_ptr[i];
	}
	
	double fx = f(x);
	for (size_t i = 0; i < nn; ++i) {
		exix_ptr[i] = 0;

		for (size_t j = 0; j < nn; ++j) {
			x_00_ptr[i] -= eps;
			x_00_ptr[j] -= eps;
			x_01_ptr[i] -= eps;
			x_01_ptr[j] += eps;
			x_10_ptr[i] += eps;
			x_10_ptr[j] -= eps;
			x_11_ptr[i] += eps;
			x_11_ptr[j] += eps;
			double H_ij = (f(x_11) - f(x_10) - f(x_01) + f(x_00)) / (4*eps*eps);
			
			exix_ptr[i] += H_ij * etax_ptr[j];

			x_00_ptr[i] = x_ptr[i];
			x_00_ptr[j] = x_ptr[j];
			x_01_ptr[i] = x_ptr[i];
			x_01_ptr[j] = x_ptr[j];
			x_10_ptr[i] = x_ptr[i];
			x_10_ptr[j] = x_ptr[j];
			x_11_ptr[i] = x_ptr[i];
			x_11_ptr[j] = x_ptr[j];
		}
	}

	delete x_00;
	delete x_01;
	delete x_10;
	delete x_11;
}

