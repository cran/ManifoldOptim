/*
This file defines the class of the Riemannian steepest descent method. See [AMS2008]
	[AMS2008]: P.-A. Absil, R. Mahony, and R. Sepulchre. Optimization algorithms on matrix manifolds.
	Princeton University Press, Princeton, NJ, 2008.

Solvers --> SolversLS --> RNewton

---- WH
*/

#ifndef RSD_H
#define RSD_H

#include "SolversLS.h"
#include "def.h"

/*Define the namespace*/
namespace ROPTLIB{

	class RSD : public SolversLS{
	public:
		/*The contructor of RSD method. It calls the function Solvers::Initialization.
		INPUT : prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		RSD(const Problem *prob, const Variable *initialx);

		/*Call Solvers::SetProbX function and indicate RCG does not need action of Hessian.
		INPUT:	prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		virtual void SetProbX(const Problem *prob, const Variable *initialx);

		/*Setting parameters (member variables) to be default values */
		virtual void SetDefaultParams();
	protected:
		/*Set the search direction to be negative gradient*/
		virtual void GetSearchDir();
	};
} /*end of ROPTLIB namespace*/
#endif // end of RSD_H
