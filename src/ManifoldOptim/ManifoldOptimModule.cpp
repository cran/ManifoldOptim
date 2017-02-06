#include <RcppArmadillo.h>
#include "RProblem.h"

RCPP_MODULE(ManifoldOptim_module) {
	class_<RProblem>("RProblem")
	.constructor<Function,Function,Function>()
	.constructor<Function,Function>()
	.constructor<Function>()
	.field_readonly("objFun", &RProblem::m_objFun)
	.field_readonly("gradFun", &RProblem::m_gradFun)
	.field_readonly("hessEtaFun", &RProblem::m_hessEtaFun)
	;
}

