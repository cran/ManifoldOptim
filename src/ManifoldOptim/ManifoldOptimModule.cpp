#include <RcppArmadillo.h>
#include "RProblem.h"

RCPP_MODULE(ManifoldOptim_module) {
	class_<RProblem>("RProblem")
	.constructor<Function,Function,Function>()
	.constructor<Function,Function>()
	.constructor<Function>()
	.field_readonly("objFun", &RProblem::_objFun)
	.field_readonly("gradFun", &RProblem::_gradFun)
	.field_readonly("hessEtaFun", &RProblem::_hessEtaFun)
	;
}

