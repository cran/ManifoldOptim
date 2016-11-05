#ifndef MANIFOLD_OPTIM_EXCEPTION_H
#define MANIFOLD_OPTIM_EXCEPTION_H

#include <exception>
#include <string>

class ManifoldOptimException : public std::exception
{
public:
	ManifoldOptimException(const std::string& msg)
		: _msg(msg)
	{
	}

	~ManifoldOptimException() throw() {}

	virtual const char* what() const throw()
	{
		return _msg.c_str();
	}

private:
	std::string _msg;
};

#endif
