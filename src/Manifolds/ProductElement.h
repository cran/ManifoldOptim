/*
This file defines the class for storage. An object of ProductElement type can be
used as a point on a product of manifold or a tangent vector in a tangent space of
a product of manifold. ProductElement does not simply put Elements together and make
a product of elements. It also reorganize all the element such that the memories used
of elements are consecutive. This will explore the locality of cpu and speed up the
algorithm.

SmartSpace --> Element --> ProductElement

---- WH
*/

#ifndef PRODUCTELEMENT_H
#define PRODUCTELEMENT_H

#include "Element.h"
#include "Manifold.h"
#include <sstream>

/*Define the namespace*/
namespace ROPTLIB{

	/*Declaration of Element. Element has been defined somewhere.*/
	class Element;

	class ProductElement : public Element{
	public:

		/*Constructor of ProductElement.
		An example of using this constructor to generate an empty point on St(2,3)^2 \times Euc(2) is:
		integer n = 3, p = 2, m = 2;
		StieVariable StieX(n, p);
		EucVariable EucX(m);
		Element **elements = new Element* [3]; elements[0] = &StieX; elements[1] = &StieX; elements[2] = &EucX;
		integer inpowsinterval = {0, 2, 3};
		ProductManifold ProdMani(elements, 3, inpowsinterval, 2);
		* The first argument indicates that there are three elements in totoal, i.e., StieX, StieX, and EucX.
		* The second argement indicates that the length of "elements" is 3.
		* The third argument indicates the indices of all kinds of elements. The number of St(2, 3) is inpowsinterval[1] - inpowsinterval[0] = 2
		and the number of Euc(2) is inpowsinterval[2] - inpowsinterval[1] = 1. Therefore, the product element is a point on St(2, 3)^2 \times Euc(2).
		* The fourth argument indicates the number of kinds of manifolds, i.e., numoftypes = 2. */
		ProductElement(Element **elements, integer numofelements, integer *powsinterval, integer numoftypes);

		/*Constructor of ProductElement.
		An example of using this constructor to generate an empty point on St(2,3)^2 \times Euc(2) is:
		integer n = 3, p = 2, m = 2;
		integer numofmanis = 2;
		integer numofmani1 = 2;
		integer numofmani2 = 1;
		StieVariable StieX(n, p);
		EucVariable EucX(m);
		ProductElement ProdX(numofmanis, &StieX, numofmani1, &EucX, numofmani2);
		* The first argument indicates that there are two kinds of manifolds St(2, 3) and Euc(2).
		* The second argement indicates that first kind of point is a point on St(2, 3).
		* The third argument indicates the number for the previous element. The number of St(2, 3) is numofmani1 = 2
		* The fourth argement indicates that second kind of point is a point on Euc(2).
		* The fifth argument indicates the number for the previous element. The number of Euc(2) is numofmani2 = 1 */
		ProductElement(integer numberoftypes, ...);

		/*Destructor of ProductElement*/
		virtual ~ProductElement();

		/*Create an object of ProductElement with same size as this ProductElement.*/
		virtual ProductElement *ConstructEmpty(void) const;

		/*Copy this ProductElement to "eta" ProductElement. After calling this function,
		this ProductElement and "eta" ProductElement will use same space to store data. */
		virtual void CopyTo(Element *eta) const;

		/*Randomly create this ProductElement. In other words, the space will be allocated based
		on the size. Then each entry in the space will be generated by the uniform distribution in [start, end].
		Note that all the temporary data are also removed.*/
		virtual void RandUnform(double start = 0, double end = 1);

		/*Randomly create this Element. In other words, the space will be allocated based
		on the size. Then each entry in the space will be generated by the normal distribution with mean and variance.
		Note that all the temporary data are also removed*/
		virtual void RandGaussian(double mean = 0, double variance = 1);

		/*Print the data. The string "name" is to mark the output such that user can find the output easily.
		If isonlymain is true, then only output the data without outputing temporary data. Otherwise,
		all the temporary data are also output.*/
		virtual void Print(const char *name = "", bool isonlymain = true) const;

		/*When the ProductElement is instantiated as a point on manifold, then this function needs to be overloaded and randomly
		generate a point on the manifold.*/
		virtual void RandInManifold();

		/*Obtain this ProductElement's pointer which points to the data;
		Users are encouraged to call this function if they want to overwrite the data without caring about its original data.
		If the data is shared with other ProductElement, then new memory are allocated without copying the data to the new memory.
		Note that all the temporary data are also removed. */
		virtual double *ObtainWriteEntireData();

		/*Obtain this Element's pointer which points to the data;
		If the data is shared with other Element, then new memory are allocated and the data are copied to the new memory.
		Note that all the temporary data are also removed. */
		virtual double *ObtainWritePartialData();

		/*If the data is shared with other SmartSpace, then new memory are allocated without copying the data to the new memory.*/
		virtual void NewMemoryOnWrite();

		/*If the data is shared with other SmartSpace, then new memory are allocated and the data are copied to the new memory.*/
		virtual void CopyOnWrite();

		/*Check whether the memory of ProductElement is constant with each individual manifold*/
		virtual void CheckMemory(const char *info = "") const;

		/*Get the idx-th Element of this ProductElement*/
		inline Element *GetElement(integer idx) const { return elements[idx]; };

		/*Get the number of elements of the ProductElement*/
		inline integer GetNumofElement(void) const { return numofelements; };

	protected:
		/*Constructor of ProductElement.
		An example of using this constructor to generate an empty point on St(2,3)^2 \times Euc(2) is:
		integer n = 3, p = 2, m = 2;
		StieVariable StieX(n, p);
		EucVariable EucX(m);
		Element **elements = new Element* [3]; elements[0] = &StieX; elements[1] = &StieX; elements[2] = &EucX;
		integer inpowsinterval = {0, 2, 3};
		ProductManifold ProdMani(elements, 3, inpowsinterval, 2);
		* The first argument indicates that there are two elements in totoal, i.e., StieX, StieX, and EucX.
		* The second argement indicates that the length of "elements" is 3.
		* The third argument indicates the indices of all kinds of elements. The number of St(2, 3) is inpowsinterval[1] - inpowsinterval[0] = 2
		and the number of Euc(2) is inpowsinterval[2] - inpowsinterval[1] = 1. Therefore, the product element is a point on St(2, 3)^2 \times Euc(2).
		* The fourth argument indicates the number of kinds of manifolds, i.e., numoftypes = 2. */
		virtual void ProductElementInitialization(Element **elements, integer numofelements, integer *powsinterval, integer numoftypes);

		/*Reset the memory of all elements of ProductElement such that their pointers to data are consistant.*/
		virtual void ResetMemoryofElementsAndSpace();

		/*Create an empty ProductElement without size information*/
		ProductElement();

		integer numoftypes; /*the number of types of elements*/
		integer *powsinterval; /*the number of each type of elements*/

		Element **elements; /*the pointers to all the element*/
		integer numofelements; /*the total number of elements*/
	};
} /*end of ROPTLIB namespace*/
#endif
