#ifndef FACTORY_H
#define FACTORY_H

#include <vector>

// Uncomment this if you want to see what it registered
//#include <typeinfo>

// Factory template is an implementation of common "Factory method pattern"
//   https://en.wikipedia.org/wiki/Factory_method_pattern
// It stores 'workers' in a vector. Each worked can create the desired 'product'
//   on demand, based on some 'input'. We register such workers by adding
//   functions which take this 'input' and depending on it's value create a 
//   specific 'product' or return NULL, and delegate the task to the next
//   worker.


// Factory template
template <class Product, class Input>
class Factory {
	typedef Product ProductType;
	typedef Input InputType;
	// Generic worker class
	class Worker {
	public:
		virtual Product* Produce(const Input& input)=0;
	};
	// Type to store all the workers
	typedef std::vector< Worker* > Workers;
	// Class to register workers with static member notation
	template <class T>
	class RegisterMe {
	public:
		inline RegisterMe() {
// Uncomment this if you want to see what it registered
//			printf("Registering < %s >!\n",typeid(T).name());
			Staff.push_back(new T);
		};
	};
        // All the workers are kept here
	static Workers Staff;
public:
        // Main function for executing the production process
	static Product* Produce(const Input& input) {
		Product* ret;
		for (typename Workers::iterator it=Staff.begin(); it != Staff.end(); it++) {
			ret = (*it)->Produce(input);
			if (ret) return ret;
		}
		return NULL;
	};
	// Template class for registration of the worker functions
	template <Product* (*T)(const Input&)>
	class Register : public Worker {
	        // Static member, which will register this worker in it's constructor
		typedef RegisterMe< Register< T > > Idiot;
		static Idiot Dummy;
	public:
		virtual Product* Produce(const Input& input) { return T(input); };
	};	
};

// Definition of the static members:
//   All the workers ...
template <class Product, class Input>
typename Factory<Product, Input>::Workers
         Factory<Product, Input>::Staff;
//   All the Dummies
template <class Product, class Input>
template <Product* (*T)(const Input&)>
typename Factory<Product, Input>::template Register< T >::Idiot
         Factory<Product, Input>::Register< T >::Dummy;


#endif