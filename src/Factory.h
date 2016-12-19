#pragma once
#include <vector>

// Uncomment this if you want to see what it registered
//#include <typeinfo>

template <class Product, class Input>
class Factory {
	typedef Product ProductType;
	typedef Input InputType;
	class Worker {
	public:
		virtual Product* Produce(const Input& input)=0;
	};
	typedef std::vector< Worker* > Workers;
	template <class T>
	class RegisterMe {
	public:
		inline RegisterMe() {
// Uncomment this if you want to see what it registered
//			printf("Registering < %s >!\n",typeid(T).name());
			Staff.push_back(new T);
		};
	};
	static Workers Staff;
public:
	static Product* Produce(const Input& input) {
		Product* ret;
		for (typename Workers::iterator it=Staff.begin(); it != Staff.end(); it++) {
			ret = (*it)->Produce(input);
			if (ret) return ret;
		}
		return NULL;
	};
	template <Product* (*T)(const Input&)>
	class Register : public Worker {
		typedef RegisterMe< Register< T > > Idiot;
		static Idiot Dummy;
	public:
		virtual Product* Produce(const Input& input) { return T(input); };
	};	
};

template <class Product, class Input>
typename Factory<Product, Input>::Workers
         Factory<Product, Input>::Staff;

template <class Product, class Input>
template <Product* (*T)(const Input&)>
typename Factory<Product, Input>::template Register< T >::Idiot
         Factory<Product, Input>::Register< T >::Dummy;


