#include "Global.h"
#include "cross.h"

#include <typeinfo>
#include <type_traits>

template <class E, class ...Arg>
CudaGlobalFunction void Kernel(const Arg... args) {
    E e;
    e.Execute(args...);
}

std::string cxx_demangle(std::string str);

inline int ceiling_div(const int & x, const int & y) {
  return x / y + (x % y != 0);
}

/// Get maximal number of threads for all the kernels on runtime
template < class E, class ...Args > int GetThreads() {
	CudaFuncAttributes attr;
    auto attr_ptr = &attr;
	CudaFuncGetAttributes(attr_ptr, (+Kernel<E, Args...>)) ;
	debug1( "[%d] Constant mem:%ld\n", D_MPI_RANK, attr.constSizeBytes);
	debug1( "[%d] Local    mem:%ld\n", D_MPI_RANK, attr.localSizeBytes);
	debug1( "[%d] Max  threads:%d\n", D_MPI_RANK, attr.maxThreadsPerBlock);
	debug1( "[%d] Reg   Number:%d\n", D_MPI_RANK, attr.numRegs);
	debug1( "[%d] Shared   mem:%ld\n", D_MPI_RANK, attr.sharedSizeBytes);
	return attr.maxThreadsPerBlock;
}

class ThreadNumberCalculatorBase {
  typedef ThreadNumberCalculatorBase type;
  typedef std::vector< type* > list_t;
  static inline list_t& List() {
    static list_t list;
    return list;
  }
  static inline bool compare ( const type* a, const type* b ) { return a->name < b->name; }
  protected:
  dim3 thr;
  unsigned int maxthr;
  std::string name;
  public:
  static void InitAll();
  ThreadNumberCalculatorBase();
  virtual void Init() = 0;
  inline dim3 threads() { return thr; }
  void print();
};

template < class T, class ...Args > class ThreadNumberCalculator : public ThreadNumberCalculatorBase {
  public:
  virtual void Init() {
    name = cxx_demangle(typeid(T).name());
    maxthr = GetThreads< T, Args... >();
    thr.z = 1;
    int val = maxthr;
    if (maxthr < X_BLOCK) {
      thr.x = maxthr;
      thr.y = 1;
    } else {
      if (val > MAX_THREADS) {
        val = MAX_THREADS;
      }
      thr.x = X_BLOCK;
      thr.y = val/X_BLOCK;
    }
  };
};

template < class ...T > class ThreadNumber {
  typedef ThreadNumberCalculator< T... > calc_t;
  static calc_t calc;
  public:
  static inline dim3 threads() { return calc.threads(); }
};

template < class ...T > ThreadNumberCalculator<T...> ThreadNumber<T...>::calc;

/// Initialize Thread/Block number variables
int InitDim();

