#include "GetThreads.h"
#ifdef HAS_CXXABI_H
    #include <cxxabi.h>
#endif
#include <algorithm>

void ThreadNumberCalculatorBase::InitAll() {
    list_t list = List();
    for (type* ptr : list) ptr->Init();
    std::sort(list.begin(), list.end(), compare);
    for (type* ptr : list) ptr->print();
}

ThreadNumberCalculatorBase::ThreadNumberCalculatorBase() {
    List().push_back(this);
}

void ThreadNumberCalculatorBase::print() {
    if (thr.x * thr.y < maxthr) {
        notice( "  %3dx%-3d  | %s --- Reduced from maximum %d\n", thr.x, thr.y, name.c_str(), maxthr);
    } else {
        output( "  %3dx%-3d  | %s\n", thr.x, thr.y, name.c_str());
    }
}

std::string cxx_demangle(std::string str) {
#ifdef HAS_CXXABI_H
    int status;
    char *c_ret = abi::__cxa_demangle(str.c_str(), 0, 0, &status);
    if (c_ret != NULL) {
        std::string ret(c_ret);
        free(c_ret);
        if (status == 0) return ret;
    }
#endif
    return str;
}

int InitDim() {
    MPI_Barrier(MPMD.local);
    output( "  Threads  |      Action\n");
    ThreadNumberCalculatorBase::InitAll();
    MPI_Barrier(MPMD.local);
    return 0;
}
