#include "../Consts.h"
#ifdef EMBEDED_PYTHON
    #include <Python.h>
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <numpy/arrayobject.h>
#endif
#include "cbPythonCall.h"
std::string cbPythonCall::xmlname = "CallPython";
#include "../HandlerFactory.h"

int cbPythonCall::Init () {
		Callback::Init();
        static bool firstTime = true;
        if (firstTime){
            firstTime = false;
            #ifdef EMBEDED_PYTHON

            char const* syspythonpath = getenv( "PYTHONPATH" );

            std::string oldpath;
            
            if ( syspythonpath == NULL ) {
            //  Big problem...
            } else {
                oldpath = std::string( syspythonpath );
            }            
            pugi::xml_attribute pn_pythonpath = node.attribute("pythonpath");
            
            oldpath.resize (oldpath.size()+1,':');
            oldpath.resize (oldpath.size()+1,'.');
    
            if (pn_pythonpath){
                oldpath.resize (oldpath.size()+1,':');
                oldpath.append(pn_pythonpath.value());
            }
            setenv("PYTHONPATH",oldpath.c_str(),1);
            debug1("PYTHONPATH set to %s", oldpath.c_str() );

            Py_Initialize();
            _import_array();
            #else
                ERROR("No Python support at compile time (./configure ... --enable-python)");
                return -1;
            #endif
        }
		return 0;
	}


int cbPythonCall::DoIt () {
        
        pugi::xml_attribute module = node.attribute("module");
		pugi::xml_attribute function = node.attribute("function");			
        pugi::xml_attribute comp = node.attribute("densities");
        pugi::xml_attribute quan = node.attribute("quantities");


        name_set components, quantities;
        if(comp){
            components.add_from_string(comp.value(),',');
        }   
        if(quan){
            quantities.add_from_string(quan.value(),',');
        } 
		Callback::DoIt();

		if (!module || !function || (!comp && !quan)) {
            error("Missing params in PythonCall");
        }

        std::vector<real_t*> buffers;
        

#ifdef EMBEDED_PYTHON

////BEGIN PYTHON HANDLING
 

	    PyObject *pModule, *pOffsets, *pFunc, *pGlobalSize;
	    PyObject *pValue, *pArgs;

	    //pName = PyString_FromString(module.value());
	    /* Error checking of pName left out */
	
	    pModule = PyImport_ImportModule(module.value());
	    //Py_DECREF(pName);
        	
	    if (pModule != NULL) {
            int buff_id = 0; 
	        pFunc = PyObject_GetAttrString( pModule, function.value() );
	        /* pFunc is a new reference */
	
	        if (pFunc && PyCallable_Check(pFunc)) {
               
                const int extra_args = 4; 
                pArgs = PyTuple_New(components.size()+quantities.size()+extra_args);
                buffers.resize( components.size()+quantities.size() );
                long int offsets[3] = {-1,-1,-1};

                for (name_set::iterator it = components.begin(); it!=components.end(); ++it){
                    const char * component = it->c_str();
    
                    PyObject* pInputData;
                    real_t * buffer;
                    long int dims[3];

                    long int size = solver->getComponentIntoBuffer(component, buffer, dims, offsets );
    
                    if (sizeof(real_t) == sizeof(float) ) {
                        pInputData = PyArray_SimpleNewFromData(3, dims, NPY_FLOAT, buffer);
                    } else if (sizeof(real_t) == sizeof(double) ){
                        pInputData = PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, buffer);
                    }
                
                    buffers[buff_id] = buffer;

                    //pInputData reference is stolen here
                    PyTuple_SetItem(pArgs, buff_id+extra_args, pInputData);
                    
                    buff_id ++;
                }


                for (name_set::iterator it = quantities.begin(); it!=quantities.end(); ++it){
                    const char * quantity = it->c_str();
    
                    PyObject* pInputData;
                    real_t * buffer;
                    long int dims[4];

                    long int size = solver->getQuantityIntoBuffer(quantity, buffer, dims, offsets );
    
                    if (sizeof(real_t) == sizeof(float) ) {
                        pInputData = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT, buffer);
                    } else if (sizeof(real_t) == sizeof(double) ){
                        pInputData = PyArray_SimpleNewFromData(4, dims, NPY_DOUBLE, buffer);
                    }
                
                    buffers[buff_id] = buffer;

                    //pInputData reference is stolen here
                    PyTuple_SetItem(pArgs, buff_id+extra_args, pInputData);
                    buff_id++;
                }




                
                pOffsets = PyTuple_New(3);    
                for (int k=0; k < 3; k++){
                    PyTuple_SetItem(pOffsets, k, PyLong_FromLong(offsets[k]));
                }

                pGlobalSize = PyTuple_New(3);    
                PyTuple_SetItem(pGlobalSize, 0, PyLong_FromLong(solver->info.region.nx));
                PyTuple_SetItem(pGlobalSize, 1, PyLong_FromLong(solver->info.region.ny));
                PyTuple_SetItem(pGlobalSize, 2, PyLong_FromLong(solver->info.region.nz));

                //first one defines number of extra arguments used
                PyTuple_SetItem(pArgs, 0, PyLong_FromLong( extra_args ));
                PyTuple_SetItem(pArgs, 1, pOffsets);
				PyTuple_SetItem(pArgs, 2, PyLong_FromLong( solver->iter  ));
 				PyTuple_SetItem(pArgs, 3,  pGlobalSize  );
 
		        pValue = PyObject_CallObject(pFunc, pArgs);
                Py_DECREF(pArgs);
                long ret_value = 999;
	            if (pValue != NULL) {
                    ret_value = PyLong_AsLong(pValue);
	                Py_DECREF(pValue);
                } 

                
	            output("Result of Python call: %ld\n",ret_value);
                if (ret_value != 0) {
	                Py_DECREF(pFunc);
	                Py_DECREF(pModule);
	                PyErr_Print();
	                error("PythonCall failed\n");
	                return 1;
	            }
                
	        }
	        else {
	            if (PyErr_Occurred())
	                PyErr_Print();
	                error("PythonCall: Cannot find function \"%s\"\n", function.value());
	        }
	        Py_XDECREF(pFunc);
	        Py_DECREF(pModule);
            
            int all_buff = buff_id;
            buff_id = 0;

            for (name_set::iterator it = components.begin(); it!=components.end(); ++it){
                const char * component = it->c_str();
                for (int k =0; k < 10; k++){
                    debug1("PythonCall after,comp %s,  buffer %d, value %d: %f\n",component,buff_id,k, buffers[buff_id][k]);
                }
                int status = solver->loadComponentFromBuffer(component, buffers[buff_id]);
                buff_id++;
            }           
            for (; buff_id <= all_buff; buff_id++){
      //          delete [] buffers[buff_id];
            }

	    }
	    else {
	        PyErr_Print();
	        error("PythonCall: Failed to load \"%s\"\n", module.value());
	        return 1;
	    }
      

//	    Py_Finalize();
//END PYTHON HANDLING
#else
                ERROR("No Python support at compile time (./configure ... --enable-python)");
                return -1;
#endif



		return 0;
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbPythonCall > >;
