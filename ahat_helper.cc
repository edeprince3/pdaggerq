#ifndef _python_api2_h_
#define _python_api2_h_

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>

#include "data.h"
#include "ahat.h"
#include "ahat_helper.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pdaggerq {

void export_ahat_helper(py::module& m) {
    py::class_<pdaggerq::ahat_helper, std::shared_ptr<pdaggerq::ahat_helper> >(m, "ahat_helper")
        .def(py::init< >())
        .def("set_string", &ahat_helper::set_string)
        .def("set_tensor", &ahat_helper::set_tensor)
        .def("set_amplitudes", &ahat_helper::set_amplitudes)
        .def("set_factor", &ahat_helper::set_factor)
        .def("bring_to_normal_order", &ahat_helper::bring_to_normal_order)
        .def("add_new_string", &ahat_helper::add_new_string);
}

PYBIND11_MODULE(pdaggerq, m) {
    m.doc() = "Python API of pdaggerq: A code for bringing strings of creation / annihilation operators to normal order.";
    export_ahat_helper(m);
}

void removeStar(std::string &x)
{ 
  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '*');});
  x.erase(it, std::end(x));
}

ahat_helper::ahat_helper()
{

    data = (std::shared_ptr<StringData>)(new StringData());

}

ahat_helper::~ahat_helper()
{
}

void ahat_helper::set_string(std::vector<std::string> in) {
    for (int i = 0; i < (int)in.size(); i++) {
        data->string.push_back(in[i]);
    }
}
void ahat_helper::set_tensor(std::vector<std::string> in) {
    for (int i = 0; i < (int)in.size(); i++) {
        data->tensor.push_back(in[i]);
    }
}
void ahat_helper::set_amplitudes(std::vector<std::string> in) {
    std::vector<std::string> tmp;
    for (int i = 0; i < (int)in.size(); i++) {
        tmp.push_back(in[i]);
    }
    data->amplitudes.push_back(tmp);
}

void ahat_helper::set_factor(double in) {
    data->factor = in;
}

void ahat_helper::add_new_string(){

    std::shared_ptr<ahat> mystring (new ahat());

    if ( data->factor > 0.0 ) {
        mystring->sign = 1;
        mystring->data->factor = fabs(data->factor);
    }else {
        mystring->sign = -1;
        mystring->data->factor = fabs(data->factor);
    }

    for (int i = 0; i < (int)data->string.size(); i++) {
        std::string me = data->string[i];
        if ( me.find("*") != std::string::npos ) {
            removeStar(me);
            mystring->is_dagger.push_back(true);
        }else {
            mystring->is_dagger.push_back(false);
        }
        mystring->symbol.push_back(me);
    }

    for (int i = 0; i < (int)data->tensor.size(); i++) {
        mystring->data->tensor.push_back(data->tensor[i]);
    }

    for (int i = 0; i < (int)data->amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->amplitudes[i].size(); j++) {
            tmp.push_back(data->amplitudes[i][j]);
        }
        mystring->data->amplitudes.push_back(tmp);
    }

    printf("\n");
    printf("    ");
    printf("// starting string:\n");
    mystring->print();

    // rearrange strings
    mystring->normal_order(ordered);

    // alphabetize
    mystring->alphabetize(ordered);

    // cancel terms
    mystring->cleanup(ordered);

    // reset data object
    data.reset();
    data = (std::shared_ptr<StringData>)(new StringData());

}

void ahat_helper::bring_to_normal_order() {
    
    std::vector< std::shared_ptr<ahat> > out;
            
    bool *vanish = (bool*)malloc(ordered.size()*sizeof(bool));
    memset((void*)vanish,'\0',ordered.size()*sizeof(bool));
    for (int i = 0; i < (int)ordered.size(); i++) {
        for (int j = i+1; j < (int)ordered.size(); j++) {
        
    
            bool strings_differ = false;
    
            // check strings
            if ( ordered[i]->symbol.size() == ordered[j]->symbol.size() ) {
                for (int k = 0; k < (int)ordered[i]->symbol.size(); k++) {
                
                    // strings differ?
                    if ( ordered[i]->symbol[k] != ordered[j]->symbol[k] ) {
                        strings_differ = true;
                    }
            
                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;
            
            // check deltas
            if ( ordered[i]->delta1.size() == ordered[j]->delta1.size() ) {
                for (int k = 0; k < (int)ordered[i]->delta1.size(); k++) {

                    // strings differ?
                    if ( ordered[i]->delta1[k] != ordered[j]->delta1[k] || ordered[i]->delta2[k] != ordered[j]->delta2[k] ) {
                        strings_differ = true;
                    }
        
                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;
        
            // check tensors
            if ( ordered[i]->data->tensor.size() == ordered[j]->data->tensor.size() ) {
                for (int k = 0; k < (int)ordered[i]->data->tensor.size(); k++) {
    
                    // strings differ?
                    if ( ordered[i]->data->tensor[k] != ordered[j]->data->tensor[k] ) {

                        strings_differ = true;
                    }

                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;

            // at this point, we know the strings are the same.  what about the factor?
            double fac1 = ordered[i]->data->factor;
            double fac2 = ordered[j]->data->factor;
            if ( fabs(fac1 + fac2) < 1e-12 ) {
                vanish[i] = true;
                vanish[j] = true;
            }


        }
    }
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( !vanish[i] ) {
            out.push_back(ordered[i]);
        }

    }

    ordered.clear();
    for (int i = 0; i < (int)out.size(); i++) {
        ordered.push_back(out[i]);

        // check spin
        ordered[i]->check_spin();

        // check for occ/vir pairs in delta functions
        ordered[i]->check_occ_vir();

    }

    printf("\n");
    printf("    ");
    printf("// normal-ordered strings:\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        ordered[i]->print();
    }
    printf("\n");

    ordered.clear();
}


} // End namespaces

#endif
