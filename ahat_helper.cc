#ifndef _python_api2_h_
#define _python_api2_h_

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include <cctype>
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
        .def(py::init< std::string >())
        .def("set_bra", &ahat_helper::set_bra)
        .def("set_string", &ahat_helper::set_string)
        .def("set_tensor", &ahat_helper::set_tensor)
        .def("set_amplitudes", &ahat_helper::set_amplitudes)
        .def("set_factor", &ahat_helper::set_factor)
        .def("add_new_string", &ahat_helper::add_new_string)
        .def("add_operator_product", &ahat_helper::add_operator_product)
        .def("add_commutator", &ahat_helper::add_commutator)
        .def("add_double_commutator", &ahat_helper::add_double_commutator)
        .def("add_triple_commutator", &ahat_helper::add_triple_commutator)
        .def("add_quadruple_commutator", &ahat_helper::add_quadruple_commutator)
        .def("simplify", &ahat_helper::simplify)
        .def("clear", &ahat_helper::clear)
        .def("print", &ahat_helper::print)
        .def("print_fully_contracted", &ahat_helper::print_fully_contracted)
        .def("print_one_body", &ahat_helper::print_one_body)
        .def("print_two_body", &ahat_helper::print_two_body);
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

void removeParentheses(std::string &x)
{ 
  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '(');});
  x.erase(it, std::end(x));

  it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == ')');});
  x.erase(it, std::end(x));

}


ahat_helper::ahat_helper(std::string vacuum_type)
{

    if ( vacuum_type == "" ) {
        vacuum = "TRUE";
    }else if ( vacuum_type == "TRUE" || vacuum_type == "true" ) {
        vacuum = "TRUE";
    }else if ( vacuum_type == "FERMI" || vacuum_type == "fermi" ) {
        vacuum = "FERMI";
    }else {
        printf("\n");
        printf("    error: invalid vacuum type (%s)\n",vacuum_type.c_str());
        printf("\n");
        exit(1);
    }

    data = (std::shared_ptr<StringData>)(new StringData());

    bra = "VACUUM";

}

ahat_helper::~ahat_helper()
{
}

void ahat_helper::set_bra(std::string bra_type){

    if ( bra_type == "" ) {
        bra = "VACUUM";
    }else if ( bra_type == "SINGLES" || bra_type == "singles" ) {
        bra = "SINGLES";
    }else if ( bra_type == "FERMI" || bra_type == "doubles" ) {
        bra = "DOUBLES";
    }else if ( bra_type == "VACUUM" || bra_type == "vacuum" ) {
        bra = "VACUUM";
    }else {
        printf("\n");
        printf("    error: invalid bra type (%s)\n",bra_type.c_str());
        printf("\n");
        exit(1);
    }

}

void ahat_helper::add_commutator(double factor, std::vector<std::string>  in) {

    if ( in.size() != 2 ) {
        printf("\n");
        printf("    error: commutator can only involve two operators\n");
        printf("\n");
        exit(1);
    }

    add_operator_product( factor, {in[0], in[1]} );
    add_operator_product(-factor, {in[1], in[0]} );

}

void ahat_helper::add_double_commutator(double factor, std::vector<std::string>  in) {

    if ( in.size() != 3 ) {
        printf("\n");
        printf("    error: double commutator can only involve three operators\n");
        printf("\n");
        exit(1);
    }

    add_operator_product( factor, {in[0], in[1], in[2]} );
    add_operator_product(-factor, {in[1], in[0], in[2]} );
    add_operator_product(-factor, {in[2], in[0], in[1]} );
    add_operator_product( factor, {in[2], in[1], in[0]} );

}

void ahat_helper::add_triple_commutator(double factor, std::vector<std::string>  in) {

    if ( in.size() != 4 ) {
        printf("\n");
        printf("    error: triple commutator can only involve four operators\n");
        printf("\n");
        exit(1);
    }

    add_operator_product( factor, {in[0], in[1], in[2], in[3]} );
    add_operator_product(-factor, {in[1], in[0], in[2], in[3]} );
    add_operator_product(-factor, {in[2], in[0], in[1], in[3]} );
    add_operator_product( factor, {in[2], in[1], in[0], in[3]} );
    add_operator_product(-factor, {in[3], in[0], in[1], in[2]} );
    add_operator_product( factor, {in[3], in[1], in[0], in[2]} );
    add_operator_product( factor, {in[3], in[2], in[0], in[1]} );
    add_operator_product(-factor, {in[3], in[2], in[1], in[0]} );

}

void ahat_helper::add_quadruple_commutator(double factor, std::vector<std::string>  in) {

    if ( in.size() != 5 ) {
        printf("\n");
        printf("    error: quadruple commutator can only involve five operators\n");
        printf("\n");
        exit(1);
    }

    add_operator_product( factor, {in[0], in[1], in[2], in[3], in[4]} );
    add_operator_product(-factor, {in[1], in[0], in[2], in[3], in[4]} );
    add_operator_product(-factor, {in[2], in[0], in[1], in[3], in[4]} );
    add_operator_product( factor, {in[2], in[1], in[0], in[3], in[4]} );
    add_operator_product(-factor, {in[3], in[0], in[1], in[2], in[4]} );
    add_operator_product( factor, {in[3], in[1], in[0], in[2], in[4]} );
    add_operator_product( factor, {in[3], in[2], in[0], in[1], in[4]} );
    add_operator_product(-factor, {in[3], in[2], in[1], in[0], in[4]} );
    add_operator_product(-factor, {in[4], in[0], in[1], in[2], in[3]} );
    add_operator_product( factor, {in[4], in[1], in[0], in[2], in[3]} );
    add_operator_product( factor, {in[4], in[2], in[0], in[1], in[3]} );
    add_operator_product(-factor, {in[4], in[2], in[1], in[0], in[3]} );
    add_operator_product( factor, {in[4], in[3], in[0], in[1], in[2]} );
    add_operator_product(-factor, {in[4], in[3], in[1], in[0], in[2]} );
    add_operator_product(-factor, {in[4], in[3], in[2], in[0], in[1]} );
    add_operator_product( factor, {in[4], in[3], in[2], in[1], in[0]} );

}

void ahat_helper::add_operator_product(double factor, std::vector<std::string>  in){

    std::vector<std::string> tmp_string;

    if ( bra == "SINGLES" ) {

        // for singles equations: <me| = <0|m*e
        tmp_string.push_back("m*");
        tmp_string.push_back("e");

    }else if ( bra == "DOUBLES" ) {

        // for doubles equations: <mnef| = <0|m*n*fe
        tmp_string.push_back("m*");
        tmp_string.push_back("n*");
        tmp_string.push_back("f");
        tmp_string.push_back("e");

    }


    for (int i = 0; i < (int)in.size(); i++) {
        // lowercase indices
        std::transform(in[i].begin(), in[i].end(), in[i].begin(), [](unsigned char c){ return std::tolower(c); });

        // remove parentheses
        removeParentheses(in[i]);

        if ( in[i].substr(0,1) == "h" ) { // one-electron operator

            std::string tmp = in[i].substr(1,2);
            tmp_string.push_back(tmp.substr(0,1)+"*");
            tmp_string.push_back(tmp.substr(1,1));
            set_tensor({tmp.substr(0,1), tmp.substr(1,1)});

        }else if ( in[i].substr(0,1) == "g" ) { // two-electron operator

            factor *= 0.5;

            std::string tmp = in[i].substr(1,4);
            tmp_string.push_back(tmp.substr(0,1)+"*");
            tmp_string.push_back(tmp.substr(2,1)+"*");
            tmp_string.push_back(tmp.substr(3,1));
            tmp_string.push_back(tmp.substr(1,1));
            set_tensor({tmp.substr(0,1), tmp.substr(1,1), tmp.substr(2,1), tmp.substr(3,1)});

        }else if ( in[i].substr(0,1) == "t" ){

            if ( in[i].substr(1,1) == "1" ){

                std::string tmp = in[i].substr(2,2);
                tmp_string.push_back(tmp.substr(0,1)+"*");
                tmp_string.push_back(tmp.substr(1,1));
                set_amplitudes({tmp.substr(0,1), tmp.substr(1,1)});

            }else if ( in[i].substr(1,1) == "2" ){

                factor *= 0.25;

                std::string tmp = in[i].substr(2,4);
                tmp_string.push_back(tmp.substr(0,1)+"*");
                tmp_string.push_back(tmp.substr(1,1)+"*");
                tmp_string.push_back(tmp.substr(3,1));
                tmp_string.push_back(tmp.substr(2,1));
                set_amplitudes({tmp.substr(0,1), tmp.substr(1,1), tmp.substr(2,1), tmp.substr(3,1)});

            }else {
                printf("\n");
                printf("    error: only t1 or t2 amplitudes are supported\n");
                printf("\n");
                exit(1);
            }
        }else {
                printf("\n");
                printf("    error: undefined string\n");
                printf("\n");
                exit(1);
        }
        
    }

    set_factor(factor);

    set_string(tmp_string);

    add_new_string();

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

void ahat_helper::add_new_string_true_vacuum(){

    std::shared_ptr<ahat> mystring (new ahat(vacuum));

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

void ahat_helper::add_new_string() {

    if ( vacuum == "TRUE" ) {
        add_new_string_true_vacuum();
    }else {
        add_new_string_fermi_vacuum();
    }

}

void ahat_helper::add_new_string_fermi_vacuum(){

    std::vector<std::shared_ptr<ahat> > mystrings;
    mystrings.push_back( (std::shared_ptr<ahat>)(new ahat(vacuum)) );

    // if normal order is defined with respect to the fermi vacuum, we must
    // check here if the input string contains any general-index operators
    // (h, g). If it does, then the string must be split to account explicitly
    // for sums over 
    int n_gen_idx = 0;
    for (int i = 0; i < (int)data->string.size(); i++) {
        std::string me = data->string[i];
         
        if ( !mystrings[0]->is_vir(me.at(0)) && !mystrings[0]->is_occ(me.at(0)) ) {
            n_gen_idx++;
        }

    }
    //printf("number of general indices: %5i\n",n_gen_idx);
    // need number of strings to be square of number of general indices 
    if ( n_gen_idx > 0 ) {
        mystrings.clear();
        for (int i = 0; i < n_gen_idx * n_gen_idx; i++) {
            mystrings.push_back( (std::shared_ptr<ahat>)(new ahat(vacuum)) );
        }
    }

    // TODO: this function only works correctly if you go through the
    // add_operator_product function (or some function that calls that one
    // one). should generalize so set_tensor, etc. can be used directly.

    if ( n_gen_idx == 0 ) {
        n_gen_idx = 1;
    }

    for (int string_num = 0; string_num < n_gen_idx * n_gen_idx; string_num++) {

        // factors:
        if ( data->factor > 0.0 ) {
            mystrings[string_num]->sign = 1;
            mystrings[string_num]->data->factor = fabs(data->factor);
        }else {
            mystrings[string_num]->sign = -1;
            mystrings[string_num]->data->factor = fabs(data->factor);
        }

        int my_gen_idx = 0;
        for (int i = 0; i < (int)data->string.size(); i++) {
            std::string me = data->string[i];

            // fermi vacuum 
            if ( mystrings[string_num]->is_vir(me.at(0)) ) {
                if (me.find("*") != std::string::npos ){
                    mystrings[string_num]->is_dagger.push_back(true);
                    mystrings[string_num]->is_dagger_fermi.push_back(true);
                    removeStar(me);
                }else {
                    mystrings[string_num]->is_dagger.push_back(false);
                    mystrings[string_num]->is_dagger_fermi.push_back(false);
                }
                mystrings[string_num]->symbol.push_back(me);
            }else if ( mystrings[string_num]->is_occ(me.at(0)) ) {
                if (me.find("*") != std::string::npos ){
                    removeStar(me);
                    mystrings[string_num]->is_dagger.push_back(true);
                    mystrings[string_num]->is_dagger_fermi.push_back(false);
                }else {
                    mystrings[string_num]->is_dagger.push_back(false);
                    mystrings[string_num]->is_dagger_fermi.push_back(true);
                }
                mystrings[string_num]->symbol.push_back(me);
            }else {

                //two-index tensor
                // 00, 01, 10, 11
                if ( n_gen_idx == 2 ) {
                    if ( my_gen_idx == 0 ) {
                        if ( string_num == 0 || string_num == 1 ) {
                            // first index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("o");
                            mystrings[string_num]->symbol.push_back("o");
                        }else {
                            // first index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                                mystrings[string_num]->is_dagger.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("w");
                            mystrings[string_num]->symbol.push_back("w");
                        }
                    }else {
                        if ( string_num == 0 || string_num == 2 ) {
                            // second index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("t");
                            mystrings[string_num]->symbol.push_back("t");
                        }else {
                            // second index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("x");
                            mystrings[string_num]->symbol.push_back("x");
                        }
                    }
                }

                //four-index tensor

                // managing these labels is so very confusing:
                // p*q*sr (pr|qs) -> o*t*uv (ov|tu), etc.
                // p*q*sr (pr|qs) -> w*x*yz (wz|xy), etc.

                if ( n_gen_idx == 4 ) {
                    if ( my_gen_idx == 0 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num == 0 || 
                             string_num == 1 ||
                             string_num == 2 ||
                             string_num == 3 ||
                             string_num == 4 ||
                             string_num == 5 ||
                             string_num == 6 ||
                             string_num == 7 ) {

                            // first index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("o");
                            mystrings[string_num]->symbol.push_back("o");
                        }else {
                            // first index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("w");
                            mystrings[string_num]->symbol.push_back("w");
                        }
                    }else if ( my_gen_idx == 1 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num ==  0 || 
                             string_num ==  1 ||
                             string_num ==  2 ||
                             string_num ==  3 ||
                             string_num ==  8 ||
                             string_num ==  9 ||
                             string_num == 10 ||
                             string_num == 11 ) {
                            // second index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("v");
                            mystrings[string_num]->symbol.push_back("v");
                        }else {
                            // second index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("z");
                            mystrings[string_num]->symbol.push_back("z");
                        }
                    }else if ( my_gen_idx == 2 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num ==  0 || 
                             string_num ==  1 ||
                             string_num ==  4 ||
                             string_num ==  5 ||
                             string_num ==  8 ||
                             string_num ==  9 ||
                             string_num == 12 ||
                             string_num == 13 ) {
                            // third index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("t");
                            mystrings[string_num]->symbol.push_back("t");
                        }else {
                            // third index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("x");
                            mystrings[string_num]->symbol.push_back("x");
                        }
                    }else {
                        if ( string_num ==  0 || 
                             string_num ==  2 ||
                             string_num ==  4 ||
                             string_num ==  6 ||
                             string_num ==  8 ||
                             string_num == 10 ||
                             string_num == 12 ||
                             string_num == 14 ) {
                            // fourth index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("u");
                            mystrings[string_num]->symbol.push_back("u");
                        }else {
                            // fourth index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("y");
                            mystrings[string_num]->symbol.push_back("y");
                        }
                    }
                }

                my_gen_idx++;
            }

        }

        for (int i = 0; i < (int)data->amplitudes.size(); i++) {
            std::vector<std::string> tmp;
            for (int j = 0; j < (int)data->amplitudes[i].size(); j++) {
                tmp.push_back(data->amplitudes[i][j]);
            }
            mystrings[string_num]->data->amplitudes.push_back(tmp);
        }

        // now, string is complete, but labels in four-index tensors need to be reordered p*q*sr(pq|sr) -> (pr|qs)
        if ( (int)mystrings[string_num]->data->tensor.size() == 4 ) {

            std::vector<std::string> tmp;
            tmp.push_back(mystrings[string_num]->data->tensor[0]);
            tmp.push_back(mystrings[string_num]->data->tensor[3]);
            tmp.push_back(mystrings[string_num]->data->tensor[1]);
            tmp.push_back(mystrings[string_num]->data->tensor[2]);

            mystrings[string_num]->data->tensor.clear();
            mystrings[string_num]->data->tensor.push_back(tmp[0]);
            mystrings[string_num]->data->tensor.push_back(tmp[1]);
            mystrings[string_num]->data->tensor.push_back(tmp[2]);
            mystrings[string_num]->data->tensor.push_back(tmp[3]);

        }

        printf("\n");
        printf("    ");
        printf("// starting string:\n");
        mystrings[string_num]->print();

        // rearrange strings
        mystrings[string_num]->normal_order(ordered);

    }

    //for (int n_ordered = 0; n_ordered < (int)ordered.size(); n_ordered++) {
    //    ordered[n_ordered]->check_occ_vir();
    //}

    // TODO: this only seems to work with normal ordering relative to the true vacuum
    // alphabetize
    //mystring->alphabetize(ordered);

    // TODO: moved cleanup to final simplify function?
    // cancel terms. i think the work is actually done on "ordered" so only need to call once 
    //mystrings[0]->cleanup(ordered);

    // reset data object
    data.reset();
    data = (std::shared_ptr<StringData>)(new StringData());
 
}

void ahat_helper::simplify() {

    std::shared_ptr<ahat> mystring (new ahat(vacuum));

    // eliminate strings based on delta functions and use delta functions to alter tensor / amplitude labels
    for (int i = 0; i < (int)ordered.size(); i++) {

        // check spin
        ordered[i]->check_spin();

        // check for occ/vir pairs in delta functions
        ordered[i]->check_occ_vir();

        // apply delta functions
        ordered[i]->gobble_deltas();

        // replace any funny labels that were added with conventional ones (fermi vacumm only)
        if ( vacuum == "FERMI" ) {
            ordered[i]->use_conventional_labels();
        }
    }

    // try to cancel similar terms
    mystring->cleanup(ordered);
    
}

void ahat_helper::print_two_body() {

    printf("\n");
    printf("    ");
    printf("// two-body strings::\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 4 ) continue;
        ordered[i]->print();
    }
    printf("\n");

}

void ahat_helper::print_fully_contracted() {

    printf("\n");
    printf("    ");
    printf("// fully-contracted strings::\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 0 ) continue;
        ordered[i]->print();
    }
    printf("\n");

}

void ahat_helper::print_one_body() {

    printf("\n");
    printf("    ");
    printf("// one-body strings::\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 2 ) continue;
        ordered[i]->print();
    }
    printf("\n");

}

void ahat_helper::print() {

    printf("\n");
    printf("    ");
    printf("// normal-ordered strings:\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        ordered[i]->print();
    }
    printf("\n");

}

void ahat_helper::clear() {

    ordered.clear();

}


} // End namespaces

#endif
