//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: data.h
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
//
// This file is part of the pdaggerq package.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License./>.
//

#ifndef TENSOR_H
#define TENSOR_H

#include<vector>
#include<string>

namespace pdaggerq {

class tensor {

  public:

    /// constructor
    tensor(){};

    /// destructor
    ~tensor(){};

    /// tensor labels (human readable)
    std::vector<std::string> labels;

    /// tensor numerical labels
    std::vector<int> numerical_labels;

    /// spin labels (human readable)
    std::vector<std::string> spin_labels;

    /// number of permutations required to sort tensor labels
    int permutations = 0;

    /// sort tensor, keep track of permutations
    virtual void sort() {
        printf("\n");
        printf("    sort() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /// comparison between two tensors (warning: ignores spin)
    bool operator==(const tensor& rhs) {
        return ( numerical_labels == rhs.numerical_labels );
    }

    /// copy tensors
    virtual tensor operator=(const tensor& rhs) {
        printf("\n");
        printf("    operator '=' has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /// print tensor
    virtual void print(std::string symbol) {
        printf("\n");
        printf("    print() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /// print tensor to string
    virtual std::string to_string(std::string symbol) {
        printf("\n");
        printf("    to_string() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /// print tensor to string with spin labels
    virtual std::string to_string_with_spin(std::string symbol) {
        printf("\n");
        printf("    to_string_with_spin() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

};

class amplitudes: public tensor {

  public:

    /// constructor
    amplitudes(){};

    /// destructor
    ~amplitudes(){};

    /// sort amplitudes, keep track of permutations, assign total numerical value
    void sort();

    /// copy amplitudes
    amplitudes operator=(const amplitudes& rhs);

    /// print amplitudes
    void print(char symbol);

    /// print amplitudes to string
    std::string to_string(char symbol);

    /// print amplitudes to string with spin labels
    std::string to_string_with_spin(char symbol);

    /// operator order
    int order = -1;

};

class integrals: public tensor {

  public:

    /// constructor
    integrals(){};

    /// destructor
    ~integrals(){};

    /// sort integrals, keep track of permutations
    void sort();

    /// copy integrals
    integrals operator=(const integrals& rhs);

    /// print integrals
    void print(std::string symbol);

    /// print integrals to string
    std::string to_string(std::string symbol);

    /// print integrals to string with spin labels
    std::string to_string_with_spin(std::string symbol);

};

class delta_functions: public tensor {

  public:

    /// constructor
    delta_functions(){};

    /// destructor
    ~delta_functions(){};

    /// sort deltas
    void sort();

    /// copy deltas
    delta_functions operator=(const delta_functions& rhs);

    /// print deltas
    void print();

    /// print deltas to string
    std::string to_string();

    /// print deltas to string with spin labels
    std::string to_string_with_spin();

};

}

#endif
