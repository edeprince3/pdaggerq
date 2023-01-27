//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_tensor.h
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

#ifndef PQ_TENSOR_H
#define PQ_TENSOR_H

#include<vector>
#include<string>

namespace pdaggerq {

class tensor {

  public:

    /**
     *
     * constructor
     *
     */
    tensor(){};

    /**
     *
     * destructor
     *
     */
    ~tensor(){};

    /**
     *
     * human readable tensor labels
     *
     */
    std::vector<std::string> labels;

    /**
     *
     * numerical representation of tensor labels
     *
     */
    std::vector<int> numerical_labels;

    /**
     *
     * human readable spin labels
     *
     */
    std::vector<std::string> spin_labels;

    /**
     *
     * number of permutations required to sort tensor labels
     *
     */
    int permutations = 0;

    /**
     *
     * sort numerical tensor labels, keep track of permutations
     *
     */
    virtual void sort() {
        printf("\n");
        printf("    sort() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /**
     *
     * compare two tensors (warning: ignores spin)
     *
     * @param rhs: the tensor against which this one is compared
     */
    bool operator==(const tensor& rhs) {
        return ( numerical_labels == rhs.numerical_labels );
    }

    /**
     *
     * copy a target tensor into this one
     *
     * @param rhs: the target tensor
     */
    virtual tensor operator=(const tensor& rhs) {
        printf("\n");
        printf("    operator '=' has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /**
     *
     * print tensor information to stdout
     *
     * @param symbol: the tensor type
     */
    virtual void print(std::string symbol) {
        printf("\n");
        printf("    print() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /**
     *
     * print tensor information to a string
     *
     * @param symbol: the tensor type
     */
    virtual std::string to_string(std::string symbol) {
        printf("\n");
        printf("    to_string() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /**
     *
     * print tensor information to a string, including spin information
     *
     * @param symbol: the tensor type
     */
    virtual std::string to_string_with_spin(std::string symbol) {
        printf("\n");
        printf("    to_string_with_spin() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /**
     *
     * print tensor information to a string, including range information
     *
     * @param symbol: the tensor type
     */
    virtual std::string to_string_with_label_ranges(std::string symbol) {
        printf("\n");
        printf("    to_string_with_label_ranges() has not been implemented for this tensor type\n");
        printf("\n");
        exit(1);
    }

    /**
     *
     * ranges that labels span ("act", "ext")
     *
     */
    std::vector<std::string> label_ranges;

};

class amplitudes: public tensor {

  public:

    /**
     * 
     * constructor 
     * 
     */
    amplitudes(){};

    /**
     *
     * destructor
     *
     */
    ~amplitudes(){};

    /**
     *
     * sort numerical amplitudes labels, keep track of permutations
     *
     */
    void sort();

    /**
     *
     * copy target amplitudes into this one
     *
     * @param rhs: the target amplitudes
     */
    amplitudes operator=(const amplitudes& rhs);

    /**
     *
     * print amplitudes information to stdout
     *
     * @param symbol: the amplitudes type
     */
    void print(char symbol);

    /**
     *
     * print amplitudes information to a string
     *
     * @param symbol: the amplitudes type
     */
    std::string to_string(char symbol);

    /**
     *
     * print amplitudes information to a string, including spin information
     *
     * @param symbol: the amplitudes type
     */
    std::string to_string_with_spin(char symbol);

    /**
     *
     * print amplitudes information to a string, including range information
     *
     * @param symbol: the amplitudes type
     */
    std::string to_string_with_label_ranges(char symbol);

    /**
     *
     * the number of labels corresponding to creation operators, e.g., 2 for t2(ab,ij), 1 for r2(a,ij)
     *
     */
    int n_create = -1;

    /**
     *
     * the number of labels corresponding to annihilation operators, e.g., 2 for t2(ab,ij), 1 for r2(a,ij)
     *
     */
    int n_annihilate = -1;
};

class integrals: public tensor {

  public:

    /**
     * 
     * constructor 
     * 
     */
    integrals(){};

    /**
     *
     * destructor
     *
     */
    ~integrals(){};

    /**
     *
     * sort numerical integrals labels, keep track of permutations
     *
     */
    void sort();

    /**
     *
     * copy target integrals into this one
     *
     * @param rhs: the target integrals
     */
    integrals operator=(const integrals& rhs);

    /**
     *
     * print integrals information to stdout
     *
     * @param symbol: the integrals type
     */
    void print(std::string symbol);

    /**
     *
     * print integrals information to a string
     *
     * @param symbol: the integrals type
     */
    std::string to_string(std::string symbol);

    /**
     *
     * print integrals information to a string, including spin information
     *
     * @param symbol: the integrals type
     */
    std::string to_string_with_spin(std::string symbol);

    /**
     *
     * print integrals information to a string, including range information
     *
     * @param symbol: the integrals type
     */
    std::string to_string_with_label_ranges(std::string symbol);

};

class delta_functions: public tensor {

  public:

    /**
     * 
     * constructor 
     * 
     */
    delta_functions(){};

    /**
     * 
     * destructor
     * 
     */
    ~delta_functions(){};

    /**
     *
     * sort numerical deltas labels
     *
     */
    void sort();

    /**
     *
     * copy target deltas into this one
     *
     * @param rhs: the target deltas
     */
    delta_functions operator=(const delta_functions& rhs);

    /**
     *
     * print deltas information to stdout
     *
     */
    void print();

    /**
     *
     * print deltas information to a string
     *
     */
    std::string to_string();

    /**
     *
     * print deltas information to a string, including spin information
     *
     */
    std::string to_string_with_spin();

    /**
     *
     * print deltas information to a string, including range information
     *
     */
    std::string to_string_with_label_ranges();

};

}

#endif
