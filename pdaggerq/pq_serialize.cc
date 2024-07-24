#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <cctype>
#include <algorithm>


#include "pq_helper.h"
#include "pq_utils.h"
#include "pq_string.h"
#include "pq_add_label_ranges.h"
#include "pq_add_spin_labels.h"
// surpresses warnings from pybind11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma GCC diagnostic pop

using namespace pdaggerq;

void pq_helper::serialize(const std::string & filename) const {

    // open file
    std::ofstream buffer(filename, std::ios::binary | std::ios::out);

    // helper function to write a primitive in binary
    auto write_primitive = [&buffer](const auto &primitive) {
        buffer.write(reinterpret_cast<const char*>(&primitive), sizeof(primitive));
    };

    // helper function to write a string in binary
    auto write_string = [&buffer, &write_primitive](const std::string &str) {
        size_t length = str.size();
        write_primitive(length);
        buffer.write(str.data(), length);
    };

    /// ordered strings

    // serialize the number of strings
    write_primitive(ordered.size());

    // serialize each string in ordered
    for (const std::shared_ptr<pq_string> & pq_str : ordered) {
        pq_str->serialize(buffer);
    }

    // serialize the number of blocked strings
    write_primitive(ordered_blocked.size());

    // serialize each string in ordered_blocked
    for (const std::shared_ptr<pq_string> & pq_str : ordered_blocked) {
        pq_str->serialize(buffer);
    }

    /// static bools from pq_string
    write_primitive(pq_string::is_spin_blocked);
    write_primitive(pq_string::is_range_blocked);

    /// vacuum type
    write_string(vacuum);

    /// print_level
    write_primitive(print_level);

    /// right operators
    write_string(right_operators_type);

    // count number of right operators
    size_t nright_operators = right_operators.size();
    write_primitive(nright_operators);

    // serialize right operators
    for (const std::vector<std::string> & op : right_operators) {
        write_primitive(op.size());
        for (const std::string & op_str : op) {
            write_string(op_str);
        }
    }

    /// left operators
    write_string(left_operators_type);

    // count number of left operators
    write_primitive(left_operators.size());

    // serialize left operators
    for (const std::vector<std::string> & op : left_operators) {
        write_primitive(op.size());
        for (const std::string & op_str : op) {
            write_string(op_str);
        }
    }

    /// cluster commutation
    write_primitive(cluster_operators_commute);

    /// find_paired_permutations
    write_primitive(find_paired_permutations);

    // close file
    buffer.close();
}

void pq_helper::deserialize(const std::string & filename) {

    // clear pq_helper
    clear();

    // open file
    std::ifstream buffer(filename, std::ios::binary | std::ios::in);

    // test if file is open
    if (!buffer.is_open()) {
        std::cout << "Error: could not open file '" << filename << "'" << std::endl;
        exit(1);
    }

    // helper function to read a primitive in binary
    auto read_primitive = [&buffer](auto &primitive) {
        buffer.read(reinterpret_cast<char*>(&primitive), sizeof(primitive));
    };

    // helper function to read a string in binary
    auto read_string = [&buffer, &read_primitive](std::string &str) {
        size_t length;
        read_primitive(length);
        str.resize(length);
        buffer.read(str.data(), length);
    };

    /// ordered strings

    // deserialize the number of strings
    size_t nstrings;
    read_primitive(nstrings);

    // deserialize each string in ordered
    ordered.clear();
    if (nstrings > 0) {
        ordered.resize(nstrings);
        for (std::shared_ptr<pq_string> & pq_str : ordered) {
            pq_str = std::make_shared<pq_string>("");
            pq_str->deserialize(buffer);
        }
    }

    // deserialize the number of blocked strings
    size_t nstrings_blocked;
    read_primitive(nstrings_blocked);

    // deserialize each string in ordered_blocked
    ordered_blocked.clear();
    if (nstrings_blocked > 0) {
        ordered_blocked.resize(nstrings_blocked);
        for (std::shared_ptr<pq_string> & pq_str : ordered_blocked) {
            pq_str = std::make_shared<pq_string>("");
            pq_str->deserialize(buffer);
        }
    }

    /// static bools from pq_string
    read_primitive(pq_string::is_spin_blocked);
    read_primitive(pq_string::is_range_blocked);

    /// vacuum type
    read_string(vacuum);

    /// print_level
    read_primitive(print_level);

    /// right operators
    read_string(right_operators_type);

    // count number of right operators
    size_t nright_operators;
    read_primitive(nright_operators);

    // deserialize right operators
    right_operators.clear();
    if (nright_operators > 0) {
        right_operators.reserve(nright_operators);
        for (int i = 0; i < nright_operators; i++) {
            size_t nops;
            read_primitive(nops);

            std::vector<std::string> op;

            if (nops > 0) {
                op.reserve(nops);
                for (int j = 0; j < nops; j++) {
                    std::string op_str;
                    read_string(op_str);
                    op.push_back(op_str);
                }
            }
            right_operators.push_back(op);
        }
    }

    /// left operators
    read_string(left_operators_type);

    // count number of left operators
    size_t nleft_operators;
    read_primitive(nleft_operators);

    // deserialize left operators
    left_operators.clear();
    if (nleft_operators > 0) {
        left_operators.reserve(nleft_operators);
        for (int i = 0; i < nleft_operators; i++) {
            size_t nops;
            read_primitive(nops);
            std::vector<std::string> op;
            if (nops > 0) {
                op.reserve(nops);
                for (int j = 0; j < nops; j++) {
                    std::string op_str;
                    read_string(op_str);
                    op.push_back(op_str);
                }
                left_operators.push_back(op);
            }
        }
    }

    /// cluster commutation
    read_primitive(cluster_operators_commute);

    /// find_paired_permutations
    read_primitive(find_paired_permutations);

    // close file
    buffer.close();

}


void pq_string::serialize(std::ofstream &buffer) const {

    /* Objects to serialize

        Primitives:
            std::string                                              vacuum
            int                                                      sign
            bool                                                     skip
            double                                                   factor
            bool                                                     has_w0

        Vectors:
            std::vector<std::string>                                 string
            std::vector<delta_functions>                             deltas
            std::vector<std::string>                                 permutations
            std::vector<std::string>                                 paired_permutations_6
            std::vector<std::string>                                 paired_permutations_3
            std::vector<std::string>                                 paired_permutations_2
            std::vector<bool>                                        is_boson_dagger
            std::vector<bool>                                        is_dagger
            std::vector<bool>                                        is_dagger_fermi
            std::vector<std::string>                                 symbol

        Maps:
            std::unordered_map<std::string, std::vector<integrals> > ints
            std::unordered_map<char, std::vector<amplitudes> >       amps
            std::unordered_map<std::string, std::string>             non_summed_spin_labels

        Static (ignored):
            bool                                                     is_spin_blocked
            bool                                                     is_range_blocked
            char[]                                                   amplitude_types
            std::string[]                                            integral_types

     * */

    // helper function to write a primitive in binary
    auto write_primitive = [&buffer](const auto &primitive) {
        buffer.write(reinterpret_cast<const char*>(&primitive), sizeof(primitive));
    };

    // helper function to write a string in binary
    auto write_string = [&buffer, &write_primitive](const std::string &str) {
        size_t length = str.size();
        write_primitive(length);
        buffer.write(str.data(), length);
    };

    /// primitives first
    write_string(vacuum); // vacuum
    write_primitive(sign);   // sign
    write_primitive(skip);   // skip
    write_primitive(factor); // factor
    write_primitive(has_w0); // has_w0

    /// vectors

    // string
    write_primitive(string.size());
    for (const std::string &s : string) {
        write_string(s);
    }

    // deltas
    write_primitive(deltas.size());
    for (const delta_functions &d : deltas) {
        d.serialize(buffer);
    }

    // permutations
    write_primitive(permutations.size());
    for (const std::string &p : permutations) {
        write_string(p);
    }

    // paired_permutations_6
    write_primitive(paired_permutations_6.size());
    for (const std::string &p : paired_permutations_6) {
        write_string(p);
    }

    // paired_permutations_3
    write_primitive(paired_permutations_3.size());
    for (const std::string &p : paired_permutations_3) {
        write_string(p);
    }

    // paired_permutations_2
    write_primitive(paired_permutations_2.size());
    for (const std::string &p : paired_permutations_2) {
        write_string(p);
    }

    // is_boson_dagger
    write_primitive(is_boson_dagger.size());
    for (const bool &b : is_boson_dagger) {
        write_primitive(b);
    }

    // is_dagger
    write_primitive(is_dagger.size());
    for (const bool &b : is_dagger) {
        write_primitive(b);
    }

    // is_dagger_fermi
    write_primitive(is_dagger_fermi.size());
    for (const bool &b : is_dagger_fermi) {
        write_primitive(b);
    }

    // symbol
    write_primitive(symbol.size());
    for (const std::string &s : symbol) {
        write_string(s);
    }

    /// maps

    // ints
    write_primitive(ints.size());
    for (const auto & [key, value] : ints) {
        write_string(key);
        write_primitive(value.size());
        for (const integrals &integral : value) {
            integral.serialize(buffer);
        }
    }

    // amps
    write_primitive(amps.size());
    for (const auto & [key, value] : amps) {
        write_primitive(key);
        write_primitive(value.size());
        for (const amplitudes &amp : value) {
            amp.serialize(buffer);
        }
    }

    // non_summed_spin_labels
    write_primitive(non_summed_spin_labels.size());
    for (const auto & [key, value] : non_summed_spin_labels) {
        write_string(key);
        write_string(value);
    }

}

void pq_string::deserialize(std::ifstream &buffer) {

    // helper function to read a primitive in binary
    auto read_primitive = [&buffer](auto &primitive) {
        buffer.read(reinterpret_cast<char*>(&primitive), sizeof(primitive));
    };

    // helper function to read a string in binary
    auto read_string = [&buffer, &read_primitive](std::string &str) {
        size_t length;
        read_primitive(length);
        str.resize(length);
        buffer.read(str.data(), length);
    };

    /// primitives first
    read_string(vacuum); // vacuum
    read_primitive(sign);   // sign
    read_primitive(skip);   // skip
    read_primitive(factor); // factor
    read_primitive(has_w0); // has_w0


    /// vectors

    // string
    size_t string_size;
    read_primitive(string_size);

    string.clear();
    if (string_size > 0) {
        string.resize(string_size);
        for (std::string &s: string) {
            read_string(s);
        }
    }

    // deltas
    size_t deltas_size;
    read_primitive(deltas_size);

    deltas.clear();
    if (deltas_size > 0) {
        deltas.resize(deltas_size);
        for (delta_functions &d: deltas) {
            d.deserialize(buffer);
        }
    }

    // permutations
    size_t permutations_size;
    read_primitive(permutations_size);

    permutations.clear();
    if (permutations_size > 0) {
        permutations.resize(permutations_size);
        for (std::string &p: permutations) {
            read_string(p);
        }
    }

    // paired_permutations_6
    size_t paired_permutations_6_size;
    read_primitive(paired_permutations_6_size);

    paired_permutations_6.clear();
    if (paired_permutations_6_size > 0) {
        paired_permutations_6.resize(paired_permutations_6_size);
        for (std::string &p: paired_permutations_6) {
            read_string(p);
        }
    }

    // paired_permutations_3
    size_t paired_permutations_3_size;
    read_primitive(paired_permutations_3_size);

    paired_permutations_3.clear();
    if (paired_permutations_3_size > 0) {
        paired_permutations_3.resize(paired_permutations_3_size);
        for (std::string &p: paired_permutations_3) {
            read_string(p);
        }
    }

    // paired_permutations_2
    size_t paired_permutations_2_size;
    read_primitive(paired_permutations_2_size);

    paired_permutations_2.clear();
    if (paired_permutations_2_size > 0) {
        paired_permutations_2.resize(paired_permutations_2_size);
        for (std::string &p: paired_permutations_2) {
            read_string(p);
        }
    }

    // is_boson_dagger
    size_t is_boson_dagger_size;
    read_primitive(is_boson_dagger_size);

    is_boson_dagger.clear();
    if (is_boson_dagger_size > 0) {
        is_boson_dagger.resize(is_boson_dagger_size);
        for (size_t i = 0; i < is_boson_dagger_size; i++) {
            bool b;
            read_primitive(b);
            is_boson_dagger[i] = b;
        }
    }

    // is_dagger
    size_t is_dagger_size;
    read_primitive(is_dagger_size);

    is_dagger.clear();
    if (is_dagger_size > 0) {
        is_dagger.resize(is_dagger_size);
        for (size_t i = 0; i < is_dagger_size; i++) {
            bool b;
            read_primitive(b);
            is_dagger[i] = b;
        }
    }

    // is_dagger_fermi
    size_t is_dagger_fermi_size;
    read_primitive(is_dagger_fermi_size);

    is_dagger_fermi.clear();
    if (is_dagger_fermi_size > 0) {
        is_dagger_fermi.resize(is_dagger_fermi_size);
        for (size_t i = 0; i < is_dagger_fermi_size; i++) {
            bool b;
            read_primitive(b);
            is_dagger_fermi[i] = b;
        }
    }

    // symbol
    size_t symbol_size;
    read_primitive(symbol_size);

    symbol.clear();
    if (symbol_size > 0) {
        symbol.resize(symbol_size);
        for (std::string &s: symbol) {
            read_string(s);
        }
    }
    /// maps

    // ints
    size_t ints_size;
    read_primitive(ints_size);

    ints.clear();
    if (ints_size > 0) {
        for (size_t i = 0; i < ints_size; i++) {
            std::string key;
            read_string(key);

            size_t value_size;
            read_primitive(value_size);

            std::vector<integrals> value;
            if (value_size >= 0) {
                value.resize(value_size);
                for (integrals &integral: value) {
                    integral.deserialize(buffer);
                }
            }

            ints[key] = value;
        }
    }
    // amps
    size_t amps_size;
    read_primitive(amps_size);

    amps.clear();
    if (amps_size > 0) {
        for (size_t i = 0; i < amps_size; i++) {
            char key;
            read_primitive(key);

            size_t value_size;
            read_primitive(value_size);

            std::vector<amplitudes> value;
            if (value_size >= 0) {
                value.resize(value_size);
                for (amplitudes &amp: value) {
                    amp.deserialize(buffer);
                }
            }

            amps[key] = value;
        }
    }

    // non_summed_spin_labels
    size_t non_summed_spin_labels_size;
    read_primitive(non_summed_spin_labels_size);

    non_summed_spin_labels.clear();
    if (non_summed_spin_labels_size > 0) {
        for (size_t i = 0; i < non_summed_spin_labels_size; i++) {
            std::string key;
            read_string(key);

            std::string value;
            read_string(value);
            non_summed_spin_labels[key] = value;
        }
    }

}


void tensor::serialize(std::ofstream &buffer) const {
    // helper function to write a primitive in binary
    auto write_primitive = [&buffer](const auto &primitive) {
        buffer.write(reinterpret_cast<const char*>(&primitive), sizeof(primitive));
    };

    // helper function to write a string in binary
    auto write_string = [&buffer, &write_primitive](const std::string &str) {
        size_t length = str.size();
        write_primitive(length);
        buffer.write(str.data(), length);
    };

    // labels
    write_primitive(labels.size());
    for (const std::string & label : labels) {
        write_string(label);
    }

    // numerical_labels
    write_primitive(numerical_labels.size());
    for (const int & numerical_label : numerical_labels) {
        write_primitive(numerical_label);
    }

    // spin_labels
    write_primitive(spin_labels.size());
    for (const std::string & spin_label : spin_labels) {
        write_string(spin_label);
    }

    // label_ranges
    write_primitive(label_ranges.size());
    for (const std::string & label_range : label_ranges) {
        write_string(label_range);
    }

    // permutations
    write_primitive(permutations);
}

void tensor::deserialize(std::ifstream &buffer) {

    // helper function to read a primitive in binary
    auto read_primitive = [&buffer](auto &primitive) {
        buffer.read(reinterpret_cast<char*>(&primitive), sizeof(primitive));
    };

    // helper function to read a string in binary
    auto read_string = [&buffer, &read_primitive](std::string &str) {
        size_t length;
        read_primitive(length);
        str.resize(length);
        buffer.read(str.data(), length);
    };

    // labels
    size_t size;
    read_primitive(size);

    labels.clear();
    if (size > 0) {
        labels.resize(size);
        for (std::string &label: labels) {
            read_string(label);
        }
    }

    // numerical_labels
    read_primitive(size);

    numerical_labels.clear();
    if (size > 0) {
        numerical_labels.resize(size);
        for (int &numerical_label: numerical_labels) {
            read_primitive(numerical_label);
        }
    }

    // spin_labels
    read_primitive(size);
    spin_labels.clear();
    if (size > 0) {
        spin_labels.resize(size);
        for (std::string &spin_label: spin_labels) {
            read_string(spin_label);
        }
    }

    // label_ranges
    read_primitive(size);
    label_ranges.clear();
    if (size > 0) {
        label_ranges.resize(size);
        for (std::string &label_range: label_ranges) {
            read_string(label_range);
        }
    }

    // permutations
    read_primitive(permutations);
}
