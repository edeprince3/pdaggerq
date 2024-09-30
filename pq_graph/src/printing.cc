//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: printing.cc
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
// Maintainer: DePrince group
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
//  limitations under the License.
//

#include <cmath>
#include <map>
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

#include "../include/pq_graph.h"
#include "../include/term.h"


// include omp only if defined
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

using std::ostream, std::string, std::vector, std::map, std::unordered_map, std::shared_ptr, std::make_shared,
        std::set, std::unordered_set, std::pair, std::make_pair, std::to_string, std::invalid_argument,
        std::stringstream, std::cout, std::endl, std::flush, std::max, std::min, std::unordered_map, std::unordered_set;

namespace pdaggerq {

    string PQGraph::str() {

        stringstream sout; // string stream to hold output

        // add banner for PQ GRAPH results
        string h1, h2; // header 1 and header 2 padding
        if (Vertex::print_type_ == "python") {
            h1 = "####################";
            h2 = "#####";
        } else if (Vertex::print_type_ == "c++") {
            h1 = "///////////////////";
            h2 = "/////";
        } else throw invalid_argument("Invalid print type: " + Vertex::print_type_);
        
        sout << h1 << " PQ GRAPH Output " << h1 << endl << endl;
        
        PQGraph copy = this->clone(); // make a clone of pq_graph

        // if not assembled, assemble the copy
        if (!copy.is_assembled_)
            copy.assemble();

        // reindex intermediates in the copy
        copy.reindex();

        // get all terms from all equations except the scalars, and reuse_tmps
        vector<Term> all_terms;

        for (auto &[eq_name, equation] : copy.equations_) { // iterate over equations in serial

            // skip "temp" equation
            if (eq_name == "temp" || eq_name == "scalar" || eq_name == "reused")
                continue;

            vector<Term> &terms = equation.terms();

            if (terms.empty())
                continue;

//            if (!equation.is_temp_equation_) {
//                has_tmps = true;
//                continue; // skip tmps equation
//            }

            equation.rearrange(); // sort tmps in equation

            // find first term without a tmp on the rhs, make it an assigment, and bring it to the front
            for (size_t i = 0; i < terms.size(); ++i) {
                bool has_tmp = false;
                for (const auto &op : terms[i].rhs()) {
                    if (op->is_temp()) {
                        if (!op->is_scalar() && !op->is_reused()) {
                            has_tmp = true;
                            break;
                        }
                    }
                }
                if (!has_tmp) {
                    std::swap(terms[i], terms[0]);
                    break;
                }
            }

            // make first term an assignment
            terms[0].is_assignment_ = true;

            for (auto &term : terms) {
                    all_terms.push_back(term.clone());
            }
        }

        // make set of all unique base names (ignore linkages and scalars)
        set<string> names;
        for (const auto &term: all_terms) {
            ConstVertexPtr lhs = term.lhs();
            if (!lhs->is_linked() && !lhs->is_scalar())
                names.insert(lhs->name());
            for (const auto &op: term.rhs()) {
                if (!op->is_linked() && !op->is_scalar())
                    names.insert(op->name());
            }
        }

        // add tmp declarations
        names.insert("perm_tmps");
        names.insert("tmps");

        // declare a map for each base name
        sout << h2 << " Declarations " << h2 << endl << endl;
        for (const auto &name: names) {
            if (Vertex::print_type_ == "c++")
                 sout << "// initialize -> ";
            else if (Vertex::print_type_ == "python") 
                sout << "## initialize -> ";
            
            sout << name << ";" << endl;
        }
        sout << endl;

        // add scalar terms to the beginning of the equation

        // create merged equation to sort tmps
        Equation merged_eq = Equation("", all_terms);
        merged_eq.rearrange("temp"); // sort tmps in merged equation
        all_terms = merged_eq.terms(); // get sorted terms

        // print scalar declarations
        if (!copy.equations_["scalar"].empty()) {
            sout << h2 << " Scalars " << h2 << endl << endl;
            copy.equations_["scalar"].rearrange("scalar"); // sort scalars in scalars equation
            for (auto &term: copy.equations_["scalar"])
                term.comments() = {}; // remove comments from scalars

            // print scalars
            sout << copy.equations_["scalar"] << endl;
            sout << h2 << " End of Scalars " << h2 << endl << endl;
        }

        // print declarations for reuse_tmps
        if (!copy.equations_["reused"].empty()){
            sout << h2 << " Shared  Operators " << h2 << endl << endl;
            copy.equations_["reused"].rearrange("reused"); // sort reuse_tmps in reuse_tmps equation
            for (auto &term: copy.equations_["reused"])
                term.comments() = {}; // remove comments from reuse_tmps

            // print reuse_tmps
            sout << copy.equations_["reused"] << endl;
            sout << h2 << " End of Shared Operators " << h2 << endl << endl;
        }

        // for each term in tmps, add the term to the merged equation
        // where each tmp of a given id is first used
        copy.equations_["temp"].rearrange("temp"); // sort tmps in tmps equation

        auto &tempterms = copy.equations_["temp"];
        std::stable_sort(tempterms.begin(), tempterms.end(), [](const Term &a, const Term &b) {
            return as_link(a.lhs())->id_ < as_link(b.lhs())->id_;
        });

        // keep track of tmp ids that have been found
        set<long> declare_ids;

        // add declaration for each tmp
        bool found_any;
        size_t attempts = 0;

        do {
            found_any = false;
            size_t last_pos_idx = 0;
            for (long k = ((long)tempterms.size())-1; k >= 0; --k) {
                auto &tempterm = copy.equations_["temp"][k];

                if (!tempterm.lhs()->is_temp()) continue;

                ConstLinkagePtr temp = as_link(tempterm.lhs());
                long temp_id = temp->id();

                // check if tmp is already declared
                if (declare_ids.find(temp_id) != declare_ids.end()) continue;

                bool found = false;
                for (auto i = 0ul; i < all_terms.size(); ++i) {
                    const Term &term = all_terms[i];

                    // check if tmp id is in the rhs of the term
                    auto [lhs_ids, rhs_ids, term_ids] = term.term_ids('t');
                    found = term_ids.find(temp_id) != term_ids.end();

                    if (!found) continue; // tmp not found in rhs of term; continue
                    else {
                        // add tmp term before this term
                        auto last_pos = all_terms.begin() + (int) i;
                        last_pos_idx = i;
                        all_terms.insert(last_pos, tempterm);
                        declare_ids.insert(temp_id); // add tmp id to set

                        // indicate that a tmp was found
                        found_any = true;
                        break;
                    }
                }
                if (!found) {
                    // add tmp term to the last used position if not found
                    all_terms.insert(all_terms.begin() + (int)last_pos_idx, tempterm);
                    declare_ids.insert(temp_id); // add tmp id to set
                    found_any = true;
                }
            }
        } while (found_any && ++attempts < equations_["temp"].size());


        // add a term to destroy the tmp after its last use
        auto make_destructor = [](const Term &tempterm, const ConstLinkagePtr &temp) -> Term {
            // create vertex with only the linkage's name
            string newname;
            string lhs_name = temp->str(true, false);

            if (Vertex::print_type_ == "python")
                newname = "del " + lhs_name;
            else if (Vertex::print_type_ == "c++")
                newname = lhs_name + ".~TArrayD();";

            Term newterm(tempterm);
            newterm.print_override_ = newname;
            return newterm;
        };

        set<long> destroy_ids;
        map<size_t, vector<Term>, std::greater<>> destruct_terms;
        for (auto &tempterm: copy.equations_["temp"]) {
            if (!tempterm.lhs()->is_temp()) continue;

            ConstLinkagePtr temp = as_link(tempterm.lhs());
            long temp_id = temp->id();

            // determine if tmp was already inserted
            bool inserted = destroy_ids.find(temp_id) != destroy_ids.end();
            if (inserted) continue;

            bool found = false;
            long declare_pos = -1;
            for (auto i = (long int) all_terms.size() - 1; i >= 0; --i) {
                const Term &term = all_terms[i];

                // check if tmp is the lhs of the term
                if (term.lhs()->same_temp(temp)) { declare_pos = i; break; }

                string term_str = term.str();
                if (term.lhs()->is_temp()) {
                    bool test = true;
                }

                // check if tmp is in the rhs of the term
                for (const auto &op : term.rhs()) {
                    string op_str = op->str();
                    if (op->is_linked() && as_link(op)->has_temp(temp, false)) {
                        found = true; break;
                    }
                }
                if (!found) continue;

                destroy_ids.insert(temp_id); // add tmp id to set

                // Create new term with tmp in the lhs and assign zero to the rhs
                Term destruct_term = make_destructor(tempterm, temp);

                // add tmp term after this term
//                all_terms.insert(all_terms.begin() + (int) i + 1, destruct_term);
                destruct_terms[i].push_back(destruct_term);
                break; // only add once
            }

            if (!found) {
                // if a match is still not found, add after the declare term
                if (declare_pos < 0) continue; // tmp not found in any term; continue

                // add temp id
                destroy_ids.insert(temp_id);

                // create the destructor term
                Term destruct_term = make_destructor(tempterm, temp);

                // add tmp term after this term
//                all_terms.insert(all_terms.begin() + (int) declare_pos + 1, destruct_term);
                destruct_terms[declare_pos].push_back(destruct_term);
            }
        }

        // add destructor terms to all_terms
        for (const auto &[idx, terms] : destruct_terms) {
            for (const auto &term : terms) {
                all_terms.insert(all_terms.begin() + (int) idx + 1, term);
            }
        }

        // get difference of declare_ids and destroy_ids
        set<long> missing_ids;
        set_difference(declare_ids.begin(), declare_ids.end(), destroy_ids.begin(), destroy_ids.end(),
                       inserter(missing_ids, missing_ids.begin()));

        bool found_all_tmp_ids = missing_ids.empty();
        if (!found_all_tmp_ids) {
            cout << "WARNING: could not find last use of tmps with ids: ";
            for (long id : missing_ids) {
                cout << id << " ";
            }
            cout << endl;
        }

        sout << h1 << " Evaluate Equations " << h1 << endl << endl;

        // update terms in merged equation
        merged_eq.terms() = all_terms;

        // stream merged equation as string
        sout << merged_eq << endl;

        // add closing banner
        sout << h1 << h1 << h1 << endl << endl;

        // return string stream as string
        return sout.str();

    }

    void PQGraph::print(string print_type) {

        constexpr auto to_lower = [](string str) {
            // map uppercase to lowercase for output
            for (auto &letter : str) {
                static unordered_map<char, char>
                        lowercase_map = {{'A', 'a'}, {'B', 'b'}, {'C', 'c'}, {'D', 'd'}, {'E', 'e'},
                                         {'F', 'f'}, {'G', 'g'}, {'H', 'h'}, {'I', 'i'}, {'J', 'j'},
                                         {'K', 'k'}, {'L', 'l'}, {'M', 'm'}, {'N', 'n'}, {'O', 'o'},
                                         {'P', 'p'}, {'Q', 'q'}, {'R', 'r'}, {'S', 's'}, {'T', 't'},
                                         {'U', 'u'}, {'V', 'v'}, {'W', 'w'}, {'X', 'x'}, {'Y', 'y'},
                                         {'Z', 'z'}};

                if (lowercase_map.find(letter) != lowercase_map.end())
                    letter = lowercase_map[letter];
            }

            // return lowercase string
            return str;
        };

        print_type = to_lower(print_type);

        if (print_type == "python" || print_type == "einsum") {
            Vertex::print_type_ = "python";
            cout << "Formatting equations for python" << endl;
        } else if (print_type == "c++" || print_type == "cpp") {
            Vertex::print_type_ = "c++";
            cout << "Formatting equations for c++" << endl;
        } else {
            cout << "WARNING: output must be one of: python, einsum, c++, or cpp" << endl;
            cout << "         Setting output to c++" << endl; // TODO: make default for python
        }
        cout << endl;

        // print output to stdout
        cout << this->str() << endl;
    }

    vector<string> PQGraph::to_strings() {
        string tastring = str();
        vector<string> eq_strings;

        // split string by newlines
        string delimiter = "\n";
        string token;
        size_t pos = tastring.find(delimiter);
        while (pos != string::npos) {
            token = tastring.substr(0, pos);
            eq_strings.push_back(token);
            tastring.erase(0, pos + delimiter.length());
            pos = tastring.find(delimiter);
        }

        return eq_strings;
    }

    std::vector<string> Equation::to_strings() const {

        std::vector<string> output;

        // set of conditions already found. Used to avoid printing duplicate conditions
        std::set<string> current_conditions = {};
        bool closed_condition = true; // whether the current condition has been closed
        for (const auto & term : terms_) { // iterate over terms


            // check if condition is already printed
            std::set<string> conditions = term.conditions();
            if (conditions != current_conditions) { // if conditions are different, print new condition

                bool has_condition = !conditions.empty();
                bool had_condition = !current_conditions.empty();

                if (had_condition && !closed_condition) {
                    // if the previous condition was not closed, close it
                    if (Vertex::print_type_ == "c++")
                        output.emplace_back("}");
                    closed_condition = true; // indicate that the condition is closed
                }

                if (has_condition) {
                    closed_condition = false; // set condition to be closed

                    string if_block = condition_string(conditions);
                    output.push_back(if_block);
                }

                // set current conditions
                current_conditions = conditions;

            } // else do nothing

            // if override is set, print override
            bool override = !term.print_override_.empty();
            if (override) {
                string padding = !conditions.empty() ? "    " : "";
                output.push_back(padding + term.print_override_);
                continue;
            }

            string padding = !conditions.empty() ? "    " : "";
            string extra_padding = padding + "    ";

            // add comments
            bool temp_decl = term.lhs()->is_temp();
            string comment = term.make_comments(temp_decl);
            if (!comment.empty()) {

                comment.insert(0, extra_padding); // add newline to the beginning of the comment

                // replace all '\"' with '' in comment
                size_t pos = 0;
                while ((pos = comment.find('\"', pos)) != string::npos) {
                    comment = comment.replace(pos, 1, "");
                    pos += 1;
                }

                // replace all "\n" with "\n    " in comment
                pos = 0;
                while ((pos = comment.find('\n', pos)) != string::npos) {
                    comment = comment.replace(pos, 1, '\n' + padding);
                    pos += 1;
                }

                output.push_back("\n" + comment); // add comment
            }

            // get string representation of term

            string term_string;
            term_string += padding + term.str();

            // replace all "\n" with "\n    " in term_string
            size_t pos = 0;
            while ((pos = term_string.find('\n', pos)) != string::npos) {
                term_string = term_string.replace(pos, 1, "\n" + extra_padding);
                pos += 1;
            }

            output.push_back(term_string);
        }

        if (!closed_condition && Vertex::print_type_ == "c++" && !current_conditions.empty()) {
            // if the final condition was not closed, close it
            output.emplace_back("}");
        }

        return output;
    }

    string Equation::condition_string(std::set<string> &conditions) {

        // if no conditions, return empty string
        if (conditions.empty()) return "";

        string if_block;
        if (Vertex::print_type_ == "c++") {
            if_block = "if (";
            for (const string &condition: conditions)
                if_block += "includes_[\"" + condition + "\"] && ";
            if_block.resize(if_block.size() - 4);
            if_block += ") {";
        } else if (Vertex::print_type_ == "python") {
            if_block = "if ";
            for (const string &condition: conditions)
                if_block += "includes_[\"" + condition + "\"] and ";
            if_block.resize(if_block.size() - 5);
            if_block += ":";
        }
        return "\n    " + if_block;
    }

    string Term::str() const {

        if (!print_override_.empty())
            // return print override if it exists for custom printing
            return print_override_;

        string output;

        bool has_permutations = !term_perms_.empty() && perm_type_ != 0;
        if (has_permutations) { // if there are permutations

            // make intermediate vertex for the permutation
            VertexPtr perm_vertex;

            bool make_perm_tmp = rhs_.size() > 1; // if there is only one vertex, no need to create intermediate vertex

            if (make_perm_tmp) {
                // if this is a linkage, but not a temp, also make a temporary vertex (doesn't print as a single vertex)
                if (rhs_[0]->is_linked() && !rhs_[0]->is_temp()) {
                    make_perm_tmp = true;
                }
            }

            if (!make_perm_tmp) perm_vertex = rhs_[0]->clone(); // no need to create intermediate vertex if there is only one
            else { // else, create the intermediate vertex and its assignment term
                perm_vertex = lhs_->clone();
                string perm_name;

                // format permutation vertex to print as map
                perm_vertex->map_string_ = "[\"perm_" + lhs_->base_name() + "\"]";

                perm_vertex->sort(); // sort permutation vertex
                perm_vertex->update_name("tmps_"); // set name of permutation vertex

                // initialize initial permutation term
                Term perm_term = *this; // copy term
                perm_term.lhs_ = perm_vertex; // set lhs to permutation vertex
                perm_term.reset_perm();
                perm_term.is_assignment_ = true; // set term as assignment
                perm_term.coefficient_ = fabs(coefficient_); // set coefficient to absolute value of coefficient

                // add string to output
                output += perm_term.str();
                output += "\n";

            } // if only one vertex, use that vertex directly

            // initialize term to permute
            Term perm_term = *this; // copy term
            perm_term.rhs_ = {perm_vertex};
            perm_term.compute_scaling(true);  // recomputes scaling

            // remove comments from term
            perm_term.comments_.clear();

            // if more than one vertex, set coefficient to 1 or -1
            if (!make_perm_tmp)
                perm_term.coefficient_ = coefficient_ > 0 ? 1 : -1;

            // get permuted terms
            std::vector<Term> perm_terms = perm_term.expand_perms();

            // add permuted terms to output
            for (auto &permuted_term: perm_terms) {
                output += permuted_term.str();
                output += '\n';
            }

            // if an intermediate vertex was created, delete it
            if (make_perm_tmp) {
                // delete the permutation vertex
                if (Vertex::print_type_ == "c++")
                    output += perm_vertex->name() + ".~TArrayD();";
                else if (Vertex::print_type_ == "python")
                    output += "del " + perm_vertex->name() + ";";
                output += "\n";
            }

            if (!perm_terms.empty())
                output.pop_back(); // remove last newline character

            return output;
        }

        // if no permutations, continue with normal term printing

        // expand additions into separate terms
        ConstLinkagePtr term_link = term_linkage();
        if (term_link->is_addition() && !term_link->is_temp()) {
            Term left_term = *this, right_term = *this;
            ConstVertexPtr left_vertex = term_link->left();
            ConstVertexPtr right_vertex = term_link->right();

            left_term.expand_rhs(left_vertex); // expand left term
            right_term.expand_rhs(right_vertex); // expand right term

            // merge constants in right term and compute scaling. right term is not an assignment
            right_term.is_assignment_ = false;
            right_term.compute_scaling(true);

            return left_term.str() + '\n' + right_term.str();
        }

        if (Vertex::print_type_ == "python")
            return einsum_str();

        // get lhs vertex string
        output = lhs_->str();

        // get sign of coefficient
        bool is_negative = coefficient_ < 0;
        if (is_assignment_) output += "  = ";
        else if (is_negative) output += " -= ";
        else output += " += ";

        // get absolute value of coefficient
        double abs_coeff = fabs(coefficient_);

        // if the coefficient is not 1, add it to the string
        bool is_empty = rhs_.empty() || term_link->empty();

        bool added_coeff = false;
        bool negative_assignment = (is_assignment_ && is_negative);
        bool needs_coeff = fabs(abs_coeff - 1) >= 1e-8 || is_empty || negative_assignment;

        if (needs_coeff) {
            // add coefficient to string
            added_coeff = true;
            if (negative_assignment)
                output += "-";

            int precision = minimum_precision(abs_coeff);
            output += to_string_with_precision(abs_coeff, precision);

            // add multiplication sign if there are rhs vertices
            if (!is_empty)
                output += " * ";
        }

        output += term_link->str();

        // ensure the last character is a semicolon (might not be there if no rhs vertices)
        if (output.back() != ';' && Vertex::print_type_ == "c++")
            output += ';';

        size_t pos = 0;
        while (pos != string::npos) {
            pos = output.find("* 1.00 *", pos);
            if (pos != string::npos) {
                output = output.replace(pos, 8, "*");
                pos += 1;
            }
        }

        return output;
    }

    string Term::einsum_str() const {
        string output;

        // get left hand side vertex name
        if (lhs_->is_linked())
             output = as_link(lhs_)->str(true, false);
        else output = lhs_->name();

        // get sign of coefficient
        bool is_negative = coefficient_ < 0;
        if (is_assignment_) output += "  = ";
        else if (is_negative) output += " -= ";
        else output += " += ";

        // get absolute value of coefficient
        double abs_coeff = fabs(coefficient_);

        // if the coefficient is not 1, add it to the string
        bool needs_coeff = fabs(abs_coeff - 1) >= 1e-8 || rhs_.empty() || is_assignment_;

        if (needs_coeff) {
            // add coefficient to string
            if (is_assignment_ && is_negative)
                output += "-";

            int precision = minimum_precision(abs_coeff);
            output += to_string_with_precision(abs_coeff, precision);

            // add multiplication sign if there are rhs vertices
            if (!rhs_.empty())
                output += " * ";
        }

        // get string of lines
        line_vector link_lines;
        string lhs_string;

        // get string of lines from lhs vertex
        for (auto & line : lhs_->lines())
            lhs_string += line.label_.front();

        string rhs_string;

        // get string of lines from the term linkage
        for (auto & line : term_linkage(true)->lines())
            rhs_string += line.label_.front();

        // make einsum string
        string einsum_string;

        // get einsum string from term linkage
        einsum_string = term_linkage(true)->str();

        // permute tensors if needed
        if (lhs_string != rhs_string) {
            einsum_string = "einsum('" + rhs_string + "->" + lhs_string + "', " + einsum_string + " )";
        }
        output += einsum_string;

        // formatting issue needs to replace "* 1.00 *" with "*"
        size_t pos = 0;
        while (pos != string::npos) {
            pos = output.find("* 1.00 *", pos);
            if (pos != string::npos) {
                output = output.replace(pos, 8, "*");
                pos += 1;
            }
        }

        return output;
    }

    string Term::make_comments(bool only_flop, bool only_comment) const {
        if (comments_.empty())
            return "";

        string comment;
        for (const auto &vertex: rhs_) {
            if (vertex->is_linked())
                comment += as_link(vertex)->tot_str(true);
            else
                comment += vertex->str();
            if (vertex != rhs_.back())
                comment += " * ";
        }

        // add permutations to comment if there are any
        if (!term_perms_.empty()) {
            string perm_str;
            int count;
            switch (perm_type_) {
                case 0:
                    break;
                case 1:
                    for (const auto &perm: term_perms_)
                        perm_str += "P(" + perm.first + "," + perm.second + ") ";
                    break;
                case 2:
                    count = 0;
                    for (const auto &perm: term_perms_) {
                        if (count++ % 2 == 0)
                            perm_str += "PP2(" + perm.first + "," + perm.second;
                        else
                            perm_str += ";" + perm.first + "," + perm.second + ") ";
                    }
                    break;
                case 3:
                    count = 0;
                    for (const auto &perm: term_perms_) {
                        if (count % 3 == 0)
                            perm_str += "PP3(" + perm.first + "," + perm.second;
                        else
                            perm_str += ";" + perm.first + "," + perm.second;
                        if (count++ % 3 == 2)
                            perm_str += ") ";
                    }
                    break;
                case 6:
                    count = 0;
                    for (const auto &perm: term_perms_) {
                        if (count % 3 == 0)
                            perm_str += "PP6(" + perm.first + "," + perm.second;
                        else
                            perm_str += ";" + perm.first + "," + perm.second;
                        if (count++ % 3 == 2)
                            perm_str += ") ";
                    }
                    break;
            }

            comment = perm_str + comment;
        }

        // get coefficient
        double coeff = coefficient_;
        bool is_negative = coeff < 0;

        string assign_str = is_assignment_ ? "  = " : " += ";

        int precision = minimum_precision(coefficient_);
        string coeff_str = to_string_with_precision(fabs(coefficient_), precision);
        if (is_negative) coeff_str.insert(coeff_str.begin(), '-');
        coeff_str += ' ';

        if (original_pq_.empty()) {
            // add lhs to comment
            comment = "// " + lhs_->str() + assign_str + coeff_str + comment;
        } else {
            comment = "// " + lhs_->name() + assign_str + original_pq_;
        }

        if (only_flop) // clear comment if only flop is requested
            comment.clear();

        // remove all quotes from comment
        comment.erase(std::remove(comment.begin(), comment.end(), '\"'), comment.end());

        // format comment for python if needed
        if (Vertex::print_type_ == "python"){
            // turn '//' into '#'
            size_t pos = comment.find("//");
            while (pos != std::string::npos) {
                comment.replace(pos, 2, "#");
                pos = comment.find("//", pos + 2);
            }
        }

        const auto &[flop_scales, mem_scales] = term_linkage(true)->scales();
        only_comment |= flop_scales.empty() && mem_scales.empty();
        if (only_comment) return comment;

        if (!comment.empty()) comment += "\n    ";
        comment += "// flops: " + lhs_->dim().str() + assign_str;
        for (const auto & flop : flop_scales)
            comment += flop.str() + " ";

        // remove last space
        if (!flop_scales.empty())
            comment.pop_back();


        comment += "\n    //  mems: " + lhs_->dim().str() + assign_str;
        for (const auto & mem : mem_scales)
            comment += mem.str() + " ";

        // remove last space
        if (!mem_scales.empty())
            comment.pop_back();

        // format comment for python if needed
        if (Vertex::print_type_ == "python"){
            // turn '//' into '#'
            size_t pos = comment.find("//");
            while (pos != std::string::npos) {
                comment.replace(pos, 2, "#");
                pos = comment.find("//", pos + 2);
            }
        }

        return comment;
    }

    void Term::reset_comments() {
        // set comments
        comments_.push_back(to_string(coefficient_)); // add coefficient to vertex strings
        for (const auto &op : rhs_)
            comments_.push_back(op->str());
    }

    std::set<string> Term::conditions() const {

        // map that stores conditions to their related operators
        const std::map<string, std::vector<string>> &mapped_conditions = mapped_conditions_;
        if (mapped_conditions.empty()) return {}; // return empty set if no conditions

        ConstLinkagePtr linkage = term_linkage(); // get linkage representation of term
        std::set<string> conditions{}; // set to store conditions

        if (!linkage) {
            // return current conditions if no linkage
            if (rhs_.empty())
                return conditions;

            // if rhs is not empty, create a new term and get its linkage
            Term new_term = *this;
            new_term.compute_scaling(true); // force recomputation of scaling
            linkage = new_term.term_linkage(); // get linkage
        }

        if (!linkage)
            return conditions; // return current conditions if no linkage

        // create a set of operator basenames
        std::vector<ConstVertexPtr> vertices = linkage->vertices();

        // include lhs vertex in conditions (if it exists)
        if (lhs_) vertices.push_back(lhs_);
        for (const auto & vertex : vertices) {
            if (!vertex) continue; // skip if vertex is null

            // loop over named conditions
            for (const auto & [condition, restrict_ops] : mapped_conditions) {
                // check if vertex is in the list of operators
                if (std::find(restrict_ops.begin(), restrict_ops.end(), vertex->base_name()) != restrict_ops.end())
                    conditions.insert(condition); // if so, add named condition to set
            }
        }

        // return set of operator basenames that have conditions
        return conditions;
    }

}