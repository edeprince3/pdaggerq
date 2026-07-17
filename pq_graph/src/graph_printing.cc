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
#include <numeric>
#include <queue>
#include <map>
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

#include "../include/pq_graph.h"
#include "../include/term.h"
#include "../include/printers/code_printer.h"


// include omp only if defined
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_set_num_threads(n) 1
#endif

using std::ostream, std::string, std::vector, std::map, std::unordered_map, std::shared_ptr, std::make_shared,
        std::set, std::unordered_set, std::pair, std::make_pair, std::to_string, std::invalid_argument,
        std::stringstream, std::cout, std::endl, std::flush, std::max, std::min, std::unordered_map, std::unordered_set;

namespace pdaggerq {

    void reset_ir_hoist_counter(); // defined with the counter, below

    string PQGraph::str(const string &print_type) const {

        // The einsums "ir" (JSONL) export is NOT a CodePrinter backend: it emits JSON, not
        // formatted source. It reuses all of the scaffolding below (clone/assemble/reindex,
        // temp ordering, term collection); only the PER-TERM emission differs, and that is
        // handled in Term::str via Vertex::ir_mode_. printer_ is pointed at the Einsum
        // (python) backend so the banners/declarations come out '#'-commented -- lines the
        // IR parser ignores -- while each term is emitted as a JSON object by ir_term_str.
        string print_type_lc = print_type;
        for (auto &c : print_type_lc) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');

        struct IRModeGuard { ~IRModeGuard() { Vertex::ir_mode_ = false; } } ir_mode_guard;
        Vertex::ir_mode_ = (print_type_lc == "ir");

        if (Vertex::ir_mode_) {
            Vertex::set_printer("python");
            reset_ir_hoist_counter(); // fresh irN names per export (the counter is otherwise
                                      // monotonic across the process)
        } else {
            Vertex::set_printer(print_type);
        }

        stringstream sout; // string stream to hold output

        // add banner for PQ GRAPH results
        const CodePrinter* printer = Vertex::printer_;
        
        sout << printer->format_named_section(" PQ GRAPH Output ", true);
        
        PQGraph copy = this->clone(); // make a clone of pq_graph

        // if not assembled, assemble the copy
        if (!copy.is_assembled_)
            copy.assemble();

        // reindex intermediates in the copy
        copy.reindex();

        // get all terms from all equations except the scalars, and reuse_tmps
        vector<Term> all_terms;

        // make set of all unique base names (ignore linkages and scalars)
        set<string> names;

        for (auto &[eq_name, equation] : copy.equations_) { // iterate over equations in serial

            vector<Term> &terms = equation.terms();

            for (const auto &term: terms) {
                VertexPtr lhs = term.lhs();
                if (!lhs->is_linked() && !lhs->is_constant())
                    names.insert(lhs->name());
                else if (lhs->is_temp())
                    names.insert(as_link(lhs)->str(true, false));

                for (const auto &op: term.rhs()) {
                    if (!op->is_linked() && !op->is_constant())
                        names.insert(op->name());
                    else if (op->is_linked()){
                        if (op->is_temp())
                            names.insert(as_link(op)->str(true, false));

                        vertex_vector vertices = as_link(op)->vertices();
                        for (const auto &vertex: vertices)
                            if (!vertex->is_linked() && !vertex->is_constant())
                                names.insert(vertex->name());
                    }
                }

                auto temps = as_link(lhs + term.term_linkage())->get_temps(true, false);
                for (const auto &temp: temps)
                    names.insert(as_link(temp)->str(true, false));

            }

            // skip "temp" equation
            if (eq_name == "temp" || eq_name == "scalar" || eq_name == "reused")
                continue;

            if (terms.empty()) continue;

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

        // declare a map for each base name
        sout << printer->format_named_section(" Declarations ", false);
        sout << printer->format_declarations(names);
        sout << endl;

        // add scalar terms to the beginning of the equation

        // create merged equation to sort tmps
        Equation merged_eq = Equation("", all_terms);
        merged_eq.rearrange("temp"); // sort tmps in merged equation
        all_terms = merged_eq.terms(); // get sorted terms

        // print scalar declarations
        if (!copy.equations_["scalar"].empty()) {
            sout << printer->format_named_section(" Scalars ", false);

            // sort scalars in scalars equation
            vector<Term> scalar_terms = copy.equations_["scalar"].terms();
            std::sort(scalar_terms.begin(), scalar_terms.end(), [](const Term &a, const Term &b) {
                return a.max_id("scalar") < b.max_id("scalar");
            });
            copy.equations_["scalar"].terms() = scalar_terms;
            copy.equations_["scalar"].collect_scaling(true);

            for (auto &term: copy.equations_["scalar"])
                term.comments() = {}; // remove comments from scalars

            // print scalars
            sout << copy.equations_["scalar"] << endl;
            sout << printer->format_named_section(" End of Scalars ", false);
        }

        // print declarations for reuse_tmps
        if (!copy.equations_["reused"].empty()){
            sout << printer->format_named_section(" Shared  Operators ", false);

            // sort reused in reused equation
            vector<Term> reused_terms = copy.equations_["reused"].terms();
            std::sort(reused_terms.begin(), reused_terms.end(), [](const Term &a, const Term &b) {
                return a.max_id("reused") < b.max_id("reused");
            });
            copy.equations_["reused"].terms() = reused_terms;
            copy.equations_["reused"].collect_scaling(true);

            for (auto &term: copy.equations_["reused"])
                term.comments() = {}; // remove comments from reuse_tmps

            // print reuse_tmps
            sout << copy.equations_["reused"] << endl;
            sout << printer->format_named_section(" End of Shared Operators ", false);
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

                LinkagePtr temp = as_link(tempterm.lhs());
                long temp_id = temp->id();

                // check if tmp is already declared
                if (declare_ids.find(temp_id) != declare_ids.end()) continue;

                bool found = false;
                for (auto i = 0ul; i < all_terms.size(); ++i) {
                    const Term &term = all_terms[i];

                    // check if tmp id is in the rhs of the term
                    idset term_ids = term.term_ids("temp");
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

                        // We need to allocate the tmp when using c++ or cpp
                        if (Term::deallocate_) {
                            string allocatename = printer->allocate(temp->str(true, false));
                            if (!allocatename.empty()) {
                                Term allocateterm(tempterm);
                                allocateterm.print_override_ = allocatename;
                                all_terms.insert(all_terms.begin() + (int) last_pos_idx, allocateterm);
                            }
                        }

                        break;
                    }
                }
                if (!found) {
                    // add tmp term to the last used position if not found
                    all_terms.insert(all_terms.begin() + (int)last_pos_idx, tempterm);
                    declare_ids.insert(temp_id); // add tmp id to set
                    found_any = true;

                    // We need to allocate the tmp when using c++ or cpp
                    if (Term::deallocate_) {
                        string allocatename = printer->allocate(temp->str(true, false));
                        if (!allocatename.empty()) {
                            Term allocateterm(tempterm);
                            allocateterm.print_override_ = allocatename;
                            all_terms.insert(all_terms.begin() + (int) last_pos_idx, allocateterm);
                        }
                    }

                }
            }
        } while (found_any && ++attempts < copy.equations_["temp"].size());


        // add a term to destroy the tmp after its last use
        auto make_destructor = [&printer](const Term &tempterm, const LinkagePtr &temp) -> Term {
            string newname = printer->deallocate(temp->str(true, false));

            Term newterm(tempterm);
            newterm.print_override_ = newname;
            return newterm;
        };

        // destructor insertion frees each tmp after its last use (python `del` /
        // c++ `~TArrayD()`); the IR has no such notion, so skip it for "ir".
        if (!Vertex::ir_mode_) {
        set<long> destroy_ids;
        map<size_t, vector<Term>, std::greater<>> destruct_terms;
        for (auto &tempterm: copy.equations_["temp"]) {
            if (!tempterm.lhs()->is_temp()) continue;

            LinkagePtr temp = as_link(tempterm.lhs());
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

                // check if tmp is in the rhs of the term
                // (this loop previously built a full term.str()/op->str() rendering
                // per iteration and discarded it -- pure wasted work in an O(n^2) scan)
                for (const auto &op : term.rhs()) {
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
        if (Term::deallocate_) {
            for (const auto &[idx, terms]: destruct_terms) {
                for (const auto &term: terms) {
                    all_terms.insert(all_terms.begin() + (int) idx + 1, term);
                }
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
        } // end destructor insertion (skipped for "ir")

        sout << printer->format_named_section(" Evaluate Equations ", true);

        // update terms in merged equation
        merged_eq.terms() = all_terms;

        // stream merged equation as string
        sout << merged_eq << endl;

        // add closing banner
        sout << printer->format_closing_banner();

        // return string stream as string
        return sout.str();

    }

    void PQGraph::print(const string &print_type) const {
        // print output to stdout
        cout << this->str(print_type) << endl;
    }

    vector<string> PQGraph::to_strings(const string &print_type) const {
        string tastring = str(print_type);
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
                    const string closer = Vertex::printer_->condition_closer();
                    if (!closer.empty()) output.emplace_back(closer);
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

            int indent = !conditions.empty() ? 1 : 0;

            // if override is set, print override
            bool override = !term.print_override_.empty();
            if (override) {
                output.push_back(Vertex::printer_->padding(indent) + term.print_override_);
                continue;
            }

            // add comments
            string comment = Vertex::printer_->format_comment(
                term.make_comments(term.lhs()->is_temp()), indent);
            if (!comment.empty()) output.push_back(comment);

            // add term line
            output.push_back(Vertex::printer_->format_term_line(term.str(), indent));
        }

        if (!closed_condition && !current_conditions.empty()) {
            // if the final condition was not closed, close it
            const string closer = Vertex::printer_->condition_closer();
            if (!closer.empty()) output.emplace_back(closer);
        }

        return output;
    }

    ostream &operator<<(ostream &os, const Equation &eq) {
        for (const string &s : eq.to_strings())
            os << Vertex::printer_->padding(1) << s << endl;
        return os;
    }

    string Equation::condition_string(std::set<string> &conditions) {
        if (conditions.empty()) return "";
        return Vertex::printer_->condition_open(conditions);
    }

    // every line appearing anywhere in a statement -- the lhs plus the whole rhs linkage
    // tree, contracted lines included, since those get subscripts too. Order is
    // deterministic (lhs first, then a breadth-first walk), which is what makes the
    // resulting subscript assignment reproducible.
    static line_vector statement_lines(const VertexPtr &lhs, const VertexPtr &rhs) {
        line_vector lines;
        if (lhs) lines.insert(lines.end(), lhs->lines().begin(), lhs->lines().end());
        if (!rhs) return lines;

        std::queue<VertexPtr> queue;
        queue.push(rhs);
        while (!queue.empty()) {
            VertexPtr cur = queue.front(); queue.pop();
            if (!cur) continue;
            lines.insert(lines.end(), cur->lines().begin(), cur->lines().end());
            if (cur->is_linked()) {
                queue.push(as_link(cur)->left());
                queue.push(as_link(cur)->right());
            }
        }
        return lines;
    }

    // Self-contained IR (einsums JSONL) emission for one term, used when Vertex::ir_mode_ is
    // set. Mirrors Term::str's structure but emits JSON objects (via ir_str) and, for an
    // antisymmetrized term, annotates every expanded statement with the permutation that
    // generated it. Recursion goes back through str(), which re-enters here in ir_mode_.
    string Term::ir_term_str() const {

        // give each distinct label in this statement its own subscript (see line.hpp).
        Line::SubscriptScope subscripts(statement_lines(lhs_, term_linkage(true)));

        string output;

        bool has_permutations = !term_perms_.empty() && perm_type_ != 0;
        if (has_permutations) {
            // carry the antisymmetrizer that generated the expanded statements so a consumer
            // can recognize "target is (anti)symmetrized under P(...)".
            string perm_json = "\"perm\":{\"type\":" + std::to_string(perm_type_) + ",\"pairs\":[";
            bool first = true;
            for (const auto &p : term_perms_) {
                if (!first) perm_json += ",";
                first = false;
                perm_json += "[\"" + p.first + "\",\"" + p.second + "\"]";
            }
            perm_json += "]}";

            MutableVertexPtr perm_vertex;
            bool perm_as_rhs = rhs_.size() == 1;
            if (perm_as_rhs && rhs_[0]->is_linked() && !rhs_[0]->is_temp())
                perm_as_rhs = false;

            if (perm_as_rhs) perm_vertex = rhs_[0]->clone();
            else {
                perm_vertex = lhs_->clone();
                perm_vertex->vertex_type_ = 'p';
                perm_vertex->sort();
                perm_vertex->update_name("tmps_");

                Term perm_term = *this;
                perm_term.lhs_ = perm_vertex;
                perm_term.reset_perm();
                perm_term.is_assignment_ = true;
                perm_term.coefficient_ = fabs(coefficient_);
                perm_term.ir_perm_json_ = perm_json; // annotate the group's definition (IR only)

                output += perm_term.str();
                output += "\n";
            }

            Term perm_term = *this;
            perm_term.rhs_ = {perm_vertex};
            perm_term.ir_perm_json_ = perm_json; // inherited by every expanded copy (IR only)
            perm_term.compute_scaling(true);
            perm_term.comments_.clear();
            if (!perm_as_rhs)
                perm_term.coefficient_ = coefficient_ > 0 ? 1 : -1;

            std::vector<Term> perm_terms = perm_term.expand_perms();
            for (auto &permuted_term: perm_terms) {
                output += permuted_term.str();
                output += '\n';
            }
            if (!perm_terms.empty())
                output.pop_back(); // remove last newline character
            return output;
        }

        // expand additions into separate terms
        LinkagePtr term_link = term_linkage();
        if (term_link->is_addition() && !term_link->is_temp()) {
            Term left_term = *this, right_term = *this;
            left_term.expand_rhs(term_link->left());
            right_term.expand_rhs(term_link->right());
            right_term.is_assignment_ = false;
            right_term.compute_scaling(true);
            return left_term.str() + '\n' + right_term.str();
        }

        return ir_str();
    }

    string Term::str() const {

        if (!print_override_.empty())
            // return print override if it exists for custom printing
            return print_override_;

        if (Vertex::ir_mode_)
            return ir_term_str(); // self-contained IR (JSONL) emission

        // give each distinct label in this statement its own subscript; the natural
        // label->char map is not injective (see Line::natural_einsum_char).
        Line::SubscriptScope subscripts(statement_lines(lhs_, term_linkage(true)));

        string output;

        // format for permutations if any
        bool has_permutations = !term_perms_.empty() && perm_type_ != 0;
        if (has_permutations) {

            // for the IR export, carry the antisymmetrizer that generated the expanded
            // statements: every statement of this group (the unpermuted definition and
            // each permuted accumulate) is annotated with the perm type and index pairs,
            // so a consumer can recognize "target is (anti)symmetrized under P(...)"
            // instead of seeing unrelated accumulates.
            string perm_json;
            if (Vertex::ir_mode_) {
                perm_json = "\"perm\":{\"type\":" + std::to_string(perm_type_) + ",\"pairs\":[";
                bool first = true;
                for (const auto &p : term_perms_) {
                    if (!first) perm_json += ",";
                    first = false;
                    perm_json += "[\"" + p.first + "\",\"" + p.second + "\"]";
                }
                perm_json += "]}";
            }

            // make intermediate vertex for the permutation
            MutableVertexPtr perm_vertex;

            bool perm_as_rhs = rhs_.size() == 1; // if there is only one vertex, no need to create intermediate vertex

            if (perm_as_rhs) {
                // if this is a linkage, but not a temp, also make a temporary vertex (doesn't print as a single vertex)
                if (rhs_[0]->is_linked() && !rhs_[0]->is_temp()) {
                    perm_as_rhs = false;
                }
            }

            if (perm_as_rhs) perm_vertex = rhs_[0]->clone(); // no need to create intermediate vertex if there is only one
            else { // else, create the intermediate vertex and its assignment term
                perm_vertex = lhs_->clone();
                string perm_name;
                perm_vertex->vertex_type_ = 'p'; // sets printing for permutation vertex
                perm_vertex->sort(); // sort permutation vertex
                perm_vertex->update_name(Vertex::printer_->scratch_prefix('t')); // set name of permutation vertex

                // initialize initial permutation term
                Term perm_term = *this; // copy term
                perm_term.lhs_ = perm_vertex; // set lhs to permutation vertex
                perm_term.reset_perm();
                perm_term.is_assignment_ = true; // set term as assignment
                perm_term.coefficient_ = fabs(coefficient_); // set coefficient to absolute value of coefficient
                perm_term.ir_perm_json_ = perm_json; // annotate the group's definition (IR only)

                // add string to output
                output += perm_term.str();
                output += "\n";

            } // if only one vertex, use that vertex directly

            // initialize term to permute
            Term perm_term = *this; // copy term
            perm_term.rhs_ = {perm_vertex};
            perm_term.ir_perm_json_ = perm_json; // inherited by every expanded copy (IR only)
            perm_term.compute_scaling(true);  // recomputes scaling

            // remove comments from term
            perm_term.comments_.clear();

            // if more than one vertex, set coefficient to 1 or -1
            if (!perm_as_rhs)
                perm_term.coefficient_ = coefficient_ > 0 ? 1 : -1;

            // get permuted terms
            std::vector<Term> perm_terms = perm_term.expand_perms();

            // add permuted terms to output
            for (auto &permuted_term: perm_terms) {
                output += permuted_term.str();
                output += '\n';
            }

            // if an intermediate vertex was created, delete it
            if (!perm_as_rhs && Term::deallocate_) {
                string del = Vertex::printer_->perm_delete(perm_vertex->name());
                if (!del.empty()) output += del;
            }

            if (!perm_terms.empty())
                output.pop_back(); // remove last newline character

            return output;
        }

        // if no permutations, continue with normal term printing

        // expand additions into separate terms
        LinkagePtr term_link = term_linkage();
        if (term_link->is_addition() && !term_link->is_temp()) {
            Term left_term = *this, right_term = *this;
            VertexPtr left_vertex = term_link->left();
            VertexPtr right_vertex = term_link->right();

            left_term.expand_rhs(left_vertex); // expand left term
            right_term.expand_rhs(right_vertex); // expand right term

            // merge constants in right term and compute scaling. right term is not an assignment
            right_term.is_assignment_ = false;
            right_term.compute_scaling(true);

            return left_term.str() + '\n' + right_term.str();
        }

        // If user or printing method requires binarization or a multiplication of addition terms,
        // ensure only two operations within any term. create intermediates as needed.
        bool needs_binarization = Term::binarize_;
        ///TODO: uncomment line below to prevent multiple in-line additions. Do so after chronusq parser is adjusted. 
        // needs_binarization |= !term_link->is_temp() && (term_link->left()->is_addition() || term_link->right()->is_addition());
        if (needs_binarization) {

            // determine if binarization is still needed
            bool made_any_change = false;
            Term binarized_term = clone(); // copy of current term to modify
            needs_binarization = binarized_term.rhs_.size() > 2;

            int count = 1;

            // helper to create intermediate vertex/term and update binarized_term
            auto make_interm = [&](const std::vector<VertexPtr> &verts, size_t erase_pos, size_t erase_count, size_t insert_pos) {
                MutableVertexPtr interm_vertex;
                if (verts.size() == 2)
                    interm_vertex = make_shared<Vertex>(Vertex::printer_->scratch_prefix(), (verts[0] * verts[1])->lines());
                else
                    interm_vertex = make_shared<Vertex>(Vertex::printer_->scratch_prefix(), verts[0]->lines());

                interm_vertex->vertex_type_ = (char)count + '0';
                interm_vertex->sort();
                interm_vertex->update_name();

                Term interm_term = binarized_term;
                interm_term.reset_perm();
                interm_term.coefficient_ = 1.0;
                interm_term.comments_.clear();
                interm_term.is_assignment_ = true;

                interm_term.lhs_ = interm_vertex;
                interm_term.rhs_ = verts;
                interm_term.compute_scaling(true);

                output += interm_term.str();
                output += "\n";

                for (size_t e = 0; e < erase_count; ++e)
                    binarized_term.rhs_.erase(binarized_term.rhs_.begin() + erase_pos);
                binarized_term.rhs_.insert(binarized_term.rhs_.begin() + insert_pos, interm_vertex);
                binarized_term.compute_scaling(true);

                made_any_change = true;
                ++count;
            };

            do {
                size_t n = binarized_term.rhs_.size();
                needs_binarization = n > 2;

                if (needs_binarization) {
                    VertexPtr &left = binarized_term.rhs_[0], &right = binarized_term.rhs_[1];

                    // determine which intermediate is larger: first two or last two
                    VertexPtr &left_end = binarized_term.rhs_[n - 2];
                    VertexPtr &right_end = binarized_term.rhs_[n - 1];

                    // prefer to binarize larger intermediate first. prefer left for ties
                    bool first_smaller = (left*right)->shape_ <= (left_end*right_end)->shape_;

                    // create intermediate from first two vertices
                    if (first_smaller)
                        make_interm({left, right}, 0, 2, 0);
                    else make_interm({left_end, right_end}, n - 2, 2, n - 2);

                } else if (binarized_term.rhs_.size() == 2) {
                    // check if left or right is an addition that needs to be binarized
                    VertexPtr &left = binarized_term.rhs_[0], &right = binarized_term.rhs_[1];
                    bool left_is_add  =  left->is_expandable(false, true);
                    bool right_is_add = right->is_expandable(false, true);

                    if (left_is_add)
                        make_interm({left}, 0, 1, 0);

                    if (right_is_add)
                        make_interm({right}, 1, 1, 1);
                }
            } while (needs_binarization);

            // now print the final binarized term if a change was made
            if (made_any_change) {
                output += binarized_term.str();
                return output;
            } // else we continue to print the original term
        }

        return Vertex::printer_->format_term(*this);
    }

    // JSON-escape a string (names like tmps_["12_vvoo"] contain quotes).
    static string ir_jesc(const string &s) {
        string o; o.reserve(s.size() + 2);
        for (char c : s) {
            if (c == '\\' || c == '"') o += '\\';
            o += c;
        }
        return o;
    }

    // Whether a vertex is a generated intermediate (routes to the intermediate
    // store, not inputs/outputs). is_temp() catches fusion intermediates (which
    // are Linkages with an id), but permutation tmps are plain Vertices named
    // "tmps_"; match those (and reused_/scalars_) by name prefix too.
    static bool ir_is_intermediate(const VertexPtr &v) {
        if (v->is_temp()) return true;
        // A LINKAGE that is not a temp is an inline expression -- e.g. a fused ADDITION whose
        // name() happens to start with its left temp's name ("tmps_[..] + einsum(..)"). It must
        // be hoisted into its own temp, not referenced by name (there is no tensor of that
        // name). Only non-linkage vertices (already-named temp references carried as plain
        // vertices) are classified by the name prefix.
        if (v->is_linked()) return false;
        const string &n = v->name();
        return n.rfind("tmps_", 0) == 0 || n.rfind("perm_tmps", 0) == 0
            || n.rfind("reused_", 0) == 0 || n.rfind("scalars_", 0) == 0;
    }

    // Serialize {name, indices, classes, is_intermediate} from a name + lines.
    // indices = einsum subscript chars (excited-state lines skipped, matching the
    // python printer); classes = Line::type() (o/v electron, O/V proton, Q aux,
    // L excited) for the dims lookup.
    static string ir_vertex_json_core(const string &name, const line_vector &lines,
                                      bool is_intermediate, bool is_target = false) {
        string istr = "[", cstr = "[";
        bool first = true;
        std::set<char> target_seen;
        for (const auto &line : lines) {
            if (line.sig_ && !Vertex::use_trial_index) continue;
            if (!first) { istr += ","; cstr += ","; }
            first = false;
            const char sub = line.einsum_char();
            // A repeated subscript on the TARGET is not expressible: it would define the
            // tensor on that diagonal only, leaving every off-diagonal element whatever the
            // consumer's allocator happened to leave there. A repeat on an OPERAND is
            // legitimate (a trace), which is why this is checked only for the target.
            // Cheap invariant, and the bug it guards against is invisible downstream: a
            // consumer writes the diagonal, reports no error, and returns a wrong answer.
            if (is_target && !target_seen.insert(sub).second)
                throw runtime_error(
                    "ir_emit: repeated target index '" + string(1, sub) + "' in " + name
                    + " -- two distinct lines were given the same subscript, which would "
                      "define this tensor on its diagonal only.");
            istr += string("\"") + sub + "\"";
            cstr += string("\"") + line.type() + "\"";
        }
        istr += "]"; cstr += "]";
        return string("{\"name\":\"") + ir_jesc(name) + "\""
             + ",\"indices\":" + istr
             + ",\"classes\":" + cstr
             + ",\"is_intermediate\":" + (is_intermediate ? "true" : "false") + "}";
    }

    // Name for a vertex as it must appear in the IR. Vertex::name() for a merged/fused
    // intermediate that is an ADDITION returns the expanded expression ("tmps_[...] +
    // einsum(...)") rather than the temp's own name; str(true,false) always gives the temp
    // identity (tmps_["id_shape"]). A temp DEFINITION target and every USE of it must agree
    // on this identity -- if a use is named by name() it carries a stray einsum string and no
    // tensor of that name exists. Input tensors (not intermediates) keep name().
    static string ir_ident(const VertexPtr &v) {
        if (v->is_linked() && ir_is_intermediate(v))
            return as_link(v)->str(true, false);
        return v->name();
    }

    static string ir_vertex_json(const VertexPtr &v) {
        return ir_vertex_json_core(ir_ident(v), v->lines(), ir_is_intermediate(v));
    }

    // Fresh names for synthetic hoisted intermediates (see ir_emit). Monotonic, so
    // unique within an export; deterministic across fresh processes (like the
    // optimizer's own tmp ids).
    static long ir_hoist_counter = 0;
    void reset_ir_hoist_counter() { ir_hoist_counter = 0; }

    // Optimal binary contraction tree over the emitted operand list, as a nested
    // JSON array of operand indices -- e.g. [[0,1],[2,3]]. Exact subset DP, costed
    // with the same scaling_map comparison the optimizer uses (dimension-aware when
    // "dims" is set, lexicographic line-count otherwise), with a deterministic
    // string tie-break. Computed at emission from the operands' line sets rather
    // than walked off the term's linkage tree: several passes leave rhs_ in a
    // canonicalised (e.g. name-sorted, see fusion's LinkTracker) rather than
    // cost-optimal order, and Term::ir_str refolds the linkage from rhs_ anyway --
    // so the stored tree is always the left fold of the emitted operand order,
    // which for a canonicalised order can start with an outer product (operands
    // sharing no index). The DP guarantees every emitted plan is outer-product
    // free and optimal under the active cost model regardless of upstream order.
    // Consumers without pairing support simply left-fold in list order.
    static string ir_optimal_pairing(const std::vector<line_vector> &op_lines,
                                     const line_vector &tlines) {
        const size_t n = op_lines.size();
        if (n < 3 || n > 12) return "";  // 2 operands need no plan; cap the 3^n DP

        // deduplicated line set per operand (a repeated line within one operand is
        // a summed diagonal; one copy carries its class for costing)
        std::vector<line_vector> ols(n);
        auto contains = [](const line_vector &v, const Line &l) {
            return std::find(v.begin(), v.end(), l) != v.end();
        };
        for (size_t i = 0; i < n; i++)
            for (const Line &l : op_lines[i])
                if (!contains(ols[i], l)) ols[i].push_back(l);
        line_vector tset;
        for (const Line &l : tlines)
            if (!contains(tset, l)) tset.push_back(l);

        const size_t full = (1ul << n) - 1;

        // external lines of subset S: lines within S still needed by the target or
        // by an operand outside S (everything else is summed out inside S)
        auto kept = [&](size_t S) {
            line_vector lines;
            for (size_t i = 0; i < n; i++) {
                if (!(S >> i & 1ul)) continue;
                for (const Line &l : ols[i]) {
                    if (contains(lines, l)) continue;
                    bool needed = contains(tset, l);
                    for (size_t j = 0; !needed && j < n; j++)
                        if (!(S >> j & 1ul) && contains(ols[j], l)) needed = true;
                    if (needed) lines.push_back(l);
                }
            }
            return lines;
        };

        struct Best { scaling_map cost; string tree; line_vector ext; bool set = false; };
        std::vector<Best> best(full + 1);
        for (size_t i = 0; i < n; i++) {
            Best &b = best[1ul << i];
            b.set = true;
            b.tree = std::to_string(i);
            b.ext = kept(1ul << i);
        }

        // ascending mask order visits every proper subset before its superset
        for (size_t S = 3; S <= full; S++) {
            if (__builtin_popcountl(S) < 2) continue;
            Best &b = best[S];
            const size_t low = S & (~S + 1ul);
            // enumerate splits S = A | B canonically (A holds the lowest bit)
            for (size_t A = (S - 1) & S; A; A = (A - 1) & S) {
                if (!(A & low)) continue;
                const size_t B = S & ~A;
                const Best &ba = best[A], &bb = best[B];

                // flops of contracting the two results: all their external lines
                shape s;
                line_vector uni = ba.ext;
                for (const Line &l : bb.ext)
                    if (!contains(uni, l)) uni.push_back(l);
                for (const Line &l : uni) s += l;

                scaling_map cost = ba.cost;
                cost += bb.cost;
                cost[s] += 1;

                string tree = "[" + ba.tree + "," + bb.tree + "]";
                int cmp = b.set ? cost.compare(b.cost) : scaling_map::this_better;
                if (cmp == scaling_map::this_better ||
                    (cmp == scaling_map::this_same && tree < b.tree)) {
                    b.set = true;
                    b.cost = std::move(cost);
                    b.tree = std::move(tree);
                }
            }
            b.ext = kept(S);
        }
        return best[full].tree;
    }

    // Emit one-or-more JSONL statements for "target [assign] coeff * <rhs_link>",
    // the target given as (tname, tlines, tinterm) so synthetic temps can be
    // targets too. A top-level addition A+B is split into "target = c*A" then
    // "target += c*B"; contractions become one statement over the linkage's
    // link_vector. An operand that is itself an inline (un-named) non-temp linkage
    // -- the (A+B)/nested-contraction sub-expressions the expression backends
    // inline but einsums cannot -- is HOISTED: it gets a synthetic tmps_ name, its
    // defining statement(s) are emitted first, and the operand is replaced by a
    // reference to that temp. So the IR carries every operand's definition.
    static string ir_emit(const string &tname, const line_vector &tlines, bool tinterm,
                          bool assign, double coeff, const VertexPtr &rhs_link,
                          const std::set<string> &conds, const string &extra = "") {

        // split additions (a named temp used as an operand is left intact)
        if (rhs_link->is_linked() && as_link(rhs_link)->is_addition() && !rhs_link->is_temp()) {
            LinkagePtr al = as_link(rhs_link);
            return ir_emit(tname, tlines, tinterm, assign, coeff, al->left(), conds, extra)
                 + "\n" + ir_emit(tname, tlines, tinterm, false, coeff, al->right(), conds, extra);
        }

        // gather operands (contraction order, mirroring the python printer's walk)
        vertex_vector ops;
        if (rhs_link->is_linked() && !rhs_link->is_temp())
            ops = as_link(rhs_link)->link_vector();
        else
            ops = { rhs_link };

        // Collect the operands BEFORE serializing any of them: a hoisted operand recurses
        // into ir_emit, which assigns its own statement's subscripts, so nothing may be
        // written out until every nested emission is done and this statement's own
        // assignment is in scope.
        struct IROperand { string name; line_vector lines; bool interm; };
        string prefix;                     // hoisted temp definitions, emitted first
        std::vector<IROperand> operands;
        for (const auto &op : ops) {
            if (op->empty()) continue;
            if (!op->is_linked() && op->is_constant()) {
                // constant-scalar vertex (created by intermediate fusion's
                // ratio*vertex): fold into the statement coefficient instead of
                // emitting an operand named "0.500000000000" with no indices,
                // which every IR consumer would misread as an input tensor.
                coeff *= op->value();
                continue;
            }
            if (op->is_linked() && !ir_is_intermediate(op)) {
                // inline (A+B)/nested-contraction operand -> materialise into a temp
                string tn = "tmps_[\"ir" + std::to_string(ir_hoist_counter++) + "\"]";
                prefix += ir_emit(tn, op->lines(), true, true, 1.0, op, conds) + "\n";
                operands.push_back({tn, op->lines(), true});
            } else {
                operands.push_back({ir_ident(op), op->lines(), ir_is_intermediate(op)});
            }
        }

        // Give every distinct label in this statement its own subscript. Without this the
        // natural label->char map is not injective (see Line::natural_einsum_char) and two
        // distinct nuclear lines can print as one index -- a silently wrong contraction.
        line_vector stmt_lines = tlines;
        for (const auto &op : operands)
            stmt_lines.insert(stmt_lines.end(), op.lines.begin(), op.lines.end());
        Line::SubscriptScope subscripts(stmt_lines);

        std::vector<string> operand_jsons;
        std::vector<line_vector> op_lines; // per emitted operand, for the pairing DP
        for (const auto &op : operands) {
            operand_jsons.push_back(ir_vertex_json_core(op.name, op.lines, op.interm));
            op_lines.push_back(op.lines);
        }

        string out = string("{\"target\":") + ir_vertex_json_core(tname, tlines, tinterm, true);
        out += string(",\"is_assignment\":") + (assign ? "true" : "false");
        { std::ostringstream cs; cs.precision(17); cs << coeff;
          out += ",\"coeff\":" + cs.str(); }

        // exact small-rational annotation: repeating fractions print as 17-digit
        // decimals above (e.g. 1/3 -> 0.33333333333333331); when the coefficient is a
        // small p/q, also emit it exactly so consumers can recover the factor.
        if (std::fabs(coeff - std::round(coeff)) > 1e-12) {          // non-integer only
            for (long q : {2L, 3L, 4L, 6L, 8L, 12L, 16L, 24L, 32L, 48L, 64L}) {
                double pd = coeff * (double) q;
                double pr = std::round(pd);
                if (std::fabs(pd - pr) < 1e-12) {
                    long p = (long) pr;
                    long g = std::gcd(std::labs(p), q);
                    out += ",\"coeff_rational\":\"" + std::to_string(p / g) + "/"
                         + std::to_string(q / g) + "\"";
                    break;
                }
            }
        }

        // caller-supplied annotation (e.g. the originating antisymmetrizer group)
        if (!extra.empty())
            out += "," + extra;

        // term-level spin-block / state conditions, if any (NEO spin blocking)
        if (!conds.empty()) {
            out += ",\"conditions\":[";
            bool first = true;
            for (const auto &c : conds) {
                if (!first) out += ",";
                first = false;
                out += string("\"") + ir_jesc(c) + "\"";
            }
            out += "]";
        }

        out += ",\"operands\":[";
        bool first = true;
        for (const auto &oj : operand_jsons) {
            if (!first) out += ",";
            first = false;
            out += oj;
        }
        out += "]";

        // the optimal binary contraction order over the operands (only meaningful
        // for 3+ operands; consumers without support ignore the field and left-fold
        // in list order). Computed by ir_optimal_pairing's subset DP from the
        // operands' line sets, so it is emitted for EVERY multi-operand statement
        // (including temp/scalar declarations) and is guaranteed outer-product free
        // even when an upstream pass canonicalised the operand order.
        if (operand_jsons.size() >= 3) {
            string tree = ir_optimal_pairing(op_lines, tlines);
            if (!tree.empty() && tree.front() == '[')
                out += ",\"pairing\":" + tree;
        }

        out += "}";
        return prefix + out;
    }

    // Structured JSONL line(s) describing this term as flat codegen statement(s):
    // target, assignment flag, coefficient, and ordered operands -- each with its
    // index labels and orbital-class chars. Consumed by the einsums/codegen
    // lowering (see neocc/codegen/einsums_printer_plan.md). Permutation operators
    // are already expanded into separate terms by Term::str() before this point.
    string Term::ir_str() const {
        return ir_emit(ir_ident(lhs_), lhs_->lines(), ir_is_intermediate(lhs_),
                       is_assignment_, coefficient_, term_linkage(true), conditions(),
                       ir_perm_json_);
    }

}