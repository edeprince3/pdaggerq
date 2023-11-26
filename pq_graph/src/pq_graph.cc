//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_graph.cc
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

#include "../include/pq_graph.h"
#include "iostream"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "../../pdaggerq/pq_string.h"
#include "../../pdaggerq/pq_helper.h"
#include <omp.h>
#include <memory>
#include <sstream>

namespace py = pybind11;
using namespace pybind11::literals;

using std::ostream;
using std::string;
using std::vector;
using std::map;
using std::unordered_map;
using std::shared_ptr;
using std::make_shared;
using std::set;
using std::unordered_set;
using std::pair;
using std::make_pair;
using std::to_string;
using std::invalid_argument;
using std::stringstream;
using std::cout;
using std::endl;
using std::flush;
using std::max;

namespace pdaggerq {

    void PQGraph::export_pq_graph(pybind11::module &m) {
        // add tabuilder pybind class
        py::class_<pdaggerq::PQGraph, std::shared_ptr<pdaggerq::PQGraph> >(m, "pq_graph")
                .def(py::init<const pybind11::dict&>())
                .def("set_options", &pdaggerq::PQGraph::set_options)
                .def("add", [](PQGraph& self, const pq_helper &pq, const std::string& equation_name) {
                    return self.add(pq, equation_name);
                }, py::arg("pq") = pq_helper(), py::arg("equation_name") = "")
                .def("print", &pdaggerq::PQGraph::print)
                .def("str", &pdaggerq::PQGraph::str)
                .def("reorder", &pdaggerq::PQGraph::reorder)
                .def("optimize", &pdaggerq::PQGraph::optimize)
                .def("analysis", &pdaggerq::PQGraph::analysis)
                .def("clear", &pdaggerq::PQGraph::clear)
                .def("to_strings", &pdaggerq::PQGraph::to_strings)
                .def("write_dot", &pdaggerq::PQGraph::write_dot);
    }

    void PQGraph::set_options(const pybind11::dict& options) {
        cout << endl << "####################" << " PQ GRAPH " << "####################" << endl << endl;

        // set defaults
        int max_temps = -1;
        int max_linkage_ops = -1;
        int depth = -1;
        Equation::t1_transform_ = false;
        verbose = true;
        make_scalars_ = true;
        batched_ = false;
        allow_merge_ = false;
        reuse_permutations_ = false;
        Equation::permuted_merge_ = false;
        Vertex::allow_permute_ = true; // TODO: add flag in options
        size_t max_num_threads = omp_get_max_threads();
        num_threads_ = (int) max_num_threads;

        if(options.contains("max_temps")) {
            max_temps = options["max_temps"].cast<int>();
            max_temps_ = max_temps;
        }

        if (options.contains("t1_transform")) Equation::t1_transform_ = options["t1_transform"].cast<bool>();
        if (options.contains("verbose")) verbose = options["verbose"].cast<bool>();
        if (options.contains("output")) {
            if (options["output"].cast<string>() == "python") {
                Term::make_einsum = true;
                cout << "Formatting equations for python" << endl;
            } else if (options["output"].cast<string>() == "c++") {
                Term::make_einsum = false;
                cout << "Formatting equations for c++" << endl;
            } else {
                cout << "WARNING: output must be one of: python, einsum, c++" << endl;
                cout << "         Setting output to c++" << endl;
            }
        }
        if (options.contains("make_scalars")) make_scalars_ = options["make_scalars"].cast<bool>();

        if (options.contains("max_contractions"))
            max_linkage_ops = options["max_contractions"].cast<int>();
        Term::max_linkages = (size_t) max_linkage_ops;

        if (options.contains("depth")) {
            depth = options["depth"].cast<int>();
        }
        Term::depth_ = (size_t) depth;
        if (depth > 1 && depth != static_cast<size_t>(-1))
            throw invalid_argument("Depth must be +-1, or 0. Custom depth beyond 1 is not supported.");

        if (options.contains("batched")) batched_ = options["batched"].cast<bool>();
        if (options.contains("allow_merge")) allow_merge_ = options["allow_merge"].cast<bool>();
        if (options.contains("permuted_merge")) Equation::permuted_merge_ = options["permuted_merge"].cast<bool>();
        if (options.contains("reuse_permutations")) reuse_permutations_ = options["reuse_permutations"].cast<bool>();

        if (options.contains("occ_labels"))
            Line::occ_labels_ = options["occ_labels"].cast<unordered_set<char>>();
        if (options.contains("virt_labels"))
            Line::virt_labels_ = options["virt_labels"].cast<unordered_set<char>>();
        if (options.contains("sig_labels"))
            Line::sig_labels_ = options["sig_labels"].cast<unordered_set<char>>();
        if (options.contains("den_labels"))
            Line::den_labels_ = options["den_labels"].cast<unordered_set<char>>();


        if (options.contains("num_threads")) {
            if (num_threads_ > max_num_threads) {
                cout << "Warning: number of threads is larger than the maximum number of threads on this machine. "
                        "Using the maximum number of threads instead." << endl;
                num_threads_ = (int) max_num_threads;
            } else num_threads_ = options["num_threads"].cast<int>();

            Equation::num_threads_ = num_threads_;
        }
        if (options.contains("conditions")) {
            // this is a list of strings of vertex names
            try {
                auto conditions = options["conditions"].cast<set<string>>();
                for (const string &condition: conditions) Term::conditions_.insert(condition);
            } catch (...) {
                cout << "WARNING: conditions must be a list of strings." << endl;
            }

            if (!Term::conditions_.empty()){
                cout << "Conditions: ";
                for (const string &condition: Term::conditions_) cout << condition << " ";
                cout << endl;
            }
        }

        if (options.contains("store_trials"))
            store_trials_ = options["store_trials"].cast<bool>();

        omp_set_num_threads(1); // set to 1 to speed up non-parallel code

        cout << "Options:" << endl;
        cout << "--------" << endl;
        cout << "    verbose: " << (verbose ? "true" : "false")
             << "  // whether to print out verbose analysis (default: true)" << endl;
        cout << "    make_einsum: " << (Term::make_einsum ? "true" : "false") << "  // whether to print equations in einsum format with python (default: false). Requires some manual formatting at the beginning"
             << endl;
        cout << "    make_scalars_: " << (make_scalars_ ? "true" : "false")
             << "  // whether to format dot products and traces as scalars (default: true; encouraged)" << endl;
        cout << "    t1_transform: " << (Equation::t1_transform_ ? "true" : "false")
             << "  // removes t1 terms from the equations when using t1-transformed integrals (default: false)" << endl;
        cout << "    max_temps: " << max_temps
             << "  // maximum number of intermediates to find (default: -1 for no limit)" << endl;
        cout << "    max_contractions: " << max_linkage_ops
             << "  // maximum number of contractions in an intermediate in `tmps` (default: -1 for no limit)" << endl;
        cout << "    depth: " << depth
             << "  maximum number of nested precontractions. -1 means no limit (default)." << endl;
        cout << "    conditions: " << flush;
        for (const string& condition : Term::conditions_)
            cout << condition << ", " << flush;
        cout << "  // tensors whose names contain one of these strings (i.e. r3) will be nested in a conditional statement such as \"if (include_r2) { ... }\" (default: None)" << endl;
        if (store_trials_) {
            cout << "    store_trials: " << (store_trials_ ? "true" : "false")
                 << "  // whether to store trial vectors as an additional index/dimension for tensors in a sigma-vector build (default: false)" << endl;
        }
        cout << "    batched: " << (batched_ ? "true" : "false") << "  // whether to substitute tmps in batches for faster generation. (default: false)" << endl;
        cout << "    allow_merge_: " << (allow_merge_ ? "true" : "false") << "  // whether to merge similar terms during optimization (default: true)" << endl;
        cout << "    permuted_merge: " << (Equation::permuted_merge_ ? "true" : "false")
             << "  // whether to merge similar terms with permuted tensors (default: true)" << endl;
        cout << "    reuse_permutations: " << (reuse_permutations_ ? "true" : "false")
             << "  // This determines whether to reuse containers for permutations instead of permuting each time a term with permutations occurs (default: true)" << endl;
        cout << "    num_threads: " << num_threads_ << "  // number of threads to use (default: 1 | available: " << max_num_threads << ")" << endl;
        cout << endl;
    }

    void PQGraph::add(const pq_helper& pq, const std::string &equation_name) {
        // check if equation already exists; if so, print warning
        if (equations_.find(equation_name) != equations_.end()) {
            cout << "WARNING: equation '" << equation_name << "' already exists. Overwriting." << endl;
        }

        if (equations_.empty()) {
            flop_map_.clear();
            mem_map_.clear();
            flop_map_init_.clear();
            mem_map_init_.clear();
        }

        // check if equation name has a '(' in it. If so, we change the construction procedure.
        bool name_is_formatted = equation_name.find('(') != string::npos;

        // if no name is provided for the equation, use an arbitrary one.
        std::string assigment_name;
        if (equation_name.empty())
             assigment_name = "eq_" + to_string(equations_.size());
        else assigment_name = equation_name;

        // make terms directly from pq_strings in pq_helper
        vector<Term> terms;

        // get strings
        bool has_blocks = pq_string::is_spin_blocked || pq_string::is_range_blocked;
        const std::vector<std::shared_ptr<pq_string>> &ordered = pq.get_ordered_strings(has_blocks);
        if (ordered.empty()){
            cout << "WARNING: no pq_strings found in pq_helper. Skipping equation '" << equation_name << "'." << endl;
            return;
        }

        // loop over each pq_string
        bool is_sigma_equation = false;
        for (const auto& pq_string : ordered) {

            // skip if pq_string is empty
            if (pq_string->skip)
                continue;


            Term term;
            if (name_is_formatted) {
                // create term from string
                term = Term(equation_name, pq_string);
            } else {
                // create term with an empty string
                term = Term("", pq_string);
            }

            // format self-contractions
            term.apply_self_links();

            // use the term to build the assignment vertex
            if (!name_is_formatted || equation_name.empty()) {
                VertexPtr assigment = make_shared<Vertex>(*term.term_linkage_);
                assigment->update_name(assigment_name);

                term.lhs() = assigment;
                term.eq() = assigment;
            }


            // check if any operator in term is a sigma operator
            for (const auto &op : term.rhs()) {
                if (op->is_sigma_) {
                    // mark that this equation has sigma vectors
                    has_sigma_vecs_ = true;

                    // mark that there is at least one equation in the pq_graph that has sigma vectors
                    is_sigma_equation = true;
                    break;
                }
            }

            terms.push_back(term);
        }


        // build equation
        Equation& new_equation = equations_[assigment_name];
        new_equation = Equation(assigment_name, terms);

        // save initial scaling
        new_equation.collect_scaling();

        const scaling_map &eq_flop_map_ = new_equation.flop_map();
        const scaling_map &eq_mem_map_  = new_equation.mem_map();

        flop_map_      += eq_flop_map_;
        mem_map_       += eq_mem_map_;

        flop_map_init_ += eq_flop_map_;
        mem_map_init_  += eq_mem_map_;

    }

    void PQGraph::print() {
        // print output to stdout
        cout << this->str() << endl;
    }

    string PQGraph::str() {

        stringstream sout; // string stream to hold output

        // add banner for PQ GRAPH results
        sout << "####################" << " PQ GRAPH Output " << "####################" << endl << endl;

        // get all terms from all equations except the tmps, scalars, and reuse_tmps equation
        vector<Term> all_terms;

        bool has_tmps = false;
        for (const auto &eq_pair : equations_) { // iterate over equations in serial
            const string &eq_name = eq_pair.first;
            const Equation &equation = eq_pair.second;
            if (!equation.allow_substitution_) {
                has_tmps = true;
                continue; // skip tmps equation
            }

            const vector<Term> &terms = equation.terms();
            all_terms.insert(all_terms.end(), terms.begin(), terms.end());
        }

        // make set of all unique base names (ignore linkages and scalars)
        set<string> base_names;
        for (const auto &term: all_terms) {
            VertexPtr lhs = term.lhs();
            if (!lhs->is_linked() && !lhs->is_scalar())
                base_names.insert(lhs->base_name());
            for (const auto &op: term.rhs()) {
                if (!op->is_linked() && !op->is_scalar())
                    base_names.insert(op->base_name());
            }
        }

        // add tmp declarations
        base_names.insert("perm_tmps");
        base_names.insert("tmps");

        // declare a map for each base name
        sout << " #####  Declarations  ##### " << endl << endl;
        for (const auto &base_name: base_names) {
            sout << "// initialize as --> std::map<std::string, TA::TArrayD> " << base_name + "_" << ";" << endl;
        }
        if (!equations_["tmps"].empty()) {
            sout << "TA::TArrayD reset_tmp();" << endl;
        }
        sout << endl;

        // add scalar terms to the beginning of the equation

        // create merged equation to sort tmps
        Equation merged_eq = Equation("", all_terms);
        sort_tmps(merged_eq); // sort tmps in merged equation
        all_terms = merged_eq.terms(); // get sorted terms

        // print scalar declarations
        if (!equations_["scalars"].empty()) {
            sout << " #####  Scalars  ##### " << endl << endl;
            sout << "std::map<std::string, double> scalars_;" << endl << endl;
            sort_tmps(equations_["scalars"]);
            sout << equations_["scalars"] << endl;
            sout << " ### End of Scalars ### " << endl << endl;
        }

        // print declarations for reuse_tmps
        if (!equations_["reuse_tmps"].empty()){
            sout << " #####  Shared  Operators  ##### " << endl << endl;
            sout << "std::map<std::string, TA::TArrayD> reuse_tmps_;" << endl;
            sort_tmps(equations_["reuse_tmps"]);
            sout << equations_["reuse_tmps"] << endl;
            sout << " ### End of Shared Operators ### " << endl << endl;
        }

        // for each term in tmps, add the term to the merged equation
        // where each tmp of a given id is first used

        sort_tmps(equations_["tmps"]); // sort tmps in tmps equation
        vector<bool> found_tmp_ids(temp_counts_["tmps"]); // keep track of tmp ids that have been found
        bool found_all_tmp_ids = false; // flag to check if all tmp ids have been found

        size_t retry = 0;
        size_t max_retry = equations_["tmps"].size();


        /** while not all tmp ids have been found:
         *      make declarations for tmps in order of first use;
         *      make destructors in order of last use.
         */
        do {
            size_t last_pos = 0; // tmps are inserted in order of first use; use this to reduce search time
            for (auto &tempterm: equations_["tmps"]) {
                if (!tempterm.lhs()->is_linked()) continue;

                LinkagePtr temp = as_link(tempterm.lhs());
                size_t temp_id = temp->id_;

                // check if temp_id has been found
                if (found_tmp_ids[temp_id-1]) continue;

                bool made_tmp = false;
                for (size_t i = last_pos; i < all_terms.size(); ++i) {
                    const Term &term = all_terms[i];

                    // check if tmp is in the rhs of the term
                    bool found = false;
                    for (const auto &op: term.rhs()) {
                        bool is_tmp = op->is_linked(); // must be a tmp
                        if (!is_tmp) continue;

                        LinkagePtr link = as_link(op);
                        is_tmp = is_tmp && !link->is_scalar(); // must not be a scalar (already in scalars_)
                        is_tmp = is_tmp && !link->is_reused_; // must not be reused (already in reuse_tmps)

                        if (is_tmp && link->id_ == temp_id) {
                            found = true;
                            break; // true if we found first use of tmp with this id
                        }
                    }

                    // tmp not found in rhs of term; continue
                    if (!found) continue;

                    // add tmp term before this term
                    all_terms.insert(all_terms.begin() + (int) i, tempterm);
                    found_tmp_ids[temp_id-1] = true; // mark tmp id as found
                    last_pos = i; break; // update last position and go to next tmp term
                }

                // if tmp id was not found, continue
                if (!found_tmp_ids[temp_id-1]) continue;

                for (auto i = (long int) all_terms.size() - 1; i >= 0; --i) {
                    const Term &term = all_terms[i];

                    // check if tmp is in the rhs of the term
                    bool found = false;
                    for (const auto &op: term.rhs()) {
                        bool is_tmp = op->is_linked(); // must be a tmp
                        if (!is_tmp) continue;

                        LinkagePtr link = as_link(op);
                        is_tmp = is_tmp && !link->is_scalar(); // must not be a scalar (already in scalars_)
                        is_tmp = is_tmp && !link->is_reused_; // must not be reused (already in reuse_tmps)

                        if (is_tmp && link->id_ == temp_id) {
                            found = true; break; // true if we found first use of tmp with this id
                        }
                    }

                    if (!found) continue; // tmp not found in rhs of term; continue

                    // Create new term with tmp in the lhs and assign zero to the rhs

                    // create vertex with only the linkage's name
                    std::string lhs_name = temp->str(true, false);
                    VertexPtr vert = make_shared<Vertex>(lhs_name);

                    // create term
                    Term newterm = Term(vert, {make_shared<Vertex>("reset_tmp")}, 1.0);
                    newterm.is_assignment_ = true;
                    newterm.comments() = {};

                    // add tmp term after this term
                    all_terms.insert(all_terms.begin() + (int) i + 1, newterm);
                    break; // only add once
                }
            }

            found_all_tmp_ids = true;
            for (auto found : found_tmp_ids) {
                if (!found) { found_all_tmp_ids = false; break; }
            }
        } while (!found_all_tmp_ids && retry++ < max_retry);

        if (!found_all_tmp_ids) {
            cout << "WARNING: could not find first use of tmps with ids: ";
            for (size_t id = 0; id < found_tmp_ids.size(); ++id) {
                if (!found_tmp_ids[id]) cout << id+1 << " ";
            }
            cout << endl;
        }

        sout << " ##########  Evaluate Equations  ########## " << endl << endl;

        // update terms in merged equation
        merged_eq.terms() = all_terms;

        // stream merged equation as string
        sout << merged_eq << endl;

        // add closing banner
        sout << "####################" << "######################" << "####################" << endl << endl;

        // return string stream as string
        return sout.str();

    }

    void PQGraph::collect_scaling(bool recompute) {

        // reset scaling maps
        flop_map_.clear(); // clear flop scaling map
        mem_map_.clear(); // clear memory scaling map

        // reset bottleneck scaling (using the first equation)
        bottleneck_flop_ = equations_.begin()->second.bottleneck_flop();
        bottleneck_mem_ = equations_.begin()->second.bottleneck_mem();

        for (auto & eq_pair : equations_) { // iterate over equations
            // collect scaling for each equation
            Equation &equation = eq_pair.second;
            equation.collect_scaling(recompute);

            const auto & flop_map = equation.flop_map(); // get flop scaling map
            const auto & mem_map = equation.mem_map(); // get memory scaling map

            flop_map_ += flop_map; // add flop scaling
            mem_map_ += mem_map; // add memory scaling

            // get bottlenecks
            const shape & bottleneck_flop = equation.bottleneck_flop(); // get flop bottleneck
            const shape & bottleneck_mem = equation.bottleneck_mem(); // get memory bottleneck

            if (bottleneck_flop > bottleneck_flop_)
                // if flop bottleneck is more expensive
                bottleneck_flop_ = bottleneck_flop; // set flop bottleneck

            if (bottleneck_mem > bottleneck_mem_)
                // if memory bottleneck is more expensive
                bottleneck_mem_ = bottleneck_mem; // set memory bottleneck
        }
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

    vector<string> PQGraph::get_equation_keys() {
        vector<string> eq_keys(equations_.size());
        transform(equations_.begin(), equations_.end(), eq_keys.begin(),
                  [](const pair<string, Equation> &p) { return p.first; });
        return std::move(eq_keys);
    }

    void PQGraph::reorder() { // verbose if not already reordered

        // save initial scaling
        static bool initial_saved = false;
        if (!initial_saved) {
            flop_map_init_ = flop_map_;
            mem_map_init_ = mem_map_;
            initial_saved = true;
        }

        if (!is_reordered_) reorder_timer.start();

        if (!is_reordered_) cout << endl << "Reordering equations..." << flush;

        // get list of keys in equations
        vector<string> eq_keys = get_equation_keys();

        omp_set_num_threads(num_threads_); // set number of threads
        #pragma omp parallel for schedule(guided) shared(equations_, eq_keys) default(none)
        for (const auto& eq_name : eq_keys) { // iterate over equations in parallel
            equations_[eq_name].reorder(true); // reorder terms in equation
        }
        omp_set_num_threads(1); // set to 1 to speed up non-parallel code

        if (!is_reordered_) cout << " Done" << endl << endl;

        // collect scaling
        if (!is_reordered_) cout << "Collecting scalings of each equation...";
        collect_scaling(); // collect scaling of equations
        if (!is_reordered_) cout << " Done" << endl;

        if (!is_reordered_) reorder_timer.stop();
        if (!is_reordered_) cout << "Reordering time: " << reorder_timer.elapsed() << endl << endl;

        is_reordered_ = true; // set reorder flag to true
        if (flop_map_pre_.empty()) flop_map_pre_ = flop_map_;
        if (mem_map_pre_.empty()) mem_map_pre_ = mem_map_;
    }

    void PQGraph::analysis() const {
        cout << "####################" << " PQ GRAPH Analysis " << "####################" << endl << endl;

        // print total time elapsed
        long double total_time = build_timer.get_runtime() + reorder_timer.get_runtime()
                                 + substitute_timer.get_runtime() + update_timer.get_runtime();
        cout << "Net time: " << Timer::format_time(total_time) << endl << endl;

        // get total number of linkages
        size_t n_flop_ops = flop_map_.total();
        size_t n_flop_ops_pre = flop_map_pre_.total();

        size_t number_of_terms = 0;
        for (const auto & eq_pair : equations_) {
            const Equation &equation = eq_pair.second;
            const auto &terms = equation.terms();
            number_of_terms += terms.size();
        }

        cout << "Total Number of Terms: " << number_of_terms << endl;
        cout << "Total Contractions: (last) " << n_flop_ops_pre << " -> (new) " << n_flop_ops << endl << endl;
        cout << "Total FLOP scaling: " << endl;
        cout << "------------------" << endl;
        size_t last_order;
        print_new_scaling(flop_map_init_, flop_map_pre_, flop_map_);

        cout << endl << "Total MEM scaling: " << endl;
        cout << "------------------" << endl;

        print_new_scaling(mem_map_init_, mem_map_pre_, mem_map_);
        cout << endl << endl;
        cout << "####################" << "######################" << "####################" << endl << endl;

    }

    void PQGraph::print_new_scaling(const scaling_map &original_map, const scaling_map &previous_map, const scaling_map &current_map) {
        printf("%10s : %8s | %8s | %8s || %10s | %10s\n", "Scaling", "initial", "reorder", "optimize", "init diff", "opt diff");

        scaling_map diff_map = current_map - previous_map;
        scaling_map tot_diff_map = current_map - original_map;

        auto last_order = static_cast<size_t>(-1);
        for (const auto & key : original_map + previous_map + current_map) {
            shape cur_shape = key.first;
            size_t new_order = cur_shape.n_;
            if (new_order < last_order) {
                printf("%10s : %8s | %8s | %8s || %10s | %10s\n" , "--------", "--------", "--------", "--------", "----------", "----------");
                last_order = new_order;
            }
            printf("%10s : %8zu | %8zu | %8zu || %10ld | %10ld \n", cur_shape.str().c_str(), original_map[cur_shape],
                   previous_map[cur_shape], current_map[cur_shape], tot_diff_map[cur_shape], diff_map[cur_shape]);
        }

        printf("%10s : %8s | %8s | %8s || %10s | %10s\n" , "--------", "--------", "--------", "--------", "----------", "----------");
        printf("%10s : %8zu | %8zu | %8zu || %10ld | %10ld \n", "Total", original_map.total(), previous_map.total(), current_map.total(),
               tot_diff_map.total(), diff_map.total());

    }

    void PQGraph::optimize() {
        reorder(); // reorder contractions in equations
//        if (allow_merge_) merge_terms(); // merge similar terms
        substitute(); // find and substitute intermediate contractions
//        if (allow_merge_) merge_terms(); // merge similar terms
//        if (reuse_permutations_)
//           merge_permutations(); // merge similar permutations
        analysis(); // analyze equations
    }

    linkage_set PQGraph::make_test_set() {

        if (!batched_) return tmp_candidates_; // if not batched, return all linkages

        linkage_set test_linkages(1024); // set of linkages to test (start with medium n_ops)

        shape worst_scale; // worst cost (start with zero)
        for (const auto & linkage : tmp_candidates_) { // get worst cost
            const auto &link_vec = linkage->to_vector();
            auto [flop_scales, mem_scales] = Linkage::scale_list(link_vec);

            shape contr_scale;
            for (auto & scale : flop_scales) {
                if (scale > contr_scale) contr_scale = scale;
            }

            if (contr_scale > worst_scale) worst_scale = contr_scale;
        }

        size_t max_size = 0; // maximum n_ops of linkage found (start with 0)
        for (const auto & linkage : tmp_candidates_) { // iterate over all linkages
            const auto &link_vec = linkage->to_vector();
            auto [flop_scales, mem_scales] = Linkage::scale_list(link_vec);

            shape contr_scale;
            for (auto & scale : flop_scales) {
                if (scale > contr_scale) contr_scale = scale;
            }

            if (contr_scale >= worst_scale) {
                if (link_vec.size() >= max_size ) { // if the linkage is large enough,
                    // some smaller linkages may be added, but the test set is somewhat random anyway
                    test_linkages.insert(linkage); // add linkage to the test set
                    max_size = link_vec.size(); // update maximum n_ops
                }
            }
        }

        return test_linkages; // return test linkages
    }

    void PQGraph::add_tmp(const LinkagePtr& precon, Equation &equation) {

        // check that term is not already in equation
//        for (const auto &term : equation) {
//            if (!term.lhs()->is_linked()) continue; // skip if lhs is not linked
//            if (as_link(term.lhs())->id_ == precon->id_)
//                return; // return if the term is already in the equation
//        }

        // make term of tmp
        Term precon_term = Term(precon);
        precon_term.reorder(); // reorder term
        equation.insert(precon_term, 0); // add term to tmp equation

    }

    void PQGraph::sort_tmps(Equation &equation) {

        // no terms, return
        if ( equation.terms().empty() ) return;

        // to sort the tmps while keeping the order of terms without tmps, we need to
        // make a map of the equation terms and their index in the equation and sort that (so annoying)
        std::vector<std::pair<Term*, size_t>> indexed_terms;
        for (size_t i = 0; i < equation.terms().size(); ++i)
            indexed_terms.emplace_back(&equation.terms()[i], i);

        // sort the terms by the maximum id of the tmps in the term, then by the index of the term
        stable_sort(indexed_terms.begin(), indexed_terms.end(), [](const auto &a, const auto &b) {

            const Term &a_term = *a.first;
            const Term &b_term = *b.first;

            const VertexPtr &a_lhs = a_term.lhs();
            const VertexPtr &b_lhs = b_term.lhs();

            // get max ids from lhs and rhs
            long a_max_temp_id = -1, b_max_temp_id = -1;

            // get ids of lhs
            auto get_lhs_id = [](const Term &term) {
                long id = -1;
                if (term.lhs()->is_linked()) {
                    LinkagePtr link = as_link(term.lhs());

                    // ignore reuse_tmp linkages
                    if (!link->is_reused_ && !link->is_scalar())
                        id = (long) link->id_;
                }
                return id;
            };

            long a_lhs_id = get_lhs_id(a_term),
                 b_lhs_id = get_lhs_id(b_term);

            // get max ids from rhs
            auto get_rhs_max_id = [](const Term &term) {
                long max_id = -1;
                for (const auto &op: term.rhs()) {
                    if (op->is_linked()) {
                        LinkagePtr link = as_link(op);

                        // ignore reuse_tmp linkages
                        if (!link->is_reused_ && !link->is_scalar())
                            max_id = std::max(max_id, (long) link->id_);
                    }
                }
                return max_id;
            };

            long a_rhs_id = get_rhs_max_id(a_term),
                 b_rhs_id = get_rhs_max_id(b_term);

            // get total max id for each term
            a_max_temp_id = max(a_lhs_id, a_rhs_id);
            b_max_temp_id = max(b_lhs_id, b_rhs_id);

            // sort by max id of tmps, then by index of term
            if (a_max_temp_id == b_max_temp_id) {
                return a.second < b.second;
            } else return a_max_temp_id < b_max_temp_id;
        });

        // replace the terms in the equation with the sorted terms
        std::vector<Term> sorted_terms;
        sorted_terms.reserve(indexed_terms.size());
        for (const auto &indexed_term : indexed_terms) {
            sorted_terms.push_back(*indexed_term.first);
        }

        equation.terms() = sorted_terms;
    }

} // pdaggerq
