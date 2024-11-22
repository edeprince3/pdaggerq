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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "../include/pq_graph.h"

// include omp only if defined
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_max_threads() 1
    #define omp_set_num_threads(n) 1
#endif
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;

using std::ostream, std::string, std::vector, std::map, std::unordered_map, std::shared_ptr, std::make_shared,
      std::set, std::unordered_set, std::pair, std::make_pair, std::to_string, std::invalid_argument,
      std::stringstream, std::cout, std::endl, std::flush, std::max, std::min, std::unordered_map, std::unordered_set;

namespace pdaggerq {

    void PQGraph::export_pq_graph(pybind11::module &m) {
        // add tabuilder pybind class
        py::class_<pdaggerq::PQGraph, std::shared_ptr<pdaggerq::PQGraph> >(m, "pq_graph")
                .def(py::init<const pybind11::dict&>())
                .def("set_options", &pdaggerq::PQGraph::set_options)
                .def("add", [](PQGraph& self, const pq_helper &pq, const std::string& equation_name, const vector<string> &label_order) {
                    return self.add(pq, equation_name, label_order);
                }, py::arg("pq") = pq_helper(), py::arg("equation_name") = "", py::arg("label_order") = vector<string>())
                .def("print", [](PQGraph& self, const std::string &print_type) {
                    return self.print(print_type);
                }, py::arg("print_type") = "")
                .def("__str__", [](PQGraph& self) { return self.str("python"); })
                .def("str", [](PQGraph& self, const std::string &print_type) {
                    return self.str(print_type);
                }, py::arg("print_type") = "")
                .def("to_strings", [](PQGraph& self, const std::string &print_type) {
                    return self.to_strings(print_type);
                }, py::arg("print_type") = "")
                .def("assemble", &pdaggerq::PQGraph::assemble)
                .def("analysis", &pdaggerq::PQGraph::analysis)
                .def("clear", &pdaggerq::PQGraph::clear)
                .def("write_dot", &pdaggerq::PQGraph::write_dot)
                .def("reorder", [](PQGraph& self) {
                    bool old_opt_level = self.opt_level_; self.opt_level_ = 1;
                    self.reorder();                       self.opt_level_ = old_opt_level;
                })
                .def("substitute", [](PQGraph& self, bool separate_sigma) {
                    bool old_opt_level = self.opt_level_; self.opt_level_ = separate_sigma ? 3 : 2;
                    self.substitute(separate_sigma, true);
                    if (separate_sigma)
                        self.substitute(true, false);
                    self.substitute(false, false);
                    self.opt_level_ = old_opt_level;
                }, py::arg("separate_sigma") = false)
                .def("prune", [](PQGraph& self) {
                    bool old_opt_level = self.opt_level_; self.opt_level_ = 4;
                    self.prune(false);                   self.opt_level_ = old_opt_level;
                })
                .def("merge", [](PQGraph& self) {
                    bool old_opt_level = self.opt_level_; self.opt_level_ = 5;
                    self.merge_terms();                   self.opt_level_ = old_opt_level;
                })
                .def("fusion", [](PQGraph& self) {
                    bool old_opt_level = self.opt_level_; self.opt_level_ = 6;
                    self.merge_intermediates();           self.opt_level_ = old_opt_level;
                })
                .def("optimize", &pdaggerq::PQGraph::optimize);
    }

    void PQGraph::set_options(const pybind11::dict& options) {
        string h1, h2; // header 1 and header 2 padding
        if (Vertex::print_type_ == "python") {
            h1 = "####################";
            h2 = "#####";
        } else if (Vertex::print_type_ == "c++") {
            h1 = "///////////////////";
            h2 = "/////";
        } else throw invalid_argument("Invalid print type: " + Vertex::print_type_);

        cout << endl << h1 << " PQ GRAPH " << h1 << endl << endl;

        if (options.contains("print_level")) {
            print_level_ = options["print_level"].cast<int>();
            if (print_level_ > 2) {
                print_level_ = 2;
            } else if (print_level_ < 0) {
                print_level_ = 0;
            }
        }

        if (options.contains("opt_level")){
            opt_level_ = options["opt_level"].cast<int>();
            if (opt_level_ > 6) {
                opt_level_ = 6;
            } else if (opt_level_ < 0) {
                opt_level_ = 0;
            }
        }

        if(options.contains("max_temps")) {
            max_temps_ = (size_t) options["max_temps"].cast<long>();
        }

        if( options.contains("max_depth")) {
                Term::max_depth_ = (size_t) options["max_depth"].cast<long>();
                if (Term::max_depth_ < 1ul) {
                    cout << "WARNING: max_depth must be greater than 1. Setting to 2." << endl;
                    Term::max_depth_ = 2ul;
                }
        }

        if (options.contains("permute_eri"))
            Vertex::permute_eri_ = options["permute_eri"].cast<bool>();

        if (options.contains("density_fitting"))
            use_density_fitting_ = options["density_fitting"].cast<bool>();

        if (options.contains("no_scalars")) {
            Equation::no_scalars_ = options["no_scalars"].cast<bool>();
            cout << "'no_scalars' is set to true. Scalars will not be included in the final equations." << endl;
        }


        if (options.contains("max_shape_map")) {
            std::map<string, long> max_shape_map;
            try {
                max_shape_map = options["max_shape_map"].cast<std::map<string, long>>();
            } catch (const std::exception &e) {
                throw invalid_argument("max_shape_map must be a map with 'o' or 'v' as keys to int values");
            }

            // throw error if max_shape_map contains an invalid key
            for (const auto &[key, val] : max_shape_map) {
                if (key != "o" && key != "v") {
                    throw invalid_argument("max_shape_map must contain only 'o' and 'v' keys; found key: " + key);
                }
            }
            
            // set max occupied lines
            size_t max_o = static_cast<size_t>(-1l)/2l;
            size_t max_v = static_cast<size_t>(-1l)/2l;
            if (max_shape_map.find("o") != max_shape_map.end())
                max_o = static_cast<size_t>(max_shape_map.at("o"));
            
            // set max virtual lines
            if (max_shape_map.find("v") != max_shape_map.end()) {
                max_v = static_cast<size_t>(max_shape_map.at("v"));
            }

            Term::max_shape_.n_  = max_o + max_v;
            Term::max_shape_.o_  = max_o;
            Term::max_shape_.oa_ = max_o;
            Term::max_shape_.v_  = max_v;
            Term::max_shape_.va_ = max_v;

        } else {
            auto n_max = static_cast<size_t>(-1l);
            Term::max_shape_.n_  = n_max;
            Term::max_shape_.o_  = n_max;
            Term::max_shape_.oa_ = n_max;
            Term::max_shape_.v_  = n_max;
            Term::max_shape_.va_ = n_max;
        }

        if (options.contains("low_memory")) {
            Linkage::low_memory_ = options["low_memory"].cast<bool>();
        }

        if (options.contains("batch_size")) {
            batch_size_ = static_cast<size_t>(options["batch_size"].cast<long>());
            if (batch_size_ < 1ul) {
                cout << "WARNING: batch_size must be greater than 1. Setting to 100." << endl;
                batch_size_ = 100ul;
            } else if (batch_size_ == 1ul) {
                cout << "WARNING: batch_size of 1 is equivalent to no batching." << endl;
            }

        }

        if (options.contains("batched")) {
            batched_ = options["batched"].cast<bool>();
            if (batched_) batch_size_ = 1ul;
        }

        if (options.contains("occ_labels"))
            Line::occ_labels_ = options["occ_labels"].cast<std::array<char, 32>>();
        if (options.contains("virt_labels"))
            Line::virt_labels_ = options["virt_labels"].cast<std::array<char, 32>>();
        if (options.contains("sig_labels"))
            Line::sig_labels_ = options["sig_labels"].cast<std::array<char, 32>>();
        if (options.contains("den_labels"))
            Line::den_labels_ = options["den_labels"].cast<std::array<char, 32>>();


        if (options.contains("nthreads")) {
            nthreads_ = options["nthreads"].cast<int>();
            if (nthreads_ > omp_get_max_threads()) {
                cout << "Warning: number of threads is larger than the maximum number of threads on this machine. "
                        "Using the maximum number of threads instead." << endl;
                nthreads_ = (int) omp_get_max_threads();
            } else if (nthreads_ < 0) {
                nthreads_ = (int) omp_get_max_threads();
            }
            Equation::nthreads_ = nthreads_;
            omp_set_num_threads(nthreads_);
            
        } else {
            // use OMP_NUM_THREADS if available
            char *omp_num_threads = getenv("OMP_NUM_THREADS");
            if (omp_num_threads != nullptr) {
                nthreads_ = std::stoi(omp_num_threads);
                if (nthreads_ > omp_get_max_threads()) {
                    cout << "Warning: OMP_NUM_THREADS is larger than the maximum number of threads on this machine. "
                            "Using the maximum number of threads instead." << endl;
                    nthreads_ = (int) omp_get_max_threads();
                }
                Equation::nthreads_ = nthreads_;
                omp_set_num_threads(nthreads_);
            }
        }

        // set defined conditions
        if (options.contains("conditions")){

            map<string, vector<string>> conditions;

            // check if conditions is a map of strings to vectors of strings
            try {
                conditions = options["conditions"].cast<map<string, vector<string>>>();
            } // else, check if it is a vector of strings; if so, make a map to the condition with single element vector
            catch (const std::exception &e) {
                auto conditions_vec = options["conditions"].cast<vector<string>>();
                for (const auto &cond : conditions_vec) {
                    conditions[cond] = {cond};
                }
            }

            Term::mapped_conditions_.clear();
            for (const auto &[condition, restrict_ops] : conditions) {
                Term::mapped_conditions_[condition] = restrict_ops;
            }
            cout << "Defined conditions: ";
            for (const auto &[condition, restrict_ops] : Term::mapped_conditions_) {
                cout << condition << " -> [";
                for (const auto &op : restrict_ops) {
                    cout << op;
                    if (op != restrict_ops.back())
                        cout << ", ";
                }
                cout << "]\n";
            }
            cout << endl;
        }

        if (options.contains("use_trial_index"))
            Vertex::use_trial_index = options["use_trial_index"].cast<bool>();

        if (options.contains("separate_sigma"))
            separate_sigma_ = options["separate_sigma"].cast<bool>();

        cout << "Options:" << endl;
        cout << "--------" << endl;
        cout << "    print_level: " << print_level_
             << "  // verbosity level:" << endl;
        cout << "                    // 0: no printing of optimization steps (default)" << endl;
        cout << "                    // 1: print optimization steps without fusion or merging" << endl;
        cout << "                    // 2: print optimization steps with fusion and merging" << endl;

        cout << "    permute_eri: " << (Vertex::permute_eri_ ? "true" : "false")
             << "  // whether to permute two-electron integrals to common order (default: true)" << endl;

        cout << "    no_scalars: " << (Equation::no_scalars_ ? "true" : "false")
             << "  // whether to skip the scalar terms in the final equations (default: false)" << endl;

        cout << "    use_trial_index: " << (Vertex::use_trial_index ? "true" : "false")
             << "  // whether to store trial vectors as an additional index/dimension for "
             << "tensors in a sigma-vector build (default: false)" << endl;
        cout << "    separate_sigma: " << (separate_sigma_ ? "true" : "false")
                << "  // whether to separate reusable intermediates for sigma-vector build (default: false)" << endl;
        cout << "    opt_level: " << opt_level_
             << "  // optimization level:" << endl;
        cout << "                  // 0: no optimization" << endl;
        cout << "                  // 1: single-term optimization only (reordering)" << endl;
        cout << "                  // 2: reordering and subexpression elimination (substitution)" << endl;
        cout << "                  // 3: reordering, substitution, and separation of reusable intermediates (for sigma vectors)" << endl;
        cout << "                  // 4: reordering, substitution, and separation; unused intermediates are removed (pruning)" << endl;
        cout << "                  // 5: reordering, substitution, separation, pruning, and merging of equivalent terms" << endl;
        cout << "                  // 6: reordering, substitution, separation, pruning, merging, and fusion of intermediates (default)" << endl;

        cout << "    batched: " << (batched_ ? "true" : "false")
             << "  // candidate substitutions are applied in batches rather than one at a time. (default: false)" << endl;
        cout << "                   // Generally faster, but may not yield optimal results compared to single substitutions." << endl;

        cout << "    batch_size: " << (long) batch_size_
             << "  // size of the batch for batched substitution (default: 10; -1 for no limit)" << endl;

        cout << "    max_temps: " << (long) max_temps_
             << "  // maximum number of intermediates to find (default: -1 for no limit)" << endl;

        cout << "    max_depth: " << (long) Term::max_depth_
             << "  // maximum depth for chain of contractions (default: -1 for no limit)" << endl;

        cout << "    max_shape: " << Term::max_shape_.str() << " // a map of maximum sizes for each line type in an intermediate (default: {o: 255, v: 255}, "
                                                               "for no limit.): " << endl;

        cout << "    low_memory: " << (Linkage::low_memory_ ? "true" : "false")
             << "  // whether to recompute or save all possible permutations of each term in memory (default: false)" << endl
             << "                       // if true, permutations are recomputed on the fly. Recommended if memory runs out." << endl;

        cout << "    nthreads: " << nthreads_
             << "  // number of threads to use (default: OMP_NUM_THREADS | available: "
             << omp_get_max_threads() << ")" << endl;

        cout << endl;
    }

    void PQGraph::add(const pq_helper& pq, const std::string &equation_name, vector<std::string> label_order) {

        total_timer.start(); // start timer
        build_timer.start(); // start timer

        // check if equation already exists; if so, print warning
        bool equation_exists = equations_.find(equation_name) != equations_.end();
        if (equation_exists) {
            cout << "WARNING: equation '" << equation_name << "' already exists. "
                     "The terms will be merged with the existing equation." << endl;
        }

        // print that custom label order is being used
        if (!label_order.empty()) {
            cout << "Using custom label order: ";
            for (const auto &label : label_order) {
                cout << label << " ";
            }
            cout << endl;
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

        auto reorder_labels = [&label_order](MutableVertexPtr &vertex) {
            vertex->sort();
            if (label_order.empty()) return;

            // reorder lines according to label_order
            const auto &lines = vertex->lines();
            size_t rank = vertex->rank();
            vector<Line> new_lines;

            bool found[rank];
            for (size_t i = 0; i < rank; ++i)
                found[i] = false;

            for (const auto &label: label_order) {
                auto pos = find_if(lines.begin(), lines.end(), [&label](const Line &line) {
                    return line.label_ == label;
                });

                if (pos != lines.end()) {
                    new_lines.push_back(*pos);
                    size_t index = pos - lines.begin();
                    found[index] = true;
                }
            }

            for (size_t i = 0; i < rank; ++i) {
                const Line &line = lines[i];
                if (line.sig_) {
                    // check if beginning of line is a sigma operator
                    auto first_line = new_lines.begin();

                    // if beginning line has label 'L', insert sigma line after it
                    if (first_line != new_lines.end() && first_line->label_ == "L") {
                        new_lines.insert(first_line + 1, line);
                    } else {
                        // if sigma line is not found, insert it at the beginning
                        new_lines.insert(new_lines.begin(), line);
                    }
                }
                else if (!found[i])
                    new_lines.push_back(line);
            }

            vertex->update_lines(new_lines);
        };


        // loop over each pq_string
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
            bool has_self_link = term.apply_self_links();

            // skip term if it has a self-link and scalars are not allowed
            if (has_self_link && Equation::no_scalars_)
                continue;

            // use the term to build the assignment vertex
            MutableVertexPtr assignment;
            if (!name_is_formatted || equation_name.empty())
                 assignment = make_shared<Vertex>(*term.term_linkage()->shallow());
            else assignment = term.lhs()->clone();

            reorder_labels(assignment);

            // update name of assignment vertex
            assignment->vertex_type_ = '\0'; // prevents printing as a map
            assignment->update_name(assigment_name);

            // update term with assignment vertex
            term.lhs() = assignment;
            term.eq()  = assignment;


            // check if any operator in term is a sigma operator
            for (const auto &op : term.rhs()) {
                if (op->is_sigma_) {
                    // mark that this equation has sigma vectors
                    has_sigma_vecs_ = true; break;
                }
            }

            if (use_density_fitting_){
                vector<Term> density_fitted_terms = term.density_fitting();
                terms.insert(terms.end(), density_fitted_terms.begin(), density_fitted_terms.end());
            } else {
                terms.push_back(term);
            }
        }


        // build equation
        Equation& new_equation = equations_[assigment_name];
        MutableVertexPtr assignment_vertex = terms.back().lhs()->clone();

        // do not format assignment vertices as a map
        assignment_vertex->vertex_type_ = '\0'; // prevents printing as a map

        if (equation_exists) // TODO: have a check for assignment vertex consistency
             new_equation.terms().insert(new_equation.terms().end(), terms.begin(), terms.end());
        else new_equation = Equation(assignment_vertex, terms);

        // save initial scaling
        new_equation.collect_scaling();

        const scaling_map &eq_flop_map_ = new_equation.flop_map();
        const scaling_map &eq_mem_map_  = new_equation.mem_map();

        flop_map_ += eq_flop_map_;
        mem_map_  += eq_mem_map_;

        flop_map_init_ = flop_map_;
        mem_map_init_  = mem_map_;

        build_timer.stop(); // stop timer
        total_timer.stop(); // stop timer
    }

    void PQGraph::collect_scaling(bool recompute, bool include_reuse) {

        include_reuse = true;

        // reset scaling maps
        flop_map_.clear(); // clear flop scaling map
        mem_map_.clear(); // clear memory scaling map

        for (auto & [name, equation] : equations_) { // iterate over equations
            if (name == "reused" && !include_reuse)
                continue; // skip reuse_tmps equation (TODO: only include for analysis)

            // collect scaling for each equation
            equation.collect_scaling(recompute);

            const auto & flop_map = equation.flop_map(); // get flop scaling map
            const auto & mem_map = equation.mem_map(); // get memory scaling map

            flop_map_ += flop_map; // add flop scaling
            mem_map_ += mem_map; // add memory scaling
        }
    }

    vector<string> PQGraph::get_equation_keys() {
        vector<string> eq_keys(equations_.size());
        transform(equations_.begin(), equations_.end(), eq_keys.begin(),
                  [](const pair<string, Equation> &p) { return p.first; });
        return std::move(eq_keys);
    }

    vector<Term *> PQGraph::every_term() {
        vector<Term*> terms;
        size_t num_terms = 0;
        for(auto &eq : equations_){
            num_terms += eq.second.size();
        }

        terms.reserve(num_terms);
        for(auto &eq : equations_){
            for(auto &term : eq.second.terms()){
                terms.push_back(&term);
            }
        }
        return terms;
    }

    void PQGraph::reorder(bool regenerate) { // verbose if not already reordered

        total_timer.start(); // start timer
        reorder_timer.start(); // start timer

        static bool print_reordering = print_level_ >= 1; // flag to check if first reordering is printed

        print_guard guard;
        if (!print_reordering) {
            guard.lock();
        }

        // save initial scaling if never saved
        if (flop_map_init_.empty()) {
            flop_map_init_ = flop_map_;
            mem_map_init_ = mem_map_;
        }

        cout << "Reordering equations..." << flush;

        // get address of every term
        vector<Term *> terms = every_term();
        #pragma omp parallel for schedule(guided) shared(terms, regenerate) default(none)
        for (Term *term : terms) {
            term->reorder(regenerate); // reorder terms in equation
        }

        cout << " Done" << endl << endl;

        // collect scaling
        cout << "Collecting scalings of each equation...";
        collect_scaling(true); // collect scaling of equations
        cout << " Done" << endl;

        reorder_timer.stop();
        cout << "Reordering time: " << reorder_timer.elapsed() << endl << endl;

        // set reorder flags to true
        print_reordering = true;
        is_reordered_ = true;

        // save scaling after reorder
        if (flop_map_pre_.empty()) {
            flop_map_pre_ = flop_map_;
            mem_map_pre_ = mem_map_;
        }
        total_timer.stop();
    }

    void PQGraph::analysis() const {
        string h1, h2; // header 1 and header 2 padding
        if (Vertex::print_type_ == "python") {
            h1 = "####################";
            h2 = "#####";
        } else if (Vertex::print_type_ == "c++") {
            h1 = "///////////////////";
            h2 = "/////";
        } else throw invalid_argument("Invalid print type: " + Vertex::print_type_);

        cout << h1 << " PQ GRAPH Analysis " << h1 << endl << endl;

        // print total time elapsed
        long double total_time = total_timer.get_runtime();
        cout << "Net time: " << Timer::format_time(total_time) << endl << endl;

        // get total number of linkages
        size_t n_flop_ops = flop_map_.total();
        size_t n_flop_ops_pre = flop_map_pre_.total();

        size_t number_of_terms = get_num_terms();

        cout << "Total Number of Terms: " << number_of_terms;
        if (number_of_terms != num_terms_init_)
            cout << " (initial: " << num_terms_init_ << ")";
        cout << endl;
        cout << "Total Contractions: (last) " << n_flop_ops_pre << " -> (new) " << n_flop_ops << endl << endl;
        cout << "Total FLOP scaling: " << endl;
        cout << "------------------" << endl;
        print_new_scaling(flop_map_init_, flop_map_pre_, is_optimized_ ? flop_map_ : flop_map_pre_);

        cout << endl << "Total MEM scaling: " << endl;
        cout << "------------------" << endl;

        print_new_scaling(mem_map_init_, mem_map_pre_, is_optimized_ ? mem_map_ : mem_map_pre_);
        cout << endl << endl;
        cout << h1 << h1 << h1 << endl << endl;

    }

    void PQGraph::print_new_scaling(const scaling_map &original_map, const scaling_map &previous_map, const scaling_map &current_map) {
        printf("%8s : %5s | %5s | %5s || %5s | %5s\n", "Scaling", "  I  ", "  R  ", "  F  ", " F-I ", " F-R ");

        // merge spins within the scaling maps
        scaling_map orig_merged = original_map.merge_spins();
        scaling_map prev_merged = previous_map.merge_spins();
        scaling_map curr_merged = current_map.merge_spins();

        scaling_map diff_map = curr_merged - prev_merged;
        scaling_map tot_diff_map = curr_merged - orig_merged;

        auto last_order = static_cast<size_t>(-1);
        for (const auto & key : orig_merged + prev_merged + curr_merged) {
            shape cur_shape = key.first;
            size_t new_order = cur_shape.n_;
            if (new_order < last_order) {
                printf("%8s : %5s | %5s | %5s || %5s | %5s\n" , "--------", "-----", "-----", "-----", "-----", "----");
                last_order = new_order;
            }
            printf("%8s : %5zu | %5zu | %5zu || %5ld | %5ld \n", cur_shape.str().c_str(), orig_merged[cur_shape],
                   prev_merged[cur_shape], curr_merged[cur_shape], tot_diff_map[cur_shape], diff_map[cur_shape]);
        }

        printf("%8s : %5s | %5s | %5s || %5s | %5s\n" , "--------", "-----", "-----", "-----", "-----", "----");
        printf("%8s : %5zu | %5zu | %5zu || %5ld | %5ld \n", "Total", orig_merged.total(), prev_merged.total(), curr_merged.total(),
               tot_diff_map.total(), diff_map.total());

    }

    void PQGraph::assemble() {
        total_timer.start();

        // determine if sigma vectors should be formatted
        separate_sigma_ &= has_sigma_vecs_ && opt_level_ >= 3;

        // find scalars in each equation
        make_scalars();

        // set assembled flag to true
        is_assembled_ = true;
        total_timer.stop();
    }

    void PQGraph::optimize() {

        if (is_optimized_) {
            cout << "Equations have already been optimized." << endl;
            return;
        }

        print_guard guard;
        if (print_level_ < 1) {
            guard.lock();
        }

        if (flop_map_init_.empty() || mem_map_init_.empty()) {
            flop_map_init_ = flop_map_;
            mem_map_init_ = mem_map_;
        }

        // set initial number of terms
        if (num_terms_init_ == 0)
            num_terms_init_ = get_num_terms();

        // set initial scaling and format scalars
        if (!is_assembled_)
            assemble();

        // merge similar terms
        merge_terms();

        // reorder contractions in equations
        reorder();

        // save scaling after reorder
        flop_map_pre_ = flop_map_;
        mem_map_pre_ = mem_map_;

        // substitute scalars first
        if (opt_level_ >= 1) {
            cout << "----- Substituting scalars -----" << endl;
            substitute(false, true);
        }

        if (opt_level_ >= 2) {

            // find and substitute intermediate contractions
            if (separate_sigma_)
                cout << "----- Separating Intermediates for sigma-vector build -----" << endl;
            else cout << "----- Substituting intermediates -----" << endl;

            substitute(separate_sigma_, false);

            if (separate_sigma_) {
                // apply substitutions again without separating intermediates
                cout << "----- Substituting all intermediates -----" << endl;
                substitute(false, false);
            }
        }

        // clean up unused intermediates
        update_timer.start();
        merge_terms();
        prune(false);

        // set optimized flag to true
        is_optimized_ = true;

        // recollect scaling of equations
        collect_scaling(true, true);
        update_timer.stop();

        // analyze equations
        analysis();

    }

} // pdaggerq
