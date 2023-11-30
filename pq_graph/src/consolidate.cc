//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: consolidate.cc
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
#include <omp.h>

using std::ostream, std::string, std::vector, std::map, std::unordered_map, std::shared_ptr, std::make_shared,
        std::set, std::unordered_set, std::pair, std::make_pair, std::to_string, std::invalid_argument,
        std::stringstream, std::cout, std::endl, std::flush, std::max, std::min;

using namespace pdaggerq;

void PQGraph::generate_linkages(bool recompute) {

    if (recompute)
        tmp_candidates_.clear(); // clear all prior candidates

    vector<string> eq_keys = get_equation_keys();
    size_t num_subs = 0; // number of substitutions made

    omp_set_num_threads(nthreads_);

    #pragma omp parallel for schedule(guided) default(none) shared(equations_, tmp_candidates_, eq_keys) \
    firstprivate(recompute)
    for (auto & eq_name : eq_keys) { // iterate over equations in parallel
        Equation &equation = equations_[eq_name]; // get equation

        // get all linkages of equation
        linkage_set linkages = equation.generate_linkages(recompute);

        // iterate over linkages and test if they are a valid candidate
        for (const auto& contr : linkages) {
            #pragma omp critical
            {
                // add linkage to all linkages
                tmp_candidates_.insert(contr);
            }
        }

    }
    omp_set_num_threads(1);

    collect_scaling(); // collect scaling of all linkages

}

void PQGraph::substitute(bool format_sigma) {

    // reorder if not already reordered
    if (!is_reordered_) reorder();

    static Timer total_timer;

    update_timer.start();

    // save original scaling
    static bool prior_saved = false;
    if (!prior_saved) {
        flop_map_pre_ = flop_map_;
        mem_map_pre_ = mem_map_;
        prior_saved = true;
    }

    /// ensure necessary equations exist
    bool missing_temp_eq   = equations_.find("tmps")       == equations_.end();
    bool missing_reuse_eq  = equations_.find("reuse_tmps") == equations_.end();
    bool missing_scalar_eq = equations_.find("scalars")    == equations_.end();

    vector<string> missing_eqs;
    if (missing_temp_eq)   missing_eqs.emplace_back("tmps");
    if (missing_reuse_eq)  missing_eqs.emplace_back("reuse_tmps");
    if (missing_scalar_eq) missing_eqs.emplace_back("scalars");

    // add missing equations
    for (const auto& missing : missing_eqs) {
        equations_[missing] = Equation(missing);
        equations_[missing].is_temp_equation_ = true; // do not allow substitution of tmp declarations
    }


    /// format contracted scalars

    if (format_sigma) {
        for (auto &eq_pair: equations_) {
            const string &eq_name = eq_pair.first;
            Equation &equation = eq_pair.second;
            equation.form_dot_products(all_linkages_["scalars"], temp_counts_["scalars"]);
        }
        for (const auto &scalar: all_linkages_["scalars"])
            // add term to scalars equation
            add_tmp(scalar, equations_["scalars"]);
        for (Term &term: equations_["scalars"].terms())
            term.comments() = {}; // comments should be self-explanatory
    }


    /// generate all possible linkages from all arrangements
    if (verbose) cout << "Generating all possible contractions from all combinations of tensors..."  << flush;
    generate_linkages(true); // generate all possible linkages
    if (verbose) cout << " Done" << endl;

    size_t num_terms = 0;
    for (const auto& eq_pair : equations_) {
        const Equation& equation = eq_pair.second;
        num_terms += equation.size();
    }

    cout << endl;
    cout << " ==> Substituting linkages into all equations <==" << endl;
    cout << "     Total number of terms: " << num_terms << endl;
    cout << "        Total contractions: " << flop_map_.total() << endl;
    cout << "    Possible Intermediates: " << tmp_candidates_.size() << endl;
    cout << "       Use batch algorithm: " << (batched_ ? "Yes" : "No") << endl;
    cout << " ===================================================="  << endl << endl;

    static size_t total_num_merged = 0;
//    if (allow_merge_) {
        size_t num_merged = merge_terms();
        total_num_merged += num_merged;
//    }

    // initialize best flop map for all equations
    scaling_map best_flop_map = flop_map_;

    // set of linkages to ignore (start with large n_ops)
    linkage_set ignore_linkages(1024);

    // get linkages with the highest scaling (use all linkages for first iteration, regardless of batched)
    // this helps remove impossible linkages from the set without regenerating all linkages as often
    linkage_set test_linkages = tmp_candidates_;
    bool first_pass = true;

    update_timer.stop();

    bool makeSub = true; // flag to make a substitution
    static size_t totalSubs = 0;
    string temp_type = format_sigma ? "reuse_tmps" : "tmps"; // type of temporary to substitute
//    temp_counts_[temp_type] = 0; // number of temporary rhs
    while (!tmp_candidates_.empty() && temp_counts_[temp_type] < max_temps_) {
        substitute_timer.start();
        if (verbose) {
            cout << "  Remaining Test combinations: " << test_linkages.size() << endl;
//            cout << " Total Remaining combinations: " << tmp_candidates_.size() << endl;
            cout << endl << endl;
        }
        if (verbose)

        makeSub = false; // reset flag
        bool allow_equality = true; // flag to allow equality in flop map
        size_t n_linkages = test_linkages.size(); // get number of linkages
        LinkagePtr bestPreCon; // best linkage to substitute

        // populate with pairs of flop maps with linkage for each equation
        vector<pair<scaling_map, LinkagePtr>> test_data(n_linkages);

        /**
         * Iterate over all linkages in parallel and test if they can be substituted into the equations.
         * If they can, save the flop map for each equation.
         * If the flop map is better than the current best flop map, save the linkage.
         */
        omp_set_num_threads(nthreads_);
#pragma omp parallel for schedule(guided) default(none) shared(test_linkages, test_data, \
            ignore_linkages, equations_) firstprivate(n_linkages, temp_counts_, temp_type, allow_equality, format_sigma)
        for (int i = 0; i < n_linkages; ++i) {
            LinkagePtr linkage = as_link(copy_vert(test_linkages[i])); // copy linkage
            bool is_scalar = linkage->is_scalar(); // check if linkage is a scalar

            size_t temp_id;

            // set id of linkage
            linkage->id_ = (long) temp_id;

            if (format_sigma) {
                // when formatting for sigma vectors,
                // we only keep linkages without a sigma vector and are not scalars
                if (linkage->is_sigma_ || is_scalar)
                    continue;
                linkage->is_reused_ = true;
            } else {
                linkage->is_reused_ = false;
            }

            if (is_scalar)
                 temp_id = temp_counts_["scalars"] + 1; // get number of scalars
            else temp_id = temp_counts_[temp_type] + 1; // get number of temps

            scaling_map test_flop_map; // flop map for test equation
            size_t numSubs = 0; // number of substitutions made
            for (auto & eq_pair : equations_) { // iterate over equations

                // if the substitution is possible and beneficial, collect the flop map for the test equation
                const string& eq_name = eq_pair.first;

                Equation equation = eq_pair.second; // create copy to prevent thread conflicts (expensive)
                numSubs += equation.test_substitute(linkage, test_flop_map, allow_equality || is_scalar);
            }

            // add to test scalings if we found a tmp that occurs in more than one term
            // or that occurs at least once and can be reused / is a scalar

            // include declaration for scaling?
            bool include_declaration = !is_scalar && !format_sigma;

            // test if we made a valid substitution
            bool testSub = numSubs > 0;
            if (testSub) {

                // make term of tmp declaration
                if (include_declaration) {
                    Term precon_term = Term(linkage, 1.0);
                    precon_term.reorder(); // reorder term

                    // add term scaling to test the flop map
                    test_flop_map += precon_term.flop_map();
                }

                // save this test flop map and linkage for serial testing
                test_data[i] = make_pair(test_flop_map, linkage);

            } else { // if we didn't make a substitution, add linkage to ignore linkages
# pragma omp critical
                {
                    ignore_linkages.insert(linkage);
                }
            }
        } // end iterations over all linkages
        omp_set_num_threads(1);

        /**
         * Iterate over all test scalings and find the best flop map.
         */
        for (auto &test_pair : test_data) {

            scaling_map &test_flop_map = test_pair.first; // get flop map
            LinkagePtr  &test_linkage  = test_pair.second; // get linkage

            // skip empty linkages
            if (test_linkage == nullptr) continue;
            if (test_linkage->empty()) continue;

            bool is_scalar = test_linkage->is_scalar(); // check if linkage is a scalar

            // test if this is the best flop map seen
            int comparison = test_flop_map.compare(best_flop_map);
            bool keep     = comparison == scaling_map::this_better;
            bool is_equiv = comparison == scaling_map::is_same;

            if (!keep) {
                if ((is_equiv && (allow_equality || is_scalar)) || // keep if equivalent and allow equality
                        (!makeSub && format_sigma && test_linkage->is_reused_)) // keep if formatting for sigma build
                    keep = true;
            }

            if (keep) {
                bestPreCon = test_linkage; // save linkage
                best_flop_map = test_flop_map; // set best flop map
                makeSub = true; // set make substitution flag to true
            }
        }
        substitute_timer.stop(); // stop timer for substitution

        if (makeSub) {

            /**
             * we made a substitution, so we need to update the equations.
             * we need to:
             *     actually substitute the linkage in all equations
             *     store the declarations for the tmps.
             *     update the flop map and memory map.
             *     update the total number of substitutions.
             *     update the total number of terms.
             *     generate a new test set without this linkage.
             */

            // check if precon is a scalar
            bool is_scalar = bestPreCon->is_scalar();

            // get number of temps for this type
            string eq_type = is_scalar ? "scalars"
                                       : temp_type;

            // set linkage id
            size_t temp_id = ++temp_counts_[eq_type];
            bestPreCon->id_ = (long) temp_id;

            update_timer.start();

            scaling_map old_flop_map = flop_map_;

            /// substitute linkage in all equations

            omp_set_num_threads(nthreads_);
            vector<string> eq_keys = get_equation_keys();
            size_t num_subs = 0; // number of substitutions made

            #pragma omp parallel for schedule(guided) default(none) firstprivate(allow_equality, bestPreCon) \
            shared(equations_, eq_keys) reduction(+:num_subs)
            for (const auto& eq_name : eq_keys) { // iterate over equations in parallel
                // get equation
                Equation &equation = equations_[eq_name]; // get equation
                size_t this_subs = equation.substitute(bestPreCon, allow_equality);
                bool madeSub = this_subs > 0;
                if (madeSub) {
                    // sort tmps in equation
                    sort_tmps(equation);
                    num_subs += this_subs;
                }
            }
            omp_set_num_threads(1); // reset number of threads (for improved performance of non-parallel code)
            totalSubs += num_subs; // add number of substitutions to total


            // format contractions
            vector<Term *> tmp_terms = get_matching_terms(bestPreCon);

            // find common coefficients and permutations
            double common_coeff = common_coefficient(tmp_terms);

            // modify coefficients of terms
            //TODO: this messes up comments for nested tmps since the coefficients are unknown just given the linkage
            // this is no big deal, but can make the output less readable (because, you know, it wasn't already)
            for (Term* term_ptr : tmp_terms)
                term_ptr->coefficient_ /= common_coeff;

            // add linkage to equations
            const Term &precon_term = add_tmp(bestPreCon, equations_[eq_type], common_coeff);

            // print linkage
            if (verbose){
                cout << " ====> Substitution " << to_string(temp_id) << " <==== " << endl;
                cout << " ====> " << precon_term << endl << endl;
            }

            // add linkage to this set
            all_linkages_[eq_type].insert(bestPreCon); // add tmp to tmps
            ignore_linkages.insert(bestPreCon); // add linkage to ignore list

            // collect new scalings
            collect_scaling();

            num_terms = 0;
            for (const auto& eq_pair : equations_) {
                const Equation &equation = eq_pair.second;
                num_terms += equation.size();
            }

            generate_linkages(false); // add new possible linkages to test set
            tmp_candidates_ -= ignore_linkages; // remove ignored linkages
            test_linkages.clear(); // clear test set
            test_linkages = make_test_set(); // make new test set

            // remove all saved linkages
            for (const auto & link_pair : all_linkages_) {
                const linkage_set & linkages = link_pair.second;
                test_linkages -= linkages;
            }

            update_timer.stop();

            // print flop map
            if (verbose) {

                // print total time elapsed
                total_timer = substitute_timer + update_timer + build_timer + reorder_timer;

                cout << "                  Net time: "  << total_timer.elapsed() << endl;
                cout << "               Update Time: "  << update_timer.get_time() << endl;
                cout << "              Reorder Time: "  << reorder_timer.get_time() << endl;
                cout << "                 Sub. Time: "  << substitute_timer.get_time() << endl;
                cout << "         Average Sub. Time: "  << substitute_timer.average_time() << endl;
                cout << "           Number of terms: "  << num_terms << endl;
                cout << "    Number of Contractions: "  << flop_map_.total() << endl;
                cout << "        Substitution count: " << num_subs << endl;
                cout << "  Total Substitution count: " << totalSubs << endl;
                cout << endl;

//                    cout << "Total Flop scaling: " << endl;
//                    cout << "------------------" << endl;
//                    print_new_scaling(flop_map_init_, flop_map_pre_, flop_map_);
//
//                    cout << endl << endl;
//                    cout << "              Substitution Time: " << substitute_timer.get_time() << endl;
//                    cout << "      Average Substitution Time: " << substitute_timer.average_time() << endl;
//                    cout << "        Total Substitution Time: " << substitute_timer.elapsed() << endl;
//                    cout << "              Total Update Time: " << update_timer.elapsed() << endl;


            }
        }

        update_timer.start();
        // add all test linkages to ignore linkages if no substitution made
        if (!makeSub)
            ignore_linkages += test_linkages;

        // remove all saved linkages
        for (const auto & link_pair : all_linkages_) {
            const linkage_set & linkages = link_pair.second;
            ignore_linkages += linkages;
        }

        // regenerate all valid linkages
        bool remake_test_set = test_linkages.empty() || first_pass;
        if(remake_test_set){

            // merge terms
//            if (allow_merge_) {
                num_merged = merge_terms();
                total_num_merged += num_merged;
//            }

            // reapply substitutions to equations
            for (const auto & precon : all_linkages_[temp_type]) {
                for (auto &[name, equation] : equations_) {
                    if (equation.is_temp_equation_)
                        continue;
                    equation.substitute(precon, true);
                }
            }
            // repeat for scalars
            for (const auto & precon : all_linkages_["scalars"]) {
                for (auto &[name, equation] : equations_) {
                    if (equation.is_temp_equation_)
                        continue;
                    equation.substitute(precon, true);
                }
            }


            if (verbose) cout << endl << "Regenerating test set..." << flush;
            generate_linkages(true); // generate all possible linkages
            if (verbose) cout << " Done ( " << flush;

            tmp_candidates_ -= ignore_linkages; // remove ignored linkages
            test_linkages.clear(); // clear test set
            test_linkages = make_test_set(); // make new test set

            update_timer.stop();
            if (verbose) cout << update_timer.get_time() << " )" << endl;
            first_pass = false;


        } else update_timer.stop();

        // remove ignored linkages
        test_linkages   -= ignore_linkages;
        tmp_candidates_ -= ignore_linkages;

    } // end while linkage
    tmp_candidates_.clear();
    substitute_timer.stop(); // stop timer for substitution

    // recollect scaling of equations (now including sigma vectors)
    collect_scaling(true, true);

    // print total time elapsed
    total_timer = substitute_timer + update_timer + build_timer + reorder_timer;

    if (temp_counts_[temp_type] >= max_temps_)
        cout << "WARNING: Maximum number of substitutions reached. " << endl << endl;

    cout << "===> Substitution Summary <===" << endl;

    num_terms = 0;
    for (const auto& eq_pair : equations_) {
        const Equation &equation = eq_pair.second;
        num_terms += equation.size();
    }
    for (const auto & [type, count] : temp_counts_) {
        if (count == 0)
            continue;
        cout << "    Found " << count << " " << type << endl;
    }
    cout << "    Total Time: " << total_timer.elapsed() << endl;
    cout << "    Total number of terms: " << num_terms << endl;
    cout << "    Total terms merged: " << total_num_merged << endl;
    cout << "    Total contractions: " << flop_map_.total() << (format_sigma ? " (ignoring assignments of intermediates)" : "") << endl;
    cout << endl;

    cout << " ===================================================="  << endl << endl;
}

void PQGraph::sort_tmps(Equation &equation) {

    // no terms, return
    if ( equation.terms().empty() ) return;

    // to sort the tmps while keeping the order of terms without tmps, we need to
    // make a map of the equation terms and their index in the equation and sort that (so annoying)
    std::vector<pair<Term*, size_t>> indexed_terms;
    for (size_t i = 0; i < equation.terms().size(); ++i)
        indexed_terms.emplace_back(&equation.terms()[i], i);

    // sort the terms by the maximum id of the tmps in the term, then by the index of the term

    auto is_in_order = [](const pair<Term*, size_t> &a, const pair<Term*, size_t> &b) {

        const Term &a_term = *a.first;
        const Term &b_term = *b.first;

        size_t a_idx = a.second;
        size_t b_idx = b.second;

        const VertexPtr &a_lhs = a_term.lhs();
        const VertexPtr &b_lhs = b_term.lhs();

        // recursive function to get min/max id of temp ids from a vertex
        std::function<void(const VertexPtr&, long&, bool)> test_vertex;
        test_vertex = [&test_vertex](const VertexPtr &op, long& id, bool get_max) {
            if (op->is_temp()) {
                LinkagePtr link = as_link(op);
                long link_id = link->id_;

                // ignore reuse_tmp linkages
                if (!link->is_reused_) {
                    if (get_max)
                         id = std::max(id,  link_id);
                    else id = std::max(id, -link_id);
                }

                // recurse into nested tmps
                for (const auto &nested_op: link->to_vector(false, false)) {
                    test_vertex(nested_op, id, get_max);
                }
            }
        };

        // get min id of temps from lhs
        auto get_lhs_id = [&test_vertex](const Term &term, bool get_max) {
            long id = get_max ? -1l : -__FP_LONG_MAX;

            test_vertex(term.lhs(), id, get_max);

            if (get_max) return id;
            else return -id;
        };

        // get min id of temps from rhs
        auto get_rhs_id = [&test_vertex](const Term &term, bool get_max) {
            long id = get_max ? -1l : -__FP_LONG_MAX;

            for (const auto &op: term.rhs())
                test_vertex(op, id, get_max);

            if (get_max) return id;
            else return -id;
        };

        long a_max_id  = max(get_lhs_id(a_term, true),
                             get_rhs_id(a_term, true));
        long a_min_id  = min(get_lhs_id(a_term, false),
                             get_rhs_id(a_term, false));

        long b_max_id  = max(get_lhs_id(b_term, true),
                             get_rhs_id(b_term, true));
        long b_min_id  = min(get_lhs_id(b_term, false),
                             get_rhs_id(b_term, false));

        bool a_has_temp = a_max_id != -1l;
        bool b_has_temp = b_max_id != -1l;

        // if no temps, sort by index
        if (!a_has_temp && !b_has_temp)
            return a_idx < b_idx;

        // if only one has temps, keep temp last
        if (a_has_temp ^ b_has_temp)
            return b_has_temp;

        if ( a_min_id == b_min_id ) {
            if (a_max_id == b_max_id) {
                if (a.first->is_assignment_ ^ b.first->is_assignment_)
                    return a.first->is_assignment_;
                return a.second < b.second;
            }
            else return a_max_id < b_max_id;
        } else return a_min_id < b_min_id;

        // should never get here
        return true;

    };

    stable_sort(indexed_terms.begin(), indexed_terms.end(), is_in_order);

    // replace the terms in the equation with the sorted terms
    std::vector<Term> sorted_terms;
    sorted_terms.reserve(indexed_terms.size());
    for (const auto &indexed_term : indexed_terms) {
        sorted_terms.push_back(*indexed_term.first);
    }

    equation.terms() = sorted_terms;
}

void PQGraph::remove_redundant_tmps() {// remove redundant contractions (only used in one term)


    //TODO: this function is not working properly.
    // it will not substitute the correct labels and does not reindex the linkages. It has other problems as well.

//    std::map<std::string, set<std::vector<Term>::iterator>> to_remove;
//
//    for (auto & [type, contractions] : all_linkages_) {
//
//        std::map<std::vector<Term>::iterator, LinkagePtr> to_replace;
//        for (const auto &contraction : contractions) {
//            std::vector<Term>::iterator term_it = std::vector<Term>::iterator();
//            bool only_one = true;
//
//            // find contractions that are only used in one term
//            bool found = false;
//            for (auto &[name, eq] : equations_) {
//                vector<Term*> terms = eq.get_temp_terms(contraction);
//                if (terms.size() == 1 && !found) {
//                    // use pointer to find iterator in the equation
//                    term_it = find_if(eq.terms().begin(), eq.terms().end(),
//                                      [&terms](const Term &term) { return &term == terms[0]; });
//                    found = true;
//                } else if (terms.size() > 1 || found) {
//                    only_one = false; break;
//                }
//            }
//
//            if (only_one && found)
//                to_replace[term_it] = contraction;
//        }
//
//
//
//        // replace contractions
//        Equation &tmp_eq = equations_[type];
//        for (auto & [term_it, contraction] : to_replace) {
//            Term &term = *term_it;
//
//            // find declarations to remove
//            double original_coeff = 1;
//            bool found = false;
//            for (size_t i = 0; i < tmp_eq.size(); i++) {
//                Term &tmpterm = tmp_eq.terms()[i];
//                if (tmpterm.lhs()->is_temp()) {
//                    const LinkagePtr &link = as_link(tmpterm.lhs());
//                    if (link->id_ == contraction->id_) {
//                        // get iterator of this term
//                        auto declare_it = tmp_eq.terms().begin() + i;
//                        found = true;
//                        original_coeff = tmpterm.coefficient_;
//                        to_remove[type].insert(declare_it);
//                    }
//                }
//            }
//            if (!found)
//                throw std::runtime_error("Could not find declaration for contraction: " + contraction->str());
//
//
//            // remove contraction from rhs of term
//            std::vector<VertexPtr> new_rhs;
//            new_rhs.reserve(term.rhs().size() + contraction->depth());
//            for (const auto &vertex : term.rhs()) {
//                if (vertex->is_temp()){
//                    const LinkagePtr &link = as_link(vertex);
//                    if (link->id_ == contraction->id_) {
//                        const auto &new_verts = link->to_vector();
//                        new_rhs.insert(new_rhs.end(), new_verts.begin(), new_verts.end());
//                        continue;
//                    }
//                }
//                new_rhs.push_back(vertex);
//            }
//
//            // set new rhs
//            term.rhs() = new_rhs;
//            term.coefficient_ *= original_coeff;
//            term.reorder(true);
//            term.reset_comments();
//        }
//    }
//
//    // remove declarations
//    for (auto & [type, decls] : to_remove) {
//        Equation &tmp_eq = equations_[type];
//        vector<Term> &terms = tmp_eq.terms();
//
//        // get distance of each declaration from the beginning of the equation
//        set<long> distances;
//        for (const auto &decl : decls) {
//            distances.insert(std::distance(terms.begin(), decl));
//        }
//
//        // remove declarations in reverse order
//        for (auto it = distances.rbegin(); it != distances.rend(); it++) {
//            terms.erase(terms.begin() + *it);
//            temp_counts_[type]--;
//        }
//    }
//
//    collect_scaling(true); // collect new scalings

//
//    // sort tmps
//    for (auto & [type, eq] : equations_) {
//        sort_tmps(eq);
//    }
//
//    // reindex all linkages
//    for (auto & [type, contractions] : all_linkages_) {
//        linkage_set new_linkages;
//
//        std::map<size_t, LinkagePtr> id_map;
//        for (const auto &contraction : contractions) {
//            // find first occurence of this contraction
//            bool found = false;
//            for (size_t i = 0; i < equations_[type].size(); i++) {
//                Term &term = equations_[type].terms()[i];
//                if (term.lhs()->is_temp()) {
//                    const LinkagePtr &link = as_link(term.lhs());
//                    if (link->id_ == contraction->id_) {
//                        id_map[i] = contraction;
//                        found = true;
//                        break;
//                    }
//                }
//            }
//        }
//
//        // sort id_map by id
//        std::vector<std::pair<size_t, LinkagePtr>> id_vec(id_map.begin(), id_map.end());
//        std::sort(id_vec.begin(), id_vec.end());
//
//        // reindex linkages
//        long id = 1;
//        for (auto & [old_id, linkage] : id_vec) {
//            linkage->id_ = id++;
//            new_linkages.insert(linkage);
//
//            // replace every occurence of this contraction with the new id
//            for (auto & [name, eq] : equations_) {
//                for (auto &term : eq.terms()) {
//                    if (term.lhs() == linkage) {
//                        term.lhs() = linkage;
//                    }
//                    for (auto &vertex : term.rhs()) {
//                        if (vertex == linkage) {
//                            vertex = linkage;
//                        }
//                    }
//                }
//            }
//        }
//
//        contractions = new_linkages;

}

vector<Term *> PQGraph::get_matching_terms(const LinkagePtr &contraction) {// grab all terms with this tmp

    // initialize vector of term pointers
    vector<Term*> tmp_terms;

    omp_set_num_threads(nthreads_);
    vector<string> eq_keys = get_equation_keys();
    #pragma omp parallel for schedule(guided) default(none) shared(equations_, eq_keys, tmp_terms, contraction)
    for (const auto& eq_name : eq_keys) { // iterate over equations in parallel
        // get equation
        Equation &equation = equations_[eq_name]; // get equation

        // get all terms with this tmp
        vector<Term*> tmp_terms_local = equation.get_temp_terms(contraction);
        #pragma omp critical
        {
            // add terms to tmp_terms
            tmp_terms.insert(tmp_terms.end(),
                             tmp_terms_local.begin(), tmp_terms_local.end());
        }

    }
    omp_set_num_threads(1); // reset number of threads (for improved performance of non-parallel code)
    return tmp_terms;
}

void PQGraph::expand_permutations(){
    //TODO: make each permutation into a separate equation
    for (auto & [name, eq] : equations_) {
        eq.expand_permutations();
    }
}

size_t PQGraph::merge_terms() {

    if (verbose) cout << "Merging similar terms:" << endl;

    // iterate over equations and merge terms
    size_t num_fuse = 0;
    omp_set_num_threads(nthreads_);
    vector<string> eq_keys = get_equation_keys();
    #pragma omp parallel for reduction(+:num_fuse) default(none) shared(equations_, eq_keys)
    for (const auto &key: eq_keys) {
        Equation &eq = equations_[key];
        if (eq.name() == "tmps") continue; // skip tmps equation
        if (eq.assignment_vertex()->rank() == 0) continue; // skip if lhs vertex is scalar
        num_fuse += eq.merge_terms(); // merge terms with same rhs up to a permutation
    }
    omp_set_num_threads(1);
    collect_scaling(); // collect new scalings

    if (verbose) cout << "Done (" << num_fuse << " terms merged)" << endl << endl;

    return num_fuse;
}

double PQGraph::common_coefficient(vector<Term*> &terms) {

    // make a count_ of the reciprocal of the coefficients of the terms
    map<size_t, size_t> reciprocal_counts;
    for (Term* term_ptr: terms) {
        Term& term = *term_ptr;
        auto reciprocal = static_cast<size_t>(round(1.0 / fabs(term.coefficient_)));
        reciprocal_counts[reciprocal]++;
    }

    // find the most common reciprocal
    size_t most_common_reciprocal = 1; // default to 1
    size_t most_common_reciprocal_count = 1;
    for (const auto &reciprocal_count: reciprocal_counts) {
        if (reciprocal_count.first <= 0) continue; // skip 0 values (generally doesn't happen)
        if (reciprocal_count.second > most_common_reciprocal_count) {
            most_common_reciprocal = reciprocal_count.first;
            most_common_reciprocal_count = reciprocal_count.second;
        }
    }
    double common_coefficient = 1.0 / static_cast<double>(most_common_reciprocal);
    return common_coefficient;
}

perm_list PQGraph::common_permutations(const vector<Term *>& terms) {
    vector<pair<string, string>> common_perms;
    size_t perm_type = 0;

    for (Term* term_ptr: terms) {
        Term& term = *term_ptr;
        perm_list term_perms = term.term_perms();

        if (term_perms.empty()) // no common permutations possible
            return {};

        if (common_perms.empty()) {
            // we haven't found any permutations yet
            // so initialize the common permutations with this one
            common_perms = term_perms;
            perm_type = term.perm_type();
            continue;
        }

        if (perm_type != term.perm_type()) {
            // the permutation type has changed
            // so we can't have any common permutations
            return {};
        }

        // find common permutations
        for (size_t i = 0; i < common_perms.size(); i++) {
            pair<string, string> perm = common_perms[i];

            // check if this permutation is in the common permutations
            bool found = false;
            for (const auto &term_perm : term_perms) {
                if (perm == term_perm) {
                    found = true;
                    break;
                }
            }

            // if not found, remove from common permutations
            if (!found) {
                common_perms.erase(common_perms.begin() + i);
                i--;
            }

            // no common permutations found
            if (common_perms.empty())
                return {};
        }
    }

    return common_perms;
}

//}
