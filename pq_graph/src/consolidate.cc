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

using std::logic_error;
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

using namespace pdaggerq;

void PQGraph::generate_linkages(bool recompute) {

    if (recompute)
        tmp_candidates_.clear(); // clear all prior candidates

    // iterate over equations
    for (auto & eq_pair : equations_) {
        Equation &equation = eq_pair.second;

        // get all linkages of equation
        linkage_set linkages = equation.generate_linkages(recompute);

        // iterate over linkages and test if they are a valid candidate
        for (const auto& contr : linkages) {

            // add linkage to all linkages
            tmp_candidates_.insert(contr);
        }

    }

    collect_scaling(); // collect scaling of all linkages

}

void PQGraph::substitute() {

    // reorder if not already reordered
    if (!is_reordered_) reorder();

    substitute_timer.start();

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
        equations_[missing].allow_substitution_ = false; // do not allow substitution of tmp declarations
    }


    /// format contracted scalars
    for (auto &eq_pair : equations_){
        const string& eq_name = eq_pair.first;
        Equation& equation = eq_pair.second;
        equation.form_dot_products(all_linkages_["scalars"], temp_counts_["scalars"]);
    }
    for (const auto & scalar : all_linkages_["scalars"])
        // add term to scalars equation
        add_tmp(scalar, equations_["scalars"]);
    for (Term& term : equations_["scalars"].terms())
        term.comments() = {}; // comments should be self-explanatory



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
    size_t total_num_merged = 0;

    // initialize best flop map for all equations
    scaling_map best_flop_map = flop_map_;

    // set of linkages to ignore (start with large n_ops)
    linkage_set ignore_linkages(1024);

    // get linkages with the highest scaling (use all linkages for first iteration, regardless of batched)
    // this helps remove impossible linkages from the set without regenerating all linkages as often
    linkage_set test_linkages = tmp_candidates_;
    bool first_pass = true;

    substitute_timer.stop();

    bool makeSub = true; // flag to make a substitution
    size_t totalSubs = 0;
    string temp_type = "tmps"; // type of temporary to substitute
    temp_counts_[temp_type] = 0; // number of temporary rhs
    while (!tmp_candidates_.empty() && temp_counts_[temp_type] < max_temps_) {
        substitute_timer.start();
        if (verbose) cout << "  Remaining Test combinations: " << test_linkages.size() << endl;
        if (verbose) cout << " Total Remaining combinations: " << tmp_candidates_.size() << endl << endl << endl;

        makeSub = false; // reset flag
        bool allow_equality = false; // flag to allow equality in flop map
        size_t n_linkages = test_linkages.size(); // get number of linkages
        LinkagePtr bestPreCon; // best linkage to substitute

        // populate with pairs of flop maps with linkage for each equation
        vector<pair<scaling_map, LinkagePtr>> test_data(n_linkages);

        /**
         * Iterate over all linkages in parallel and test if they can be substituted into the equations.
         * If they can, save the flop map for each equation.
         * If the flop map is better than the current best flop map, save the linkage.
         */
        omp_set_num_threads(num_threads_);
#pragma omp parallel for schedule(guided) default(none) shared(test_linkages, test_data, \
            ignore_linkages, equations_) firstprivate(n_linkages, temp_counts_, temp_type, allow_equality)
        for (int i = 0; i < n_linkages; ++i) {
            LinkagePtr linkage = as_link(copy_vert(test_linkages[i])); // copy linkage
            bool is_scalar = linkage->is_scalar(); // check if linkage is a scalar

            size_t temp_id;
            if (is_scalar)
                temp_id = temp_counts_["scalars"] + 1; // get number of scalars
            else temp_id = temp_counts_[temp_type] + 1; // get number of temps

            // set id of linkage
            linkage->id_ = (long) temp_id;

            scaling_map test_flop_map; // flop map for test equation
            size_t numSubs = 0; // number of substitutions made
            for (auto & eq_pair : equations_) { // iterate over equations

                // if the substitution is possible and beneficial, collect the flop map for the test equation
                const string& eq_name = eq_pair.first;
                if (!eq_pair.second.allow_substitution_) continue; // skip tmps equation

                Equation equation = eq_pair.second; // create copy to prevent thread conflicts (expensive)
                numSubs += equation.test_substitute(linkage, test_flop_map, allow_equality || is_scalar);
            }

            // add to test scalings if we found a tmp that occurs in more than one term
            // or that occurs at least once and can be reused / is a scalar

            // include declaration for scaling?
            bool include_declaration = !is_scalar;

            // test if we made a valid substitution
            bool testSub = include_declaration ? numSubs > 1 : numSubs > 0;
            if (testSub) {
                // make term of tmp declaration
                if (include_declaration) {
                    Term precon_term = Term(linkage);
                    precon_term.reorder(); // reorder term

                    // add term scaling to test the flop map
                    test_flop_map += precon_term.flop_map();
                }

                // save this test flop map and linkage for serial testing
                test_data[i] = make_pair(test_flop_map, linkage);

            } else if (numSubs == 0) { // if we didn't make a substitution, add linkage to ignore linkages
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

            if ( is_equiv && (allow_equality || is_scalar) )
                keep = true;

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

            // print linkage
            if (verbose){
                cout << " ====> Substitution " << to_string(temp_id) << " <==== " << endl;

                // format as a term
                Term precon_term = Term(bestPreCon);
                precon_term.comments() = {}; // clear comments
                cout << " ====> " << precon_term << endl << endl;
            }
            update_timer.start();

            scaling_map old_flop_map = flop_map_;

            /// substitute linkage in all equations

            omp_set_num_threads(num_threads_);
            vector<string> eq_keys = get_equation_keys();
            size_t num_subs = 0; // number of substitutions made
#pragma omp parallel for schedule(guided) default(none) firstprivate(allow_equality, bestPreCon) \
                shared(equations_, eq_keys) reduction(+:num_subs)
            for (const auto& eq_name : eq_keys) { // iterate over equations in parallel
                // get equation name
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

            // add linkage to equations
            add_tmp(bestPreCon, equations_[eq_type]); // add tmp to equations

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
            update_timer.stop();

            // print flop map
            if (verbose) {

                // print total time elapsed
                long double total_time = build_timer.get_runtime() + reorder_timer.get_runtime()
                                         + substitute_timer.get_runtime() + update_timer.get_runtime();
                cout << "                      Time: "  << substitute_timer.get_time() << endl;
                cout << "                  Net time: "  << Timer::format_time(total_time) << endl;
                cout << "              Average Time: "  << Timer::format_time(
                        total_time / (double) substitute_timer.count()
                ) << endl;
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
        if (!makeSub) ignore_linkages += test_linkages;

        // remove ignored linkages from test set
        test_linkages -= ignore_linkages;

        // remove all saved linkages
        for (const auto & link_pair : all_linkages_) {
            const linkage_set & linkages = link_pair.second;
            test_linkages -= linkages;
        }

        // regenerate all valid linkages
        bool remake_test_set = test_linkages.empty() || first_pass;
        if(remake_test_set){

            // reapply substitutions to equations
            substitute_timer.start();
            for (const auto & precon : all_linkages_[temp_type]) {
                for (auto &eq_pair : equations_) {
                    Equation &equation = eq_pair.second;
                    equation.substitute(precon, true);
                }
            }
            // repeat for scalars
            for (const auto & precon : all_linkages_["scalars"]) {
                for (auto &eq_pair : equations_) {
                    Equation &equation = eq_pair.second;
                    equation.substitute(precon, true);
                }
            }

            if (verbose) cout << endl << "Regenerating test set..." << flush;
            generate_linkages(false); // generate all possible linkages
            if (verbose) cout << " Done ( " << flush;

            tmp_candidates_ -= ignore_linkages; // remove ignored linkages
            test_linkages.clear(); // clear test set
            test_linkages = make_test_set(); // make new test set

            update_timer.stop();
            if (verbose) cout << update_timer.get_time() << " )" << endl;
            first_pass = false;


        } else update_timer.stop();
    } // end while linkage
    tmp_candidates_.clear();
    substitute_timer.stop(); // stop timer for substitution

    // print total time elapsed
    long double total_time = build_timer.get_runtime() + reorder_timer.get_runtime()
                             + substitute_timer.get_runtime() + update_timer.get_runtime();

    if (temp_counts_[temp_type] >= max_temps_)
        cout << "WARNING: Maximum number of substitutions reached. " << endl << endl;

    for (const auto & [type, count] : temp_counts_) {
        if (count == 0)
            continue;
        cout << "     Found " << count << " " << type << endl;
    }

    num_terms = 0;
    for (const auto& eq_pair : equations_) {
        const Equation &equation = eq_pair.second;
        num_terms += equation.size();
    }
    cout << "     Total number of terms: " << num_terms << endl;
    cout << endl;

    cout << " ===================================================="  << endl << endl;
}

void PQGraph::expand_permutations(){
    for (auto & [name, eq] : equations_) {
        eq.expand_permutations();
    }
}




//
//
//    size_t PQGraph::merge_terms() {
//
//        if (verbose) cout << "Merging similar terms:" << endl;
//
//        // iterate over equations and merge terms
//        size_t num_fuse = 0;
//        omp_set_num_threads(num_threads_);
//        vector<string> eq_keys = get_equation_keys();
//        #pragma omp parallel for reduction(+:num_fuse) default(none) shared(equations_, eq_keys)
//        for (const auto &key: eq_keys) {
//            Equation &eq = equations_[key];
//            if (eq.name() == "tmps") continue; // skip tmps equation
//            if (eq.assignment_vertex()->rank() == 0) continue; // skip if lhs vertex is scalar
//            num_fuse += eq.merge_terms(); // merge terms with same rhs up to a permutation
//        }
//        omp_set_num_threads(1);
//        collect_scaling(); // collect new scalings
//
//        if (verbose) cout << "Done (" << num_fuse << " terms merged)" << endl << endl;
//
//        return num_fuse;
//    }
//
//    void PQGraph::merge_permutations() {
//
//        /*
//         * This function merges the permutation containers of the equations in the tabuilder.
//         * This way, we can just add terms to the same type of permutation and permute at the very end.
//         */
//        if (is_reused_) throw logic_error("Cannot merge permutation containers of reused equations.");
//
//        // merge permutation containers for each equation
//        omp_set_num_threads(num_threads_);
//        vector<string> eq_keys = get_equation_keys();
//        #pragma omp parallel for default(none) shared(equations_, eq_keys)
//        for (const auto &key: eq_keys) {
//            Equation &eq = equations_[key];
//            eq.merge_permutations();
//        }
//        omp_set_num_threads(1);
//
//        has_perms_merged_ = true;
//
//    }
//
//    double PQGraph::common_coefficient(vector<Term> &terms) {
//
//        // make a count_ of the reciprocal of the coefficients of the terms
//        map<size_t, size_t> reciprocal_counts;
//        for (const Term &term: terms) {
//            auto reciprocal = static_cast<size_t>(round(1.0 / fabs(term.coefficient_)));
//            reciprocal_counts[reciprocal]++;
//        }
//
//        // find the most common reciprocal
//        size_t most_common_reciprocal = 1; // default to 1
//        size_t most_common_reciprocal_count = 1;
//        for (const auto &reciprocal_count: reciprocal_counts) {
//            if (reciprocal_count.first <= 0) continue; // skip 0 values (generally doesn't happen)
//            if (reciprocal_count.second > most_common_reciprocal_count) {
//                most_common_reciprocal = reciprocal_count.first;
//                most_common_reciprocal_count = reciprocal_count.second;
//            }
//        }
//        double common_coefficient = 1.0 / static_cast<double>(most_common_reciprocal);
//        return common_coefficient;
//    }
//}
