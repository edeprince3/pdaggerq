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

void PQGraph::generate_linkages(bool recompute, bool format_sigma) {

    if (recompute)
        tmp_candidates_.clear(); // clear all prior candidates

    size_t num_subs = 0; // number of substitutions made

    omp_set_num_threads(nthreads_);
    for (auto & [eq_name, equation] : equations_) { // iterate over equations in parallel
        // get all linkages of equation and add to candidates
        tmp_candidates_ += equation.generate_linkages(recompute);
    }

    omp_set_num_threads(1);

    // collect scaling of all linkages
    collect_scaling(true);

}

linkage_set PQGraph::make_test_set() {
      /// deprecated

//    if (!batched_)
        return tmp_candidates_; // if not batched, return all candidates
//
//    static linkage_set test_linkages(1024); // set of linkages to test (start with medium n_ops)
//    test_linkages.clear();
//
//    shape worst_scale; // worst cost (start with zero)
//    for (const auto & linkage : tmp_candidates_) { // get worst cost
//        if (linkage->flop_scale() > worst_scale)
//            worst_scale = linkage->flop_scale();
//    }
//
//    size_t max_size = __FP_LONG_MAX; // maximum n_ops of linkage found (start with 0)
//    for (const auto & linkage : tmp_candidates_) { // iterate over all linkages
//
//        if (linkage->flop_scale() >= worst_scale) {
//            if (linkage->depth() <= max_size ) { // we want to grab the smallest linkages first (easier to nest)
//                test_linkages.insert(linkage); // add linkage to the test set
//                max_size = linkage->depth(); // update maximum n_ops
//            }
//        }
//    }
//
//    return test_linkages; // return test linkages
}

Term& PQGraph::add_tmp(const ConstLinkagePtr& precon, Equation &equation, double coeff) {
    // make term with tmp
    equation.terms().insert(equation.end(), Term(precon, coeff));
    return equation.terms().back();
}

void PQGraph::substitute(bool format_sigma) {

    // begin timings
    static Timer total_timer;
    total_timer.start();

    // reorder if not already reordered
    if (!is_reordered_) reorder();

    update_timer.start();

    /// ensure necessary equations exist
    bool missing_temp_eq   = equations_.find("tmps")    == equations_.end();
    bool missing_reuse_eq  = equations_.find("reuse")   == equations_.end();
    bool missing_scalar_eq = equations_.find("scalars") == equations_.end();

    vector<string> missing_eqs;
    if (missing_temp_eq)   missing_eqs.emplace_back("tmps");
    if (missing_reuse_eq)  missing_eqs.emplace_back("reuse");
    if (missing_scalar_eq) missing_eqs.emplace_back("scalars");

    // add missing equations
    for (const auto& missing : missing_eqs) {
        equations_[missing] = Equation(missing);
        equations_[missing].is_temp_equation_ = true; // do not allow substitution of tmp declarations
    }


    /// format contracted scalars
    static bool made_scalars = false;
    if (!made_scalars) {
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

        made_scalars = true;
    }


    /// generate all possible linkages from all arrangements
    if (verbose) cout << "Generating all possible contractions from all combinations of tensors..."  << flush;
    generate_linkages(true, format_sigma); // generate all possible linkages
    if (verbose) cout << " Done" << endl;

    size_t num_terms = 0;
    for (const auto& eq_pair : equations_) {
        const Equation& equation = eq_pair.second;
        num_terms += equation.size();
    }

    size_t num_contract = flop_map_.total();

    cout << endl;
    cout << " ==> Substituting linkages into all equations <==" << endl;
    cout << "     Total number of terms: " << num_terms << endl;
    cout << "        Total contractions: " << flop_map_.total() << endl;
    cout << "     Use batched algorithm: " << (batched_ ? "yes" : "no") << endl;
    if (batched_)
        cout << "                Batch size: " << ((long) batch_size_ == -1 ? "no limit" : to_string(batch_size_)) << endl;
    cout << "         Max linkage depth: " << ((long)Term::max_depth_ == -1 ? "no limit" : to_string(Term::max_depth_)) << endl;
    cout << "    Possible intermediates: " << tmp_candidates_.size() << endl;
    cout << "    Number of threads used: "  << nthreads_ << endl;
    cout << " ===================================================="  << endl << endl;

    // give user a warning if the number of possible linkages is large
    // suggest using the batch algorithm, making the max linkage smaller, or increasing number of threads
    if (tmp_candidates_.size()*num_contract > 1000*10000) {
        cout << "WARNING: There are a large number of contractions and candidate intermediates." << endl;
        cout << "         This may take a long time to run." << endl;
        cout << "         Consider increasing the number of threads, making the max depth smaller, or using the batch algorithm." << endl;
        cout << endl; //185
    }

    static size_t total_num_merged = 0;
    size_t num_merged = 0;
    if (allow_merge_ && !format_sigma) {
        num_merged = merge_terms();
        total_num_merged += num_merged;
    }

    // initialize best flop map for all equations
    collect_scaling(true);

    // set of linkages to ignore
    linkage_set ignore_linkages(tmp_candidates_.size());

    // get linkages with the highest scaling (use all linkages for first iteration, regardless of batched)
    // this helps remove impossible linkages from the set without regenerating all linkages as often
    linkage_set test_linkages = tmp_candidates_;
    bool first_pass = true;

    update_timer.stop();

    scaling_map best_flop_map = flop_map_;
    static size_t totalSubs = 0;
    string temp_type = format_sigma ? "reuse" : "tmps"; // type of temporary to substitute

    bool makeSub = false; // flag to make a substitution
    while (!tmp_candidates_.empty() && temp_counts_[temp_type] < max_temps_) {
        substitute_timer.start();

        makeSub = false; // reset flag
        bool allow_equality = true; // flag to allow equality in flop map
        size_t n_linkages = test_linkages.size(); // get number of linkages
        LinkagePtr link_to_sub; // best linkage to substitute

        // populate with pairs of flop maps with linkage for each equation
        vector<pair<scaling_map, LinkagePtr>> test_data(n_linkages);


        // print ratio for showing progress
        size_t print_ratio = n_linkages / 20;
        bool print_progress = n_linkages > 200 && verbose;

        if (print_progress)
            cout << "PROGRESS:" << endl;

        /**
         * Iterate over all linkages in parallel and test if they can be substituted into the equations.
         * If they can, save the flop map for each equation.
         * If the flop map is better than the current best flop map, save the linkage.
         */
        omp_set_num_threads(nthreads_);
#pragma omp parallel for schedule(guided) default(none) shared(test_linkages, test_data, \
            ignore_linkages, equations_, stdout) firstprivate(n_linkages, temp_counts_, temp_type, allow_equality, \
            format_sigma, print_ratio, print_progress)
        for (int i = 0; i < n_linkages; ++i) {

            // copy linkage
            LinkagePtr linkage = as_link(test_linkages[i]->deep_copy_ptr());
            bool is_scalar = linkage->is_scalar(); // check if linkage is a scalar


            if (format_sigma && linkage->is_sigma_) {
                // when formatting for sigma vectors,
                // we only keep linkages without a sigma vector and are not scalars
                ignore_linkages.insert(linkage);
                continue;
            }
            linkage->is_reused_ = format_sigma;

            // set id of linkage
            size_t temp_id;
            if (is_scalar)
                 temp_id = temp_counts_["scalars"] + 1; // get number of scalars
            else temp_id = temp_counts_[temp_type] + 1; // get number of temps
            linkage->id_ = (long) temp_id;

            scaling_map test_flop_map; // flop map for test equation
            size_t numSubs = 0; // number of substitutions made
            for (auto & eq_pair : equations_) { // iterate over equations

                const string &eq_name = eq_pair.first; // get equation name
                Equation &equation = eq_pair.second; // get equation

                // if the substitution is possible and beneficial, collect the flop map for the test equation
                numSubs += equation.test_substitute(linkage, test_flop_map, allow_equality || is_scalar || format_sigma);
            }

            // add to test scalings if we found a tmp that occurs in more than one term
            // or that occurs at least once and can be reused / is a scalar

            // include declaration for scaling?
            bool keep_declaration = !is_scalar && !format_sigma;

            // test if we made a valid substitution
            int thresh = keep_declaration ? 1 : 0;
            if (numSubs > thresh ) {

                // make term of tmp declaration
                Term precon_term = Term(linkage, 1.0);
                precon_term.reorder(); // reorder term

                if (keep_declaration) {
                    // add term scaling to test the flop map
                    test_flop_map += precon_term.flop_map();
                } else {
                    // remove from test flop map (since it will be extracted)
                    test_flop_map -= precon_term.flop_map();
                }

                // save this test flop map and linkage for serial testing
                test_data[i] = make_pair(test_flop_map, linkage);

            } else { // if we didn't make a substitution, add linkage to ignore linkages
                ignore_linkages.insert(linkage);
            }

            if (print_progress && i % print_ratio == 0) {
                printf("  %2.1lf%%", (double) i / (double) n_linkages * 100);
                std::fflush(stdout);
            }

        } // end iterations over all linkages
        omp_set_num_threads(1);
        std::cout << std::endl << std::endl;



        /**
         * Iterate over all test scalings, remove incompatible ones, and sort them
         */

        std::multimap<scaling_map, LinkagePtr> sorted_test_data;
        for (auto &[test_flop_map, test_linkage] : test_data) {

            // skip empty linkages
            if (test_linkage == nullptr) continue;
            if (test_linkage->empty()) continue;

            if (test_flop_map > flop_map_) {
                // remove the linkage completely if the scaling only got worse
                ignore_linkages.insert(test_linkage);
                continue;
            }

            bool is_scalar = test_linkage->is_scalar(); // check if linkage is a scalar

            // test if this is the best flop map seen
            int comparison = test_flop_map.compare(flop_map_);
            bool is_equiv = comparison == scaling_map::is_same;
            bool keep = comparison == scaling_map::this_better;

            // if we haven't made a substitution yet and this is either a
            // scalar or a sigma vector, keep it
            if (!makeSub && (format_sigma || is_scalar)) keep = true;

            // if the scaling is the same and it is allowed, set keep to true
            if (!keep && is_equiv && allow_equality) keep = true;


            if (keep) {
//                link_to_sub = test_linkage; // save linkage
//                best_flop_map = test_flop_map; // set best flop map
//                makeSub = true; // set make substitution flag to true

                sorted_test_data.insert(make_pair(test_flop_map, test_linkage));
            } else {
                ignore_linkages.insert(test_linkage); // add linkage to ignore linkages
            }
        }
        substitute_timer.stop(); // stop timer for substitution

        makeSub = !sorted_test_data.empty();
        if (makeSub) {

            /**
             * we found substitutions, so we need to update the equations.
             * we need to:
             *     actually substitute the linkage in all equations
             *     store the declarations for the tmps.
             *     update the flop map and memory map.
             *     update the total number of substitutions.
             *     update the total number of terms.
             *     generate a new test set without this linkage.
             * we do this for every linkage that would improve the scaling in order of the linkage that
             * improves the scaling the most.
             */

            update_timer.start();

            size_t batch_count = 0;
            for (const auto &[found_flop, found_linkage] : sorted_test_data){

                substitute_timer.start();

                link_to_sub = found_linkage;

                // check if link is a scalar
                bool is_scalar = link_to_sub->is_scalar();

                // get number of temps for this type
                string eq_type = is_scalar ? "scalars"
                                           : temp_type;

                // set linkage id
                size_t temp_id = ++temp_counts_[eq_type];
                link_to_sub->id_ = (long) temp_id;

                scaling_map last_flop_map = flop_map_;

                /// substitute linkage in all equations

                omp_set_num_threads(nthreads_);
                vector<string> eq_keys = get_equation_keys();
                size_t num_subs = 0; // number of substitutions made

                #pragma omp parallel for schedule(guided) default(none) firstprivate(allow_equality, link_to_sub) \
                shared(equations_, eq_keys) reduction(+:num_subs)
                for (const auto& eq_name : eq_keys) { // iterate over equations in parallel
                    // get equation
                    Equation &equation = equations_[eq_name]; // get equation
                    size_t this_subs = equation.substitute(link_to_sub, allow_equality);
                    bool madeSub = this_subs > 0;
                    if (madeSub) {
                        // sort tmps in equation
                        sort_tmps(equation);
                        num_subs += this_subs;
                    }
                }
                omp_set_num_threads(1); // reset number of threads (for improved performance of non-parallel code)
                totalSubs += num_subs; // add number of substitutions to total

                // prepares the next batch of substitutions
                auto remake_candidates = [this, &ignore_linkages, & test_linkages, format_sigma](){

                    // collect new scalings
                    collect_scaling();

                    // add new possible linkages to test set
                    generate_linkages(false, format_sigma);
                    tmp_candidates_ -= ignore_linkages; // remove ignored linkages
                    test_linkages.clear(); // clear test set
                    test_linkages = make_test_set(); // make new test set

                    // remove all saved linkages
                    for (const auto & link_pair : all_linkages_) {
                        const linkage_set & linkages = link_pair.second;
                        test_linkages -= linkages;
                    }
                };


                if (num_subs == 0) {
                    // if we didn't make a substitution, add linkage to ignore linkages
                    ignore_linkages.insert(link_to_sub);
                    temp_counts_[eq_type]--;
                    remake_candidates();
                    substitute_timer.stop();
                    continue;
                }

                // collect new scaling
                collect_scaling();

                // format contractions
                vector<Term *> tmp_terms = get_matching_terms(link_to_sub);

                // find common coefficients and permutations
                double common_coeff = common_coefficient(tmp_terms);

                // modify coefficients of terms
                for (Term* term_ptr : tmp_terms)
                    term_ptr->coefficient_ /= common_coeff;

                // add linkage to equations
                const Term &precon_term = add_tmp(link_to_sub, equations_[eq_type], common_coeff);

                // print linkage
                if (verbose){
                    cout << " ====> Substitution " << to_string(temp_id) << " <==== " << endl;
                    cout << " ====> " << precon_term << endl;
                    cout << " Difference: " << flop_map_ - last_flop_map << std::endl << endl;
                }

                // add linkage to this set
                all_linkages_[eq_type].insert(link_to_sub); // add tmp to tmps
                ignore_linkages.insert(link_to_sub); // add linkage to ignore list

                num_terms = 0;
                for (const auto& eq_pair : equations_) {
                    const Equation &equation = eq_pair.second;
                    num_terms += equation.size();
                }

                // prepare the next batch of substitutions
                remake_candidates();

                // print flop map
                if (verbose) {

                    // print total time elapsed
                    substitute_timer.stop();
                    update_timer.stop();
                    total_timer.stop();

                    cout << "                  Net time: " << total_timer.elapsed() << endl;
                    cout << "              Reorder Time: " << reorder_timer.elapsed() << endl;
                    cout << "               Update Time: " << update_timer.elapsed() << endl;
                    cout << "         Average Sub. Time: " << substitute_timer.average_time() << endl;
                    cout << "           Number of terms: " << num_terms << endl;
                    cout << "    Number of Contractions: " << flop_map_.total() << endl;
                    cout << "        Substitution count: " << num_subs << endl;
                    cout << "  Total Substitution count: " << totalSubs << endl;
                    cout << "      Remaining candidates: " << test_linkages.size() << endl;
                    cout << endl;
                }

                total_timer.start();
                update_timer.start();

                // break if not batching substitutions or if we have reached the batch size
                // at batch_size_=1 this will only substitute the best link found and then completely regenerate the results.
                // otherwise it will substitute the best batch_size_ number of linkages and then regenerate the results.
                if (!batched_ || ++batch_count >= batch_size_ || temp_counts_[temp_type] > max_temps_) {
                    substitute_timer.stop();
                    break;
                }
            }

            update_timer.stop();
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
        if(remake_test_set) {

            // remove intermediates that only occur once for printing
            remove_redundant_tmps();

            num_merged = 0;
            if (allow_merge_ && !format_sigma) {
                num_merged = merge_terms();
                total_num_merged += num_merged;
            }

            if (verbose) cout << endl << "Regenerating test set..." << std::flush;
            generate_linkages(true, format_sigma); // generate all possible linkages
            tmp_candidates_ -= ignore_linkages; // remove ignored linkages
            test_linkages.clear(); // clear test set
            test_linkages = make_test_set(); // make new test set

            update_timer.stop();
            if (verbose) cout << " Done ( " << flush;
            if (verbose) cout << update_timer.get_time() << " )" << endl;
            first_pass = false;

        } else update_timer.stop();

        // remove ignored linkages
        test_linkages   -= ignore_linkages;
        tmp_candidates_ -= ignore_linkages;

    } // end while linkage
    cout << endl;

    // remove intermediates that only occur once for printing
    remove_redundant_tmps();

    // resort tmps
    for (auto & [type, eq] : equations_) {
        sort_tmps(eq);
    }

    tmp_candidates_.clear();
    substitute_timer.stop(); // stop timer for substitution
    update_timer.stop();

    // recollect scaling of equations (now including sigma vectors)
    collect_scaling(true, true);

    // print total time elapsed

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

    total_timer.stop();
    cout << "    Total Time: " << total_timer.elapsed() << endl;
    total_timer.start();

    cout << "    Total number of terms: " << num_terms << endl;
    if (allow_merge_ && !format_sigma)
        cout << "    Total terms merged: " << total_num_merged << endl;
    cout << "    Total contractions: " << flop_map_.total() << (format_sigma ? " (ignoring assignments of intermediates)" : "") << endl;
    cout << endl;

    cout << " ===================================================="  << endl << endl;

    total_timer.stop();
}

void PQGraph::sort_tmps(Equation &equation) {

    // no terms, return
    if ( equation.terms().empty() ) return;

    // to sort the tmps while keeping the order of terms without tmps, we need to
    // make a map of the equation terms and their index in the equation and sort that (so annoying)
    std::vector<pair<Term*, size_t>> indexed_terms;
    size_t eq_size = equation.terms().size();
    indexed_terms.reserve(eq_size);
    for (size_t i = 0; i < eq_size; ++i)
        indexed_terms.emplace_back(&equation.terms()[i], i);

    // sort the terms by the maximum id of the tmps in the term, then by the index of the term

    auto is_in_order = [](const pair<Term*, size_t> &a, const pair<Term*, size_t> &b) {

        const Term &a_term = *a.first;
        const Term &b_term = *b.first;

        size_t a_idx = a.second;
        size_t b_idx = b.second;

        const ConstVertexPtr &a_lhs = a_term.lhs();
        const ConstVertexPtr &b_lhs = b_term.lhs();

        // TODO: keep entire list of ids and use that to sort

        // recursive function to get nested temp ids from a vertex
        std::function<set<long>(const ConstVertexPtr&)> test_vertex;
        test_vertex = [&test_vertex](const ConstVertexPtr &op) {

            set<long> ids;
            if (op->is_temp()) {
                ConstLinkagePtr link = as_link(op);
                long link_id = link->id_;

                // ignore reuse_tmp linkages
                if (!link->is_reused_ && !link->is_scalar()) {
                    ids.insert(link_id);
                }

                // recurse into nested temps
                for (const auto &nested_op: link->to_vector()) {
                    set<long> sub_ids = test_vertex(nested_op);
                    ids.insert(sub_ids.begin(), sub_ids.end());
                }
            }

            return ids;
        };

        // get min id of temps from lhs
        auto get_lhs_id = [&test_vertex](const Term &term) {
            return test_vertex(term.lhs());
        };

        // get min id of temps from rhs
        auto get_rhs_id = [&test_vertex](const Term &term) {

            set<long> ids;
            for (const auto &op: term.rhs()) {
                set<long> sub_ids = test_vertex(op);
                ids.insert(sub_ids.begin(), sub_ids.end());
            }
            return ids;
        };

        set<long> a_lhs_ids = get_lhs_id(a_term);
        set<long> b_lhs_ids = get_lhs_id(b_term);

        set<long> a_rhs_ids = get_rhs_id(a_term);
        set<long> b_rhs_ids = get_rhs_id(b_term);

        set<long> a_total_ids = a_lhs_ids, b_total_ids = b_lhs_ids;
        a_total_ids.insert(a_rhs_ids.begin(), a_rhs_ids.end());
        b_total_ids.insert(b_rhs_ids.begin(), b_rhs_ids.end());


        bool a_lhs_is_tmp = a_term.lhs()->is_temp();
        bool b_lhs_is_tmp = b_term.lhs()->is_temp();

        bool a_has_temp = a_lhs_is_tmp || !a_lhs_ids.empty() || !a_rhs_ids.empty();
        bool b_has_temp = b_lhs_is_tmp || !b_lhs_ids.empty() || !b_rhs_ids.empty();

        // if no temps, keep order
        if (!a_has_temp && !b_has_temp)
            return a_idx < b_idx;

        // keep in total lexicographical order
        if (a_total_ids > b_total_ids)
            return false;
        else if (a_total_ids == b_total_ids) {
            // if total ids are equal, keep assignments first
            if (a_term.is_assignment_ ^ b_term.is_assignment_)
                return a_term.is_assignment_;
            else if (a_term.is_assignment_ && b_term.is_assignment_) {

                // Keep temp assignments first if total ids are equal
                if (a_lhs_is_tmp ^ b_lhs_is_tmp)
                    return a_lhs_is_tmp;

                // if both are assignments sort by lhs id
                if (a_lhs_ids > b_lhs_ids)
                    return false;
                else if (a_lhs_ids == b_lhs_ids) {
                    // if lhs ids are equal, sort by rhs ids
                    if (a_rhs_ids > b_rhs_ids)
                        return false;
                    else if (a_rhs_ids == b_rhs_ids) {
                        // if rhs ids are equal, sort by index
                        return a_idx < b_idx;
                    }
                }
            }

            // preserve order if all else is equal
            return a_idx < b_idx;
        }

        // if total ids of a are less than b, keep order
        return true;
    };

    sort(indexed_terms.begin(), indexed_terms.end(), is_in_order);

    // replace the terms in the equation with the sorted terms
    std::vector<Term> sorted_terms;
    sorted_terms.reserve(indexed_terms.size());
    for (const auto &indexed_term : indexed_terms) {
        sorted_terms.push_back(*indexed_term.first);
    }

    equation.terms() = sorted_terms;
}

void PQGraph::remove_redundant_tmps() {
    // remove redundant contractions (only used in one term and its assignment)

    return; // TODO: fix this

    cout << "Removing redundant temps..." << endl << flush;

    std::unordered_map<ConstLinkagePtr, vector<Term*>, LinkageHash, LinkageEqual> to_replace;
    std::unordered_map<ConstLinkagePtr, pair<Term, string>, LinkageHash, LinkageEqual> to_remove;

    // iterate over all linkages and determine which linkages only occur in one term and its assignment
    for (const auto & [type, linkage_set] : all_linkages_) {
        for (const auto & linkage : linkage_set) {

            // get all terms with this tmp
            vector<Term*> tmp_terms = get_matching_terms(linkage);

            // if found in only one term, find assignment and add to to_replace
            if (tmp_terms.size() == 1){

                if (type == "reuse") {
                    if (!tmp_terms.front()->lhs()->is_temp()){
                        // for reuse tmps, we only want to remove tmps that are assigned to a tmp
                        continue;
                    }
                }

                // add term to be replaced
                to_replace[linkage].push_back(tmp_terms.front());

                // find assignment
                for (auto term_it = equations_[type].terms().begin(); term_it != equations_[type].terms().end(); ++term_it){
                    if (term_it->lhs()->is_temp() && as_link(term_it->lhs())->id_ == linkage->id_){
                        to_remove[linkage] = {*term_it, type};
                        break;
                    }
                }
            }
        }
    }

    // helper function to recursively remove nested tmps with a given id
    std::function<ConstVertexPtr(const ConstVertexPtr&, long)> remove_nested_id;
    remove_nested_id = [&remove_nested_id](const ConstVertexPtr &op, long id) {

        VertexPtr ret_op = op->deep_copy_ptr();

        if (ret_op->is_temp()) {
            vector<ConstVertexPtr> expanded_tmp;

            for (auto &sub_op : as_link(ret_op)->to_vector()) {
                ConstVertexPtr new_op = sub_op->deep_copy_ptr();

                // fully replace nested tmps
                if (sub_op->is_temp() && as_link(sub_op)->id_ == id) {
                    for (auto &nested_op : as_link(sub_op)->to_vector()) {
                        expanded_tmp.push_back(nested_op);
                    }
                    continue;

                // remove nested tmps
                } else if (sub_op->is_temp()) {
                    new_op = remove_nested_id(new_op, id);
                }

                // add to expanded tmp
                expanded_tmp.push_back(new_op);
            }

            ret_op = Linkage::link(expanded_tmp);
            as_link(ret_op)->copy_misc(as_link(op));
        }

        return ret_op;
    };

    // loop over all linkages to be replaced
    //TODO: making these substitutions could invalidate the terms in to_replace
    // we need regenerate to_replace after each substitution
    for (const auto & [linkage, terms] : to_replace) {
        // get assignment
        auto &[assignment_term, type] = to_remove[linkage];

        // get id of tmp
        long link_id = linkage->id_;

        // get coefficient of assignment
        double coeff = assignment_term.coefficient_;

        // loop over all terms to be replaced
        for (Term* term : terms) {
            // replace rhs of term
            vector<ConstVertexPtr> new_rhs;

            for (const ConstVertexPtr &op : term->rhs()){
                if (op->is_temp() && as_link(op)->id_ == link_id){
                    // replace tmp with assignment
                    for (const ConstVertexPtr &new_op : as_link(op)->to_vector())
                        new_rhs.push_back(new_op);
                } else {
                    // keep op
                    new_rhs.push_back(op);
                }
            }

            // replace rhs
            term->rhs() = new_rhs;

            // find everywhere this tmp is nested and replace it
            for (auto& [name, equation] : equations_) {
                for (auto &ref_term: equation.terms()) {

                    // store copy
                    ConstVertexPtr original_lhs = ref_term.lhs()->deep_copy_ptr();

                    // replace lhs
                    ref_term.lhs() = as_link(remove_nested_id(ref_term.lhs(), link_id)->deep_copy_ptr());

                    // if lhs is a linkage, remove from all linkages
                    if (ref_term.lhs()->is_temp()) {

                        // replace in all linkages
                        all_linkages_[name].erase(as_link(original_lhs));
                        all_linkages_[name].insert(as_link(ref_term.lhs()));
                    }

                    // replace rhs
                    for (auto &ref_op: ref_term.rhs())
                        ref_op = remove_nested_id(ref_op, link_id);
                }
            }

            // multiply coefficient by assignment
            term->coefficient_ *= coeff;

            term->request_update();
            term->compute_scaling(true);
        }
    }

    for (const auto & [linkage, term_type] : to_remove) {
        // remove all occurrences of this tmp
        const Term& assignment_term = term_type.first;
        size_t link_id = linkage->id_;
        const string &type = term_type.second;


        // find the assignment in the equation
        std::vector<Term>& eq_terms = equations_[type].terms();
        auto term_it = find_if(eq_terms.begin(), eq_terms.end(), [&assignment_term, link_id](const Term& term){
            return term.lhs()->is_temp() && as_link(term.lhs())->id_ == link_id;
        });

        // remove assignment
        eq_terms.erase(term_it);

        // print to screen
        cout << "    Removed intermediate " << link_id << endl;

        // remove from all linkages
        all_linkages_[type].erase(linkage);
    }

    cout << "Done" << endl << endl;

}

vector<Term *> PQGraph::get_matching_terms(const ConstLinkagePtr &contraction) {
    // grab all terms with this tmp

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
    for (auto & [name, eq] : equations_) {
        eq.expand_permutations();
    }
}

size_t PQGraph::merge_terms() {

//    if (!allow_merge_)
        return 0; // do not merge terms if not allowed

    if (verbose) cout << "Merging similar terms:" << endl;

    // iterate over equations and merge terms
    size_t num_fuse = 0;
    omp_set_num_threads(nthreads_);
    vector<string> eq_keys = get_equation_keys();
    #pragma omp parallel for reduction(+:num_fuse) default(none) shared(equations_, eq_keys)
    for (const auto &key: eq_keys) {
        Equation &eq = equations_[key];
        if (eq.is_temp_equation_) continue; // skip tmps equation

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

        if ((fabs(term.coefficient_) - 1e-10) < 1e-10)
            continue; // skip terms with coefficient of 0

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

    // do not modify coefficients if any are close to 0
    if (most_common_reciprocal == 0)
        return 1.0;

    double common_coefficient = 1.0 / static_cast<double>(most_common_reciprocal);

    // do not modify coefficients if any are close to 0
    if (common_coefficient == 0)
        return 1.0;

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
