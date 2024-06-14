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

#include <memory>

#include "../include/pq_graph.h"
#include "iostream"
// include omp only if defined
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

using std::ostream, std::string, std::vector, std::map, std::unordered_map, std::shared_ptr, std::make_shared,
        std::set, std::unordered_set, std::pair, std::make_pair, std::to_string, std::invalid_argument,
        std::stringstream, std::cout, std::endl, std::flush, std::max, std::min;

using namespace pdaggerq;



void PQGraph::make_all_links(bool recompute) {

    if (recompute)
        all_links_.clear(); // clear all prior candidates

    linkage_set candidate_linkages; // set of linkages
    for (auto &[eq_name, equation]: equations_) {
        // get all linkages of equation and add to candidates
        all_links_ += equation.make_all_links(recompute);
    }
}

Term& PQGraph::add_tmp(const ConstLinkagePtr& precon, Equation &equation, double coeff) {
    // make term with tmp
    equation.terms().insert(equation.end(), Term(precon, coeff));
    return equation.terms().back();
}

void PQGraph::substitute(bool format_sigma, bool only_scalars) {

    // begin timings
    total_timer.start();

    // reorder if not already reordered
    if (!is_reordered_) reorder();

    update_timer.start();

    /// ensure necessary equations exist
    bool missing_temp_eq = equations_.find("tmps") == equations_.end();
    bool missing_reuse_eq = equations_.find("reuse") == equations_.end();
    bool missing_scalar_eq = equations_.find("scalars") == equations_.end();

    vector<string> missing_eqs;
    if (missing_temp_eq) missing_eqs.emplace_back("tmps");
    if (missing_reuse_eq) missing_eqs.emplace_back("reuse");
    if (missing_scalar_eq) missing_eqs.emplace_back("scalars");

    // add missing equations
    for (const auto &missing: missing_eqs) {
        equations_.emplace(missing, Equation(make_shared<Vertex>(missing), {}));
        equations_[missing].is_temp_equation_ = true; // do not allow substitution of tmp declarations
    }

    /// generate all possible linkages from all arrangements


    if (verbose_) cout << "Generating all possible linkages..." << flush;

    size_t org_max_depth = Term::max_depth_;

    if (batched_)
        Term::max_depth_ = 2; // set max depth to 2 for initial linkage generation
    size_t current_depth = 2; // current depth of linkages

    make_all_links(true); // generate all possible linkages
    if (verbose_) cout << " Done" << endl;

    size_t num_terms = 0;
    for (const auto &eq_pair: equations_) {
        const Equation &equation = eq_pair.second;
        num_terms += equation.size();
    }

    size_t num_contract = flop_map_.total();

    cout << " ==> Substituting linkages into all equations <==" << endl;
    cout << "     Total number of terms: " << num_terms << endl;
    cout << "        Total contractions: " << flop_map_.total() << endl;
    cout << "     Use batched algorithm: " << (batched_ ? "yes" : "no") << endl;
    if (batched_)
        cout << "                Batch size: " << ((long) batch_size_ == -1 ? "no limit" : to_string(batch_size_))
             << endl;
    cout << "         Max linkage depth: " << ((long) Term::max_depth_ == -1 ? "no limit" : to_string(Term::max_depth_))
         << endl;
    cout << "    Possible intermediates: " << all_links_.size() << endl;
    cout << "    Number of threads used: " << nthreads_ << endl;
    cout << " ====================================================" << endl << endl;

    // give user a warning if the number of possible linkages is large
    // suggest using the batch algorithm, making the max linkage smaller, or increasing number of threads
    if (all_links_.size() * num_contract > 1000 * 10000) {
        cout << "WARNING: There are a large number of contractions and candidate intermediates." << endl;
        cout << "         This may take a long time to run." << endl;
        cout
                << "         Consider increasing the number of threads, making the max depth smaller, or using the batch algorithm."
                << endl;
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
    linkage_set ignore_linkages(all_links_.size());

    // get linkages with the highest scaling (use all linkages for first iteration, regardless of batched)
    // this helps remove impossible linkages from the set without regenerating all linkages as often
    linkage_set test_linkages = all_links_;

    update_timer.stop();

    bool first_pass = true;
    scaling_map best_flop_map = flop_map_;
    static size_t totalSubs = 0;
    string temp_type = format_sigma ? "reuse" : "tmps"; // type of temporary to substitute

    bool makeSub; // flag to make a substitution
    bool found_any = false; // flag to check if we found any linkages
    while ((!test_linkages.empty() || first_pass) && temp_counts_["all"] < max_temps_) {
        substitute_timer.start();

        makeSub = false; // reset flag
        bool allow_equality = true; // flag to allow equality in flop map
        size_t n_linkages = test_linkages.size(); // get number of linkages
        LinkagePtr link_to_sub; // best linkage to substitute

        // populate with pairs of flop maps with linkage for each equation
        vector<pair<scaling_map, LinkagePtr>> test_data(n_linkages);


        // print ratio for showing progress
        size_t print_ratio = n_linkages / 20;
        bool print_progress = n_linkages > 200 && verbose_;

        if (print_progress)
            cout << "PROGRESS:" << endl;

        /**
         * Iterate over all linkages in parallel and test if they can be substituted into the equations.
         * If they can, save the flop map for each equation.
         * If the flop map is better than the current best flop map, save the linkage.
         */
#pragma omp parallel for schedule(guided) default(none) shared(test_linkages, test_data, \
            ignore_linkages, equations_, stdout) firstprivate(n_linkages, temp_counts_, temp_type, allow_equality, \
            format_sigma, print_ratio, print_progress, only_scalars)
        for (int i = 0; i < n_linkages; ++i) {

            // copy linkage
            LinkagePtr linkage = as_link(test_linkages[i]->clone());
            bool is_scalar = linkage->is_scalar(); // check if linkage is a scalar

            if (is_scalar) {
                // if no scalars are allowed, skip this linkage
                if (Equation::no_scalars_) {
                    ignore_linkages.insert(linkage);
                    continue;
                }
            }


            if ((format_sigma && linkage->is_sigma_) || (only_scalars && !is_scalar)) {
                // when formatting for sigma vectors,
                // we only keep linkages without a sigma vector and are not scalars
                ignore_linkages.insert(linkage);
                continue;
            }
            linkage->is_reused_ = format_sigma;

            // set id of linkage
            size_t temp_id = temp_counts_["all"] + 1; // get number of temps
            linkage->id() = (long) temp_id;

            scaling_map test_flop_map; // flop map for test equation
            size_t numSubs = 0; // number of substitutions made
            for (auto &[eq_name, equation]: equations_) { // iterate over equations

                // skip scalar equations
                if (eq_name == "scalars") continue;


                // if the substitution is possible and beneficial, collect the flop map for the test equation
                numSubs += equation.test_substitute(linkage, test_flop_map,
                                                    allow_equality || is_scalar || format_sigma);
            }

            // add to test scalings if we found a tmp that occurs in more than one term
            // or that occurs at least once and can be reused / is a scalar

            // include declaration for scaling?
            bool keep_declaration = !is_scalar && !format_sigma;

            // test if we made a valid substitution
            int thresh = keep_declaration ? 1 : 0;
            if (numSubs > thresh) {

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
        if (print_progress) std::cout << "  Done" << std::endl << std::endl;



        /**
         * Iterate over all test scalings, remove incompatible ones, and sort them
         */

        std::multimap<scaling_map, LinkagePtr> sorted_test_data;
        for (auto &[test_flop_map, test_linkage]: test_data) {

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
            if (!makeSub && (format_sigma || (is_scalar && !Equation::no_scalars_))) keep = true;

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
            for (const auto &[found_flop, found_linkage]: sorted_test_data) {

                substitute_timer.start();

                link_to_sub = found_linkage;

                // check if link is a scalar
                bool is_scalar = link_to_sub->is_scalar();

                // get number of temps for this type
                string eq_type = is_scalar ? "scalars"
                                           : temp_type;

                // set linkage id
                size_t temp_id = ++temp_counts_["all"];
                link_to_sub->id() = (long) temp_id;

                scaling_map last_flop_map = flop_map_;

                /// substitute linkage in all equations

                vector<string> eq_keys = get_equation_keys();
                size_t num_subs = 0; // number of substitutions made

#pragma omp parallel for schedule(guided) default(none) firstprivate(allow_equality, link_to_sub) \
                shared(equations_, eq_keys) reduction(+:num_subs)
                for (const auto &eq_name: eq_keys) { // iterate over equations in parallel
                    // get equation
                    Equation &equation = equations_[eq_name]; // get equation
                    size_t this_subs = equation.substitute(link_to_sub, allow_equality);
                    bool madeSub = this_subs > 0;
                    if (madeSub) {
                        // sort tmps in equation
                        equation.rearrange();
                        num_subs += this_subs;
                    }
                }
                totalSubs += num_subs; // add number of substitutions to total

                // add linkage to ignore linkages
                ignore_linkages.insert(link_to_sub);
                test_linkages.erase(link_to_sub);

                // collect new scaling
                collect_scaling();

                if (num_subs == 0) {
                    temp_counts_["all"]--;
                    continue;
                }
                else {
                    // format contractions
                    auto [tmp_terms, tmp_decl_terms] = get_matching_terms(link_to_sub);

                    // add tmp declaration to tmps_terms (should not have more than 1 term)
                    tmp_terms.insert(tmp_terms.end(), tmp_decl_terms.begin(), tmp_decl_terms.end());

                    // find common coefficients and permutations
                    double common_coeff = common_coefficient(tmp_terms);

                    // modify coefficients of terms
                    for (Term *term_ptr: tmp_terms)
                        term_ptr->coefficient_ /= common_coeff;

                    // add linkage to equations
                    const Term &precon_term = add_tmp(link_to_sub, equations_[eq_type], common_coeff);

                    // print linkage
                    if (verbose_) {
                        cout << " ====> Substitution " << to_string(temp_id) << " <==== " << endl;
                        cout << " ====> " << precon_term << endl;
                        cout << " Difference: " << flop_map_ - last_flop_map << std::endl << endl;
                    }

                    // add linkage to this set
                    saved_linkages_[eq_type].insert(link_to_sub); // add tmp to tmps
                    found_any = true; // set found any flag to true
                }

                num_terms = 0;
                for (const auto &eq_pair: equations_) {
                    const Equation &equation = eq_pair.second;
                    num_terms += equation.size();
                }

                // print flop map
                if (verbose_) {

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
                    cout << "  Total Substitution count: " << totalSubs << endl << endl;
                    cout << "---------------------------- Remaining candidates: " << test_linkages.size();
                    cout << " ----------------------------" << endl << endl;
                }

                total_timer.start();
                update_timer.start();

                // break if not batching substitutions or if we have reached the batch size
                // at batch_size_=1 this will only substitute the best link found and then completely regenerate the results.
                // otherwise it will substitute the best batch_size_ number of linkages and then regenerate the results.
                if (!batched_ || ++batch_count >= batch_size_ || temp_counts_["all"] > max_temps_) {
                    substitute_timer.stop();
                    break;
                }
            }

            update_timer.stop();
        }

        update_timer.start();

        // remove all prior substituted linkages
        for (const auto &link_pair: saved_linkages_) {
            const linkage_set &linkages = link_pair.second;
            ignore_linkages += linkages;
        }

        // update test linkages
        test_linkages = all_links_ - ignore_linkages;

        bool recompute = test_linkages.empty();
        bool last_empty = recompute;

        if (recompute) {
            // merge terms if allowed
            num_merged = merge_terms();
            total_num_merged += num_merged;
        }

        // gradually increase max depth if we have not found any linkages (start from lowest depth; only if batching)
        while (test_linkages.empty() && recompute) {

            Term::max_depth_ = ++current_depth; // increase max depth

            if (verbose_)
                cout << endl << "Regenerating test set with max depth (" << current_depth << ") ... " << std::flush;

            // regenerate all valid linkages with the new depth
            make_all_links(true);

            // update test linkages
            test_linkages = all_links_ - ignore_linkages;

            if (current_depth >= org_max_depth)
                break; // break if we have reached the maximum depth

            if (last_empty && test_linkages.empty()) {
                current_depth = org_max_depth - 1; // none found twice, so test up to max depth

                if (!batched_) break; // break if not batching
            }
            last_empty = test_linkages.empty();
        }

        if (verbose_ && recompute)
            cout << " Done (" << "found " << test_linkages.size() << ")" << endl;

        // remove tmps that are not used
        remove_unused_tmps();

        update_timer.stop();
        if (recompute && verbose_) cout << "Updates Done ( " << update_timer.get_time() << " )" << endl << endl;

        first_pass = false;

//        static bool called_once = false;
//        if (!only_scalars && !called_once) {
            // merge intermediates
            merge_intermediates();
//            called_once = true;
//        }

    // end while loop when no more substitutions can be made, or we have reached the maximum number of temps
    }

    Term::max_depth_ = org_max_depth;

    // resort tmps
    for (auto & [type, eq] : equations_)
        eq.rearrange();

    substitute_timer.stop(); // stop timer for substitution
    update_timer.stop();

    // recollect scaling of equations (now including sigma vectors)
    collect_scaling(true, true);

    // print total time elapsed

    if (temp_counts_[temp_type] >= max_temps_)
        cout << "WARNING: Maximum number of substitutions reached. " << endl << endl;

    if (!found_any) {
        cout << "No substitutions found." << endl;
        return;
    }

    cout << "=================================> Substitution Summary <=================================" << endl;

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

void PQGraph::remove_unused_tmps() {
    // remove unused contractions (only used in one term and its assignment)

//    if (!prune_tmps_)
    return; // do not remove unused temps if pruning is disabled

    cout << "Removing unused temps..." << endl << flush;
    if (verbose_)
        cout << " WARNING: This may yield incorrect results if there are too many nested temps. set"
                " prune_tmps=false to disable this (recommended for large problems)." << endl << flush;



    std::set<size_t> unused_ids; // ids of unused temps

    // iterate over all linkages and determine which linkage ids only occur in one term
    for (const auto & [type, linkage_set] : saved_linkages_) {
        for (const auto & linkage : linkage_set) {

            // ignore scalar linkages (they do not use much memory)
            if (linkage->is_scalar())
                continue;

            // get all terms with this tmp
            auto [tmp_terms, tmp_decl_terms] = get_matching_terms(linkage);

            // add tmp declaration to tmps_terms (should not have more than 1 term)
            tmp_terms.insert(tmp_terms.end(), tmp_decl_terms.begin(), tmp_decl_terms.end());

            // if found in less than one term, find assignment and add to to_replace
            if (tmp_terms.size() <= 2){

                if (type == "reuse") {
                    if (!tmp_terms.front()->lhs()->is_temp()){
                        // for reuse tmps, we only want to remove tmps that are assigned to a tmp
                        continue;
                    }
                }

                // add to unused ids
                unused_ids.insert(linkage->id());

                // multiply coefficient of matching term by coefficient of assignment
                if (tmp_terms.size() == 2) {
                    Term *assignment = tmp_terms.front();
                    Term *term = tmp_terms.back();
                    term->coefficient_ *= assignment->coefficient_;
                }
            }
        }
    }

    cout << "Found " << unused_ids.size() << " unused temps." << endl << flush;
    cout << "Removing temps: " << endl;
    size_t count = 0, wrap = 10;
    for (const auto & id : unused_ids) {
        cout << id << " ";
        if (++count % wrap == 0) cout << endl;
    }
    cout << endl;

    std::function<VertexPtr(const ConstVertexPtr&)> remove_redundance;
    remove_redundance = [&remove_redundance, &unused_ids](const ConstVertexPtr& refop) {
        bool has_unused = true;

        // if not a temp, return
        if (!refop->is_temp()) { return refop->clone(); }

        bool ref_is_unused = unused_ids.find(as_link(refop)->id()) != unused_ids.end();

        ConstLinkagePtr reflink = as_link(refop);
        vector<ConstVertexPtr> new_ops = reflink->link_vector(true);

        while(has_unused) {
            vector<ConstVertexPtr> trial_ops = new_ops;
            new_ops.clear();

            has_unused = false;
            for (auto &op : trial_ops) {
                if (!op->is_temp()) {
                    new_ops.push_back(op);
                    continue;
                }

                ConstLinkagePtr link = as_link(op);
                bool link_unused = unused_ids.find(link->id()) != unused_ids.end();

                // this is a temp. Now we recursively replace the left and the right.
                ConstVertexPtr leftlink = link->left(), rightlink = link->right();

                // check if left and right are unused
                bool left_unused =
                        leftlink->is_temp() && unused_ids.find(as_link(leftlink)->id()) != unused_ids.end();
                bool right_unused =
                        rightlink->is_temp() && unused_ids.find(as_link(rightlink)->id()) != unused_ids.end();

                has_unused |= left_unused || right_unused || link_unused;

                // remove unused ids
                auto left_uniqued = remove_redundance(leftlink);
                auto right_uniqued = remove_redundance(rightlink);
                if (!link_unused) {
                    // build new link from left and right
                    VertexPtr new_link = left_uniqued * right_uniqued;
                    as_link(new_link)->copy_misc(link);
                    new_ops.push_back(new_link);
                }
                else {
                    new_ops.push_back(left_uniqued);
                    new_ops.push_back(right_uniqued);
                }
            }
        }

        // construct new linkage from new_ops
        LinkagePtr new_link = Linkage::link(new_ops);
        if (!ref_is_unused)
            new_link->copy_misc(reflink); // copy misc data from reflink

        return new_link->clone();
    };

    // iterate over all equations and remove unused ids from ops
    for (auto & [type, equation] : equations_) {
        vector<Term> new_terms;
        new_terms.reserve(equation.size());
        for (auto & term : equation.terms()) {
            // check if lhs is unused
            if (term.lhs()->is_temp() && unused_ids.find(as_link(term.lhs())->id()) != unused_ids.end()) {
                continue; // do not add term
            }

            Term new_term = term;
            new_term.lhs() = remove_redundance(new_term.lhs());
            vector<ConstVertexPtr> new_rhs;
            for (auto & op : new_term.rhs()) {
                op = remove_redundance(op);
                if (op->is_linked() && !op->is_temp()){
                    auto sub_ops = as_link(op)->link_vector(true);
                    new_rhs.insert(new_rhs.end(), sub_ops.begin(), sub_ops.end());
                } else {
                    new_rhs.push_back(op);
                }
            }

            // compute scaling of new term
            new_term.rhs() = new_rhs;
            new_term.request_update();
            new_term.compute_scaling(true);
            new_term.rhs() = new_term.term_linkage()->link_vector(true);
            new_term.reorder(true);

            // add term to new terms
            new_terms.push_back(new_term);
        }
        equation.terms() = new_terms;
    }

    // rebuild all linkages
    for (auto & [type, linkages] : saved_linkages_) {
        linkage_set new_linkage_set;
        for (auto & linkage : linkages) {
            if (unused_ids.find(linkage->id()) != unused_ids.end())
                continue;
            ConstVertexPtr new_linkage = remove_redundance(linkage);
            new_linkage_set.insert(as_link(new_linkage));
        }
        saved_linkages_[type] = new_linkage_set;
    }

    // recompute scaling and reorder terms
    reorder();
    collect_scaling(true);

    cout << "Done" << endl << endl;

}

pair<vector<Term *>, vector<Term *>> PQGraph::get_matching_terms(const ConstLinkagePtr &intermediate) {
    // grab all terms with this tmp

    // initialize vector of term pointers
    vector<Term*> tmp_terms;

    vector<string> eq_keys = get_equation_keys();
    #pragma omp parallel for schedule(guided) default(none) shared(equations_, eq_keys, tmp_terms, intermediate)
    for (const auto& eq_name : eq_keys) { // iterate over equations in parallel
        // get equation
        Equation &equation = equations_[eq_name]; // get equation

        // get all terms with this tmp
        vector<Term*> tmp_terms_local = equation.get_temp_terms(intermediate);
        #pragma omp critical
        {
            // add terms to tmp_terms
            tmp_terms.insert(tmp_terms.end(),
                             tmp_terms_local.begin(), tmp_terms_local.end());
        }

    }

    vector<Term*> tmp_decl_terms;
    vector<Term*> pruned_tmp_terms;
    for (auto &term : tmp_terms) {
        if (term->lhs()->same_temp(intermediate))
             tmp_decl_terms.push_back(term);
        else pruned_tmp_terms.push_back(term);
    }

    return {pruned_tmp_terms, tmp_decl_terms};
}

void PQGraph::expand_permutations(){
    for (auto & [name, eq] : equations_) {
        eq.expand_permutations();
    }
}

size_t PQGraph::merge_terms() {

    if (!allow_merge_)
        return 0; // do not merge terms if not allowed

    if (verbose_) cout << "Merging similar terms..." << flush;

    // iterate over equations and merge terms
    size_t num_fuse = 0;
    vector<string> eq_keys = get_equation_keys();
    #pragma omp parallel for reduction(+:num_fuse) default(none) shared(equations_, eq_keys)
    for (const auto &key: eq_keys) {
        Equation &eq = equations_[key];
        if (eq.is_temp_equation_) continue; // skip tmps equation

        num_fuse += eq.merge_terms(); // merge terms with same rhs up to a permutation
    }
    collect_scaling(); // collect new scalings

    if (verbose_) cout << "Done (" << num_fuse << " terms merged)" << endl;

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

PQGraph PQGraph::clone() const {
    // make initial copy
    PQGraph copy = *this;

    // copy equations and make deep copies of terms
    copy.equations_.clear();
    for (auto & [name, eq] : equations_) {
        copy.equations_[name] = eq.clone();
    }

    // copy all linkages
    copy.saved_linkages_.clear();
    for (const auto & [type, linkages] : saved_linkages_) {
        linkage_set new_linkages;
        for (const auto & linkage : linkages) {
            ConstLinkagePtr link = as_link(linkage->shallow());
            new_linkages.insert(link) ;
        }
        copy.saved_linkages_[type] = new_linkages;
    }

    return copy;

}

void PQGraph::make_scalars() {

    // find scalars in all equations and substitute them
    linkage_set scalars = saved_linkages_["scalars"];
    for (auto &[name, eq]: equations_) {
        // do not make scalars in scalar equation
        if (name == "scalars") continue;
        eq.make_scalars(scalars, temp_counts_["scalars"]);
    }

    // create new equation for scalars if it does not exist
    if (equations_.find("scalars") == equations_.end()) {
        equations_.emplace("scalars", Equation(make_shared<Vertex>("scalars"), {}));
        equations_["scalars"].is_temp_equation_ = true;
    }

    if (Equation::no_scalars_) {

        cout << "Removing scalars from equations..." << endl;

        // remove scalar equation
        equations_.erase("scalars");

        // remove scalars from all equations
        vector<string> to_remove;
        for (auto &[name, eq]: equations_) {
            vector<Term> new_terms;
            for (auto &term: eq.terms()) {
                bool has_scalar = false;
                for (auto &op: term.rhs()) {
                    if (op->is_linked() && op->is_scalar()) {
                        has_scalar = true;
                        break;
                    }
                }

                if (!has_scalar)
                    new_terms.push_back(term);
            }
            // if no terms left, remove equation
            if (new_terms.empty())
                to_remove.push_back(name);
            else
                eq.terms() = new_terms;
        }

        // remove equations
        for (const auto &name: to_remove) {
            cout << "Removing equation: " << name << " (no terms left after removing scalars)" << endl;
            equations_.erase(name);
        }

        // remove scalars from saved linkages
        scalars.clear();
    }

    // add new scalars to all linkages and equations
    for (const auto &scalar: scalars) {
        // add term to scalars equation
        add_tmp(scalar, equations_["scalars"]);
        saved_linkages_["scalars"].insert(scalar);
    }

    // remove comments from scalars
    for (Term &term: equations_["scalars"].terms())
        term.comments() = {}; // comments should be self-explanatory

    // collect scaling
    collect_scaling(true);

}

void PQGraph::merge_intermediates() {

    return merge_intermediates2();

    // get all intermediates
    linkage_set intermediates;
    for (auto &[name, eq] : equations_) {
        for (const auto &term : eq.terms()) {
            for (const auto &op : term.rhs()) {
                if (op->is_temp()) {
                    intermediates.insert(as_link(op));
                }
            }
        }
    }

    // iterate over all equations and find intermediates that have the same connectivity with the rhs
    typedef std::unordered_map<ConstLinkagePtr, vector<Term*>,
            LinkageHash, LinkageEqual> intermediate_map;

    intermediate_map intermediate_terms;
    for (auto & link : intermediates) {
        // get all terms with this intermediate
        auto [tmp_terms, tmp_decl_terms] = get_matching_terms(link);

        // add tmp declaration to tmps_terms (should not have more than 1 term)
        tmp_terms.insert(tmp_terms.end(), tmp_decl_terms.begin(), tmp_decl_terms.end());

        intermediate_terms[link] = tmp_terms;
    }

    // for each intermediate, find intermediates with the same connectivity
    std::unordered_map<ConstLinkagePtr, vector<ConstLinkagePtr>,
            LinkageHash, LinkageEqual> merge_map;
    linkage_set tested_linkages;
    set<Term*> terms_to_merge;
    for (const auto & [this_intermediate, this_terms] : intermediate_terms) {
        // check if this intermediate has already been tested
        if (tested_linkages.find(as_link(this_intermediate->clone())) != tested_linkages.end()) continue;

        // check if term has already been merged
        if (terms_to_merge.find(this_terms.front()) != terms_to_merge.end()) continue;

        vector<ConstLinkagePtr> to_merge;
        for (const auto & [other_intermediate, other_terms] : intermediate_terms) {
            // check if this term has already been merged
            if (terms_to_merge.find(other_terms.front()) != terms_to_merge.end()) continue;

            // skip same this_intermediate
            if (this_intermediate == other_intermediate) continue;
            // skip if not same number of this_terms
            if (this_terms.size() != other_terms.size()) continue;
            // check if this intermediate has already been tested
            if (tested_linkages.find(as_link(other_intermediate->clone())) != tested_linkages.end()) continue;
            // if the shapes of the intermediates are different, skip
            if (this_intermediate->shape_ != other_intermediate->shape_) continue;

            // check if connectivity of all this_terms is the same
            bool can_merge = true;
            for (auto this_term : this_terms) {
                bool found = false;
                // skip if the lhs of this term is this intermediate
                if (this_term->lhs()->is_linked()){
                    if (this_term->lhs()->same_temp(this_intermediate))
                        continue;
                }

                for (auto other_term : other_terms) {
                    // skip if the lhs of the other term is the other intermediate
                    if (other_term->lhs()->same_temp(other_intermediate)) continue;

                    // make sure lhs is the same
                    bool lhs_same = *this_term->lhs() == *other_term->lhs();
                    if (!lhs_same) continue;

                    // make sure permutation is the same
                    bool perm_same = this_term->perm_type() == other_term->perm_type() &&
                                     this_term->term_perms() == other_term->term_perms();
                    if (!perm_same) continue;

                    // get this_rhs of term and other term
                    vector<ConstVertexPtr> this_rhs = this_term->rhs();
                    vector<ConstVertexPtr> other_rhs = other_term->rhs();

                    // remove this_intermediate from this_rhs by looking at the id
                    vector<ConstVertexPtr> new_this_rhs = this_rhs;
                    new_this_rhs.reserve(this_rhs.size());
                    for (auto &rhs_op: this_rhs) {
                        if (as_link(rhs_op)->same_temp(this_intermediate)) {
                            continue; // skip if same intermediate
                        }

                        if (rhs_op->has_temp(this_intermediate)) {
                            ConstVertexPtr expanded = as_link(rhs_op)->expand_to_temp(this_intermediate);

                            // add all expanded vertices except this_intermediate
                            vector<ConstVertexPtr> expanded_verts = as_link(expanded)->link_vector(true);
                            for (auto &expanded_vert : expanded_verts) {
                                if (this_intermediate->same_temp(expanded_vert)) continue;
                                new_this_rhs.push_back(expanded_vert);
                            }
                        }
                        new_this_rhs.push_back(rhs_op);
                    }
                    this_rhs = new_this_rhs;

                    // remove other this_intermediate from this_rhs by looking at the id
                    vector<ConstVertexPtr> new_other_rhs = other_rhs;
                    new_other_rhs.reserve(other_rhs.size());
                    for (auto &rhs_op: other_rhs) {
                        if (as_link(rhs_op)->same_temp(other_intermediate)) {
                            continue; // skip if same intermediate
                        }
                        if (rhs_op->has_temp(other_intermediate)) {
                            ConstVertexPtr expanded = as_link(rhs_op)->expand_to_temp(other_intermediate);
                            vector<ConstVertexPtr> expanded_verts = as_link(expanded)->link_vector(true);
                            for (auto &expanded_vert : expanded_verts) {
                                if (other_intermediate->same_temp(expanded_vert)) continue;
                                new_other_rhs.push_back(expanded_vert);
                            }
                        }
                        new_other_rhs.push_back(rhs_op);
                    }
                    other_rhs = new_other_rhs;

                    // cannot merge if different number of rhs
                    if (this_rhs.size() != other_rhs.size()) continue;

                    // sort this_rhs by name
                    std::sort(this_rhs.begin(), this_rhs.end(), [](const auto &a, const auto &b) {
                        return a->name() < b->name();
                    });
                    std::sort(other_rhs.begin(), other_rhs.end(), [](const auto &a, const auto &b) {
                        return a->name() < b->name();
                    });

                    // check if any rhs ops are in tested_linkages
                    bool skip = false;
                    for (auto &rhs_op: this_rhs) {
                        if (rhs_op->is_temp() && tested_linkages.find(as_link(rhs_op->clone())) != tested_linkages.end()) {
                            skip = true;
                            break;
                        }
                    }
                    if (skip) continue;
                    for (auto &rhs_op: other_rhs) {
                        if (rhs_op->is_temp() && tested_linkages.find(as_link(rhs_op->clone())) != tested_linkages.end()) {
                            skip = true;
                            break;
                        }
                    }

                    // check if connectivity is the same with this this_intermediate
                    ConstVertexPtr this_link, other_link;
                    if (this_rhs.size() == 1) {
                        this_link = this_rhs.front();
                        other_link = other_rhs.front();
                    } else if (this_rhs.empty()) {
                        this_link = {};
                        other_link = {};
                    } else {
                        this_link = Linkage::link(this_rhs);
                        other_link = Linkage::link(other_rhs);
                    }

                    // connect both links with this_intermediate
                    this_link  = this_intermediate * this_link;
                    other_link = this_intermediate * other_link;

                    // check if links are the same
                    if (*as_link(this_link) != *as_link(other_link)) {
                        found = true;
                        break;
                    }

                    // all checks passed for this term
                }
                if (!found) {
                    can_merge = false;
                    break;
                }
            }

            // if same connectivity, add to to_merge
            if (can_merge) {

                // skip if any of the terms have already been queued for merging
                bool skip = false;
                for (auto &term : other_terms) {
                    if (terms_to_merge.find(term) != terms_to_merge.end()) {
                        skip = true;
                        break;
                    }
                }
                if (skip) continue;

                // add terms to merge
                for (auto & term : other_terms) {
                    terms_to_merge.insert(term);
                }

                // add to to_merge
                to_merge.push_back(other_intermediate);
                tested_linkages.insert(other_intermediate);
            }
        }

        // if any intermediates to merge, add to merge_map
        if (!to_merge.empty()) {
            merge_map[this_intermediate] = to_merge;
            for (auto & term : this_terms) {
                terms_to_merge.insert(term);
            }
        }

        // ignore this intermediate when checking for other intermediates
        tested_linkages.insert(this_intermediate);

    }

    // create a set of terms to remove
    set<Term*> terms_to_remove;

    // map intermediates to terms to add
    std::unordered_map<ConstLinkagePtr, vector<Term>, LinkageHash, LinkageEqual> new_inter_terms_map;

    // now we need to add all the terms of the other intermediates to this intermediate
    for (const auto & [this_intermediate, other_intermediates] : merge_map) {
        // add all intermediate terms to this intermediate
        VertexPtr new_intermediate = this_intermediate->clone();
        as_link(new_intermediate)->id() = -1;

        vector<Term*> other_terms;
        for (const auto & other_intermediate : other_intermediates) {
            // add terms associated with other intermediate
            other_terms.insert(other_terms.end(),
                               intermediate_terms[other_intermediate].begin(),
                               intermediate_terms[other_intermediate].end());
        }

        vector<Term> new_inter_terms; new_inter_terms.reserve(other_intermediates.size());
        for (auto & other_intermediate : other_intermediates) {
            // find term with lhs as this intermediate
            for (auto & other_term : intermediate_terms[other_intermediate]) {
                if (other_term->lhs()->same_temp(other_intermediate)) {
                    // add term with new intermediate
                    new_inter_terms.push_back(*other_term);
                }
                terms_to_remove.insert(other_term);
            }
        }
        new_inter_terms_map[this_intermediate] = new_inter_terms;

        // find equation with this intermediate
        string inter_type;
        for (const auto & [type, linkages] : saved_linkages_) {
            // find intermediate in saved linkages
            bool found = linkages.find(as_link(this_intermediate->clone())) != linkages.end();
            if (found) {
                inter_type = type;
                break;
            }
        }

        // remove old intermediates from saved linkages
        string intermediate_type;
        for (auto & [type, linkages] : saved_linkages_) {
            for (const auto & other_intermediate : other_intermediates) {
                linkages.erase(as_link(other_intermediate->clone()));
            }
        }
    }

    // build new intermediates
    std::unordered_map<ConstLinkagePtr, ConstLinkagePtr> old_to_new_links;
    for (const auto & [this_intermediate, other_intermediates] : merge_map) {
        // add all intermediate terms to this intermediate
        VertexPtr new_intermediate = this_intermediate->clone();
        as_link(new_intermediate)->id() = -1;

        vector<Term *> other_terms;
        long max_id = as_link(this_intermediate)->id();

        printf("Merge intermediate %s = %s with:\n", this_intermediate->str().c_str(), this_intermediate->tot_str().c_str());
        for (const auto &other_intermediate: other_intermediates) {
            LinkagePtr formatted_other = as_link(other_intermediate->clone());
            formatted_other->id() = -1;
            formatted_other->replace_lines(this_intermediate->lines_);
            new_intermediate = new_intermediate + formatted_other;

            printf("\t%s = %s\n", other_intermediate->str().c_str(), formatted_other->tot_str().c_str());

            // keep smaller id
            long other_id = as_link(other_intermediate)->id();
            if (other_id > max_id)
                max_id = (long) other_id;
        }
        printf("\n");
        as_link(new_intermediate)->copy_misc(as_link(this_intermediate));
//        as_link(new_intermediate)->id() = max_id;
        old_to_new_links[this_intermediate] = as_link(new_intermediate);

    }

    // remove all terms in terms_to_remove
    for (auto & [name, eq] : equations_) {
        vector<Term> new_terms;
        for (auto & term : eq.terms()) {
            if (terms_to_remove.find(&term) == terms_to_remove.end()) {
                new_terms.push_back(term);
            }
        }
        eq.terms() = new_terms;
    }

    // add old intermediate terms to kept intermediate terms
    for (auto & [this_intermediate, new_inter_terms] : new_inter_terms_map) {
        const ConstVertexPtr &new_intermediate = old_to_new_links[this_intermediate];

        // find type of this intermediate
        string inter_type;
        for (auto & [type, linkages] : saved_linkages_) {
            // find intermediate in saved linkages
            bool found = linkages.find(as_link(this_intermediate->clone())) != linkages.end();
            if (found) {
                inter_type = type;

                // remove from saved linkages and add new intermediate
                linkages.erase(as_link(this_intermediate->clone()));
                linkages.insert(as_link(new_intermediate->clone()));

                break;
            }
        }

        // format new terms
        for (auto &term : new_inter_terms) {
            LinkagePtr new_link = as_link(old_to_new_links[this_intermediate]->clone());

            // get line mapping
            auto line_map = LineHash::map_lines(term.lhs()->lines_, new_link->lines_);

            // replace the lhs with this intermediate
            term.lhs() = new_link;
            term.eq() = new_link;

            // replace lines in the rhs
            vector<ConstVertexPtr> term_ops;
            for (auto &op: term.rhs()) {
                VertexPtr new_op = op->clone();
                new_op->replace_lines(line_map);
                term_ops.push_back(new_op);
            }

            // set new term rhs
            term.rhs() = term_ops;

            term.request_update();
            term.is_assignment_ = false; // do not treat as an assignment (we are merging)
        }

        // add new terms to the end of the equation
        equations_[inter_type].terms().insert(equations_[inter_type].terms().end(),
                                              new_inter_terms.begin(), new_inter_terms.end());

        // sort terms
        equations_[inter_type].rearrange();

        // now replace the old intermediate with the new intermediate in all equations
        for (auto & [name, eq] : equations_) {

            // replace the ids of the old intermediates with the new intermediate
            for (auto &term : eq.terms()) {
                for (auto &[link, merge_links]: merge_map) {
                    LinkagePtr new_link = as_link(old_to_new_links[link]->clone());
                    for (auto &merge_link: merge_links) {
                        for (auto &op: term.rhs()) {
                            if (op->is_linked()) {
                                VertexPtr new_op = op->clone();
                                // find old link and replace with new link
                                ConstVertexPtr old_link = as_link(op)->find_link(link);

                                // if found, replace
                                if (old_link) {
                                    new_link->replace_lines(old_link->lines_);
                                    as_link(new_op)->replace_link(link, new_link);
                                }
                            }
                        }

                        // now the lhs
                        if (term.lhs()->is_linked()) {
                            VertexPtr new_lhs = term.lhs()->clone();
                            ConstVertexPtr old_link = as_link(term.lhs())->find_link(link);
                            if (old_link) {
                                new_link->replace_lines(old_link->lines_);
                                as_link(new_lhs)->replace_link(link, new_link);
                                term.lhs() = new_lhs;
                            }
                        }

                        // now the eq
                        if (term.eq() && term.eq()->is_linked()) {
                            VertexPtr new_eq = term.eq()->clone();
                            ConstVertexPtr old_link = as_link(term.eq())->find_link(link);
                            if (old_link) {
                                new_link->replace_lines(old_link->lines_);
                                as_link(new_eq)->replace_link(link, new_link);
                                term.eq() = new_eq;
                            }
                        }
                    }
                }
            }
            eq.rearrange();
        }
    }

    // compute new scaling
    collect_scaling(true);
}

























