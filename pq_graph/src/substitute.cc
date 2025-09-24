//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: substitute.cc
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

#include <algorithm>
#include <map>
#include <iostream>
#include <memory>
#include "../include/pq_graph.h"

using std::next_permutation;
using std::string;
using std::vector;
using std::map;
using std::pair;
using std::make_shared;
using std::shared_ptr;
using std::to_string;
using std::cout;
using std::flush;
using std::endl;
using std::max;

using namespace pdaggerq;

void PQGraph::make_all_links(bool recompute) {

    if (recompute)
        all_links_.clear(); // clear all prior candidates

    linkage_set candidate_linkages; // set of linkages
    for (auto &[eq_name, equation]: equations_) {
        // get all linkages of equation and add to candidates
        all_links_ += equation.make_all_links(recompute);
    }

    // clear history of all linkages
    for (auto &linkage: all_links_)
        linkage->forget(true);

}

linkage_set Equation::make_all_links(bool compute_all) {

    linkage_set all_linkages(2048); // all possible linkages in the equations (start with large bucket n_ops)

#pragma omp parallel for schedule(guided) shared(terms_, all_linkages) default(none) firstprivate(compute_all)
    for (auto & term : terms_) { // iterate over terms

        // skip term if it is optimal, and we are not computing all linkages
        if (!compute_all && term.generated_linkages_)
            continue;

        term.reorder(); // reorder term (only if necessary)
        all_linkages += term.make_all_links(); // nerate linkages in term and add to the set of all linkages

        term.generated_linkages_ = true; // set term to have generated linkages

    } // iterate over terms

    return all_linkages;
}

linkage_set Term::make_all_links() const {

    if (rhs_.empty())
        return {}; // if constant, return an empty set of linkages

    // initialize vector of linkages
    linkage_set linkages;

    if (term_linkage()->is_temp()) return {}; // the term_linkage is already a temp, no need to test it.

    // generate all subgraphs of the term
    auto subgraphs = term_linkage()->subgraphs(Term::max_depth_);

    // insert all subgraphs of a given deoth into the set of linkages
    for (const auto &subgraph : subgraphs) {
        if (subgraph->shape_ > Term::max_shape_) continue; // skip if subgraph shape is too large
        if (subgraph->empty()) continue; // skip if subgraph is empty
        if (subgraph->is_temp()) continue; // the subgraph is already a temp, no need to test it.

        // do not make intermediates with pure scalars (unless adding)
        const auto &subleft = subgraph->left(), subright = subgraph->right();
        bool with_scalar;
        with_scalar  =  subleft->is_scalar() && !subleft->is_linked();
        with_scalar |= subright->is_scalar() && !subright->is_linked();
        if (with_scalar && !subgraph->is_addition()) continue;

        // get best permutation of subgraph and relabel with generic lines
        LinkagePtr best_perm = as_link(subgraph->best_permutation()->relabel());
        subgraph->forget();
        best_perm->forget(); // clear the history of the best permutation

        // insert the best subperm into the set of linkages
        linkages.insert(best_perm);
    }

    return linkages;
}

void PQGraph::substitute(bool format_sigma, bool only_scalars) {

    // begin timings
    total_timer.start();

    // reorder if not already reordered
    if (!is_reordered_) reorder();

    update_timer.start();

    /// ensure necessary equations exist
    bool missing_temp_eq = equations_.find("temp") == equations_.end();
    bool missing_reuse_eq = equations_.find("reused") == equations_.end();
    bool missing_scalar_eq = equations_.find("scalar") == equations_.end();

    vector<string> missing_eqs;
    if (missing_temp_eq) missing_eqs.emplace_back("temp");
    if (missing_reuse_eq) missing_eqs.emplace_back("reused");
    if (missing_scalar_eq) missing_eqs.emplace_back("scalar");

    // add missing equations
    for (const auto &missing: missing_eqs) {
        equations_.emplace(missing, Equation(make_shared<Vertex>(missing), {}));
        equations_[missing].is_temp_equation_ = true; // do not allow substitution of tmp declarations
    }

    /// generate all possible linkages from all arrangements


    cout << "Generating all possible linkages..." << flush;

    size_t org_max_depth = Term::max_depth_;
    size_t current_depth;
    if (batched_) {
        Term::max_depth_ = 1; // set max depth to 1 for initial linkage generation
        current_depth = 1;
    } else {
        current_depth = Term::max_depth_;
    }


    make_all_links(true); // generate all possible linkages
    cout << " Done" << endl;

    size_t num_terms = get_num_terms();


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
    size_t num_merged = merge_terms();
    total_num_merged += num_merged;

    // initialize best flop map for all equations
    collect_scaling(true);

    // set of linkages to ignore
    linkage_set ignore_linkages(all_links_.size());

    // add all saved linkages to ignore linkages
    for (const auto &[type, linkages]: saved_linkages_) {
        ignore_linkages += linkages;
    }

    // get linkages with the highest scaling (use all linkages for first iteration, regardless of batched)
    // this helps remove impossible linkages from the set without regenerating all linkages as often
    linkage_set test_linkages = all_links_ - ignore_linkages;

    update_timer.stop();

    bool first_pass = true;
    scaling_map best_flop_map = flop_map_;
    static size_t totalSubs = 0;
    string temp_type = format_sigma ? "reused" : "temp"; // type of temporary to substitute
    temp_type = only_scalars ? "scalar" : temp_type; // type of equation to substitute into

    bool makeSub; // flag to make a substitution
    bool found_any = false; // flag to check if we found any linkages
    size_t retries = 0; // number of retries
    while ((!test_linkages.empty() || first_pass) && temp_counts_[temp_type] < max_temps_) {
        substitute_timer.start();

        makeSub = false; // reset flag
        bool allow_equality = true; // flag to allow equality in flop map
        size_t n_linkages = test_linkages.size(); // get number of linkages
        MutableLinkagePtr link_to_sub; // best linkage to substitute

        // populate with pairs of flop maps with linkage for each equation
        vector<pair<scaling_map, MutableLinkagePtr>> test_data(n_linkages);


        // print ratio for showing progress
        size_t print_ratio = n_linkages / 20;
        bool print_progress = n_linkages > 2000;

        if (print_progress)
            cout << "PROGRESS:" << endl;

        /**
         * Iterate over all linkages in parallel and test if they can be substituted into the equations.
         * If they can, save the flop map for each equation.
         * If the flop map is better than the current best flop map, save the linkage.
         */
#pragma omp parallel for schedule(guided) default(none) shared(test_linkages, test_data, \
            ignore_linkages, equations_, stdout) firstprivate(n_linkages, temp_counts_, temp_type, allow_equality, \
            format_sigma, print_ratio, print_progress, only_scalars, separate_sigma_)
        for (int i = 0; i < n_linkages; ++i) {

            // copy linkage
            MutableLinkagePtr linkage = as_link(test_linkages[i]->shallow());
            bool is_scalar = linkage->is_scalar(); // check if linkage is a scalar
            bool is_sigma = linkage->is_sigma_;

            string eq_type; // get equation type
            if (is_scalar){
                eq_type = "scalar";
            } else if (!is_sigma && separate_sigma_) {
                eq_type = "reused";
                linkage->reused_ = true;
            } else {
                eq_type = "temp";
            }

            if (is_scalar) {
                // if no scalars are allowed, skip this linkage
                if (Equation::no_scalars_) {
                    linkage->forget(); // clear linkage history
                    ignore_linkages.insert(linkage); // add linkage to ignore linkages
                    continue;
                }
            }

            if ((format_sigma && is_sigma) || (only_scalars && !is_scalar)) {
                // when formatting for sigma vectors,
                // we only keep linkages without a sigma vector and are not scalars
                linkage->forget(); // clear linkage history
                ignore_linkages.insert(linkage);
                continue;
            }

            // check if this linkage is in the ignore set
            if (ignore_linkages.find(linkage) != ignore_linkages.end()) {
                linkage->forget(); // clear linkage history
                continue;
            }

            // set id of linkage
            long temp_id = temp_counts_[eq_type] + 1; // get number of temps
            linkage->id() = temp_id;

            scaling_map test_flop_map; // flop map for test equation
            size_t numSubs = 0; // number of substitutions made
            for (auto &[eq_name, equation]: equations_) { // iterate over equations

                if (eq_name == "scalar" || eq_name == "reused") continue; // skip scalar and reuse equations

                // if the substitution is possible and beneficial, collect the flop map for the test equation
                numSubs += equation.test_substitute(linkage, test_flop_map);
            }

            // add to test scalings if we found a tmp that occurs in more than one term
            // or that occurs at least once and can be reused / is a scalar

            // include declaration for scaling?
            bool keep_declaration = eq_type != "scalar" && eq_type != "reused";

            // test if we made a valid substitution
            if (numSubs > 0) {

                if (keep_declaration) {
                    // make term of tmp declaration
                    Term precon_term = Term(linkage, 1.0);
                    precon_term.compute_scaling();
                    // add scaling of declaration term to the test flop map if we are keeping the declaration
                    test_flop_map += precon_term.flop_map();
                }

                // set any negative values to zero
                test_flop_map.all_positive();

                // save this test flop map and linkage for serial testing
                test_data[i] = make_pair(test_flop_map, linkage);

            } else { // if we didn't make a substitution, add linkage to ignore linkages
                linkage->forget(); // clear linkage history
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

        std::multimap<scaling_map, MutableLinkagePtr> sorted_test_data;
        for (auto &[test_flop_map, test_linkage]: test_data) {

            // skip empty linkages
            if (test_linkage == nullptr) continue;
            if (test_linkage->empty()) continue;
            test_linkage->forget(true); // clear linkage history

            if (test_flop_map > flop_map_) {
                // remove the linkage completely if the scaling only got worse
                ignore_linkages.insert(test_linkage);
                continue;
            }

            bool is_scalar = test_linkage->is_scalar(); // check if linkage is a scalar
            bool is_reused = test_linkage->reused_; // check if linkage is reused

            // test if this is the best flop map seen
            int comparison = test_flop_map.compare(flop_map_);
            bool is_equiv = comparison == scaling_map::this_same;
            bool keep = comparison == scaling_map::this_better;

            // if we haven't made a substitution yet and this is either a
            // scalar or a sigma vector, keep it
            if (is_reused || (is_scalar && !Equation::no_scalars_)) keep = true;

            // if the scaling is the same and it is allowed, set keep to true
            if (!keep && is_equiv && allow_equality) keep = true;


            if (keep) {
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

                // get number of temps for this type
                string eq_type = link_to_sub->type();

                // set linkage id
                long temp_id = ++temp_counts_[eq_type];
                link_to_sub->id() = temp_id;

                scaling_map last_flop_map = flop_map_;

                /// substitute linkage in all equations

                vector<string> eq_keys = get_equation_keys();
                size_t num_subs = 0; // number of substitutions made

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
                link_to_sub->forget(true); // clear linkage history
                ignore_linkages.insert(link_to_sub);
                test_linkages.erase(link_to_sub);

                // collect new scaling
                collect_scaling();

                if (num_subs == 0) {
                    temp_counts_[eq_type]--;
                    continue;
                }
                else {

                    // add linkage to equations
                    const Term &precon_term = add_tmp(link_to_sub, equations_[eq_type], 1.0);

                    // print linkage
                    {
                        cout << " ====> Substitution " << to_string(temp_id) << " <==== " << endl;
                        cout << " ====> " << precon_term << endl;
                        cout << " Difference: " << flop_map_ - last_flop_map << std::endl << endl;
                    }

                    // add linkage to this set
                    saved_linkages_[eq_type].insert(link_to_sub); // add tmp to tmps
                    found_any = true; // set found any flag to true
                }

                num_terms = get_num_terms(); // get number of terms

                // print flop map
                {

                    // print total time elapsed
                    substitute_timer.stop();
                    update_timer.stop();
                    total_timer.stop();

                    cout << "---------------------------- Remaining candidates: " << test_linkages.size();
                    cout << " ----------------------------" << endl << endl;

                    cout << "                  Net time: " << total_timer.elapsed() << endl;
                    cout << "              Reorder Time: " << reorder_timer.elapsed() << endl;
                    cout << "               Update Time: " << update_timer.elapsed() << endl;
                    cout << "                 Sub. Time: " << substitute_timer.elapsed() << endl;
                    cout << "         Average Sub. Time: " << substitute_timer.average_time() << endl;
                    cout << "           Number of terms: " << num_terms << endl;
                    cout << "    Number of Contractions: " << flop_map_.total() << endl;
                    cout << "        Substitution count: " << num_subs << endl;
                    cout << "  Total Substitution count: " << totalSubs << endl << endl;
                }

                total_timer.start();
                update_timer.start();
                // break if not batching substitutions or if we have reached the batch size
                // at batch_size_=1 this will only substitute the best link found and then completely regenerate the results.
                // otherwise it will substitute the best batch_size_ number of linkages and then regenerate the results.
                if (!batched_ || ++batch_count >= batch_size_ || temp_counts_[eq_type] > max_temps_) {
                    break;
                }
            }

            update_timer.stop();
        }

        update_timer.start();

        // remove all prior substituted linkages
        for (const auto &[type, linkages]: saved_linkages_) {
            ignore_linkages += linkages;
        }

        // update test linkages
        test_linkages = all_links_ - ignore_linkages;

        bool recompute = test_linkages.empty();
        bool last_empty = recompute;

        if (recompute) {

            // synchronize all pointers in graph
            forget();

            // merge terms if allowed
            num_merged = merge_terms();
            total_num_merged += num_merged;

            size_t num_fused = merge_intermediates();
            if (num_fused > 0) {
                total_num_merged += num_fused;
                cout << "Fused " << num_fused << " terms." << endl;
            }

            prune();

            // gradually increase max depth if we have not found any linkages (start from lowest depth; only if batching)
            while (test_linkages.empty()) {

                if (++current_depth == 0) --current_depth; // reset depth if overflow
                Term::max_depth_ = current_depth; // increase max depth

                {
                    cout << "Regenerating test set with depth " << flush;
                    if (current_depth >= org_max_depth)
                        cout << "(max) ... " << flush;
                    else cout << "(" << current_depth << ") ... " << flush;
                }

                // regenerate all valid linkages with the new depth
                make_all_links(true);

                // update test linkages
                test_linkages = all_links_ - ignore_linkages;

                // clear linkages within ignore set and test set
                for (auto &linkage: ignore_linkages)
                    linkage->forget(true);
                for (auto &linkage: test_linkages)
                    linkage->forget(true);

                cout << " Done (" << "found " << test_linkages.size() << ")" << endl;

                if (current_depth >= org_max_depth)
                    break; // break if we have reached the maximum depth

                if (last_empty && test_linkages.empty()) {
                    current_depth = org_max_depth - 1; // none found twice, so test up to max depth

                    if (!batched_) break; // break if not batching
                }
                last_empty = test_linkages.empty();
            }
            if (test_linkages.size() <= 5) {
                // if all candidates are additions, we can stop
                bool all_additions = std::all_of(test_linkages.begin(), test_linkages.end(), [](const LinkagePtr &link) {
                    return link->is_addition();
                });
                if (all_additions) break;
            }

            // exit excessive retries for substitution.
            if (current_depth == org_max_depth && !makeSub) {
                retries++;
                if (retries > 5) {
                    cout << "Could not find any more substitutions." << endl;
                    break;
                }
            } else if (makeSub) {
                // reset retries if we had found a substitution
                retries = 0;
            }
        }

        update_timer.stop();
        first_pass = false;

        // end while loop when no more substitutions can be made, or we have reached the maximum number of temps
    }

    // merge terms if allowed
    num_merged = merge_terms();
    total_num_merged += num_merged;

    // merge intermediates
    size_t num_fused = merge_intermediates();
    if (num_fused > 0) {
        total_num_merged += num_fused;
        cout << "Fused " << num_fused << " terms." << endl;
    }

    // prune intermediates, but also remove single use intermediates
    prune();

    Term::max_depth_ = org_max_depth;

    // resort tmps
    for (auto & [type, eq] : equations_)
        eq.rearrange();

    substitute_timer.stop(); // stop timer for substitution
    update_timer.stop();

    // recollect scaling of equations (now including sigma vectors)
    collect_scaling(true, true);


    if (temp_counts_[temp_type] >= max_temps_)
        cout << "WARNING: Maximum number of substitutions reached. " << endl << endl;

    if (!found_any) {
        cout << "No substitutions found." << endl << endl;
        return;
    }

    // print total time elapsed
    cout << endl << "=================================> Substitution Summary <=================================" << endl;

    num_terms = get_num_terms();
    for (const auto & [type, count] : temp_counts_) {
        if (count == 0)
            continue;
        cout << "    Found " << count << " " << type << endl;
    }

    total_timer.stop();
    cout << "    Total Time: " << total_timer.elapsed() << endl;
    total_timer.start();

    cout << "    Total number of terms: " << num_terms << endl;
    cout << "    Total terms merged: " << total_num_merged << endl;
    cout << "    Total contractions: " << flop_map_.total() << (format_sigma ? " (ignoring assignments of intermediates)" : "") << endl;
    cout << endl;

    cout << " ===================================================="  << endl << endl;

    total_timer.stop();
}

size_t Equation::substitute(const LinkagePtr &linkage, bool allow_equality) {

    /// iterate over terms and substitute
    size_t num_terms = terms_.size();
    size_t num_subs = 0; // number of substitutions

    // scaling of the linkage cannot be more than the equation
    if (linkage->netscales().first > flop_map()) return 0;

    #pragma omp parallel for schedule(guided) shared(terms_, linkage) firstprivate(num_terms, allow_equality) \
                             reduction(+:num_subs) default(none)
    for (int i = 0; i < num_terms; i++) {
        Term &term = terms_[i]; // get term

        // check if linkage is compatible with term
        if (!term.is_compatible(linkage)) continue; // skip term if linkage is not compatible

        /// substitute linkage in term
        bool madeSub;
        madeSub = term.substitute(linkage);

        /// increment number of substitutions if substitution was successful
        if (madeSub) {
            ++num_subs;
            term.request_update(); // set term to be updated
        }
    } // substitute linkage in term

    return num_subs;
}

size_t Equation::test_substitute(const MutableLinkagePtr &linkage, scaling_map &test_flop_map, bool allow_equality) {

    // scaling of the linkage cannot be more than the equation
    if (linkage->netscales().first > flop_map()) return 0;

    /// iterate over terms and substitute
    size_t num_terms = terms_.size();
    size_t num_subs = 0; // number of substitutions
    test_flop_map += flop_map_; // test memory scaling map
    for (int i = 0; i < num_terms; i++) {
        // skip term if linkage is not compatible
        if (!terms_[i].is_compatible(linkage)) continue;

        // get term copy
        Term term = terms_[i];
        term.term_linkage() = as_link(term.term_linkage()->shallow()); // deep copy of term linkage

        // It's faster to subtract the old scaling and add the new scaling than
        // to recompute the scaling map from scratch
        test_flop_map -= term.flop_map(); // subtract flop scaling map for term

        // substitute linkage in term copy
        bool madeSub = term.substitute(linkage);
        term.term_linkage()->forget(); // clear the linkage history for lazy evaluation
        test_flop_map += term.flop_map(); // add new flop scaling map for term

        // increment number of substitutions if substitution was successful
        if (madeSub) ++num_subs; // increment number of substitutions

    } // substitute linkage in term copy

    return num_subs;
}

bool Term::substitute(const LinkagePtr &linkage) {

    if (rhs_.empty())
        return false;

    // recompute the flop and memory cost of the term if necessary
    compute_scaling();

    // break out of loops if a substitution was made
    bool madeSub = false; // initialize boolean to track if substitution was made

    // generate every permutation of the term
    const linkage_vector &graph_perms = term_linkage()->permutations();

    // iterate over all possible orderings of vertex subsets
    LinkagePtr best_linkage = as_link(term_linkage()->shallow());
    for (const auto &graph_perm : graph_perms) {
        // substitute the linkage in the permutation (if possible)
        graph_perm->forget();
        auto matching_linkages = graph_perm->find_links(linkage);
        if (matching_linkages.empty()) continue; // skip if linkage is not found in permutation

        // otherwise, make the substitution for each matching linkage
        LinkagePtr new_term_linkage = graph_perm;
        for (const auto &found_linkage : matching_linkages) {
            MutableVertexPtr new_link = found_linkage->shallow();
            as_link(new_link)->copy_misc(linkage);
            new_term_linkage = as_link(new_term_linkage->replace(found_linkage, new_link).first);
        }

        new_term_linkage = as_link(new_term_linkage)->best_permutation();
        const auto &[new_flop, new_mem] = new_term_linkage->netscales();
        const auto &[best_flop, best_mem] = best_linkage->netscales();

        if (new_flop > best_flop) continue; // new_linkage is worse than the best one
        if (new_flop == best_flop) {
            // for the same flop cost, prefer the one with less memory
            if (new_mem > best_mem) continue; // new_linkage is worse than the best one
        }

        // create the best permutation of the substitution and break
        best_linkage = new_term_linkage;
        madeSub = true;
        break;
    }

    // if a substitution was made, replace the linkage in the term
    if (madeSub) {
        // replace the rhs with the best linkage (if it is a temp or addition, we should not expand into a vector)
        expand_rhs(best_linkage);
        request_update(); // set flags for optimization
        compute_scaling(true); // recompute the flop and memory cost of the term
    }

    return madeSub;

}

bool Term::is_compatible(const LinkagePtr &linkage) const {

    // if no possible linkages, return false
    if (rhs_.empty()) return false;

    if (lhs_->is_temp()){

        // do not allow substitution to intermediates with smaller ids unless they are different types
        if(lhs_->id() <= linkage->id()) {
            if (lhs_->type() == linkage->type()) return false;
        }

        // do not allow substitution of reused intermediates with non-reused intermediates
        if (lhs_->type() != "temp" && linkage->type() == "temp") return false;
    }

    // scaling of the linkage cannot be more than the term
    if (linkage->netscales().first > flop_map()) return false;

    // get total vector of linkage vertices (without expanding nested linkages)
    vertex_vector link_list = linkage->link_vector();
    vertex_vector term_list = term_linkage()->link_vector();

    // sort lists by name
    sort(link_list.begin(), link_list.end(), [](const VertexPtr &a, const VertexPtr &b) {
        return a->name_ < b->name_;
    });
    sort(term_list.begin(), term_list.end(), [](const VertexPtr &a, const VertexPtr &b) {
        return a->name_ < b->name_;
    });

    // check if all vertex names are found in the term
    bool all_found = std::includes(term_list.begin(), term_list.end(), link_list.begin(), link_list.end(),
                                  [](const VertexPtr &a, const VertexPtr &b) {
                                      return a->name_ < b->name_;
                                  });

    // return true if all linkages are found in the term
    return all_found;

}

void PQGraph::make_scalars() {

    cout << "Finding scalars..." << flush;
    if ( opt_level_ >= 2 ) {
        // use substitution to find scalars
        print_guard guard; guard.lock();
        substitute(false, true);
    }

    // find scalars in all equations and substitute them without reordering terms
    for (auto &[name, eq]: equations_) {
        // do not make scalars in scalar equation
        if (name == "scalar") continue;
        eq.make_scalars(saved_linkages_["scalar"], temp_counts_["scalar"]);
    }
    cout << " Done" << endl;

    // create new equation for scalars if it does not exist
    if (equations_.find("scalar") == equations_.end()) {
        equations_.emplace("scalar", Equation(make_shared<Vertex>("scalar"), {}));
        equations_["scalar"].is_temp_equation_ = true;
    }

    if (Equation::no_scalars_)
        remove_scalars();

    linkage_vector scalars_vec(saved_linkages_["scalar"].begin(), saved_linkages_["scalar"].end());
    // sort by the id of the scalars
    sort(scalars_vec.begin(), scalars_vec.end(), [](const LinkagePtr &a, const LinkagePtr &b) {
        return a->id() < b->id();
    });

    // add new scalars to all linkages and equations
    for (const auto &scalar: scalars_vec) {
        // add term to scalars equation if it is not already there
        bool already_added = false;
        for (const auto &term: equations_["scalar"].terms()) {
            if (*term.lhs() == *scalar) { already_added = true; break; }
        }

        if (!already_added)
            add_tmp(scalar, equations_["scalar"]);

        // print scalar
        cout << scalar->str() << " = " << *scalar << endl;
    }

    // remove comments from scalars
    for (Term &term: equations_["scalar"].terms())
        term.comments() = {}; // comments should be self-explanatory

    cout << endl;

    // collect scaling
    collect_scaling(true);
    is_assembled_ = true;
}

void Equation::make_scalars(linkage_set &scalars, long &n_temps) {

    // iterate over terms
    #pragma omp parallel for schedule(guided) shared(terms_, scalars, n_temps) default(none)
    for (auto & term : terms_) {

        // make scalars in term
        bool made_scalar = true;
        while (made_scalar) {
            // make scalars in term
            made_scalar = term.make_scalars(scalars, n_temps);

        } // eventually no more scalars will be made
    }
}

bool Term::make_scalars(linkage_set &scalars, long &id) {

    if (rhs_.empty())
        return false; // do nothing if term is empty

    // break out of loops if a substitution was made
    bool made_scalar = false; // initialize boolean to track if substitution was made

    const linkage_vector &graph_perms = term_linkage()->permutations();
    linkage_map<linkage_set> term_scalars;
    for (const auto &graph_perm : graph_perms) {
        const auto perm_scalars = graph_perm->find_scalars();
        auto &perm_entry = term_scalars[graph_perm];
        for (const auto &scalar : perm_scalars) {
            if (!scalar->is_scalar()) continue; // skip if scalar is not actually a scalar (should not happen)
            if (scalar->is_temp()) continue;    // skip if scalar is already a temp
            if (!scalar->is_linked()) continue; // skip if scalar is not linked
            perm_entry.insert(scalar);
        }
    }
    if (term_scalars.empty()) return false; // do nothing if no scalars are found

    LinkagePtr new_linkage = as_link(term_linkage()->shallow());
    for (const auto& [perm_linkage, perm_scalars] : term_scalars) {
        for (const auto &scalar : perm_scalars) {

            // reorder scalar for the best permutation
            LinkagePtr scalar_link = as_link(scalar)->best_permutation();
            MutableLinkagePtr new_scalar = as_link(scalar_link->shallow());

            // check if scalar is already in set of scalars for setting the id
            #pragma omp critical(ScalarSubstitution)
            {
                long new_id = id + 1;
                auto scalar_pos = scalars.find(new_scalar);
                if (scalar_pos != scalars.end())
                    new_id = scalar_pos->get()->id(); // if scalar is already in set of scalars, change the id
                else ++id; // if scalar is not in set of scalars, increment the id for the next scalar

                new_scalar->id_ = new_id;

                // replace scalar in the term linkage
                auto [subbed_linkage, replaced] = as_link(perm_linkage)->replace(scalar, new_scalar);
                if (replaced) {
                    new_linkage = as_link(subbed_linkage);
                    scalars.insert(new_scalar); // insert scalar into set of scalars
                    made_scalar = true;
                }
            }
            if (made_scalar) break;
        }
        if (made_scalar) break;
    }

    // replace the rhs with the best linkage (if it is a temp, we should not expand into a vector)
    expand_rhs(new_linkage);
    request_update(); // set flags for optimization
    compute_scaling(true); // recompute the flop and memory cost of the term
    return made_scalar;
}

void PQGraph::remove_scalars() {
    cout << "Removing scalars from equations..." << endl;

    // remove scalar equation
    equations_.erase("scalar");
    saved_linkages_["scalar"].clear();

    // remove scalars from all equations
    vector<string> to_remove;
    for (auto &[name, eq]: equations_) {
        vector<Term> new_terms;
        for (auto &term: eq.terms()) {
            bool has_scalar = false;
            for (auto &op: term.rhs()) {
                if (op->is_linked() && op->is_scalar()) {
                    has_scalar = true; break;
                }
            }
            if (!has_scalar)
                new_terms.push_back(term);
        }
        // if no terms left, remove equation
        if (new_terms.empty())
            to_remove.push_back(name);
        else eq.terms() = new_terms;
    }

    // remove equations
    for (const auto &name: to_remove) {
        cout << "Removing equation: " << name << " (no terms left after removing scalars)" << endl;
        equations_.erase(name);
    }
}