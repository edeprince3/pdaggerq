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
#include <iostream>

#include "../include/pq_graph.h"
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
        std::stringstream, std::cout, std::endl, std::flush, std::max, std::min;

using namespace pdaggerq;

struct LinkInfo {
    LinkagePtr link;
    Term* term;
    Term trunc_term;
};

struct LinkTracker {
    linkage_map<vector<LinkInfo>> link_track_map_{}; // map of linkages to terms and trunc terms
    linkage_map<set<Term*>> link_declare_map_{}; // map of linkages to their declarations
    map<string, long> max_ids_;

    LinkTracker(){
        // reserve bins for the link maps
        link_track_map_.reserve(1024);
        link_declare_map_.reserve(1024);
    }

    void insert(Term* term) {

        // if lhs is a declaration, store that separately
        if (term->lhs()->is_temp()) {
            auto link = as_link(term->lhs()->shallow());
            max_ids_[link->type()] = std::max(max_ids_[link->type()], link->id());
            link->forget(true); // forget the link history for memory efficiency
            link_declare_map_[link].insert(term);
            return;
        }

        // extract all linkages within the term
        VertexPtr dummy = 0.0 * std::make_shared<Vertex>("dummy");
        for (auto &vertex: term->rhs()) {

            // vertex in term is fusable only if it is linked. if linked, it must be a temp or not an addition
            bool fusable = vertex->has_any_temp();
            if (fusable) {
                auto all_temps = as_link(vertex)->get_temps(true, false);
                for (auto &temp: all_temps) {
                    LinkagePtr temp_link = as_link(temp);
                    max_ids_[temp_link->type()] = max(max_ids_[temp_link->type()], temp_link->id());
                    
                    // create a new link info
                    LinkInfo link_info;
                    link_info.link = temp_link;
                    link_info.link->forget(true); // forget the link history for memory efficiency
                    link_info.term = term;

                    // create a term without the link
                    Term trunc_term = *term;
                    trunc_term.term_linkage() = nullptr;
                    vertex_vector trunc_rhs;
                    for (auto &other_vertex: trunc_term.rhs()) {
                        VertexPtr new_vertex = other_vertex;
                        if (other_vertex->is_linked()) {
                            new_vertex = as_link(other_vertex)->replace(temp_link, dummy).first;
                        }
                        trunc_rhs.push_back(new_vertex);
                    }

                    // sort the trunc term by name and update
                    std::sort(trunc_rhs.begin(), trunc_rhs.end(),
                              [](const VertexPtr &a, const VertexPtr &b) { return a->name() < b->name(); });
                    trunc_term.rhs() = trunc_rhs;
                    trunc_term.compute_scaling(true);
                    trunc_term.term_linkage()->forget(true); // forget the link history for memory efficiency
                    
                    link_info.trunc_term = trunc_term;

                    // insert the trunc term into the trunc map
                    link_track_map_[link_info.link].push_back(link_info);
                }
            }
        }
    }

    void populate(PQGraph& pq_graph) {
        clear();
        for (auto & [name, eq] : pq_graph.equations()) {
            for (auto &term : eq.terms()) {
                insert(&term);
            }
        }

        auto redundant_idxs = [](const vector<LinkInfo> &vec) {
            set<size_t, std::greater<>> idxs;
            set<Term*> terms;
            for (size_t i = 0; i < vec.size(); i++) {
                Term *term = vec[i].term;
                if (terms.find(term) != terms.end()) {
                    idxs.insert(i);
                }
                terms.insert(term);
            }
            return idxs;
        };

        // get redundant indices
        for (auto &[link, link_infos]: link_track_map_) {
            auto idxs = redundant_idxs(link_infos);
            
            vector<LinkInfo> new_link_infos;
            for (size_t i = 0; i < link_infos.size(); i++) {
                if (idxs.find(i) == idxs.end()) {
                    new_link_infos.push_back(link_infos[i]);
                }
            }
            link_infos = new_link_infos;
        }

        // now we sort the trunc and track terms by string representation
        for (auto &[link, link_infos]: link_track_map_) {

            vector<pair<string, size_t>> argsorted_infos;
            for (size_t i = 0; i < link_infos.size(); i++) {
                // use trunc_term.str() as primary key, and term's full str() as secondary
                // to break ties consistently across different link info vectors
                string sort_key = link_infos[i].trunc_term.str();
                sort_key += "|";
                sort_key += link_infos[i].term->str();
                argsorted_infos.emplace_back(sort_key, i);
            }
            std::sort(argsorted_infos.begin(), argsorted_infos.end());

            // reorder the info vectors by the sorted indices
            vector<LinkInfo> new_link_infos;
            for (auto & [_, i] : argsorted_infos) {
                new_link_infos.push_back(link_infos[i]);
            }

            link_infos = new_link_infos;
        }
    }

    void clear() {
        link_track_map_.clear();
        link_declare_map_.clear();
    }

    void prune() {

        linkage_map<vector<LinkInfo>> new_link_track_map;
        linkage_map<set<Term*>> new_link_declare_map;

        for (auto &[link, link_infos]: link_track_map_) {

            bool remove_link = false;

            // remove all linkages that have no declaration
            remove_link |= link_declare_map_[link].empty();

            // remove all linkages that have no track terms
            remove_link |= link_infos.empty();

            // ensure that all tracked link infos for this linkage have consistent permutations of their lines so that connectivity can be compared meaningfully
            perm_list ref_perms;
            for (auto &info : link_infos) {
                // ensure the lines within each tracked link info are consistently permuted
                perm_list tracked_perms;
                if (info.term->perm_type() != 0) {
                    set<string> seen_lines;
                    for (auto &line : info.link->lines()) {
                        seen_lines.insert(line.label_);
                    }                    
                    for (auto &perm_pair : info.term->term_perms()) {
                        if (seen_lines.find(perm_pair.first) != seen_lines.end()) {
                            tracked_perms.push_back(perm_pair);
                        } else if (seen_lines.find(perm_pair.second) != seen_lines.end()) {
                            tracked_perms.emplace_back(perm_pair.second, perm_pair.first);
                        }
                    }
                }

                if (ref_perms.empty()) ref_perms = tracked_perms;
                else if (ref_perms != tracked_perms) remove_link = true;
                if (remove_link) break;
            }
            

            if (!remove_link) {
                new_link_track_map.insert({link, link_infos});
                new_link_declare_map.insert({link, link_declare_map_[link]});
            }
            
            link->forget(true); // forget the link history for memory efficiency
        }

        // overwrite the link maps
        link_track_map_ = new_link_track_map;
        link_declare_map_ = new_link_declare_map;
    }

};

struct LinkMerger {
    LinkTracker link_tracker_;
    PQGraph& pq_graph_;
    linkage_map<linkage_vector> link_merge_map_;

    explicit LinkMerger(PQGraph& pq_graph) : pq_graph_(pq_graph){
        link_merge_map_.reserve(10 * pq_graph_.saved_linkages().size());
        link_tracker_.populate(pq_graph_);
        link_tracker_.prune();
    }

    void populate() {
        // find all linkages that can be merged (same connectivity with all trunc terms)
        VertexPtr dummy = 0.0 * std::make_shared<Vertex>("dummy");

        // extract trunc map to separate vectors to parallelize
        linkage_vector all_links; all_links.reserve(link_tracker_.link_track_map_.size());
        vector<vector<LinkInfo>> all_infos; all_infos.reserve(link_tracker_.link_track_map_.size());
        for (auto &[link, link_infos]: link_tracker_.link_track_map_) {
            link->forget(true); // forget the link history for memory efficiency
            if (!link_infos.empty()) {
                all_links.push_back(link);
                all_infos.push_back(link_infos);
            }
        }

        // per-index results to avoid critical section
        vector<vector<pair<LinkagePtr, LinkagePtr>>> per_k_results(all_links.size());

        #pragma omp parallel for schedule(guided) default(none) shared(all_links, all_infos, dummy, per_k_results)
        for (size_t k = 0; k < all_links.size(); k++) {
            auto &link1 = all_links[k];
            auto &link1_info = all_infos[k];

            for (size_t l = k + 1; l < all_links.size(); l++) {
                auto &link2 = all_links[l];
                auto &link2_info = all_infos[l];

                // skip does not have the same number of trunc terms
                if (link1_info.size() != link2_info.size()) continue;

                // shapes must be the same
                if (link1->dim() != link2->dim()) continue;

                // if the first linkage is reused, then the second linkage must be the same (cannot reuse a non-reused)
                if (link1->is_reused() && !link2->is_reused()) continue;
                // check if the trunc terms have the same connectivity
                bool same_connectivity = true;

                double link_ratio = 0.0;
                set<string> ref_conditions;
                perm_list ref_perms;

                VertexPtr reflink1_2 = nullptr;
                for (size_t i = 0; i < link1_info.size(); i++) {

                    // extract link info
                    auto &link1_trunc = link1_info[i].link;
                    auto &link1_term  = link1_info[i].term;
                    auto &trunc_term1 = link1_info[i].trunc_term;

                    auto &link2_trunc = link2_info[i].link;
                    auto &link2_term = link2_info[i].term;
                    auto &trunc_term2 = link2_info[i].trunc_term;

                    // ensure both trunc links have the same exact lines
                    if (link1_trunc->lines() != link2_trunc->lines()) { same_connectivity = false; break; }

                    // ensure link is not within an addition (cannot merge, intermediate must be top level term)
                    auto term1_temps = link1_term->term_linkage()->get_temps(true, false);
                    auto term2_temps = link2_term->term_linkage()->get_temps(true, false);

                    // find link1 and link2 in the term temps
                    bool found1 = false, found2 = false;
                    for (auto &temp: term1_temps) {
                        if (*temp == *link1_trunc) { found1 = true; break; }
                    }
                    if (!found1) { same_connectivity = false; break; }

                    for (auto &temp: term2_temps) {
                        if (*temp == *link2_trunc) { found2 = true; break; }
                    }
                    if (!found2) { same_connectivity = false; break; }

                    // ensure connectivity of the two trunc links are the same for all terms
                    VertexPtr link1_2 = link1_trunc + link2_trunc;
                    if (reflink1_2 == nullptr) reflink1_2 = link1_2;
                    else if (*link1_2 != *reflink1_2) { same_connectivity = false; break; }

                    // determine if permutation is the same
                    if (link1_term->perm_type()  != link2_term->perm_type())  { same_connectivity = false; break; }
                    if (link1_term->term_perms() != link2_term->term_perms()) { same_connectivity = false; break; }

                    // check that the conditions of the full terms are the same
                    auto term1_conditions = link1_term->conditions();
                    auto term2_conditions = link2_term->conditions();
                    if (term1_conditions != term2_conditions) { same_connectivity = false; break; }

                    // determine if coefficient ratio is the same (should be)
                    double cur_ratio = link2_term->coefficient_ / link1_term->coefficient_;
                    if (link_ratio == 0.0) link_ratio = cur_ratio;
                    else if (fabs(cur_ratio - link_ratio) > 1e-10) { same_connectivity = false; break; }

                    // replace the replacement vertex with the trunc vertex
                    VertexPtr term1_link = trunc_term1.term_linkage()->replace(dummy, link1_trunc).first;
                    VertexPtr term2_link = trunc_term2.term_linkage()->replace(dummy, link1_trunc).first;

                    term1_link = link1_term->lhs() + term1_link;
                    term2_link = link2_term->lhs() + term2_link;

                    // ensure both the links have the same exact lines
                    if (term1_link->lines() != term2_link->lines()) { same_connectivity = false; break; }

                    if (*term1_link != *term2_link) { same_connectivity = false; break; }

                    // now check the other trunc term
                    term1_link = trunc_term1.term_linkage()->replace(dummy, link2_trunc).first;
                    term2_link = trunc_term2.term_linkage()->replace(dummy, link2_trunc).first;

                    term1_link = link1_term->lhs() + term1_link;
                    term2_link = link2_term->lhs() + term2_link;

                    // ensure both the links have the same exact lines
                    if (term1_link->lines() != term2_link->lines()) { same_connectivity = false; break; }

                    if (*term1_link != *term2_link) { same_connectivity = false; break; }

                }

                if (!same_connectivity) continue;
                else {
                    per_k_results[k].push_back({link1, link2});
                }
            }
        }

        // serial merge of per-index results into link_merge_map_
        for (auto &results : per_k_results) {
            for (auto &[target, merge] : results) {
                target->forget(true);
                merge->forget(true);
                link_merge_map_[target].push_back(merge);
            }
        }
    }

    set<Term *> extract_terms(const LinkagePtr& target_link) {// add the track terms to the visited terms
        set<Term*> visited_terms;
        for (auto &info: link_tracker_.link_track_map_[target_link]) {
            if (info.term) visited_terms.insert(info.term);
        }
        for (auto &term: link_tracker_.link_declare_map_[target_link]) {
            if (term) visited_terms.insert(term);
        }

        return visited_terms;
    }

    void prune() {
        // remove merge linkages that have shared terms
        set<Term*> unique_terms;

        // prune each linkage in the merge map
        linkage_map<linkage_vector> new_link_merge_map;
        new_link_merge_map.reserve(link_merge_map_.size());

        vector<pair<LinkagePtr,linkage_vector>> sorted_link_infos;
        sorted_link_infos.reserve(link_merge_map_.size());
        for (const auto& merge_entry: link_merge_map_)
            if (merge_entry.first) {
                sorted_link_infos.insert(std::lower_bound(
                         sorted_link_infos.begin(), sorted_link_infos.end(), merge_entry,
                        [this](const pair<LinkagePtr,linkage_vector> &a, const pair<LinkagePtr,linkage_vector> &b) {
                            scaling_map a_scales, b_scales;

                            // get scaling of terms for a
                            for (auto &info: link_tracker_.link_track_map_[a.first])
                                a_scales += info.term->flop_map();
                            for (auto &a_merger: a.second) {
                                for (auto &info: link_tracker_.link_track_map_[a_merger])
                                    a_scales += info.term->flop_map();
                            }

                            // get scaling of terms for b
                            for (auto &info: link_tracker_.link_track_map_[b.first])
                                b_scales += info.term->flop_map();
                            for (auto &b_merger: b.second) {
                                for (auto &info: link_tracker_.link_track_map_[b_merger])
                                    b_scales += info.term->flop_map();
                            }

                            // keep more expensive linkages first
                            if (a_scales != b_scales) return a_scales > b_scales;
                            else return a.first->id() < b.first->id();

                       }), merge_entry);
            }

        linkage_vector sorted_links;
        sorted_links.reserve(sorted_link_infos.size());
        for (auto &[link, _]: sorted_link_infos) {
            sorted_links.push_back(link);
        }
        sorted_link_infos.clear();


        for (auto &target_link: sorted_links) {

            auto &merge_links =  link_merge_map_[target_link];
            if (merge_links.empty()) continue;

            auto target_terms = extract_terms(target_link);
            unique_terms.insert(target_terms.begin(), target_terms.end());

            linkage_vector new_merge_links;
            for (auto &merge_link: merge_links) {
                auto merge_terms = extract_terms(merge_link);

                // check if any of the terms are unique
                bool unique = true;
                for (auto &term: merge_terms) {
                    unique = unique_terms.find(term) == unique_terms.end();
                    if (!unique) break;
                }

                // if unique, add to new merge links
                if (unique) {
                    new_merge_links.push_back(merge_link);
                    unique_terms.insert(merge_terms.begin(), merge_terms.end());
                }
            }

            // overwrite the merge links
            new_link_merge_map[target_link] = new_merge_links;
        }

        // overwrite the link merge map
        link_merge_map_ = new_link_merge_map;

        // rebuild sorted links
        sorted_links.clear();
        sorted_links.reserve(link_merge_map_.size());
        for (auto &[link, _]: link_merge_map_)
            if (link) sorted_links.push_back(link);
        std::sort(sorted_links.begin(), sorted_links.end(), [](const LinkagePtr &a, const LinkagePtr &b) { return a->id() > b->id(); });

        // now prune between the merge links
        unique_terms.clear();
        new_link_merge_map.clear();
        unordered_set<LinkagePtr, LinkageHash, LinkageEqual> visited_links;
        for (auto &ref_link: sorted_links) {
            auto &ref_merge_links = link_merge_map_[ref_link];
            auto ref_terms = extract_terms(ref_link);
            for (auto &ref_merge_link: ref_merge_links) {
                auto merge_terms = extract_terms(ref_merge_link);
                ref_terms.insert(merge_terms.begin(), merge_terms.end());
            }

            unique_terms.insert(ref_terms.begin(), ref_terms.end());
            visited_links.insert(ref_link);

            // skip if no merge links
            if (ref_merge_links.empty()) continue;

            // add to new link merge map
            new_link_merge_map[ref_link] = ref_merge_links;

            for (auto &comp_link : sorted_links) {
                auto &comp_merge_links = link_merge_map_[comp_link];

                // skip if link has been visited
                if (visited_links.find(comp_link) != visited_links.end()) continue;

                if (comp_merge_links.empty()) continue;

                auto comp_terms = extract_terms(comp_link);
                for (auto &comp_merge_link: comp_merge_links) {
                    auto merge_terms = extract_terms(comp_merge_link);
                    comp_terms.insert(merge_terms.begin(), merge_terms.end());
                }

                // check if any of the terms are unique
                bool unique = false;
                for (auto &term: comp_terms) {
                    unique = unique_terms.find(term) == unique_terms.end();
                    if (!unique) break;
                }

                // if unique, add to new link merge map
                if (unique) {
                    new_link_merge_map[comp_link] = comp_merge_links;
                    unique_terms.insert(comp_terms.begin(), comp_terms.end());
                }
            }
        }

        // overwrite the link merge map
        link_merge_map_ = new_link_merge_map;
    }

    void print() {
        for (auto &[target_link, merge_links]: link_merge_map_) {
            cout << "Fusion Targets: " << endl;
            Term target_as_term(as_link(target_link));
            string target_str = target_as_term.str();
            // replace all newlines with newlines and spaces
            auto pos = target_str.find('\n');
            while (pos != string::npos) {
                target_str.replace(pos, 1, "\n    ");
                pos = target_str.find('\n', pos + 5);
            }
            cout << "    " << target_str << endl;

            for (auto &merge_link: merge_links) {
                Term merge_as_term(as_link(merge_link));
                string merge_str = merge_as_term.str();
                // replace all newlines with newlines and spaces
                pos = merge_str.find('\n');
                while (pos != string::npos) {
                    merge_str.replace(pos, 1, "\n    ");
                    pos = merge_str.find('\n', pos + 5);
                }
            }
        }
    }

    void merge() {
        // initialize new declarations
        vector<pair<string, Term>> new_declarations;

        // merge the linkages
        for (auto &[target, merge_links]: link_merge_map_) {

            // get target linkage
            LinkagePtr target_link = target;

            // update max id
            link_tracker_.max_ids_[target_link->type()]++;

            // get the info for the target and merge trunc terms
            const vector<LinkInfo> &target_infos = link_tracker_.link_track_map_[target_link];
            vector<vector<LinkInfo>> merge_infos;
            for (auto &merge_link: merge_links) {
                auto &merge_info = link_tracker_.link_track_map_[merge_link];
                merge_infos.push_back(merge_info);
            }

            // merge the trunc terms
            vector<Term> new_terms(target_infos.size());
            vector<MutableVertexPtr> merged_vertices(target_infos.size());
            vector<long> max_ids(target_infos.size());
            MutableVertexPtr merged_vertex_init;
            string link_type = target_infos[0].link->type();

            #pragma omp parallel for schedule(guided) default(none) shared(target_infos, merge_infos, new_terms, merged_vertices, max_ids, merged_vertex_init, link_type, target_link, pq_graph_, link_tracker_)
            for (size_t i = 0; i < target_infos.size(); i++) {
                // build merged vertex
                MutableVertexPtr merged_vertex = target_infos[i].link->shallow();
                long max_id = link_tracker_.max_ids_[target_link->type()];

                Term *target_term = target_infos[i].term;
                string merged_pq = target_term->original_pq_;
                for (auto &merge_info: merge_infos) {
                    MutableLinkagePtr target_vertex = as_link(merge_info[i].link->shallow());
                    Term *merge_term = merge_info[i].term;
                    max_id = std::max(max_id, target_vertex->id());

                    // get ratio of coefficients
                    double ratio = merge_term->coefficient_ / target_term->coefficient_;
                    if (fabs(ratio - 1.0) > 1e-10)
                         merged_vertex = merged_vertex + ratio * target_vertex;
                    else merged_vertex = merged_vertex + target_vertex;

                    // add the pq string to track evaluation
                    merged_pq += "\n    " + Vertex::printer_->comment_prefix() + " ";
                    merged_pq += string(merge_term->lhs()->name().size(), ' ');
                    merged_pq += " += " + merge_term->original_pq_;
                }

                as_link(merged_vertex)->factor();
                bool last_add_bool = merged_vertex->is_addition();
                as_link(merged_vertex)->copy_misc(target_infos[i].link);
                as_link(merged_vertex)->is_addition() = last_add_bool;
                as_link(merged_vertex)->id() = max_id;

                if (i == 0) {
                    merged_vertex_init = merged_vertex->relabel()->shallow();
                    as_link(merged_vertex_init)->factor();
                    as_link(merged_vertex_init)->copy_misc(target_infos[i].link);
                    as_link(merged_vertex_init)->is_addition() = last_add_bool;
                    as_link(merged_vertex_init)->id() = max_id;
                }

                // build the new term
                Term new_term = target_infos[i].trunc_term;
                new_term.original_pq_ = merged_pq;

                // find replacement vertex
                for (auto& vertex : new_term.rhs()) {
                    if (vertex->is_linked()) {
                        VertexPtr new_vertex = as_link(vertex)->replace(0.0 * std::make_shared<Vertex>("dummy"), merged_vertex).first;
                        vertex = new_vertex;
                    }
                }
                new_term.request_update();
                new_term.reorder();
                new_terms[i] = new_term.shallow();

                // store for batch update after parallel loop
                merged_vertices[i] = as_link(merged_vertex);
                max_ids[i] = max_id;
            }

            // batch update saved linkages and temp counts (no critical section needed)
            for (size_t i = 0; i < merged_vertices.size(); i++) {
                pq_graph_.saved_linkages()[link_type].insert(merged_vertices[i]);
                pq_graph_.temp_counts()[link_type] = std::max(pq_graph_.temp_counts()[link_type], max_ids[i]);
            }

            // overwrite the target terms with the new terms
            size_t idx = 0;
            for (auto &link_info: link_tracker_.link_track_map_[target_link]) {
                *link_info.term = new_terms[idx++];
            }

            set<Term*> declare_term = link_tracker_.link_declare_map_[target_link];
            if (!declare_term.empty()) {
                // build new declaration
                Term new_def = (*declare_term.begin())->shallow();
                new_def.eq()  = merged_vertex_init;
                new_def.lhs() = merged_vertex_init;

                MutableLinkagePtr merged_vertex_copy = as_link(merged_vertex_init->shallow());
                merged_vertex_copy->id() = -1;
                new_def.expand_rhs(merged_vertex_copy);

                new_def.request_update();
                new_def.compute_scaling(true);
                new_def.term_linkage()->forget(); // forget the link history for memory efficiency

                // add to new declarations
                new_declarations.emplace_back(link_type, new_def);
            } else {
                // add to new declarations
                new_declarations.emplace_back(link_type, Term(as_link(merged_vertex_init)));
            }

            // set merged terms to null lhs
            for (auto &merge_link: merge_links) {
                merge_link->forget(true); // forget the link history for memory efficiency
                for (auto &link_info: link_tracker_.link_track_map_[merge_link]) {
                    if (link_info.term) link_info.term->lhs() = nullptr;
                }
            }
        }

        // remove null terms and merge constants
        for (auto & [name, eq] : pq_graph_.equations()) {
            vector<Term> new_terms;
            for (auto &term: eq.terms()) {
                if (term.lhs() != nullptr) {
                    new_terms.push_back(term);
                }
            }
            eq.terms() = new_terms;
            eq.collect_scaling(true);
        }

        for (auto &[link_type, new_declaration]: new_declarations) {

            // substitute the new merged linkages
            LinkagePtr new_merged_link = as_link(new_declaration.lhs());
            for (auto &[name, eq]: pq_graph_.equations()) {
                eq.substitute(new_merged_link, true);
            }

            // add new declarations
            auto &link_equation = pq_graph_.equations()[link_type];
            link_equation.terms().insert(link_equation.begin(), new_declaration);
            link_equation.rearrange();
        }
    }

    void clear() {
        link_tracker_.clear();
        link_merge_map_.clear();
    }

};


size_t PQGraph::merge_intermediates(){
    if (opt_level_ < 6)
        return 0;

    print_guard guard;
    if (print_level_ < 2) {
        guard.lock();
    }

    // count terms in pq_graph
    size_t num_terms = get_num_terms();

    LinkMerger link_merger(*this);
    link_merger.populate();
    link_merger.prune();
    link_merger.print();
    link_merger.merge();
    link_merger.clear();

    size_t fused_terms = get_num_terms();
    fused_terms = num_terms - fused_terms;
    collect_scaling();

    size_t num_fused_total = fused_terms;
    while (fused_terms > 0) {
        fused_terms = merge_intermediates(); // recursively merge intermediates until no more terms are fused
        num_fused_total += fused_terms;
    }

    return num_fused_total;
}