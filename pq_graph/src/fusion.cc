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

struct LinkInfo {
    ConstLinkagePtr link;
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

        // extract all linkages within the term TODO: do this for all arbitrary linkages (not just temps)
        ConstVertexPtr dummy = 0.0 * std::make_shared<Vertex>("dummy");
        for (auto &vertex: term->rhs()) {
            if (vertex->has_any_temp()) {
                auto all_temps = as_link(vertex)->get_temps();
                for (auto &temp: all_temps) {
                    ConstLinkagePtr temp_link = as_link(temp);
                    max_ids_[temp_link->type()] = max(max_ids_[temp_link->type()], temp_link->id());
                    
                    // create a new link info
                    LinkInfo link_info;
                    link_info.link = temp_link;
                    link_info.link->forget(true); // forget the link history for memory efficiency
                    link_info.term = term;
                    
//                    link_track_map_[temp_link].emplace_back(temp_link, term);

                    Term trunc_term = *term;
                    trunc_term.term_linkage() = nullptr;
                    vector<ConstVertexPtr> trunc_rhs;
                    for (auto &other_vertex: trunc_term.rhs()) {
                        ConstVertexPtr new_vertex = other_vertex;
                        if (other_vertex->is_linked()) {
                            new_vertex = as_link(other_vertex)->replace(temp_link, dummy, true).first;
                        }
                        trunc_rhs.push_back(new_vertex);
                    }

                    // sort the trunc term by name and update
                    std::sort(trunc_rhs.begin(), trunc_rhs.end(),
                              [](const ConstVertexPtr &a, const ConstVertexPtr &b) { return a->name() < b->name(); });
                    trunc_term.rhs() = trunc_rhs;
                    trunc_term.compute_scaling(true);
                    
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
                argsorted_infos.emplace_back(link_infos[i].trunc_term.str(), i);
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
    linkage_map<vector<ConstLinkagePtr>> link_merge_map_;

    explicit LinkMerger(PQGraph& pq_graph) : pq_graph_(pq_graph){
        link_merge_map_.reserve(10 * pq_graph_.saved_linkages().size());
        link_tracker_.populate(pq_graph_);
        link_tracker_.prune();
    }

    void populate() {
        // find all linkages that can be merged (same connectivity with all trunc terms)
        ConstVertexPtr dummy = 0.0 * std::make_shared<Vertex>("dummy");

        // extract trunc map to separate vectors to parallelize
        vector<ConstLinkagePtr> all_links; all_links.reserve(link_tracker_.link_track_map_.size());
        vector<vector<LinkInfo>> all_infos; all_infos.reserve(link_tracker_.link_track_map_.size());
        for (auto &[link, link_infos]: link_tracker_.link_track_map_) {
            link->forget(true); // forget the link history for memory efficiency
            if (!link_infos.empty()) {
                all_links.push_back(link);
                all_infos.push_back(link_infos);
            }
        }

        #pragma omp parallel for schedule(guided) default(none) shared(all_links, all_infos, link_merge_map_, dummy)
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
                for (size_t i = 0; i < link1_info.size(); i++) {

                    // extract link info
                    auto &link1_trunc = link1_info[i].link;
                    auto &link1_term  = link1_info[i].term;
                    auto &trunc_term1 = link1_info[i].trunc_term;
                    
                    auto &link2_trunc = link2_info[i].link;
                    auto &link2_term = link2_info[i].term;
                    auto &trunc_term2 = link2_info[i].trunc_term;

                    // determine if permutation is the same
                    if (link1_term->perm_type()  != link2_term->perm_type())  { same_connectivity = false; break; }
                    if (link1_term->term_perms() != link2_term->term_perms()) { same_connectivity = false; break; }

                    // determine if coefficient ratio is the same (should be)
                    double cur_ratio = link2_term->coefficient_ / link1_term->coefficient_;
                    if (link_ratio == 0.0) link_ratio = cur_ratio;
                    if (fabs(cur_ratio - link_ratio) > 1e-10) { same_connectivity = false; break; }

                    // replace the replacement vertex with the trunc vertex
                    ConstVertexPtr term1_link = trunc_term1.term_linkage()->replace(dummy, link1_trunc).first;
                    ConstVertexPtr term2_link = trunc_term2.term_linkage()->replace(dummy, link1_trunc).first;

                    term1_link = link1_term->lhs() + term1_link;
                    term2_link = link2_term->lhs() + term2_link;

                    if (*term1_link != *term2_link) { same_connectivity = false; break; }

                    // now check the other trunc term
                    term1_link = trunc_term1.term_linkage()->replace(dummy, link2_trunc).first;
                    term2_link = trunc_term2.term_linkage()->replace(dummy, link2_trunc).first;

                    term1_link = link1_term->lhs() + term1_link;
                    term2_link = link2_term->lhs() + term2_link;

                    if (*term1_link != *term2_link) { same_connectivity = false; break; }

                    // check that the conditions of the full terms are the same
                    auto term1_conditions = link1_term->conditions();
                    auto term2_conditions = link2_term->conditions();
                    if (term1_conditions != term2_conditions) { same_connectivity = false; break; }

                }

                if (!same_connectivity) continue;
                else {
                    #pragma omp critical
                    {
                        link1->forget(true); // forget the link history for memory efficiency
                        link2->forget(true); // forget the link history for memory efficiency
                        link_merge_map_[link1].push_back(link2); 
                    }
                }
            }
        }
    }

    set<Term *> extract_terms(const ConstLinkagePtr& target_link) {// add the track terms to the visited terms
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
        linkage_map<vector<ConstLinkagePtr>> new_link_merge_map;
        new_link_merge_map.reserve(link_merge_map_.size());

        vector<ConstLinkagePtr> sorted_links;
        sorted_links.reserve(link_merge_map_.size());
        for (auto &[link, _]: link_merge_map_)
            if (link) sorted_links.push_back(link);
        std::sort(sorted_links.begin(), sorted_links.end(), [](const ConstLinkagePtr &a, const ConstLinkagePtr &b) { return a->id() > b->id(); });

        for (auto &target_link: sorted_links) {

            auto &merge_links =  link_merge_map_[target_link];
            if (merge_links.empty()) continue;

            auto target_terms = extract_terms(target_link);
            unique_terms.insert(target_terms.begin(), target_terms.end());

            vector<ConstLinkagePtr> new_merge_links;
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
        std::sort(sorted_links.begin(), sorted_links.end(), [](const ConstLinkagePtr &a, const ConstLinkagePtr &b) { return a->id() > b->id(); });

        // now prune between the merge links
        unique_terms.clear();
        new_link_merge_map.clear();
        unordered_set<ConstLinkagePtr, LinkageHash, LinkageEqual> visited_links;
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
            cout << "Target Terms: " << endl;
            for (auto &link_info: link_tracker_.link_track_map_[target_link]) {
                auto &term = link_info.term;
                string term_str = term->str();
                // replace all newlines with newlines and spaces
                pos = term_str.find('\n');
                while (pos != string::npos) {
                    term_str.replace(pos, 1, "\n    ");
                    pos = term_str.find('\n', pos + 5);
                }
                cout << "    " << term_str << endl;
            }
            for (auto &merge_link: merge_links) {
                for (auto &link_info: link_tracker_.link_track_map_[merge_link]) {
                    auto &term = link_info.term;
                    string term_str = term->str();
                    // replace all newlines with newlines and spaces
                    pos = term_str.find('\n');
                    while (pos != string::npos) {
                        term_str.replace(pos, 1, "\n    ");
                        pos = term_str.find('\n', pos + 5);
                    }
                    cout << "    " << term_str << endl;
                }
            }
            cout << endl;
        }
    }

    void merge() {
        // initialize new declarations
        vector<pair<string, Term>> new_declarations;

        // merge the linkages
        for (auto &[target, merge_links]: link_merge_map_) {

            // get target linkage
            ConstLinkagePtr target_link = target;

            // update max id
            link_tracker_.max_ids_[target_link->type()]++;

            // get the info for the target and merge trunc terms
            const vector<LinkInfo> &target_infos = link_tracker_.link_track_map_[target_link];
            vector<vector<LinkInfo>> merge_infos;
            for (auto &merge_link: merge_links) {
                auto &merge_info = link_tracker_.link_track_map_[merge_link];
                // sort merge infos by hash string of the link
                std::sort(merge_info.begin(), merge_info.end(), [](const LinkInfo &a, const LinkInfo &b) {
                        constexpr LinkageHash link_hash;
                        return link_hash.string_hash(a.link) < link_hash.string_hash(b.link);
                        });
                merge_infos.push_back(merge_info);
            }

            // merge the trunc terms
            vector<Term> new_terms;
            VertexPtr merged_vertex_init;
            string link_type = target_infos[0].link->type();

            #pragma omp parallel for default(none) shared(target_infos, merge_infos, new_terms, merged_vertex_init, link_type, target_link)
            for (size_t i = 0; i < target_infos.size(); i++) {
                // build merged vertex
                VertexPtr merged_vertex = target_infos[i].link->shallow();
                long max_id = link_tracker_.max_ids_[target_link->type()];
                if (merged_vertex->type() == "temp")
                    // only unset the id if the vertex is a temp
                    merged_vertex->id() = -1;

                Term *target_term = target_infos[i].term;
                string merged_pq = target_term->original_pq_;
                for (auto &merge_info: merge_infos) {
                    LinkagePtr target_vertex = as_link(merge_info[i].link->shallow());
                    Term *merge_term = merge_info[i].term;
                    max_id = std::max(max_id, target_vertex->id());

                    if (target_vertex->type() == "temp")
                        // only unset the id if the vertex is a temp
                        target_vertex->id() = -1;

                    // get ratio of coefficients
                    double ratio = merge_term->coefficient_ / target_term->coefficient_;
                    if (fabs(ratio - 1.0) > 1e-10)
                         merged_vertex = merged_vertex + ratio * target_vertex;
                    else merged_vertex = merged_vertex + target_vertex;
                    as_link(merged_vertex)->fuse();

                    // add the pq string to track evaluation
                    merged_pq += Term::make_einsum ? "\n    # " : "\n    // ";
                    merged_pq += string(merge_term->lhs()->name().size(), ' ');
                    merged_pq += " += " + merge_term->original_pq_;
                }
                bool last_add_bool = merged_vertex->is_addition();
                as_link(merged_vertex)->copy_misc(target_infos[i].link);
                as_link(merged_vertex)->is_addition() = last_add_bool;
                as_link(merged_vertex)->id() = max_id;

                if (i == 0) merged_vertex_init = merged_vertex;

                // build the new term
                Term new_term = target_infos[i].trunc_term;
                new_term.original_pq_ = merged_pq;

                // find replacement vertex
                for (auto& vertex : new_term.rhs()) {
                    if (vertex->is_linked()) {
                        ConstVertexPtr new_vertex = as_link(vertex)->replace(0.0 * std::make_shared<Vertex>("dummy"), merged_vertex).first;
                        vertex = new_vertex;
                    }
                }
                new_term.request_update();
                new_term.reorder();
                new_term.term_linkage()->forget(); // forget the link history for memory efficiency

                // add merged vertex to saved linkages
                pq_graph_.saved_linkages()[link_type].insert(as_link(merged_vertex));
                #pragma omp critical
                {
                    pq_graph_.temp_counts()[link_type] = std::max(pq_graph_.temp_counts()[link_type], max_id);
                    new_terms.push_back(new_term);
                }
            }

            // overwrite the target terms with the new terms
            size_t idx = 0;
            for (auto &link_info: link_tracker_.link_track_map_[target_link]) {
                *link_info.term = new_terms[idx++];
            }

            set<Term*> declare_term = link_tracker_.link_declare_map_[target_link];
            if (!declare_term.empty()) {
                // build new declaration
                Term new_def = (*declare_term.begin())->clone();
                LinkagePtr merged_vertex_copy = as_link(merged_vertex_init->shallow());
                merged_vertex_copy->id() = -1;
                new_def.eq()  = merged_vertex_init;
                new_def.lhs() = merged_vertex_init;

                new_def.expand_rhs(merged_vertex_copy);
                new_def.request_update();
                new_def.reorder();
                new_def.term_linkage()->forget(); // forget the link history for memory efficiency

                // add to new declarations
                new_declarations.emplace_back(link_type, new_def);
            } else {
                // add to new declarations
                new_declarations.emplace_back(link_type, Term(as_link(merged_vertex_init)));
            }

            // set merged terms to null lhs
            for (auto &merge_link: merge_links) {
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
        }

        // add new declarations
        for (auto &[link_type, new_declaration]: new_declarations) {
            auto &link_equation = pq_graph_.equations()[link_type];
            link_equation.terms().insert(link_equation.begin(), new_declaration);
            link_equation.rearrange('s');
            link_equation.rearrange('r');
            link_equation.rearrange('t');
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
    size_t num_terms = 0;
    for (auto & [name, eq] : equations()) {
        num_terms += eq.terms().size();
    }

    LinkMerger link_merger(*this);
    link_merger.populate();
    link_merger.prune();
    link_merger.print();
    link_merger.merge();
    link_merger.clear();

    size_t fused_terms = 0;
    for (auto & [name, eq] : equations()) {
        fused_terms += eq.terms().size();
    }
    fused_terms = num_terms - fused_terms;
    collect_scaling();

    size_t num_fused_total = fused_terms;
    while (fused_terms > 0) {
        fused_terms = merge_intermediates(); // recursively merge intermediates until no more terms are fused
        num_fused_total += merge_intermediates();
    }

    return num_fused_total;
}

size_t PQGraph::prune() {

    if (opt_level_< 5)
        return 0; // do not remove unused temps if pruning is disabled

    print_guard guard;
    if (print_level_ < 2) {
        guard.lock();
    }

    // remove unused contractions (only used in one term and its assignment)

    // get all matching terms for each temp in saved_linkages
    linkage_map<set<Term*>> matching_terms;
    linkage_map<set<Term*>> tmp_decl_terms;
    matching_terms.reserve(10*saved_linkages_.size());
    tmp_decl_terms.reserve(10*saved_linkages_.size());
    for (const auto & [type, linkages] : saved_linkages_) {
        for (const auto &linkage : linkages) {

            if (linkage->id() == -1) continue; // skip if id is -1
            auto [tmp_terms, tmp_decls] = get_matching_terms(linkage);
            if (tmp_terms.empty() && tmp_decls.empty()) continue; // occurs nowhere in the equations; skip

            matching_terms[linkage] = tmp_terms;
            tmp_decl_terms[linkage] = tmp_decls;
        }
    }

    // remove temps that are used in only one term or are not used at all

    size_t num_removed = 0;
    linkage_set to_remove;
    for (const auto & [temp, terms] : matching_terms) {

        // remove (regardless of use) if temp has only one pure vertex
        if (temp->vertices().size() > 1) {
            // count number of occurrences of the temp in the terms
            size_t num_occurrences = 0;
            for (auto &term: terms) {
                if (term->lhs() == nullptr) continue; // skip if term has no lhs (will be removed later)
                for (auto &vertex: term->rhs()) {
                    if (vertex->is_linked()) {
                        auto all_temps = as_link(vertex)->find_links(temp);
                        num_occurrences += all_temps.size();
                    }
                }
            }

            // skip if temp is used more than once
            if (num_occurrences > 1) continue;

            if (num_occurrences == 1 ) {
                if (temp->is_reused() || temp->is_scalar() || temp->is_addition()) {
                    // if used only once and this is a reusable temp, remove only if the term declare a temp
                    bool declares_temp = (*terms.begin())->lhs() != nullptr && (*terms.begin())->lhs()->is_linked();
                    if (!declares_temp) continue;
                }
            }
        }

        num_removed++;

        // set lhs to a null pointer to mark for removal
        set<Term*> tmp_decl_term = tmp_decl_terms[temp];
        if (!tmp_decl_term.empty()) {
            for (auto &term: tmp_decl_term) {
                term->lhs() = nullptr;
            }
        }

        // add to the list of temps to remove
        to_remove.insert(temp);
    }

    if (num_removed > 0) {

        // remove all terms with lhs set to nullptr if any are found
        for (auto &[name, eq]: equations_) {
            vector<Term> new_terms;
            for (auto &term: eq.terms()) {
                if (term.lhs() != nullptr)
                    new_terms.push_back(term.clone());
            }
            eq.terms() = new_terms;
        }

        // sort to_remove by decreasing id
        vector<ConstLinkagePtr> sorted_to_remove;
        sorted_to_remove.reserve(to_remove.size());
        sorted_to_remove.insert(sorted_to_remove.begin(), to_remove.begin(), to_remove.end());
        std::sort(sorted_to_remove.begin(), sorted_to_remove.end(), [](const ConstLinkagePtr &a, const ConstLinkagePtr &b) {
            return a->id() > b->id();
        });

        map<string, linkage_set> new_saved_linkages;
        cout << "Removing unused temps:" << endl;
        for (size_t i = 0; i < sorted_to_remove.size(); i++) {
            ConstLinkagePtr temp = sorted_to_remove[i];
            Term temp_term(as_link(temp));
            string temp_str = temp_term.str();

            // replace all newlines with newlines and spaces
            auto pos = temp_str.find('\n');
            while (pos != string::npos) {
                temp_str.replace(pos, 1, "\n    ");
                pos = temp_str.find('\n', pos + 5);
            }

            cout << "    " << temp_str << endl;

            // replace nested temps to remove
            for (size_t j = i + 1; j < sorted_to_remove.size(); j++) {
                sorted_to_remove[i] = as_link(sorted_to_remove[i]->replace_id(sorted_to_remove[j], -1).first);
            }
            temp = sorted_to_remove[i];

            // unset the temp in saved_linkages
            for (auto &[type, linkages]: saved_linkages_) {
                for (const auto &link: linkages) {
                    if (*link == *temp) continue; // skip if temp is the same
                    if (link->id() == -1) continue; // skip if id is -1

                    ConstVertexPtr new_link = as_link(link)->replace_id(temp, -1).first;
                    if (new_link->id() != -1)
                        new_saved_linkages[type].insert(as_link(new_link));
                }
            }

            // unset the temp in all equations
            // get list of keys in equations
            vector<string> eq_keys = get_equation_keys();

            #pragma omp parallel for schedule(guided) shared(equations_, eq_keys, temp) default(none)
            for (const auto& name : eq_keys) {
                Equation &eq = equations_[name]; // get equation
                for (auto &term: eq.terms()) {

                    // start with lhs
                    if (term.lhs() != nullptr && term.lhs()->is_linked()) {
                        // find the temp in the lhs
                        ConstVertexPtr new_lhs = as_link(term.lhs())->replace_id(temp, -1).first;
                        term.lhs() = new_lhs;
                    }
                    if (term.eq() != nullptr && term.eq()->is_linked()) {
                        // find the temp in the eq
                        ConstVertexPtr new_eq = as_link(term.eq())->replace_id(temp, -1).first;
                        term.eq() = new_eq;
                    }

                    // now do the same for rhs
                    for (auto &op: term.rhs()) {
                        if (op != nullptr && op->is_linked()) {
                            // find the temp in the rhs
                            ConstVertexPtr new_op = as_link(op)->replace_id(temp, -1).first;
                            op = new_op;
                        }
                    }
                }
            }
        }

        // overwrite saved_linkages
        saved_linkages_ = new_saved_linkages;
        cout << endl; // print newline after all removals
    }

    // remove all terms with lhs set to nullptr if any are found
    for (auto &[name, eq]: equations_) {
        vector<Term> new_terms;
        for (auto &term: eq.terms()) {
            if (term.lhs() != nullptr) {
                term.reorder(true);
                new_terms.push_back(term);
            }
        }
        eq.terms() = new_terms;
    }

    if (opt_level_ >= 6) {
        vector<Term*> all_terms; all_terms.reserve(10*equations_.size());
        for (auto &[name, eq]: equations_) {
            for (auto &term: eq.terms()) {
                all_terms.push_back(&term);
            }
        }

        #pragma omp parallel for schedule(guided) default(none) shared(all_terms)
        for (Term *term_ptr: all_terms) {
            Term &term = *term_ptr;
            // fuse the term linkage
            LinkagePtr term_link = as_link(term.term_linkage()->clone());
            if (!term_link->is_temp()) continue;
            else term_link->fuse();

            term.expand_rhs(term_link);
            term.reorder(true);
        }
    }

    size_t num_removed_total = num_removed;
    while (num_removed > 0) {
        num_removed = prune(); // recursively prune until no more temps are removed
        num_removed_total += prune();
    }

    return num_removed_total;
}

pair<set<Term *>, set<Term*>> PQGraph::get_matching_terms(const ConstLinkagePtr &intermediate) {
    // grab all terms with this tmp

    // initialize vector of term pointers
    set<Term*> tmp_terms;

    vector<string> eq_keys = get_equation_keys();
#pragma omp parallel for schedule(guided) default(none) shared(equations_, eq_keys, tmp_terms, intermediate)
    for (const auto& eq_name : eq_keys) { // iterate over equations in parallel
        // get equation
        Equation &equation = equations_[eq_name]; // get equation

        // get all terms with this tmp
        set<Term*> tmp_terms_local = equation.get_temp_terms(intermediate);
#pragma omp critical
        {
            // add terms to tmp_terms
            tmp_terms.insert(tmp_terms_local.begin(), tmp_terms_local.end());
        }

    }

    set<Term*> tmp_decl_terms;
    set<Term*> pruned_tmp_terms;
    for (auto &term : tmp_terms) {
        if (term->lhs()->same_temp(intermediate) && term->is_assignment_)
            tmp_decl_terms.insert(term);
        else pruned_tmp_terms.insert(term);
    }

    return {pruned_tmp_terms, tmp_decl_terms};
}

size_t PQGraph::merge_terms() {

    if (opt_level_< 5)
        return 0; // do not merge terms if not allowed

    print_guard guard;
    if (print_level_ < 2) {
        guard.lock();
    }

    // iterate over equations and merge terms
    size_t num_merged = 0;
    vector<string> eq_keys = get_equation_keys();
#pragma omp parallel for reduction(+:num_merged) default(none) shared(equations_, eq_keys)
    for (const auto &key: eq_keys) {
        Equation &eq = equations_[key];
        if (eq.is_temp_equation_) continue; // skip tmps equation

        num_merged += eq.merge_terms(); // merge terms with same rhs up to a permutation
    }
    collect_scaling(); // collect new scalings

    if (num_merged > 0) cout << "Merged " << num_merged << " terms" << endl;

    return num_merged;
}

double PQGraph::common_coefficient(set<Term*> &terms) {

    return 1.0; // do not modify coefficients for now

/*    // make a count of the coefficients
    map<string, size_t> coeff_counts;
    for (Term* term_ptr: terms) {
        Term& term = *term_ptr;

        if ((fabs(term.coefficient_) - 1e-10) < 1e-10)
            continue; // skip terms with coefficient of 0
        size_t precision = minimum_precision(fabs(term.coefficient_));
        string coeff_str = to_string_with_precision(precision);
        coeff_counts[coeff_str]++;
    }

    // find the most common coefficient
    string most_common_coeff = "1.00"; // default to 1
    size_t most_common_coeff_count = 1;
    for (const auto &[coeff, count]: coeff_counts) {
        if (count > most_common_coeff_count) {
            most_common_coeff = coeff;
            most_common_coeff_count = count;
        }
    }

    // get common coefficient
    double common_coefficient = stod(most_common_coeff);
    if (common_coefficient <= 1e-8)
        return 1.0; // do not modify coefficients if any are close to 0

    return common_coefficient;*/
}