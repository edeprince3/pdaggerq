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

// Function definitions
linkage_set PQGraph::collect_intermediates() {
    linkage_set intermediates;
    for (const auto &[name, eq] : equations_) {
        for (const auto &term : eq.terms()) {
            for (const auto &op : term.rhs()) {
                if (op->is_temp()) {
                    intermediates.insert(as_link(op));
                }
            }
        }
    }

    return intermediates;
}

std::unordered_map<ConstLinkagePtr, pair<vector<Term*>, vector<Term*>>, LinkageHash, LinkageEqual>
        PQGraph::get_intermediate_terms(const linkage_set &intermediates) {
    std::unordered_map<ConstLinkagePtr, pair<vector<Term*>, vector<Term*>>, LinkageHash, LinkageEqual> intermediate_terms;
    for (const auto &link : intermediates) {
        auto terms = get_matching_terms(link);
        intermediate_terms[link] = terms;
    }
    return intermediate_terms;
}

bool PQGraph::can_merge_terms(const vector<Term*> &this_terms, const vector<Term*> &other_terms, const ConstLinkagePtr& this_intermediate, const ConstLinkagePtr& other_intermediate, const linkage_set& tested_linkages) {

    // cannot merge if not the same size and shape;
    bool same_size = this_terms.size() == other_terms.size();
    bool same_shape = this_intermediate->shape_ == other_intermediate->shape_;
    if (!same_shape || !same_size) return false;


    // cannot merge if linkage has already been tested or if the intermediates are the same
    bool same_temp = this_intermediate->same_temp(other_intermediate);
    bool already_tested = tested_linkages.contains(as_link(other_intermediate->clone()));
    if ( same_temp || already_tested )
        return false;

    string this_inter_str  = this_intermediate->str()  + this_intermediate->tot_str();
    string other_inter_str = other_intermediate->str() + other_intermediate->tot_str();

    long this_id = this_intermediate->id();
    same_temp = this_intermediate->same_temp(other_intermediate);


    for (const auto this_term : this_terms) {
        if (this_term->lhs()->same_temp(this_intermediate))
            continue;

        bool found = false;
        for (const auto other_term : other_terms) {
            if (other_term->lhs()->same_temp(other_intermediate))
                continue; // ignore declaration terms

            if (other_term->lhs()->same_temp(this_intermediate))
                return false; // cannot merge intermediate with itself

            // if different perm type or perms, then they cannot be merged
            if (this_term->perm_type() != other_term->perm_type() ||
             this_term->term_perms() != other_term->term_perms()) continue;

            auto this_rhs = remove_intermediate(this_term->rhs(), this_intermediate);
            auto other_rhs = remove_intermediate(other_term->rhs(), other_intermediate);

            if (this_rhs.size() != other_rhs.size())
                return false;

            sort_operands(this_rhs);
            sort_operands(other_rhs);

            if (any_of(this_rhs.begin(), this_rhs.end(), [&](auto rhs_op) { return rhs_op->is_temp() && tested_linkages.contains(as_link(rhs_op->clone())); }) ||
                any_of(other_rhs.begin(), other_rhs.end(), [&](auto rhs_op) { return rhs_op->is_temp() && tested_linkages.contains(as_link(rhs_op->clone())); })) {
                continue;
            }

            if (!is_same_connectivity(this_rhs, other_rhs, this_term->lhs(), other_term->lhs(), this_intermediate, other_intermediate)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

std::unordered_map<ConstLinkagePtr, vector<ConstLinkagePtr>, LinkageHash, LinkageEqual> PQGraph::find_mergeable_intermediates(const std::unordered_map<ConstLinkagePtr, pair<vector<Term*>, vector<Term*>>, LinkageHash, LinkageEqual> &intermediate_terms, linkage_set &tested_linkages) {
    std::unordered_map<ConstLinkagePtr, vector<ConstLinkagePtr>, LinkageHash, LinkageEqual> merge_map;
    set<Term*> terms_to_merge;

    for (const auto &[this_intermediate, this_terms_pair] : intermediate_terms) {
        auto &[this_terms,this_decl_terms] = this_terms_pair;
        if (tested_linkages.contains(as_link(this_intermediate->clone())) || terms_to_merge.count(this_terms.front()) > 0)
            continue;
        tested_linkages.insert(this_intermediate);

        vector<ConstLinkagePtr> to_merge;
        for (const auto &[other_intermediate, other_terms_pair] : intermediate_terms) {
            auto &[other_terms,other_decl_terms] = other_terms_pair;

            // check if any of the terms have already been merged
            bool already_merged = any_of(other_terms.begin(), other_terms.end(), [&](auto term) {
                return terms_to_merge.count(term) > 0;;
            });

            // not all terms can be merged
            if (already_merged || !can_merge_terms(this_terms, other_terms, this_intermediate, other_intermediate, tested_linkages))
                continue;

            to_merge.push_back(other_intermediate);
            tested_linkages.insert(other_intermediate);
        }

        if (!to_merge.empty()) {
            merge_map[this_intermediate] = to_merge;
            terms_to_merge.insert(this_terms.begin(), this_terms.end());
        }
    }

    return merge_map;
}

vector<ConstVertexPtr> PQGraph::remove_intermediate(const vector<ConstVertexPtr>& rhs, const ConstLinkagePtr &intermediate) {
    vector<ConstVertexPtr> new_rhs;
    new_rhs.reserve(rhs.size());
    for (const auto &rhs_op : rhs) {
        if (as_link(rhs_op)->same_temp(intermediate)) continue;

        if (rhs_op->has_temp(intermediate)) {
            auto expanded = as_link(rhs_op)->expand_to_temp(intermediate);
            auto expanded_verts = as_link(expanded)->link_vector(true);
            for (const auto &expanded_vert : expanded_verts) {
                if (!intermediate->same_temp(expanded_vert)) {
                    new_rhs.push_back(expanded_vert);
                }
            }
        }
        new_rhs.push_back(rhs_op);
    }
    return new_rhs;
}

void PQGraph::sort_operands(vector<ConstVertexPtr> &operands) {
    sort(operands.begin(), operands.end(), [](const auto &a, const auto &b) { return a->name() < b->name(); });
}

bool PQGraph::is_same_connectivity(const vector<ConstVertexPtr> &this_rhs, const vector<ConstVertexPtr> &other_rhs,
                                   const ConstVertexPtr &this_lhs, const ConstVertexPtr &other_lhs,
                                   const ConstLinkagePtr &intermediate, const ConstLinkagePtr &test_intermediate) {

    // replace lines in intermediate with old lines
    VertexPtr similar_intermediate = intermediate->clone();
    as_link(similar_intermediate)->replace_lines(test_intermediate->lines_);

    ConstVertexPtr this_link, other_link;
    if (this_rhs.size() == 1) {
        this_link = this_rhs.front();
        other_link = other_rhs.front();
    } else if (this_rhs.empty()) {
        this_link  = this_lhs + similar_intermediate;
        other_link = other_lhs + similar_intermediate;
        return *as_link(this_link) == *as_link(other_link);
    } else {
        this_link = Linkage::link(this_rhs);
        other_link = Linkage::link(other_rhs);
    }

    this_link  = this_lhs + similar_intermediate * this_link;
    other_link = other_lhs + similar_intermediate * other_link;

    return *as_link(this_link) == *as_link(other_link);
}

void PQGraph::update_saved_linkages(const ConstLinkagePtr &this_intermediate, const vector<ConstLinkagePtr> &other_intermediates) {
    for (auto &[type, linkages] : saved_linkages_) {
        for (const auto &other_intermediate : other_intermediates) {
            linkages.erase(as_link(other_intermediate->clone()));
        }
    }
}

void replace_rhs_operands(Term &term, const ConstLinkagePtr& this_intermediate, const std::unordered_map<ConstLinkagePtr, ConstLinkagePtr> &old_to_new_links) {
    auto new_link = as_link(old_to_new_links.at(this_intermediate)->clone());
    auto line_map = LineHash::map_lines(term.lhs()->lines_, new_link->lines_);

    term.lhs() = new_link;
    term.eq() = new_link;
    vector<ConstVertexPtr> new_rhs;
    for (auto &op : term.rhs()) {
        auto new_op = op->clone();
        new_op->replace_lines(line_map);
        new_rhs.push_back(new_op);
    }
    term.rhs() = new_rhs;
    term.request_update();
}

void PQGraph::add_new_terms(std::unordered_map<ConstLinkagePtr, vector<Term>, LinkageHash, LinkageEqual> &new_inter_terms_map,
                            const std::unordered_map<ConstLinkagePtr, vector<ConstLinkagePtr>, LinkageHash, LinkageEqual> &merge_map) {

    // build new intermediates
    std::unordered_map<ConstLinkagePtr, ConstLinkagePtr> old_to_new_links;
    old_to_new_links = make_new_intermediates(merge_map, old_to_new_links);

    for (auto &[this_intermediate, new_inter_terms] : new_inter_terms_map) {
        const auto &new_intermediate = old_to_new_links[this_intermediate];
        string type;
        for (auto &[link_type, linkages] : saved_linkages_) {
            if (linkages.contains(as_link(this_intermediate->clone()))) {
                linkages.erase(as_link(this_intermediate->clone()));
                linkages.insert(as_link(new_intermediate->clone()));
                type = link_type;
                break;
            }
        }

        for (Term &term : new_inter_terms) {

            term.is_assignment_ = term.lhs()->same_temp(this_intermediate);

            term.lhs() = as_link(new_intermediate->clone());
            term.eq() = as_link(new_intermediate->clone());
//            replace_rhs_operands(term, this_intermediate, old_to_new_links);
            if (term.is_assignment_) {
                LinkagePtr rhs_link = as_link(new_intermediate->clone());
                rhs_link->id() = -1;
                term.rhs() = {rhs_link};
            } else {
                continue;
            }

            term.request_update();

            // add new term to equations
            equations_[type].terms().insert(equations_[type].terms().end(), term);
        }



        // replace old intermediates with new intermediates
        replace_old_intermediate(equations_, old_to_new_links);

        // rearrange all equations
        for (auto &[name, eq] : equations_) {
            eq.rearrange();
        }
    }
}

unordered_map<ConstLinkagePtr, ConstLinkagePtr> &PQGraph::make_new_intermediates(
        const unordered_map<ConstLinkagePtr, vector<ConstLinkagePtr>, LinkageHash, LinkageEqual> &merge_map,
        unordered_map<ConstLinkagePtr, ConstLinkagePtr> &old_to_new_links) const {
    for (const auto &[this_intermediate, other_intermediates] : merge_map) {
        VertexPtr new_intermediate = this_intermediate->clone();
        as_link(new_intermediate)->id() = -1; // do not treat as a temporary intermediate for now

        for (const auto &other_intermediate : other_intermediates) {
            LinkagePtr formatted_other = as_link(other_intermediate->clone());
            formatted_other->replace_lines(*as_link(this_intermediate));
            formatted_other->id() = -1; // do not treat as a temporary intermediate

            new_intermediate = new_intermediate + as_link(formatted_other->clone());
        }

        // copy intermediate properties to new intermediate
        as_link(new_intermediate)->copy_misc(as_link(this_intermediate));
        as_link(new_intermediate)->is_addition_ = true;

        // add new intermediate to the list of intermediates
        old_to_new_links[this_intermediate] = as_link(new_intermediate);
    }
    return old_to_new_links;
}

void PQGraph::remove_terms(const set<Term*> &terms_to_remove) {
    for (auto &[name, eq] : equations_) {
        vector<Term> new_terms;
        for (auto &term : eq.terms()) {
            if (!(terms_to_remove.count(&term) > 0)) {
                new_terms.push_back(term);
            }
        }
        eq.terms() = new_terms;
    }
}

void PQGraph::process_and_merge_intermediates(const std::unordered_map<ConstLinkagePtr, vector<ConstLinkagePtr>, LinkageHash, LinkageEqual> &merge_map,
                                              std::unordered_map<ConstLinkagePtr, pair<vector<Term *>, vector<Term *>>, LinkageHash, LinkageEqual> &intermediate_terms) {
    set<Term*> terms_to_remove;
    std::unordered_map<ConstLinkagePtr, vector<Term>, LinkageHash, LinkageEqual> new_inter_terms_map;

    for (const auto &[this_intermediate, other_intermediates] : merge_map) {
        VertexPtr new_intermediate = this_intermediate->clone();
        as_link(new_intermediate)->id() = -1;

        for (const auto &other_intermediate : other_intermediates) {
            auto &[other_terms,other_decl_terms] = intermediate_terms.at(other_intermediate);

            // add declaration terms for formatting with new intermediate
            for (auto &term : other_decl_terms) {
                new_inter_terms_map[this_intermediate].emplace_back(*term);
                terms_to_remove.insert(term);
            }

            // remove terms to be merged
            for (auto &term : other_terms) {
                terms_to_remove.insert(term);
            }

            // remove declaration term for this intermediate
            auto &[this_terms,this_decl_terms] = intermediate_terms.at(this_intermediate);
            for (auto &term : this_decl_terms) {
                new_inter_terms_map[this_intermediate].emplace_back(*term);
                terms_to_remove.insert(term);
            }

        }

        update_saved_linkages(this_intermediate, other_intermediates);
    }

    remove_terms(terms_to_remove);
    add_new_terms(new_inter_terms_map, merge_map);
}

void PQGraph::replace_old_intermediate(std::map<string, Equation> &equations, const std::unordered_map<ConstLinkagePtr, ConstLinkagePtr> &old_to_new_links) {
    for (auto &[name, eq] : equations) {
        for (auto &term : eq.terms()) {
            for (const auto &[old_link, new_link] : old_to_new_links) {
                for (auto &op : term.rhs()) {
                    if (op->is_linked() && as_link(op)->find_link(old_link)) {
                        LinkagePtr new_op = as_link(op->clone());
                        LinkagePtr new_link_clone = as_link(new_link->clone());
                        new_link_clone->replace_lines(*new_op);

                        new_op->replace_link(old_link, new_link);
                        op = new_op;
                    }
                }
                if (term.lhs()->is_linked() && as_link(term.lhs())->find_link(old_link)) {
                    LinkagePtr new_lhs = as_link(term.lhs()->clone());
                    LinkagePtr new_link_clone = as_link(new_link->clone());
                    new_link_clone->replace_lines(*new_lhs);
                    new_lhs->replace_link(new_link_clone, new_link);
                    term.lhs() = new_lhs;
                }
                if (term.eq() && term.eq()->is_linked() && as_link(term.eq())->find_link(old_link)) {
                    LinkagePtr new_eq = as_link(term.eq()->clone());
                    LinkagePtr new_link_clone = as_link(new_link->clone());
                    new_link_clone->replace_lines(*new_eq);
                    new_eq->replace_link(new_link_clone, new_link);
                    term.eq() = new_eq;
                }
            }
        }
    }
}

void PQGraph::merge_intermediates2() {
    // Retrieve all intermediates
    linkage_set intermediates = collect_intermediates();
    if (intermediates.empty()) {
        cout << "Intermediates not found." << endl;
        return;
    }

    // Get all terms associated with each intermediate
    auto intermediate_terms = get_intermediate_terms(intermediates);
    if (intermediate_terms.empty()) {
        cout << "Intermediates have no terms." << endl;
        return;
    }

    // Find intermediates with the same connectivity
    linkage_set tested_linkages;
    auto merge_map = find_mergeable_intermediates(intermediate_terms, tested_linkages);
    if (merge_map.empty()) {
        cout << "Intermediates cannot be merged." << endl;
        return;
    }

    // print intermediates to merge
    cout << "Intermediates to merge:" << endl;
    for (const auto &[this_intermediate, other_intermediates] : merge_map) {
        cout << "  " << this_intermediate->str() << " -> \n";
        for (const auto &other_intermediate : other_intermediates) {
            cout << "\t" << other_intermediate->str() << " = " << other_intermediate->tot_str() << endl;
            LinkagePtr formatted_other = as_link(other_intermediate->clone());
            formatted_other->replace_lines(*this_intermediate);
            cout << "\t" << formatted_other->str() << " = " << formatted_other->tot_str() << endl;
        }
        cout << endl;
    }
    cout << flush;

    // Process and merge intermediates
    process_and_merge_intermediates(merge_map, intermediate_terms);
}
