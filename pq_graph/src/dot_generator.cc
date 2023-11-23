//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: dot_generator.cc
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
using namespace pdaggerq;

void PQGraph::write_dot(string &filepath) {
    ofstream os(filepath);
    os << "digraph G {" << endl;
    std::string padding = "    ";

    os << padding << "newrank=true;\n";
    os << padding << "rankdir=LR;\n";
    os << padding << "mode=hier;";
    os << padding << "overlap=\"20:prism\";\n";
    os << padding << "ordering=out;\n";
    os << padding << "compound=false;\n";
    os << padding << "sep=1.5;\n";
    os << padding << "K=1.0;\n";
    os << padding << "splines=spline;\n";


    os << padding << "node [fontname=\"Helvetica\"];\n";
    os << padding << "edge [fontsize=20, labelfontsize=20, concentrate=false, compound=false];\n";

    // foreach in reverse order
    padding += "    ";
    for (auto it = equations_.rbegin(); it != equations_.rend(); ++it) {
        Equation &eq = it->second;

        if (eq.terms().empty())
            continue;

        // declare subgraph
        std::string graphname = "cluster_equation_" + eq.assignment_vertex()->base_name_;
        os << padding << "subgraph " << graphname << " {\n";
        os << padding << "    style=rounded;\n";

        // write equation
        eq.write_dot(os, "black", false);


        // add label
        const auto vertex = eq.assignment_vertex();

        os << padding << "label = \"";
        os << vertex->base_name_;
        if (!vertex->lines().empty()) os << "(";
        for (const auto &line : vertex->lines()) {
            os << line.label_;
            if (line != vertex->lines().back()) {
                os << ",";
            }
        }
        if (!vertex->lines().empty()) os << ")";
        os << "\";\n";

        // add formatting
        os << padding << "color = \"black\";\n";
        os << padding << "fontsize = 32;\n";

        os << padding << "}\n";

    }
    os << "}" << endl;
    os.close();

    // reset counters
    for (auto &[name, eq] : equations_){
        eq.write_dot(os, "black", true);
    }

}

ostream &Equation::write_dot(ostream &os, const string &color, bool reset) {
    static size_t term_count = 0;
    if (reset) {
        term_count = 0;
        return os;
    }

    std::string padding = "        ";
    std::string last_graphname;
    for (Term &term : terms_) {
        if (term.rhs().empty())
            continue;

        term.compute_scaling(true);
        std::string graphname = "cluster_term" + to_string(term_count++);
        os << padding << "subgraph " << graphname << " {\n";
        os << padding << "    style=rounded;\n";
        os << padding << "    label=\"";


        // label coefficients
        if (term.coefficient_ != 1.0) {
            if (term.coefficient_ == -1.0)
                os << "-";
            else os << term.coefficient_ << " ";
        }

        // add permutations
        for (const auto &perm : term.term_perms()) {
            os << "P(";
            os << perm.first << "," << perm.second << ")";
        }
        os << " ";

        // label vertices
        for (const auto &vertex : term.term_linkage_->to_vector()) {
            if (vertex->base_name_.empty()) continue;

            os << vertex->base_name_;
            if (!vertex->lines().empty()) os << "(";
            for (const auto &line : vertex->lines()) {
                os << line.label_;
                if (line != vertex->lines().back()) {
                    os << ",";
                }
            }
            if (!vertex->lines().empty()) os << ")";
            if (vertex != term.term_linkage_->to_vector().back()) {
                os << " ";
            }
        }

        os << "\";\n";

        term.term_linkage_->write_dot(os, color, reset);
        os << padding << "}\n";

        if (last_graphname.empty()) {
            last_graphname = graphname;
        }

    }
    return os;
}

ostream &Linkage::write_dot(ostream &os, const std::string& color, bool reset) const {

    static size_t op_id = 0;
    static size_t dummy_count = 0;

    if (reset) {
        op_id = 0;
        dummy_count = 0;
        return os;
    } else { op_id++; dummy_count++; }



//    for (auto begin_it = vbegin(), end_it = vend(); begin_it->right_ != end_it->right_; begin_it = begin_it->vnext()) {
//        VertexPtr l_vert = begin_it->left_;
//        VertexPtr r_vert = begin_it->right_;
//        os << l_vert->base_name() << " " << r_vert->base_name();
//    }
    os << endl;


    // get vertices
    const vector<VertexPtr> &vertices = this->to_vector(true);

    /** write vertices as graph **/

    std::string padding = "                ";

    std::vector<std::string> node_names, null_nodes, int_edge_names, ext_edge_names;


    std::string node_style = "color=\"" + color + "\", fontsize=20, style=bold";
    std::string null_node_style = "style=invis, height=.1,width=.1";

    /// declare nodes

    // loop over vertices and declare nodes
    for (size_t i = 0; i < vertices.size(); i++) {
        // initialize current node
        const VertexPtr &current = vertices[i];
        std::string l_id = std::to_string(i) + to_string(op_id);

        if (vertices[i] == nullptr || vertices[i]->base_name().empty())
            continue;

        std::string current_node = current->base_name() + "_" + l_id;
        std::string node_signature = padding + current_node + " [label=\"" + current->base_name() + "\", ";

        if (current->base_name().empty())
            node_signature += null_node_style + "];\n";
        else
            node_signature += node_style + "];\n";
        node_names.push_back(node_signature);
    }

    std::string ext_edge_style = "color=\"" + color + "\", style=bold, len=0.01, arrowsize=1.25";

    /// plot external lines

    for (const auto &line: this->lines_) {
        for (size_t i = 0; i < vertices.size(); i++) {

            // initialize current node
            const VertexPtr &current = vertices[i];
            std::string l_id = std::to_string(i) + to_string(op_id);

            if (vertices[i] == nullptr || vertices[i]->base_name().empty())
                continue;

            std::string current_node = current->base_name() + "_" + l_id;

            /// link all vertices to external lines
            // loop over left external lines
            size_t ext_count = 0;
            const auto & current_lines = current->lines();
            size_t current_len = current_lines.size();

            // initialize dummy node name
            std::string null = "null" + std::to_string(dummy_count) + line.label_ + std::to_string(ext_count++);
            null_nodes.push_back(null + " [label=\"\", " + null_node_style + "];\n");

            // make edge label
            std::string edge_label = line.label_;

            // check if line is in bra
            auto curr_it = std::find(current_lines.begin(), current_lines.end(), line);
            if (curr_it == current_lines.end()) continue; // line not found

            size_t curr_dist = std::distance(current_lines.begin(), curr_it);

            bool curr_is_bra = curr_dist < current_len - current_len / 2;


            // determine direction of edge
            std::string direction = "forward";

            if (curr_is_bra) {
                if (!line.o_) {
                    direction = "back";
                }
                std::swap(current_node, null);
            } else {
                if (line.o_) {
                    direction = "back";
                }
            }

            std::string connnection = current_node + ":_ -> " + null;

            // write edge
            ext_edge_names.push_back(padding + connnection + " [label=\"" + edge_label + "\"," + ext_edge_style + ", dir=" + direction + "];\n");
        }
    }

    /// plot internal lines

    std::string int_edge_style = "color=\"" + color + "\"";

    for (size_t i = 0; i < vertices.size(); i++) {
        // initialize current node
        const VertexPtr &current = vertices[i];
        std::string l_id = std::to_string(i) + to_string(op_id);

        if (vertices[i] == nullptr || vertices[i]->base_name().empty())
            continue;

        for (size_t j = i+1; j < vertices.size(); j++) {
            //TODO: incorporate scalar vertices

            if (vertices[j] == nullptr || vertices[j]->base_name().empty())
                continue;


            const VertexPtr &next = vertices[j];

            // make contraction of current and next
            LinkagePtr link = as_link(current * next);

            // initialize next node
            std::string r_id = std::to_string(j) + to_string(op_id);
            std::string next_node = next->base_name() + "_" + r_id;
            std::string current_node = current->base_name() + "_" + l_id;

            // Add vertices as nodes. connect the current and next vertices with edges from the connections map
            // (-1 indicates no match and should use a dummy node)

            const auto & current_lines = current->lines();
            const auto & next_lines = next->lines();
            size_t current_len = current_lines.size();
            size_t next_len = next_lines.size();
            // loop over internal lines
            for (const auto &line: link->int_lines_) {

                // initialize edge label
                std::string edge_label = line.label_;

                // determine direction of edge
                // check if line is in bra
                auto curr_it = std::find(current_lines.begin(), current_lines.end(), line);
                size_t curr_dist = std::distance(current_lines.begin(), curr_it);

                bool curr_is_bra = curr_dist < current_len - current_len / 2;

                // determine direction of edge
                std::string direction = "forward";

                if (curr_is_bra) {
                    if (line.o_) {
                        direction = "back";
                    }
                    std::swap(current_node, next_node);
                } else {
                    if (!line.o_) {
                        direction = "back";
                    }
                }

                std::string connnection = current_node + " -> " + next_node;

                // write edge
//                os << padding << connnection << " [label=\"" << edge_label << "\"," + int_edge_style + ", dir=" + direction + "];\n";
                int_edge_names.push_back(padding + connnection + " [label=\"" + edge_label + "\"," + int_edge_style + ", dir=" + direction + "];\n");
            }
        }
    }

    /// write file

    // print nodes name
    for (const auto &node_name : node_names)
        os << node_name;

    // print external edges
    for (const auto &edge_name : ext_edge_names)
        os << edge_name;

    // make dummy nodes invisible
    for (const auto &dummy_node : null_nodes)
        os << padding << dummy_node;

    // print internal edges
    for (const auto &edge_name : int_edge_names)
        os << edge_name;


    return os;
}