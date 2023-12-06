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

//    os << padding << "newrank=true;\n";
    os << padding << "rankdir=LR;\n";
//    os << padding << "mode=hier;";
//    os << padding << "overlap=\"20:prism\";\n";
//    os << padding << "ordering=out;\n";
//    os << padding << "compound=false;\n";
//    os << padding << "sep=1.25;\n";
//    os << padding << "K=1.0;\n";
//    os << padding << "splines=spline;\n";


//    os << padding << "node [fontname=\"Helvetica\"];\n";
//    os << padding << "edge [concentrate=false];\n";

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
        os << padding << "    clusterrank=local;\n";

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
        os << padding << "    clusterrank=local;\n";
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
        for (const auto &vertex : term.term_linkage_->get_vertices()) {
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
            if (vertex != term.term_linkage_->get_vertices().back()) {
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

    static size_t term_id = 0;
    static size_t dummy_count = 0;

    if (reset) {
        term_id = 0;
        dummy_count = 0;
        return os;
    } else { term_id++; dummy_count++; }


    // get vertices
    vector<ConstVertexPtr> vertices = this->get_vertices(true, true);

    // sort vertices
    std::sort(vertices.begin(), vertices.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->base_name() < b->base_name();
    });

    bool track_temps = true; // TODO: make this a parameter
    vector<ConstVertexPtr> temps = this->get_vertices(true, false);
    vector<ConstVertexPtr> temp_verts;
    temp_verts.reserve(vertices.size());

    // now fully expand vertices in temps

    if (!this->is_temp()) {
        for (auto &temp: temps) {
            if (!temp->is_temp())
                continue;

            for (auto &temp_vert: as_link(temp)->get_vertices(true, true)) {
                temp_verts.push_back(temp_vert);
            }

        }
    }

    // sort temp vertices
    std::sort(temp_verts.begin(), temp_verts.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->base_name() < b->base_name();
    });

    /** write vertices as graph **/

    auto edge_direction = [](const Line &line, bool curr_is_bra, std::string &direction, std::string &left_node,
                             std::string &right_node) {
        std::swap(left_node, right_node);
        if (curr_is_bra) {
            if (line.o_) {
                direction = "forward";
                std::swap(left_node, right_node);
            } else {
                direction = "back";
            }
        } else {
            if (line.o_) {
                direction = "forward";
            } else {
                std::swap(left_node, right_node);
                direction = "back";
            }
        }
    };

    std::string padding = "                ";

    std::vector<std::string> node_names, temp_names, null_nodes, int_edge_names, ext_edge_names;


    std::string node_style = "color=\"" + color + "\", fontsize=20, style=bold";
    std::string null_node_style = "style=invis, shape=none, height=0.1,width=0.1";
    std::string ext_edge_style = "color=\"" + color + "\", style=bold, arrowsize=1.25";
    std::string int_edge_style = "color=\"" + color + "\", concentrate=false";

    /// declare nodes and plot internal lines

    size_t group_count = 0;
    bool began_temp = false;
    size_t temp_count = 0;
    for (size_t i = 0; i < vertices.size(); i++) {
        // initialize current node
        const ConstVertexPtr &current = vertices[i];

        // check if vertex is in temp vertices
        bool in_temp = false;
        for (auto &temp : temp_verts) {
            if (temp.get() == current.get()) {
                in_temp = true;
                break;
            }
        }

        if (in_temp && !began_temp && track_temps) {
            // add subgraph
            node_names.push_back("subgraph cluster_tmp" + to_string(temp_count++) + "_" + to_string(term_id) + "{\n");
            began_temp = true;
        } else if (!in_temp && began_temp && track_temps) {
            node_names.emplace_back("label=\"\";\n");
            node_names.emplace_back("style=dashed;\n");
            node_names.emplace_back("rank=min;\n");
            node_names.emplace_back("}\n");
            began_temp = false;
        }

        std::string l_id = std::to_string(i) + to_string(term_id);
        std::string current_node = current->base_name() + "_" + l_id;

        if (vertices[i] == nullptr || vertices[i]->base_name().empty())
            continue;


        std::string node_signature = current_node + " [label=\"" + current->base_name() + "\", " + node_style;

        // add to group so that consecutive vertices are in the same group
        if (i % 2 == 0)
            group_count++;

        node_signature += ", group=" + std::to_string(group_count) + "];\n";
        node_names.push_back(node_signature);

        for (size_t j = i+1; j < vertices.size(); j++) {
            //TODO: incorporate scalar vertices

            if (vertices[j] == nullptr || vertices[j]->base_name().empty())
                continue;


            const ConstVertexPtr &next = vertices[j];

            // make contraction of current and next
            ConstLinkagePtr link = as_link(current * next);

            // initialize next node
            std::string r_id = std::to_string(j) + to_string(term_id);
            std::string next_node = next->base_name() + "_" + r_id;

            // Add vertices as nodes. connect the current and next vertices with edges from the connections map
            // (-1 indicates no match and should use a dummy node)

            const auto & current_lines = current->lines();
            const auto & next_lines = next->lines();
            size_t current_len = current_lines.size();
            size_t next_len = next_lines.size();

            // loop over internal lines
            for (const auto &line: link->int_lines()) {

                // initialize edge label
                std::string edge_label = line.label_;

                // determine direction of edge
                // check if line is in bra
                auto curr_it = std::find(current_lines.begin(), current_lines.end(), line);
                size_t curr_dist = std::distance(current_lines.begin(), curr_it);

                bool curr_is_bra = curr_dist < current_len - current_len / 2;

                // determine direction of edge
                std::string direction;
                std::string left_node = current_node;
                std::string right_node = next_node;

                edge_direction(line, curr_is_bra, direction, left_node, right_node);

                std::string connnection = left_node + " -> " + right_node;

                // write edge
//                os << padding << connnection << " [label=\"" << edge_label << "\"," + int_edge_style + ", dir=" + direction + "];\n";
                int_edge_names.push_back(connnection + " [label=\"" + edge_label + "\"," + int_edge_style + ", dir=" + direction + "];\n");
            }
        }
    }

    if (began_temp && track_temps) {
        node_names.emplace_back("label=\"\";\n");
        node_names.emplace_back("style=dashed;\n");
        node_names.emplace_back("}\n");
        began_temp = false;
    }

    /// plot external lines

    temp_count=0;
    began_temp = false;
    for (const auto &line: this->lines_) {
        // initialize dummy node name
        std::string null_node = "null_node" + std::to_string(dummy_count) + (line.o_ ? "o": "v") + line.label_;

        if (line.sig_)
            continue;

        bool added_null = false;
        group_count = 0;
        for (size_t i = 0; i < vertices.size(); i++) {



            // initialize current node
            const ConstVertexPtr &current = vertices[i];
            std::string l_id = std::to_string(i) + to_string(term_id);

            if (vertices[i] == nullptr || vertices[i]->base_name().empty())
                continue;

            std::string current_node = current->base_name() + "_" + l_id;

            /// link all vertices to external lines
            // loop over left external lines
            size_t ext_count = 0;
            const auto & current_lines = current->lines();
            size_t current_len = current_lines.size();

            // make edge label
            std::string edge_label = line.label_;

            // add to group so that null and vertex are in the same group
            if (i % 2 == 0)
                group_count++;

            // check if line is in bra
            auto curr_it = std::find(current_lines.begin(), current_lines.end(), line);
            if (curr_it == current_lines.end()) continue; // line not found

            size_t curr_dist = std::distance(current_lines.begin(), curr_it);
            bool curr_is_bra = curr_dist < current_len - current_len / 2;

            if (!added_null) {

//                // check if vertex is in temp vertices
//                bool in_temp = false;
//                for (auto &temp: temp_verts) {
//                    if (temp.get() == current.get()) {
//                        in_temp = true;
//                        break;
//                    }
//                }
//
//                if (in_temp && !began_temp && track_temps) {
//                    // add subgraph
//                    null_nodes.push_back(
//                            "subgraph cluster_tmp" + to_string(temp_count++) + "_" + to_string(term_id) + "{\n");
//                    began_temp = true;
//                } else if (!in_temp && began_temp && track_temps) {
//                    null_nodes.emplace_back("label=\"\";\n");
//                    null_nodes.emplace_back("style=dashed;\n");
//                    null_nodes.emplace_back("rank=min;\n");
//                    null_nodes.emplace_back("}\n");
//                    began_temp = false;
//                }

                null_nodes.push_back(null_node + " [label=\"\", " + null_node_style + ", group=" + to_string(group_count) + "];\n");
                added_null = true;
            }


            // determine direction of edge
            std::string direction;
            std::string left_node = current_node;
            std::string right_node = null_node;

            edge_direction(line, curr_is_bra, direction, left_node, right_node);

            std::string connnection = left_node + " -> " + right_node;

            // write edge
            ext_edge_names.push_back(connnection + " [label=\"" + edge_label + "\"," + ext_edge_style + ", dir=" + direction + "];\n");
        }
    }

    if (began_temp && track_temps) {
        null_nodes.emplace_back("label=\"\";\n");
        null_nodes.emplace_back("style=dashed;\n");
        null_nodes.emplace_back("rank=min;\n");
        null_nodes.emplace_back("}\n");
        began_temp = false;
    }

    /// write file

    // print nodes name
    for (const auto &node_name : node_names)
        os << padding << node_name;

    // print internal edges
    for (const auto &edge_name : int_edge_names)
        os << padding << edge_name;

    // print external edges
    for (const auto &edge_name : ext_edge_names)
        os << padding << edge_name;

    // make dummy nodes invisible
    for (const auto &dummy_node : null_nodes)
        os << padding << dummy_node;

    return os;
}