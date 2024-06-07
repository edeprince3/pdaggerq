#include "../include/pq_graph.h"
using namespace pdaggerq;

string join(const vector<Line> &lines, const string &delimiter) {
    std::stringstream result;
    for (size_t i = 0; i < lines.size(); ++i) {
        result << lines[i].label_;
        if (i < lines.size() - 1) result << delimiter;
    }
    return result.str();
}

void PQGraph::write_dot(string &filepath) {
    ofstream os(filepath);
    os << "digraph G {\n";
    string padding = "    ";

    // Graph attributes
    os << padding << "rankdir=LR;\n";
    os << padding << "mode=hier;\n";
    os << padding << "compound=false;\n";
    os << padding << "splines=spline;\n";
    os << padding << "node [fontname=\"Helvetica\"];\n";
    os << padding << "edge [concentrate=false];\n";

    padding += "    ";
    for (auto &it : equations_) {
        Equation &eq = it.second;
        if (eq.terms().empty()) continue;

        string graphname = "cluster_equation_" + it.first;
        os << padding << "subgraph " << graphname << " {\n";
        os << padding << "    style=rounded;\n";
        os << padding << "    clusterrank=local;\n";
        eq.write_dot(os, "black", false);

        const auto vertex = eq.assignment_vertex();
        os << padding << "label = \"" << vertex->base_name_;
        if (!vertex->lines().empty()) os << "(" << join(vertex->lines(), ",") << ")";
        os << "\";\n";
        os << padding << "color = \"black\";\n";
        os << padding << "fontsize = 32;\n";
        os << padding << "}\n";
    }
    os << "}\n";

    for (auto &[name, eq] : equations_) {
        eq.write_dot(os, "black", true);
    }
}

string format_label(const Term &term) {
    std::stringstream label;
    int w = minimum_precision(term.coefficient_);
    if (term.coefficient_ != 1.0)
        label << (term.coefficient_ == -1.0 ? "-" : to_string_with_precision(term.coefficient_, w) + " ");
    for (const auto &perm : term.term_perms())
        label << "P(" << perm.first << "," << perm.second << ") ";

    vector<ConstVertexPtr> vertices = term.term_linkage(true)->vertices(true);
    for (const auto &vertex : vertices) {
        if (vertex && !vertex->empty()) {
            label << vertex->base_name_;
            if (!vertex->lines().empty()) label << "(" << join(vertex->lines(), ",") << ")";
            label << " ";
        }
    }
    return label.str();
}

ostream &Equation::write_dot(ostream &os, const string &color, bool reset) {
    static size_t term_count = 0;
    if (reset) {
        term_count = 0;
        return os;
    }

    string padding = "        ";
    for (Term &term : terms_) {
        if (term.rhs().empty()) continue;

        term.compute_scaling(true);
        string graphname = "cluster_term" + to_string(term_count++);
        os << padding << "subgraph " << graphname << " {\n";
        os << padding << "    style=rounded;\n";
        os << padding << "    clusterrank=local;\n";
        os << padding << "    label=\"" << format_label(term) << "\";\n";

        term.term_linkage(true)->write_dot(os, color, reset);
        os << padding << "}\n";
    }
    return os;
}

string determine_direction(bool o, bool curr_is_bra) {
    if (curr_is_bra) {
        return o ? "forward" : "back";
    } else {
        return o ? "back" : "forward";
    }
}

void append_null_nodes(const ConstLinkagePtr &link, const vector<ConstVertexPtr> &vertices, size_t &dummy_count, size_t term_id, vector<string> &null_nodes, vector<string> &ext_edge_names, size_t &group_count, const string &ext_edge_style, const string &null_node_style) {
    for (const auto &line : link->lines_) {
        if (line.sig_) continue;

        string null_node = "null_node" + to_string(dummy_count++) + (line.o_ ? "o" : "v") + line.label_;
        bool added_null = false;
        for (size_t i = 0; i < vertices.size(); ++i) {
            const auto &current = vertices[i];
            if (current->base_name().empty()) continue;

            const auto &current_lines = current->lines();
            auto curr_it = find(current_lines.begin(), current_lines.end(), line);
            if (curr_it == current_lines.end()) continue;

            if (!added_null) {
                null_nodes.push_back(null_node + " [label=\"\", " + null_node_style + ", group=" + to_string(group_count++) + "];\n");
                added_null = true;
            }

            string direction = determine_direction(line.o_, curr_it < current_lines.end() - current_lines.size() / 2);
            string connnection = (curr_it < current_lines.end() - current_lines.size() / 2 ? current->base_name() + "_" + to_string(i) + to_string(term_id) : null_node) + " -> " + (curr_it < current_lines.end() - current_lines.size() / 2 ? null_node : current->base_name() + "_" + to_string(i) + to_string(term_id));
            ext_edge_names.push_back(connnection + " [label=\"" + line.label_ + "\"," + ext_edge_style + ", dir=" + direction + "];\n");
        }
    }
}

void append_edge_links(const ConstLinkagePtr &link, const ConstVertexPtr &current, const ConstVertexPtr &next, const string &current_node, const string &next_node, vector<string> &int_edge_names, const string &int_edge_style) {
    const auto &current_lines = current->lines();
    size_t current_len = current_lines.size();
    for (const auto &line : link->int_lines()) {
        string edge_label = line.label_;
        auto curr_it = find(current_lines.begin(), current_lines.end(), line);
        size_t curr_dist = distance(current_lines.begin(), curr_it);
        bool curr_is_bra = curr_dist < current_len - current_len / 2;
        string direction = determine_direction(line.o_, curr_is_bra);
        string connnection = (curr_is_bra ? next_node : current_node) + " -> " + (curr_is_bra ? current_node : next_node);
        int_edge_names.push_back(connnection + " [label=\"" + edge_label + "\"," + int_edge_style + ", dir=" + direction + "];\n");
    }
}

void append_edges(const vector<ConstVertexPtr> &vertices, const ConstVertexPtr &current, size_t i, vector<string> &int_edge_names, const string &l_id, size_t term_id, const string &int_edge_style) {
    string current_node = current->base_name() + "_" + l_id;
    for (size_t j = i + 1; j < vertices.size(); ++j) {
        const auto &next = vertices[j];
        if (next->base_name().empty()) continue;

        string r_id = to_string(j) + to_string(term_id);
        string next_node = next->base_name() + "_" + r_id;

        const auto &link = as_link(current * next);
        append_edge_links(link, current, next, current_node, next_node, int_edge_names, int_edge_style);
    }
}

vector<ConstVertexPtr> sorted_vertices(const ConstLinkagePtr& link) {
    vector<ConstVertexPtr> vertices = link->vertices();
    sort(vertices.begin(), vertices.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->base_name() < b->base_name();
    });
    return vertices;
}

vector<ConstVertexPtr> sorted_temp_vertices(const ConstLinkagePtr& link) {
    vector<ConstVertexPtr> temp_verts;
    for (const auto &temp : link->link_vector(false)) {
        if (temp->is_temp()) {
            for (const auto &temp_vert : as_link(temp)->vertices()) {
                temp_verts.push_back(temp_vert);
            }
        }
    }
    sort(temp_verts.begin(), temp_verts.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->base_name() < b->base_name();
    });
    return temp_verts;
}

ostream &Linkage::write_dot(ostream &os, const string &color, bool reset) const {
    static size_t term_id = 0, dummy_count = 0;
    if (reset) { term_id = dummy_count = 0; return os; }

    ++term_id; ++dummy_count;
    string padding = "                ";
    vector<string> node_names, int_edge_names, ext_edge_names, null_nodes;
    string node_style = "color=\"" + color + "\", fontsize=20, style=bold";
    string null_node_style = "style=invis, shape=none, height=0.01,width=0.01";
    string ext_edge_style = "color=\"" + color + "\", style=bold, arrowsize=1.25";
    string int_edge_style = "color=\"" + color + "\", concentrate=false";

    size_t group_count = 0, temp_count = 0;
    bool began_temp = false;
    vector<ConstVertexPtr> vertices = sorted_vertices(as_link(shared_from_this()));
    vector<ConstVertexPtr> temp_verts = sorted_temp_vertices(as_link(shared_from_this()));

    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto &current = vertices[i];
        bool in_temp = find(temp_verts.begin(), temp_verts.end(), current) != temp_verts.end();

        if (in_temp && !began_temp) {
            node_names.push_back("subgraph cluster_tmp" + to_string(temp_count++) + "_" + to_string(term_id) + "{\n");
            began_temp = true;
        } else if (!in_temp && began_temp) {
            node_names.push_back("label=\"\";\nstyle=dashed;\nrank=min;\n}\n");
            began_temp = false;
        }

        string l_id = to_string(i) + to_string(term_id);
        string current_node = current->base_name() + "_" + l_id;

        if (!current->base_name().empty()) {
            string node_signature = current_node + " [label=\"" + current->base_name() + "\", " + node_style + ", group=" + to_string(group_count++) + "];\n";
            node_names.push_back(node_signature);
            append_edges(vertices, current, i, int_edge_names, l_id, term_id, int_edge_style);
        }
    }

    if (began_temp) node_names.push_back("label=\"\";\nstyle=dashed;\nrank=min;\n}\n");

    append_null_nodes(as_link(shared_from_this()), vertices, dummy_count, term_id, null_nodes, ext_edge_names, group_count, ext_edge_style, null_node_style);

    for (const auto &name : node_names) os << padding << name;
    for (const auto &edge : int_edge_names) os << padding << edge;
    for (const auto &edge : ext_edge_names) os << padding << edge;
    for (const auto &node : null_nodes) os << padding << node;

    return os;
}