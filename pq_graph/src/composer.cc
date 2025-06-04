#include "../include/composer.h"
#include <sstream>
#include <iomanip>

string TAMM_Composer::compose(const VertexPtr &vertex) {

    // this is a linked vertex
    if (vertex->is_linked())
        return compose(as_link(vertex));

    string name = vertex->name();
    name += compose_lines(vertex); // compose lines
    return name;
}

string TAMM_Composer::compose_lines(const VertexPtr &vertex, bool sort) {
    if (vertex->size() == 0) return ""; // if rank is 0, return empty string
    if (vertex->size() == 1) {
        // do not print sigma lines if use_trial_index is false for otherwise scalar vertices
        if (vertex->lines_[0].sig_ && !vertex->use_trial_index)
            return "";
    }

    // make a copy of lines that is sorted if sort is true
    line_vector lines;
    if (!sort) lines = vertex->lines();
    else {
        lines.reserve(vertex->lines().size());
        for (const Line &line : vertex->lines())
            lines.insert(
                    std::lower_bound(lines.begin(), lines.end(), line, line_compare()), line);
    }

    // loop over lines
    string line_str = "(";
    for (const Line &line : lines) {
        if (!vertex->use_trial_index && line.sig_) continue;
        line_str += line.label_;
        line_str += ",";
    }
    line_str.pop_back(); // remove last comma
    line_str += ")";
    return line_str;
}

string TAMM_Composer::compose(const LinkagePtr &linkage) {
    if (linkage->empty()) return {};

    // this is a named intermediate (a temp), so we can return a unique name for it
    if (linkage->is_temp()) {
        string name;
        // first determine the type of temp
        if (linkage->is_scalar())
            name = "scalars_";
        else if (linkage->is_reused())
            name = "reused_";
        else name = "tmps_";


        // use id to create a generic name that is indexed by the id with a map
        name += "[\"";
        string dimstring = linkage->dimstring();
        if (linkage->id() >= 0) {
            // format the id as a string (%04d)
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(4) << linkage->id();
            name += ss.str();
        }

        // if there is a dimension string, add it to the name
        if (!dimstring.empty())
            name += "_" + dimstring;

        // close the temp name with a closing bracket
        name += "\"]";

        // if lines need to be included, add them to the name
        name += compose_lines(linkage);

        return name;
    }
    // else this is a normal linkage and can be composed of its left and right vertices

    const VertexPtr& left = linkage->left();
    const VertexPtr& right = linkage->right();
    if (left->empty()) return compose(right);
    if (right->empty()) return compose(left);

    string output;
    if (linkage->is_addition()) {
        return compose(left) + " + " + compose(right);
    }

    vertex_vector scalars;
    vertex_vector tensors;
    vertex_vector link_vector = linkage->link_vector();
    for (const auto &op: link_vector) {
        if (op->empty()) continue;
        if (op->is_scalar()) {
            // pure scalars should be added first
            if (!op->is_linked()) scalars.insert(scalars.begin(), op);
            else scalars.push_back(op);
        }
        else {
            tensors.push_back(op);
        }
    }

    if (scalars.empty() && tensors.empty()) return "1.0";

    // first add scalars
    for (const auto &scalar: scalars) {
        string scalar_str = compose(scalar);
        if (scalar->is_addition() && !scalar->is_temp())
            scalar_str = "(" + scalar_str + ")";
        output += scalar_str + " * ";
    }

    if (tensors.empty()) {
        output.pop_back(); output.pop_back(); output.pop_back();
        return output;
    }

    return output;
}

string TAMM_Composer::compose(const Term &term) {

    if (!term.print_override_.empty())
        // return print override if it exists for custom printing
        return term.print_override_;

    string output;

    // format for permutations if any
    bool has_permutations = !term.term_perms().empty() && term.perm_type() != 0;
    if (has_permutations) {

        // make intermediate vertex for the permutation
        MutableVertexPtr perm_vertex;

        bool perm_as_rhs = term.rhs().size() == 1; // if there is only one vertex, no need to create intermediate vertex

        if (perm_as_rhs) {
            // if this is a linkage, but not a temp, also make a temporary vertex (doesn't print as a single vertex)
            if (term.rhs()[0]->is_linked() && !term.rhs()[0]->is_temp()) {
                perm_as_rhs = false;
            }
        }

        if (perm_as_rhs) perm_vertex = term.rhs()[0]->clone(); // no need to create intermediate vertex if there is only one
        else { // else, create the intermediate vertex and its assignment term
            perm_vertex = term.lhs()->clone();
            string perm_name;
            perm_vertex->vertex_type_ = 'p'; // sets printing for permutation vertex
            perm_vertex->sort(); // sort permutation vertex
            perm_vertex->update_name("tmps_"); // set name of permutation vertex

            // initialize initial permutation term
            Term perm_term = term; // copy term
            perm_term.lhs() = perm_vertex; // set lhs to permutation vertex
            perm_term.reset_perm();
            perm_term.is_assignment_ = true; // set term as assignment
            perm_term.coefficient_ = fabs(term.coefficient_); // set coefficient to absolute value of coefficient

            // add string to output
            output += compose(perm_term);
            output += "\n";

        } // if only one vertex, use that vertex directly

        // initialize term to permute
        Term perm_term = term; // copy term
        perm_term.rhs() = {perm_vertex};
        perm_term.compute_scaling(true);  // recomputes scaling

        // remove comments from term
        perm_term.comments().clear();

        // if more than one vertex, set coefficient to 1 or -1
        if (!perm_as_rhs)
            perm_term.coefficient_ = term.coefficient_ > 0 ? 1 : -1;

        // get permuted terms
        std::vector<Term> perm_terms = perm_term.expand_perms();

        // add permuted terms to output
        for (auto &permuted_term: perm_terms) {
            output += compose(permuted_term);
            output += '\n';
        }

        // if an intermediate vertex was created, delete it
        if (!perm_as_rhs && Term::deallocate_) {
            output += ".deallocate(" + perm_vertex->name() + ")";
            output += "\n";
        }

        if (!perm_terms.empty())
            output.pop_back(); // remove last newline character

        return output;
    }

    // if no permutations, continue with normal term printing

    // expand additions into separate terms
    LinkagePtr term_link = term.term_linkage();
    if (term_link->is_addition() && !term_link->is_temp()) {
        Term left_term = term, right_term = term;
        VertexPtr left_vertex = term_link->left();
        VertexPtr right_vertex = term_link->right();

        left_term.expand_rhs(left_vertex); // expand left term
        right_term.expand_rhs(right_vertex); // expand right term

        // merge constants in right term and compute scaling. right term is not an assignment
        right_term.is_assignment_ = false;
        right_term.compute_scaling(true);

        return compose(left_term) + '\n' + compose(right_term);
    }

    // we need binarization for c++ output. if more than 2 vertices, binarize into two terms
    bool binarize = term.rhs().size() > 2;
    if (binarize && Term::binarize_) {

        vertex_vector binarize_vertices = term.rhs();

        // extract last vertex
        VertexPtr last_vertex = binarize_vertices.back();
        binarize_vertices.pop_back();

        // make intermediate vertex for the binarization
        auto binarize_link = Linkage::link(binarize_vertices);
        MutableVertexPtr binarize_vertex = make_shared<Vertex>("tmps_", binarize_link->lines());
        binarize_vertex->vertex_type_ = 'b';    // sets printing for binarization vertex
        binarize_vertex->sort();                // sort labels of binarization vertex
        binarize_vertex->update_name();         // update name of binarization vertex

        if (term.lhs()->name() == binarize_vertex->name()) {
            binarize_vertex->vertex_type_ = 'd';
            binarize_vertex->update_name();
        }

        // initialize initial binarization term
        Term binarize_term;
        binarize_term.lhs() = binarize_vertex;     // set lhs to binarization vertex
        binarize_term.rhs() = binarize_vertices;   // set rhs to binarization vertices
        binarize_term.expand_rhs(binarize_link);          // expand rhs
        binarize_term.is_assignment_ = true;      // set term as assignment

        // add string to output
        output += compose(binarize_term) + "\n";

        // initialize term to binarize
        Term binarize_last_term = term;          // copy term
        binarize_last_term.rhs() = {binarize_vertex, last_vertex};
        binarize_last_term.expand_rhs(binarize_last_term.term_linkage(true));          // expand rhs

        output += compose(binarize_last_term);
        return output;
    }

    // get lhs vertex string
    output = compose(term.lhs());

    // get sign of coefficient
    bool is_negative = term.coefficient_ < 0;
    if (term.is_assignment_) output += "  = ";
    else if (is_negative) output += " -= ";
    else output += " += ";

    // get absolute value of coefficient
    double abs_coeff = fabs(term.coefficient_);

    // if the coefficient is not 1, add it to the string
    bool is_empty = term.rhs().empty() || term_link->empty();

    bool added_coeff = false;
    bool negative_assignment = (term.is_assignment_ && is_negative);
    bool needs_coeff = fabs(abs_coeff - 1) >= 1e-8 || is_empty || negative_assignment;

    if (needs_coeff) {
        // add coefficient to string
        added_coeff = true;
        if (negative_assignment)
            output += "-";

        int precision = minimum_precision(abs_coeff);
        output += to_string_with_precision(abs_coeff, precision);

        // add multiplication sign if there are rhs vertices
        if (!is_empty)
            output += " * ";
    }

    output += compose(term_link);

    size_t pos = 0;
    while (pos != string::npos) {
        pos = output.find("* 1.00 *", pos);
        if (pos != string::npos) {
            output = output.replace(pos, 8, "*");
            pos += 1;
        }
    }

//        return output;
    return "( " + output + " )";
}