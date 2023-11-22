#include <algorithm>
#include <map>
#include <cmath>
#include <iostream>
#include "../include/term.h"


namespace pdaggerq {

    void Term::set_perm(const string & perm_string) {// extract permutation indices
        VertexPtr perm_op = make_shared<Vertex>(perm_string); // create permutation vertex

        // check if permutation is a P, PP2, PP3, or PP6
        size_t perm_rank = perm_op->rank(); // get rank of permutation (number of indices in permutation)

        if (perm_rank == 2) { // single index permutation
            perm_type_ = 1;
            term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_); // add permutation indices to vector
        } else if (perm_rank == 4){ // PP2 permutation
            perm_type_ = 2;
            term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_);
            term_perms_.emplace_back((*perm_op)[2].label_, (*perm_op)[3].label_);
        } else if (perm_rank == 6){
            // check if PP3 or PP6 (same ranks)
            if (perm_string[2] == '3'){ // PP3
                perm_type_ = 3;
                term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_);
                term_perms_.emplace_back((*perm_op)[2].label_, (*perm_op)[3].label_);
                term_perms_.emplace_back((*perm_op)[4].label_, (*perm_op)[5].label_);
            } else if (perm_string[2] == '6'){ // PP6
                perm_type_ = 6;
                term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_);
                term_perms_.emplace_back((*perm_op)[2].label_, (*perm_op)[3].label_);
                term_perms_.emplace_back((*perm_op)[4].label_, (*perm_op)[5].label_);
            } else throw logic_error("Invalid permutation vertex: " + perm_string);
        } else throw logic_error("Invalid permutation vertex: " + perm_string);

        perm_pairs_mem_ = term_perms_; // set memory permutation pairs
        perm_type_mem_ = perm_type_; // set memory permutation type
    }

    string &
    Term::make_perm_string(string &output, const VertexPtr &perm_vertex, double abs_coeff, int perm_sign,
                           bool has_one) const {// add permutation to output

        // copy term
        Term perm_term = *this;

        // set property of this term
        perm_term.rhs_.clear();
        VertexPtr perm_vertex_copy = copy_vert(perm_vertex);

        perm_term.rhs_.push_back(perm_vertex_copy);
        perm_term.set_perm({}, 0);
        perm_term.set_perm_mem({}, 0);
        perm_term.is_assignment_ = false;

        // set sign and coefficient
        if (has_one) { // if only one vertex, add coefficient
            if (perm_sign >= 0) perm_term.coefficient_ = abs_coeff;
            else perm_term.coefficient_ = -abs_coeff;
        } else // if more than one vertex, do not add coefficient but include sign
            perm_term.coefficient_ = perm_sign;

        output += perm_term.str();
        output += "\n";

        return output;
    }

    string &
    Term::p1_permute_string(string &output, VertexPtr &perm_vertex, double abs_coeff, int perm_sign, bool has_one) const {

        make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        /// add all permutations of permutation vertex

        // create copy of original lines
        vector<Line> orig_lines = perm_vertex->lines();
        vector<Line> perm_lines = orig_lines;

        size_t num_swaps = 0; // number of swaps applied
        size_t swap_size = 1; // number of swaps to be applied at once
        const auto& perm_lines_end = perm_lines.end();
        while (swap_size <= term_perms_.size() ) {
            // update sign for each count_ of permutations
            perm_sign *= -1;
            for (const auto & perm_pair : term_perms_){

                // get lines to be permuted
                string perm_line1 = perm_pair.first;
                string perm_line2 = perm_pair.second;

                // swap lines
                for (Line & line : perm_lines) {
                    if (line.label_ == perm_line1) line.label_ = perm_line2;
                    else if (line.label_ == perm_line2) line.label_ = perm_line1;
                }

                if ( ++num_swaps % swap_size == 0){ // if permutation has not been accumulated yet
                    // update permutation vertex
                    perm_vertex->update_lines(perm_lines, false);

                    // add permutation to output
                    output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

                    // reset permutation lines
                    perm_lines = orig_lines;

                    size_t total_swaps = (1 << swap_size) - 1; // total number of swaps to be made for this swap n_ops
                    if (num_swaps >= total_swaps){ // if all permutations have been applied for this swap_size
                        swap_size += 1; // increment swap n_ops
                    }
                }
            }
        }

        // remove last newline
        output.pop_back();
        return output;
    }

    string &
    Term::p2_permute_string(string &output, VertexPtr &perm_vertex, double abs_coeff, int perm_sign, bool has_one) const {

        /// add all permutations of permutation vertex

        make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // create copy of original lines
        vector<Line> orig_lines = perm_vertex->lines();
        vector<Line> perm_lines = orig_lines;

        // get lines to be permuted
        pair<string, string> perm_pair1 = term_perms_[0];
        pair<string, string> perm_pair2 = term_perms_[1];

        string perm_line1_1 = perm_pair1.first;
        string perm_line1_2 = perm_pair1.second;
        string perm_line2_1 = perm_pair2.first;
        string perm_line2_2 = perm_pair2.second;

        // swap lines
        for (Line & line : perm_lines) {
            if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
            else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
            else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
            else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
        }

        perm_vertex->update_lines(perm_lines, false);
        output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // remove last newline
        output.pop_back();
        return output;
    }

    string &
    Term::p3_permute_string(string &output, VertexPtr &perm_vertex, double abs_coeff, int perm_sign, bool has_one) const {

        /// add all permutations of permutation vertex

        make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // create copy of original lines
        vector<Line> orig_lines = perm_vertex->lines();
        vector<Line> perm_lines = orig_lines;

        // get lines to be permuted
        pair<string, string> perm_pair1 = term_perms_[0];
        pair<string, string> perm_pair2 = term_perms_[1];
        pair<string, string> perm_pair3 = term_perms_[2];

        string perm_line1_1 = perm_pair1.first;
        string perm_line1_2 = perm_pair1.second;
        string perm_line2_1 = perm_pair2.first;
        string perm_line2_2 = perm_pair2.second;
        string perm_line3_1 = perm_pair3.first;
        string perm_line3_2 = perm_pair3.second;

        // first pair permutation
        for (Line & line : perm_lines) {
            if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
            else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
            else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
            else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
        }
        perm_vertex->update_lines(perm_lines, false);
        output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // second pair permutation
        perm_lines = orig_lines;
        for (Line & line : perm_lines) {
            if (line.label_ == perm_line1_1) line.label_ = perm_line3_1;
            else if (line.label_ == perm_line3_1) line.label_ = perm_line1_1;
            else if (line.label_ == perm_line1_2) line.label_ = perm_line3_2;
            else if (line.label_ == perm_line3_2) line.label_ = perm_line1_2;
        }
        perm_vertex->update_lines(perm_lines, false);
        output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // remove last newline
        output.pop_back();
        return output;
    }

    string &
    Term::p6_permute_string(string &output, VertexPtr &perm_vertex, double abs_coeff, int perm_sign, bool has_one) const {

        /// add all permutations of permutation vertex

        make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // create copy of original lines
        vector<Line> orig_lines = perm_vertex->lines();

        // get lines to be permuted
        pair<string, string> perm_pair1 = term_perms_[0];
        pair<string, string> perm_pair2 = term_perms_[1];
        pair<string, string> perm_pair3 = term_perms_[2];

        string perm_line1_1 = perm_pair1.first;
        string perm_line1_2 = perm_pair1.second;
        string perm_line2_1 = perm_pair2.first;
        string perm_line2_2 = perm_pair2.second;
        string perm_line3_1 = perm_pair3.first;
        string perm_line3_2 = perm_pair3.second;

        // original (abc;ijk)

        // pair permutation (acb;ikj)
        vector<Line> perm_lines1 = orig_lines;
//        swap(perm_lines1[perm_idx2_1], perm_lines1[perm_idx3_1]);
//        swap(perm_lines1[perm_idx2_2], perm_lines1[perm_idx3_2]);
        for (Line & line : perm_lines1) {
            if (line.label_ == perm_line2_1) line.label_ = perm_line3_1;
            else if (line.label_ == perm_line3_1) line.label_ = perm_line2_1;
            else if (line.label_ == perm_line2_2) line.label_ = perm_line3_2;
            else if (line.label_ == perm_line3_2) line.label_ = perm_line2_2;
        }
        perm_vertex->update_lines(perm_lines1, false);
        output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // pair permutation (bac;jik)
        vector<Line> perm_lines2 = orig_lines;
        for (Line & line : perm_lines2) {
            if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
            else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
            else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
            else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
        }
        perm_vertex->update_lines(perm_lines2, false);
        output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // pair permutation (cab;kij)
        for (Line & line : perm_lines2) {
            if (line.label_ == perm_line1_1) line.label_ = perm_line3_1;
            else if (line.label_ == perm_line3_1) line.label_ = perm_line1_1;
            else if (line.label_ == perm_line1_2) line.label_ = perm_line3_2;
            else if (line.label_ == perm_line3_2) line.label_ = perm_line1_2;
        }
        perm_vertex->update_lines(perm_lines2, false);
        output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // pair permutation (cba;kji)
        for (Line & line : perm_lines2) {
            if (line.label_ == perm_line2_1) line.label_ = perm_line3_1;
            else if (line.label_ == perm_line3_1) line.label_ = perm_line2_1;
            else if (line.label_ == perm_line2_2) line.label_ = perm_line3_2;
            else if (line.label_ == perm_line3_2) line.label_ = perm_line2_2;
        }
        perm_vertex->update_lines(perm_lines2, false);
        output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // pair permutation (bca;jki)
        for (Line & line : perm_lines2) {
            if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
            else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
            else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
            else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
        }
        perm_vertex->update_lines(perm_lines2, false);
        output = make_perm_string(output, perm_vertex, abs_coeff, perm_sign, has_one);

        // remove last newline
        output.pop_back();
        return output;
    }

} // pdaggerq
