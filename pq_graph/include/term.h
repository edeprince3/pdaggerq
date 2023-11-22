#ifndef PDAGGERQ_TERM_H
#define PDAGGERQ_TERM_H

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include "collections/scaling_map.hpp"
#include "vertex.h"
#include "linkage.h"
#include "collections/linkage_set.hpp"
#include "../../pdaggerq/pq_string.h"


namespace pdaggerq {

    typedef vector<pair<string, string>> perm_list;

    /**
     * Term class
     * A term is a product of rhs and a coefficient
     * Each vertex is a string with the vertex name and its indices
     * Term is optimized for floating point operations
     */
    class Term {

        protected:

            VertexPtr lhs_; // vertex on the left hand side of the term
            VertexPtr eq_; // vertex of the equation this term is in (usually the same as lhs_)
            vector<VertexPtr> rhs_; // rhs of the term
            vector<string> comments_; // string representation of the original rhs

            size_t rank_{}; // rank of the term

            /// scaling of the term (stored as a pair of integers, (num virtual, num occupied))
            scaling_map flop_map_; // map of flop scaling with linkage occurrence in term
            scaling_map mem_map_; // map of memory scaling with linkage occurrence in term

            shape bottleneck_flop_; // bottleneck flop scaling of the term
            shape bottleneck_mem_; // bottleneck memory scaling of the term

            /// list of permutation indices (should generalize to arbitrary number of indices)

            perm_list term_perms_;
            perm_list perm_pairs_mem_; // recall if this term used to be a permutation and what the permutation pairs were

            // perm_type_ = 0: no permutation
            // perm_type_ = 1: P(i,j) R(ij;ab) = R(ij;ab) - R(ji;ab)
            // perm_type_ = 2: PP2(i,a;j,b) R(ijk;abc) = R(ijk;abc) + (jik,bac)
            // perm_type_ = 3: PP3(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + (jik;bac) + R(kji;cba)
            // perm_type_ = 6: PP6(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + R(jik;bac) + R(jki;bca) + R(ikj;acb) + R(kji;cba) + R(kij;cab)
            size_t perm_type_ = 0; // default is no permutation
            size_t perm_type_mem_ = 0; // recall if this term used to be a permutation and what the permutation type was

    public:

            bool is_optimal_ = false; // flag for if term has optimal linkages (default is false)
            bool needs_update_ = true; // flag for if term needs to be updated (default is true)
            bool is_assignment_ = false; // true if the term is an assignment (default is false, using +=)

            static set<string> conditions_; // conditions to apply to terms, given by a vector of names for rhs
            mutable LinkagePtr term_linkage_; // linkage of the term

            /**
             * Returns a set of the conditions that the term requires
             * @return set of conditions
             */
            set<string> which_conditions() const;

            /******** Constructors ********/

            Term() = default;
            ~Term() = default;

            double coefficient_{}; // coefficient of the term

            /**
             * Constructor
             * @param name name of the assignment vertex
             * @param pq_str representation of term from pq_helper
             */
            Term(const string &name, const shared_ptr<pq_string>& pq_str);

            /**
             * Constructor
             * @param name name of the assignment vertex
             * @param vertex_strings vector of vertex string representations
             */
            Term(const string &name, const vector<string> &vertex_strings);

            /**
             * Constructor
             * @param lhs_vertex assignment vertex
             * @param vertices vector of rhs
             * @param coefficient coefficient of the term
             */
            Term(const VertexPtr &lhs_vertex, const vector<VertexPtr> &vertices, double coefficient);

            /**
             * Constructor to build assignment of a linkage
             * @param linkage linkage to assign
             */
            explicit Term(const LinkagePtr &linkage);

            /**
             * Copy constructor
             * @param other term to copy
             */
            Term(const Term &other) = default;

            /**
             * Move constructor
             * @param other term to move
             */
            Term(Term &&other) noexcept = default;

            /******** operator overloads ********/

            /**
             * Assignment operator
             * @param other term to copy
             * @return reference to this term
             */
            Term &operator=(const Term &other) = default;

            /**
             * Move assignment operator
             * @param other term to move
             * @return reference to this term
             */
            Term &operator=(Term &&other) noexcept = default;

            /**
             * return reference to vertex at index
             * @param index index of vertex
             * @return reference to vertex at index
             */
            VertexPtr &operator[](size_t index){ return rhs_[index]; }

            /**
             * return const reference to vertex at index
             * @param index index of vertex
             * @return const reference to vertex at index
             */
            const VertexPtr &operator[](size_t index) const{ return rhs_[index]; }


            /**
             * Formats permutation rhs for term
             * @param perm_string string representation of permutation vertex
             */
            void set_perm(const string & perm_string);

            /**
             * Formats permutation rhs for term
             * @param perm_pairs pairs of permutations
             * @param perm_type the type of permutation
             */
            void set_perm(const perm_list& perm_pairs, const size_t perm_type){
                term_perms_ = perm_pairs;
                perm_pairs_mem_ = perm_pairs;
                perm_type_ = perm_type;
                perm_type_mem_ = perm_type;
            }

            /**
             * Recalls if this term used to be a permutation and what the permutation pairs were
             * @return pair of permutations_pairs and permutation type
             */
            pair<perm_list, size_t> recall_perm() const {
                return make_pair(perm_pairs_mem_, perm_type_mem_);
            }

            /**
             * Set permutation memory
             * @param perm_pairs pairs of permutations
             * @param perm_type the type of permutation
             */
            void set_perm_mem(const perm_list& perm_pairs, const size_t perm_type){
                perm_pairs_mem_ = perm_pairs;
                perm_type_mem_ = perm_type;
            }

            /**
             * Get left hand side vertex
             * @return left hand side vertex
             */
            const VertexPtr &lhs() const { return lhs_; }
            VertexPtr &lhs() { return lhs_; }


            /**
             * Get vertex for the equation
             * @return vertex for the equation
             */
            const VertexPtr &eq() const { return eq_; }

            /**
             * Set left hand side vertex
             */
            void set_lhs(const VertexPtr &lhs) { lhs_ = lhs; }

            /**
             * Set vertex for the equation
             */
            void set_equation_vertex(const VertexPtr &equation_vertex) { eq_ = equation_vertex; }

            /**
             * permutation indices
             * @return permutation indices
             */
            const perm_list &term_perms() const { return term_perms_; }

            /**
             * Get type of permutation
             * @return type of permutation
             */
            size_t perm_type() const { return perm_type_; }

            /**
             * Get number of rhs
             */
             size_t size() const { return rhs_.size(); }

            /**
             * begin iterator
             */
            vector<VertexPtr>::iterator begin() { return rhs_.begin(); }

            /**
             * end iterator
             */
            vector<VertexPtr>::iterator end() { return rhs_.end(); }

            /**
             * begin const iterator
             */
            vector<VertexPtr>::const_iterator begin() const { return rhs_.begin(); }

            /**
             * end const iterator
             */
            vector<VertexPtr>::const_iterator end() const { return rhs_.end(); }

            /**
             * Get mutable reference to vertex strings
             * @return vector of vertex strings
             */
            vector<string> &comments() {
                return comments_;
            }

            /**
             * Get const reference to comments
             * @return const vector reference of comments
             */
            const vector<string> &comments() const {
                return comments_;
            }

            /**
             * Get rhs and allow modification
             * @return vector of rhs
             */
            vector<VertexPtr> &rhs() {
                return rhs_;
            }

            /**
             * Set rhs
             * @param vertices vector of rhs
             */
            void set_rhs(const vector<VertexPtr> &rhs) {
                rhs_ = rhs;
            }

            /**
             * Get const rhs
             */
            const vector<VertexPtr> &rhs() const {
                return rhs_;
            }

            /**
             * generate string for comment
             * @param only_flop if true, only flop scaling is included in comment
             * @return string for comment
             */
            string make_comments(bool only_flop = false, bool only_comment=false) const;

            /**
             * Get rank
             * @return rank
             */
            size_t rank() const { return rank_; }

            /**
             * Get flop scaling
             * @return map of flop scaling
             */
            const scaling_map &flop_map() const { return flop_map_; }

            /**
             * Get memory scaling
             * @return map of memory scaling
             */
            const scaling_map &mem_map() const { return mem_map_; }

            /**
             * Get bottleneck flop scaling
             * @return bottleneck flop scaling
             */
            const shape &bottleneck_flop() const { return bottleneck_flop_; }

            /**
             * Get bottleneck memory scaling
             * @return bottleneck memory scaling
             */
            const shape &bottleneck_mem() const { return bottleneck_mem_; }


            /**
             * Sort terms such that the terms with tempOps are first
             * @param terms
             */
            static void sort_terms(vector<Term> &terms);

            /******** Functions ********/

            /**
             * Reorder term for optimal floating point operations and store scaling
             */
            void reorder(bool recompute = false);

             /**
              * Populate flop and memory scaling maps
              * @param perm permutation of the rhs
              */
            void compute_scaling(const vector<VertexPtr> &arrangement, bool recompute = false);

             /**
              * Populate flop and memory scaling maps with identity permutation
              */
            void compute_scaling(bool recompute = false){
                compute_scaling(rhs_, recompute); // compute scaling of current rhs
            }

            /**
             * Compare flop scaling of this term to another term by overloading <
             * @param other term to compare to
             */
            bool operator<(const Term &other) const{
                return flop_map_ < other.flop_map_;
            }

            /**
             * Compare flop scaling of this term to another term by overloading >
             * @param other term to compare to
             */
            bool operator>(const Term &other) const{
                return flop_map_ > other.flop_map_;
            }

            /**
             * Compare rhs of this term to another term by overloading ==
             * @param other term to compare to
             * @note only compares flop scaling.
             *       DOES NOT compare coefficient
             */
            bool operator==(const Term &other) const{
                return rhs_ == other.rhs_;
            }

            /**
             * Compare rhs of this term to another term by overloading !=
             * @param other term to compare to
             * @note only compares flop scaling.
             *       DOES NOT compare coefficient
             */
            bool operator!=(const Term &other) const{
                return rhs_ != other.rhs_;
            }

            /**
             * Create string representation of the term
             * @return string representation of the term
             */
            string str() const;
            string einsum_str() const;
            static bool make_einsum;
            string operator+(const string &other) const{ return str() + other; }
            friend string operator+(const string &other, const Term &term){ return other + term.str(); }
            friend ostream &operator<<(ostream &os, const Term &term){
                os << term.str();
                return os;
            }

            /**
             * Represent the coefficient as a fraction
             * @param coeff coefficient to represent
             * @param threshold threshold for error of representation
             * @return pair of numerator and denominator
             */
            static pair<int,int> as_fraction(double coeff, double threshold = 1e-6);

            /**
             * create string representation of permutation assignment
             * @param output string of assignment
             * @param perm_vertex permutation vertex
             * @param abs_coeff absolute value of coefficient
             * @param perm_sign sign of permutation
             * @param has_one boolean indicating if term has one vertex
             * @return string representation of permutation assignment
             */
            string &make_perm_string(string &output, const VertexPtr &perm_vertex, double abs_coeff, int perm_sign,
                                     bool has_one) const;

            /**
             * Create string representation of single index permutation
             * @return string representation of the single index permutation
             */
            string &
            p1_permute_string(string &output, VertexPtr &perm_vertex, double abs_coeff, int perm_sign, bool has_one) const;

            /**
             * Create string representation of PP2 permutation
             * @return string representation of PP2 permutation
             */
            string &
            p2_permute_string(string &output, VertexPtr &perm_vertex, double abs_coeff, int perm_sign, bool has_one) const;

            /**
             * Create string representation of PP3 permutation
             * @return string representation of PP3 permutation
             */
            string &
            p3_permute_string(string &output, VertexPtr &perm_vertex, double abs_coeff, int perm_sign, bool has_one) const;

            /**
             * Create string representation of PP6 permutation
             * @return string representation of PP6 permutation
             */
            string &
            p6_permute_string(string &output, VertexPtr &perm_vertex, double abs_coeff, int perm_sign, bool has_one) const;


            /**
             * Execute an operation of all combinations of subterms
             */
            static void operate_subsets(
                    size_t n, // number of subterms
                    const std::function<void(const vector<size_t>&)>& op, // operation to perform on each subset
                    const std::function<bool(const vector<size_t>&)>& valid_op = nullptr, // operation to check if subset is valid
                    const std::function<bool(const vector<size_t>&)>& break_perm_op = nullptr, // operation to check if permutation should be broken
                    const std::function<bool(const vector<size_t>&)>& break_subset_op = nullptr // operation to check if subset should be broken
            );

            /**
             * Substitute linkage into the term
             * @param linkage linkage to substitute
             * @param allow_equality allow equality of scaling
             * @return boolean indicating if substitution was successful
             */
            bool substitute(const LinkagePtr &linkage, bool allow_equality = false);

            /**
             * collect all possible linkages from all equations
             */
            linkage_set generate_linkages() const;

            static size_t max_linkages; // maximum number of rhs in a linkage
            static size_t depth_; // depth of nested tempOps
            static bool permute_vertices_;

            /**
             * find best scalar linkage for a given term
             * @param id id of the scalar linkage
             * @return best scalar linkage
             */
            LinkagePtr make_dot_products(size_t id);

             /**
              * find vertices with self-contractions and format with delta functions
              */
             void apply_self_links();

            /**
             * pop_back vertex from term
             */
            void pop_back() { rhs_.pop_back(); }

            /**
             * insert vertex into term
             */
            void insert(size_t pos, const VertexPtr &op) { rhs_.insert(rhs_.begin() + (int)pos, op); }

            /**
             * check if term includes rhs of the linkage
             * @param linkage linkage to check
             * @return boolean indicating if term includes rhs of the linkage
             */
            bool is_compatible(const LinkagePtr &linkage) const;

            /**
             * swaps the sign of the term
             */
            void swap_sign() { coefficient_ *= -1; }

            /**
             * check if term is equivalent to another term
             * @param term1 the first term
             * @param term2 the second term
             * @return boolean indicating if term1 is equivalent to term2
             */
            static bool equivalent(const Term &term1, const Term &term2);


            /**
             * check if term is equivalent to another term up to a permutation (keeping track of the permutation)
             * @param ref_term the first term
             * @param compare_term the second term
             * @return pair of booleans indicating if ref_term is equivalent to compare_term and if the permutation is odd
             */
            static pair<bool, bool> same_permutation(const Term &ref_term, const Term &compare_term);

            //TODO: implement generic term.
            Term genericize() const;

    }; // end Term class

    struct TermHash { // hash functor for finding similar terms
        size_t operator()(const Term& term) const {

            if (term.term_linkage_ == nullptr){
                // get the total linkage of the term with its flop and memory scalings
                auto [term_linkage, flop_scales, mem_scales] = Linkage::link_and_scale(term.rhs());
                term.term_linkage_ = term_linkage;
            }

            // add string representation of term.
            // TODO: find all instances where term_linkage_ is used and replace with term.rhs()
            std::string term_str;
            if (term.rhs().size() <= 1)
                term_str = term.rhs()[0]->str();
            else
                term_str = term.term_linkage_->tot_str(true, false);


            // add permutation type and permutation pairs
            term_str += to_string(term.perm_type());
            for (const auto& pair : term.term_perms()) {
                term_str += pair.first;
                term_str += pair.second;
            }

            return hash<string>()(term_str);

//            string term_str;
//            // add vertex names to string and sorted line names to string
//            for (const auto& op : term) {
//                term_str += op->name();
//
//                vector<string> labels;
//                for (const auto& line : op->lines()) labels.push_back(line.label_);
//                sort(labels.begin(), labels.end());
//
//                for (const auto& label : labels) term_str += label;
//            }
//
//            // finally, add permutation type and permutation pairs
//            term_str += to_string(term.perm_type());
//            for (const auto& pair : term.term_perms()) {
//                term_str += pair.first;
//                term_str += pair.second;
//            }
//
//            // return hash of string
//            return hash<string>()(term_str);
        }
    };

    struct TermPredicate { // predicate functor for finding similar terms
        bool operator()(const Term& term1, const Term& term2) const {
            return Term::equivalent(term1, term2);
        }
    };

} // pdaggerq

#endif //PDAGGERQ_TERM_H
