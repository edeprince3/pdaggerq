#ifndef PDAGGERQ_linkage_H
#define PDAGGERQ_linkage_H

#include <set>
#include <unordered_set>
#include "vertex.h"
#include "collections/scaling_map.hpp"
#include "misc/timer.h"
#include "memory"
#include <mutex>

namespace pdaggerq {

    /// map of connections between lines
    typedef std::multiset<pair<uint8_t, uint8_t>> connection_type;

    /**
     * Class to represent contractions of a single vertex with a set of other vertices
     * The contraction itself is also a vertex and is defined by a left and right vertex
     */
    class Linkage : public Vertex {

        // define pointer type
        typedef shared_ptr<Linkage> LinkagePtr;

        // indicates the vertex is linked to another vertex
        bool is_linked() const override { return true; }

        /// cost of linkage (flops and memory) as pair of vir and occ counts
        shape flop_scale_{}; // flops
        shape mem_scale_{}; // memory

        mutable std::mutex mtx_; // mutex for thread safety
        mutable vector<VertexPtr> all_vert_; // all linkages with vertices (mutable to allow for lazy evaluation)

        /**
         * helper function to connect the lines of the linkage
         * @param left the left vertex
         * @param right the right vertex
         */
        void set_links(const VertexPtr &left, const VertexPtr &right);

        public:
            long id_ = -1; // id of the linkage (default to -1 if not set)
            size_t nvert_{}; // number of vertices in the linkage
            bool is_addition_ = false; // whether the linkage is an addition; else it is a contraction
            bool is_reused_ = false; // whether the linkage is a shared operator (can be extracted)

            /// vertices in the linkage
            VertexPtr left_; // the lhs argument of the linkage
            VertexPtr right_; // the rhs argument of the linkage

            /// internal and external lines
            std::vector<Line> int_lines_; // internal lines
            std::vector<Line> l_ext_lines_; // left external lines
            std::vector<Line> r_ext_lines_; // right external lines

            /// map of connections between lines
            connection_type connections_; // connections between lines

            /********** Constructors **********/

            Linkage();

            /**
             * Constructor
             * @param left vertex to contract with
             * @param right vertex to contract with
             */
            Linkage(const VertexPtr &left, const VertexPtr &right, bool is_addition);

            /**
             * Connects the lines of the linkage, sets the flop and memory scaling, and sets the name
             * this function will populate the Vertex base class with the result of the contraction
             */
            void connect_lines(const VertexPtr &left, const VertexPtr &right);

            /**
             * Destructor
             */
            ~Linkage() override = default;

            /**
             * Copy constructor
             * @param other linkage to copy
             */
            Linkage(const Linkage &other);

            /**
             * Move constructor
             * @param other linkage to move
             */
            Linkage(Linkage &&other) noexcept;

            /**
             * helper to move only the linkage data
             * @param other linkage to move
             */
            void move_link(Linkage &&other);

            /**
             * helper to clone only the linkage data
             * @param other linkage to clone
             */
            void clone_link(const Linkage &other);

            /****** operator overloads ******/

            /**
             * Copy assignment operator
             * @param other linkage to copy
             * @return reference to this
             */
            Linkage &operator=(const Linkage &other);

            /**
             * Move assignment operator
             * @param other linkage to move
             * @return reference to this
             */
            Linkage &operator=(Linkage &&other) noexcept;

            /**
             * Equality operator
             * @param other linkage to compare
             * @return true if equal, false otherwise
             */
            bool operator==(const Linkage &other) const;

            // TODO: we need an equality operator to test if two linkages are the same up to a permutation of the
            //  lines in the vertices and return the number of permutations made.
            pair<bool, bool> permuted_equals(const Linkage &other) const;


            /**
             * Inequality operator
             * @param other linkage to compare
             * @return true if not equal, false otherwise
             */
            bool operator!=(const Linkage &other) const;

            /**
             * Less than operator
             * @note compares flop scaling
             */
            bool operator<(const Linkage &other) const{
                return flop_scale_ < other.flop_scale_;
            }

            /**
             * Greater than operator
             * @note compares flop scaling
             */
            bool operator>(const Linkage &other) const{
                return flop_scale_ > other.flop_scale_;
            }

            /**
             * Less than or equal to operator
             * @note compares flop scaling
             */
            bool operator<=(const Linkage &other) const{
                return flop_scale_ <= other.flop_scale_;
            }

            /**
             * Greater than or equal to operator
             * @note compares flop scaling
             */
            bool operator>=(const Linkage &other) const{
                return flop_scale_ >= other.flop_scale_;
            }


            /**
             * Propogation step of finding the ith vertex within nested contractions
             * @param root the current linkage
             * @param i the index of the vertex to find
             * @param depth the current depth of the linkage
             * @return pointer to the ith vertex
             */
            static VertexPtr get(const shared_ptr<const Linkage>& root, uint8_t i, uint8_t &depth);

            /**
             * get pointer to the ith vertex within nested contractions
             * @param i the index of the vertex to find
             */
            VertexPtr get(uint8_t i) const;

            /**
             * convert the linkage to a vector of vertices in order
             * @param result vector of vertices
             * @note this function is recursive
             */
            void to_vector(vector<VertexPtr> &result, size_t &i) const;

            /**
             * convert the linkage to a const vector of vertices
             * @return vector of vertices
             * @note this function is recursive
             */
            const vector<VertexPtr> &to_vector(bool regenerate = false) const;

            /**
             * Get connections
             * @return connections
             */
            const connection_type &connections() const { return connections_; }


            /**
             * Make a series of linkages from vertices into a single linkage
             * @param op_vec list of vertices
             */
            static LinkagePtr link(const vector<VertexPtr> &op_vec);

            /**
             * Make a series of linkages from vertices and keep each linkage as a separate vertex
             * @param op_vec list of vertices
             */
            static vector<LinkagePtr> links(const vector<VertexPtr> &op_vec);

            /**
             * Returns a list of all flop and mem scales within a series of linkages
             * @return list of all flop and mem scales within a series of linkages αs a pair of vectors
             */
            static pair<vector<shape>,vector<shape>> scale_list(const vector<VertexPtr> &op_vec) ;

            /**
             * Returns a list of all flop and mem scales within a series of linkages and the linkage
             * @param op_vec list of vertices
             * @return the resulting linkage with the list of all flop and mem scales
             */
            static tuple<LinkagePtr, vector<shape>, vector<shape>> link_and_scale(const vector<VertexPtr> &op_vec);

            /**
             * Get flop cost
             * @return flop cost
             */
            const shape &flop_scale() const { return flop_scale_; }

            /**
             * Get memory cost
             * @return memory cost
             */
            const shape &mem_scale() const { return mem_scale_; }

            /**
             * Create generic string representation of linkage
             * @param make_generic if true, make generic string representation
             * @return generic string representation of linkage
             */
             string str(bool make_generic, bool include_lines = true) const;
             string str() const override {
                 // default to generic string representation when not specified
                 return str(true);
             }
             friend ostream &operator<<(ostream &os, const Linkage &linkage){
                    os << linkage.tot_str(true);
                    return os;
             }

            /**
            * Get string of contractions and additions
            * @param expand if true, expand contractions recursively
            * @return linkage string
            */
            string tot_str(bool expand = false, bool make_dot=true) const;

        /**
         * Write DOT representation of linkage to file stream (to visualize linkage in graphviz)
         * @param os output stream
         * @param linkage linkage to write
         * @return output stream
         */
        ostream &write_dot(ostream &os, const std::string& color = "black", bool reset = false) const;

            /**
             * check if linkage is empty
             * @return true if empty, false otherwise
             */
            bool empty() const override {
                if (nvert_ == 0) return true;
                else return Vertex::empty();
            }

            /**
             * Get depth of linkage
             * @return depth of linkage
             */
            size_t depth() const { return nvert_; }

    }; // class linkage

    // define pointer type
    typedef shared_ptr<Linkage> LinkagePtr;

    // define cast function from VertexPtr to LinkagePtr (base class to derived class)
    static LinkagePtr as_link(const VertexPtr& vertex) { return dynamic_pointer_cast<Linkage>(vertex); }

    /**
     * Perform linkage of two vertices by overload of * operator
     * @param other vertex to contract with
     * @return linkage of the two vertices
     * @note this is an extern function to allow for operator overloading outside of the namespace
     */
    extern VertexPtr operator*(const VertexPtr &left, const VertexPtr &right);

    /**
     * Perform linkage of two vertices by overload of + operator
     * @param other vertex to add
     * @return linkage of linkage
     * @note this is an extern function to allow for operator overloading outside of the namespace
     */
    extern VertexPtr operator+(const VertexPtr &left, const VertexPtr &right);

    /**
     * perform a deep copy of a vertex
     * @param vertex vertex to copy
     * @return pointer to the copy
     */
    extern VertexPtr copy_vert(const VertexPtr &vertex);

} // pdaggerq

#endif //PDAGGERQ_linkage_H
