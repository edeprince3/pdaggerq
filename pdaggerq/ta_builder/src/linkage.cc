#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>
#include <cstring>
#include "../include/linkage.h"

namespace pdaggerq {

    /********** Constructors **********/

    inline Linkage::Linkage(const VertexPtr &left, const VertexPtr &right, bool is_addition) : Vertex() {

        // set inputs
        if (!left->is_linked() && !right->is_linked()) {
            // a binary linkage is associative (left and right are interchangeable)
            // sort left and right vertices by name to prevent duplicates
            // TODO: make sure this works with Term::is_compatible()
            if (left->name() < right->name()) {
                 left_ =  left;
                right_ = right;
            } else {
                 left_ = right;
                right_ =  left;
            }
        } else {
            // a linkage with more than two vertices is not associative
            left_  =  left;
            right_ = right;
        }

        // count_ the left and right vertices
        nvert_ = 2;

        // determine the depth of the linkage
        if (left_->is_linked()){
            // subtract 1 and add the number of vertices in left
            --nvert_ += as_link(left_)->nvert_;
        }
        if (right_->is_linked()){
            // subtract 1 and add the number of vertices in right
            --nvert_ += as_link(right_)->nvert_;
        }

        is_addition_ = is_addition;

        // create hash for the name (should be unique and faster for comparisons)
        name_ = to_string(hash<string>()(left_->name_ + " " + right_->name_));
        base_name_ = name_;

        // build internal and external lines
        set_links(left_, right_);

        // create mapping of indices of internal and external lines from left to right vertices
        connect_lines(left, right);

        // check if linkage is a sigma vertex or density fitted vertex
        is_sigma_ = !lines_.empty() && lines_[0].sig_;
        is_den_   = !lines_.empty() && lines_[0].den_;
    }

    inline void Linkage::set_links(const VertexPtr &left, const VertexPtr &right) {

        // clear internal and external lines and connections
        int_lines_.clear();
        lines_.clear();

        // grab data from left and right vertices
        uint8_t left_size = left->size();
        uint8_t right_size = right->size();
        uint8_t total_size = left_size + right_size;

        const auto &left_lines = left->lines();
        const auto &right_lines = right->lines();

        // handle scalars
        if (left_size == 0 && right_size == 0) return; // both vertices are scalars (no lines)
        else if (left_size == 0) { // if left is a scalar, just use right_lines as linkage
            lines_ = right_lines; return;
        } else if (right_size == 0) { // if right is a scalar, just use left_lines as linkage
            lines_  = left_lines; return;
        }

        // reserve space for internal and external lines
        int_lines_.reserve(total_size);
        lines_.reserve(total_size);


        // populate left lines
        map<Line, uint8_t> line_populations;
        for (const auto &line : left_lines)
            line_populations[line]++;

        // populate right lines
        for (const auto &line : right_lines)
            line_populations[line]++;

        // use count to determine if the line is internal or external
        for (auto &[line, freq] : line_populations) {
            if (freq == 1)
                     lines_.push_back(line); // external line
            else int_lines_.push_back(line); // internal line
        }

        // lines are sorted via map insertion

        // set properties
        rank_  = lines_.size();
        shape_ = shape(lines_);
        has_blk_ = left->has_blk_ || right->has_blk_;

        // update scaling
        flop_scale_ += int_lines_;
        flop_scale_ += shape_;
        mem_scale_  += shape_;

    }

    inline void Linkage::connect_lines(const VertexPtr &left, const VertexPtr &right) {

        // clear connections
        int_connec_.clear();
        r_ext_idx_.clear();
        l_ext_idx_.clear();

        // grab data from left and right vertices
        const auto &left_lines = left->lines();
        const auto &right_lines = right->lines();

        // build internal connections
        auto hint = int_connec_.begin();
        for (const auto &line : int_lines_) {
            // find line in left and right vertices
            auto left_it = std::find(left_lines.begin(), left_lines.end(), line);
            auto right_it = std::find(right_lines.begin(), right_lines.end(), line);

            // get indices of line in left and right vertices
            uint8_t left_idx = std::distance(left_lines.begin(), left_it);
            uint8_t right_idx = std::distance(right_lines.begin(), right_it);

            // add indices to connections
            hint = int_connec_.emplace_hint(hint, left_idx, right_idx);

            // find next occurrence of line in left and right vertices
            left_it = std::find(++left_it, left_lines.end(), line);
            right_it = std::find(++right_it, right_lines.end(), line);

            bool  left_ended =  left_it ==  left_lines.end();
            bool right_ended = right_it == right_lines.end();

            while (!left_ended || !right_ended) {

                // get indices of line in left and right vertices
                left_idx  = std::distance( left_lines.begin(),  left_it);
                right_idx = std::distance(right_lines.begin(), right_it);

                // add indices to connections
                hint = int_connec_.emplace_hint(hint, left_idx, right_idx);

                // find next occurrence of line in left and right vertices
                left_ended =  left_it ==  left_lines.end();
                right_ended = right_it == right_lines.end();
                if (!left_ended)
                    left_it = std::find(++left_it, left_lines.end(), line);
                if (!right_ended)
                    right_it = std::find(++right_it, right_lines.end(), line);

            }
        }

        // build external connections
        auto r_hint = r_ext_idx_.begin();
        auto l_hint = l_ext_idx_.begin();
        for (const auto &line : lines_) {
            // find line in left and right vertices
            auto left_it = std::find(left_lines.begin(), left_lines.end(), line);
            auto right_it = std::find(right_lines.begin(), right_lines.end(), line);

            if (left_it == left_lines.end()) {
                uint8_t right_idx = std::distance(right_lines.begin(), right_it);
                r_hint = r_ext_idx_.emplace_hint(r_hint, right_idx);
            }
            if (right_it == right_lines.end()) {
                uint8_t left_idx = std::distance(left_lines.begin(), left_it);
                l_hint = l_ext_idx_.emplace_hint(l_hint, left_idx);
            }
        }
    }

    LinkagePtr Linkage::link(const vector<VertexPtr> &op_vec) {
        uint8_t op_vec_size = op_vec.size();

        if (op_vec_size == 0) {
            throw invalid_argument("Linkage::link(): op_vec must have at least two elements");
        } else if (op_vec_size == 1) {
            // this is a hack to allow for the creation of a LinkagePtr from a single VertexPtr
            // TODO: find a better way to do this
            return as_link(make_shared<Vertex>("") * op_vec[0]);
        }

        VertexPtr linkage = op_vec[0] * op_vec[1];
        for (uint8_t i = 2; i < op_vec_size; i++)
            linkage = std::move(linkage * op_vec[i]);

        return as_link(linkage);
    }

    vector<LinkagePtr> Linkage::links(const vector<VertexPtr> &op_vec){
        uint8_t op_vec_size = op_vec.size();
        if (op_vec_size <= 1) {
            throw invalid_argument("Linkage::link(): op_vec must have at least two elements");
        }

        vector<LinkagePtr> linkages(op_vec_size - 1);

        VertexPtr linkage = op_vec[0] * op_vec[1];
        linkages[0] = as_link(linkage);
        for (uint8_t i = 2; i < op_vec_size; i++) {
            linkage = linkage * op_vec[i];
            linkages[i - 1] = as_link(linkage);
        }

        return linkages;
    }

    pair<vector<shape>,vector<shape>> Linkage::scale_list(const vector<VertexPtr> &op_vec) {

        uint8_t op_vec_size = op_vec.size();
        if (op_vec_size <= 1) {
            throw invalid_argument("link(): op_vec must have at least two elements");
        }

        vector<shape> flop_list;
        vector<shape> mem_list;

        LinkagePtr linkage = as_link(op_vec[0] * op_vec[1]);
        flop_list.push_back(linkage->flop_scale_);
        mem_list.push_back(linkage->mem_scale_);

        for (uint8_t i = 2; i < op_vec_size; i++) {
            linkage = as_link(linkage * op_vec[i]);
            flop_list.push_back(linkage->flop_scale_);
            mem_list.push_back(linkage->mem_scale_);
        }

        return {flop_list, mem_list};
    }

    tuple<LinkagePtr, vector<shape>, vector<shape>> Linkage::link_and_scale(const vector<VertexPtr> &op_vec) {
        uint8_t op_vec_size = op_vec.size();
        if (op_vec_size <= 1) {
            throw invalid_argument("link(): op_vec must have at least two elements");
        }

        vector<shape> flop_list;
        vector<shape> mem_list;

        LinkagePtr linkage = as_link(op_vec[0] * op_vec[1]);
        flop_list.push_back(linkage->flop_scale_);
        mem_list.push_back(linkage->mem_scale_);

        for (uint8_t i = 2; i < op_vec_size; i++) {
            linkage = as_link(linkage * op_vec[i]);
            flop_list.push_back(linkage->flop_scale_);
            mem_list.push_back(linkage->mem_scale_);
        }

        return {linkage, flop_list, mem_list};
    }

    Linkage::Linkage() {
        id_ = -1;
        flop_scale_ = shape();
        mem_scale_ = shape();
    }

    /****** operator overloads ******/

    bool Linkage::operator==(const Linkage &other) const {

        // check if both linkage are empty or not
        if (empty()) return other.empty();

        // check if linkage type is the same
        if (is_addition_ != other.is_addition_) return false;

        // check the depth of the linkage
        if (nvert_ != other.nvert_) return false;

        // check if left and right vertices are linked in the same way
        if ( left_->is_linked() ^  other.left_->is_linked()) return false;
        if (right_->is_linked() ^ other.right_->is_linked()) return false;

        // check that scales are equal
        if (flop_scale_ != other.flop_scale_) return false;
        if (mem_scale_  !=  other.mem_scale_) return false;

        // check linkage maps
        if (l_ext_idx_  !=  other.l_ext_idx_) return false;
        if (r_ext_idx_  !=  other.r_ext_idx_) return false;
        if (int_connec_ != other.int_connec_) return false;

        // check if linkage vertices (and external lines) are equivalent
        if (!equivalent(other)) return false;

        // check if left and right vertices are equivalent
        if ( !left_->equivalent( *other.left_)) return false;
        if (!right_->equivalent(*other.right_)) return false;

        // check if left linkages are equivalent
        if (left_->is_linked() && other.left_->is_linked()) {
            if (*as_link(left_) != *as_link(other.left_))
                return false;
        }

        // check if right linkages are equivalent
        if (right_->is_linked() && other.right_->is_linked()) {
            if (*as_link(right_) != *as_link(other.right_))
                return false;
        }

        // if all tests pass, return true
        return true;
    }

    // repeat code from == operator, but invert the logic to end recursion early if possible
    bool Linkage::operator!=(const Linkage &other) const {
        return !(*this == other);
    }

    pair<bool, bool> Linkage::permuted_equals(const Linkage &other) const {
        // first test if the linkages are equal
        if (*this == other) return {true, false};

        // test if the linkages have the same number of vertices
        if (nvert_ != other.nvert_) return {false, false};

        // extract total vector of vertices
        const vector<VertexPtr> &this_vert = to_vector();
        const vector<VertexPtr> &other_vert = other.to_vector();

        // check if the vertices are isomorphic and keep track of the number of permutations
        bool swap_sign = false;
        for (size_t i = 0; i < nvert_; i++) {
            bool odd_perm = false;
            bool same_to_perm = is_isomorphic(*this_vert[i], *other_vert[i], odd_perm);
            if (!same_to_perm) return {false, false};
            if (odd_perm) swap_sign = !swap_sign;
        }

        // if the linkages are isomorphic, return true and if the permutation is odd
        return {true, swap_sign};

    }


    string Linkage::str(bool make_generic, bool include_lines) const {

        if (id_ == -1) { // TODO: this might be annoying if we want to reuse a tmp. We will see when we get there...
            // this is not an intermediate vertex (generic linkage).
            // return the str of the left and right vertices
            return tot_str(false, true);
        }

        if (!make_generic) return str();

        // prepare output string as a map of tmps (or reuse_tmps) to a generic name
        string generic_str = is_reused_ ? "reuse_tmps_[\"" : "tmps_[\"";

        // add dimension string
        generic_str += dimstring();
        generic_str += "_";

        // format for scalars
        if (is_scalar())
            generic_str = "scalars_[\"";


        // use id_ to create a generic name
        if (id_ >= 0)
            generic_str += to_string(id_);

        generic_str += "\"]";

        if (include_lines) // if lines are included, add them to the generic name (default)
            generic_str += line_str();

        // create a generic vertex that has the same lines as this linkage.
        // this adds the spin and ov strings to name
        // return its string representation
        return generic_str;
    }

    string Linkage::tot_str(bool expand, bool make_dot) const {

        if (empty()) return "";

        // do not expand linkages that are not intermediates
        if (id_ == -1) expand = false;

        // prepare output string
        string output, left_string, right_string;

        // build left string representation recursively
        if (left_->is_linked() && expand) left_string = as_link(left_)->tot_str(expand, make_dot);
        else left_string = left_->str();

        // build right string representation recursively
        if (right_->is_linked() && expand) right_string = as_link(right_)->tot_str(expand, make_dot);
        else right_string = right_->str();
        
        
        if (!is_addition_) output = left_string + " * " + right_string;
        else { output = "(" + left_string + " + " + right_string + ")"; }


        // if rank == 0, all lines are internal; requires dot() function call
        if (rank() == 0 && !is_addition_ && make_dot) {
            // add 'dot(' after '='
            output = "dot(" + output;

            // find the last star in output
            size_t last_star = output.rfind(" * ");

            // find last ' * '; replace with ', '
            output.replace(last_star, 3, ", ");

            // add closing parenthesis
            output += ")";
        }

        return output;
    }

    VertexPtr Linkage::get(const shared_ptr<const Linkage> &root, uint8_t i, uint8_t &depth) {

        // while the left vertex is also a linkage, recurse
        const VertexPtr &left = root->left_;
        if (left->is_linked()) {
            const LinkagePtr left_linkage = as_link(left);
            const VertexPtr &result = get(left_linkage, i, depth);
            if (result != nullptr)
                return result;
        }

        // if the left vertex is not a linkage, check if it is the ith vertex
        if (i == depth++)
            return left; // return the vertex

        // while the right vertex is also a linkage, recurse
        const VertexPtr &right = root->right_;
        if (right->is_linked()){
            const LinkagePtr right_linkage = as_link(right);
            const VertexPtr &result = get(right_linkage, i, depth);
            if (result != nullptr)
                return result;
        }

        // if the right vertex is not a linkage, check if it is the ith vertex
        if (i == depth++)
            return right; // return the vertex

        // if neither vertex is the ith vertex, return nullptr
        return nullptr;
    }

    VertexPtr Linkage::get(uint8_t i) const {

        // recurse through nested contractions to find the ith vertex
        uint8_t depth = 0;
        auto this_ptr = shared_ptr<const Linkage>(this);
        VertexPtr result = get(this_ptr, i, depth);
        if (result == nullptr)
            throw std::runtime_error("Linkage::get: vertex not found\n i = " + std::to_string(i) +
                                     "\n depth = " + std::to_string(depth) +
                                     "\n left = " + left_->name() +
                                     "\n right = " + right_->name());

        return result;
    }

    inline void Linkage::to_vector(vector<VertexPtr> &result, size_t &i) const {

        if (empty()) return;

        // get left vertex
        if (left_->is_linked()) {
            const LinkagePtr left_linkage = as_link(left_);
            // compute the vertices recursively and save them
            left_linkage->to_vector();

            // add left vertices to result
            for (const auto &vertex : left_linkage->all_vert_)
                result[i++] = copy_vert(vertex);

        } else result[i++] = copy_vert(left_);

        // get right vertex
        if (right_->is_linked()) {
            const LinkagePtr right_linkage = as_link(right_);
            // compute the vertices recursively and save them
            right_linkage->to_vector();

            // add right vertices to result
            void move_link(Linkage &other);

            for (const auto &vertex : right_linkage->all_vert_)
               result[i++] = copy_vert(vertex);

        } else result[i++] = copy_vert(right_);
    }

    const vector<VertexPtr> &Linkage::to_vector(bool regenerate) const {

        std::lock_guard<std::mutex> lock(mtx_);  // Lock the mutex for the scope of the function
        if (all_vert_.empty() || regenerate){ // the vertices are not known
            // compute the vertices recursively
            all_vert_ = vector<VertexPtr>(nvert_); // store the vertices in all_vert_ for next query
            size_t i = 0;
            to_vector(all_vert_, i);
            return all_vert_;
        }

        // the vertices are known
        return all_vert_;
    }


    void Linkage::clone_link(const Linkage &other) {
        // Lock the mutex for the scope of the function
        std::lock_guard<std::mutex> lock(mtx_);

        // call base class copy constructor
        this->Vertex::operator=(other);

        // fill linkage data
        left_ = copy_vert(other.left_);
        right_ = copy_vert(other.right_);

        id_ = other.id_;
        nvert_ = other.nvert_;

        int_connec_ = other.int_connec_;
        l_ext_idx_ = other.l_ext_idx_;
        r_ext_idx_ = other.r_ext_idx_;

        int_lines_ = other.int_lines_;

        flop_scale_ = other.flop_scale_;
        mem_scale_ = other.mem_scale_;

        is_addition_ = other.is_addition_;
        is_reused_ = other.is_reused_;

        all_vert_ = other.all_vert_;
        for (auto &vertex : all_vert_)
            vertex = copy_vert(vertex);
    }

    Linkage::Linkage(const Linkage &other) {
        clone_link(other);
    }

    Linkage &Linkage::operator=(const Linkage &other) {
        // check for self-assignment
        if (this == &other) return *this;
        else clone_link(other);

        return *this;
    }

    void Linkage::move_link(Linkage &&other) {
        // Lock the mutex for the scope of the function
        std::lock_guard<std::mutex> lock(mtx_);

        // call base class move constructor
        this->Vertex::operator=(other);

        // move linkage data
        left_ = std::move(other.left_);
        right_ = std::move(other.right_);

        id_ = other.id_;
        nvert_ = other.nvert_;

        int_connec_ = std::move(other.int_connec_);
        l_ext_idx_ = std::move(other.l_ext_idx_);
        r_ext_idx_ = std::move(other.r_ext_idx_);

        int_lines_ = std::move(other.int_lines_);

        flop_scale_ = std::move(other.flop_scale_);
        mem_scale_ = std::move(other.mem_scale_);

        is_addition_ = other.is_addition_;
        is_reused_ = other.is_reused_;

        all_vert_ = std::move(other.all_vert_);
    }

    Linkage::Linkage(Linkage &&other) noexcept {

        // call move constructor
        move_link(std::move(other));
    }

    Linkage &Linkage::operator=(Linkage &&other) noexcept {
        // check for self-assignment
        if (this == &other) return *this;
        else move_link(std::move(other));

        return *this;
    }

    extern VertexPtr operator*(const VertexPtr &left, const VertexPtr &right){
        return make_shared<Linkage>(left, right, false);
    }

    extern VertexPtr operator+(const VertexPtr &left, const VertexPtr &right){
        return make_shared<Linkage>(left, right, false);
    }

    extern VertexPtr copy_vert(const VertexPtr &vertex){
        if ( vertex->is_linked() )
             return make_shared<Linkage>(*as_link(vertex));
        else return make_shared<Vertex>(*vertex);
        
    }

    /**
     * Write DOT representation of linkage to file stream (to visualize linkage in graphviz)
     * @param os output stream
     * @param linkage linkage to write
     * @return output stream
     */
    ostream &Linkage::write_dot(ostream &os, const std::string& color, bool reset) const {

        static size_t op_id = 0;
        static size_t dummy_count = 0;

        if (reset) {
            op_id = 0;
            dummy_count = 0;
            return os;
        } else { op_id++; dummy_count++; }

        // get vertices
        const vector<VertexPtr> &vertices = this->to_vector(true);

        // TODO: incorporate scalar vertices and make this recursive
//        if (vertices.size() <= 1) return os; // do not write a graph for a single vertex

        std::string padding = "                ";

        std::set<std::string> node_names;
        std::set<std::string> null_nodes;

        std::string node_style = "color=\"" + color + "\", fontsize=20, style=bold";
        std::string null_node_style = "style=invis, height=.1,width=.1";

        std::string int_edge_style = "color=\"" + color + "\"";
        int_edge_style += ", labelfontsize=20";
        int_edge_style += ", fontsize=20";
        int_edge_style += ", concentrate=false";
//        int_edge_style += ", constraint=false";
//        int_edge_style += ", len=1.5";


        std::string ext_edge_style = "color=\"" + color + "\"";
        ext_edge_style += ", labelfontsize=20";
        ext_edge_style += ", fontsize=20";
        ext_edge_style += ", concentrate=false";
//        ext_edge_style += ", len=1.5";
//        ext_edge_style += ", minlen=1.0";




        /** write vertices as graph **/

        /// plot internal lines
        for (size_t i = 0; i < vertices.size(); i++) {
            // initialize current node
            const VertexPtr &current = vertices[i];
            std::string l_id = std::to_string(i) + to_string(op_id);

            for (size_t j = i+1; j < vertices.size(); j++) {
                //TODO: incorporate scalar vertices

                if (vertices[j]->base_name().empty())
                    continue;


                const VertexPtr &next = vertices[j];

                // make contraction of current and next
                LinkagePtr link = as_link(current * next);

                // initialize next node
                std::string r_id = std::to_string(j) + to_string(op_id);
                std::string next_node = next->base_name() + "_" + r_id;
                std::string current_node = current->base_name() + "_" + l_id;

                if (next->base_name() == "Id")
                    next_node = current_node;

                if (current->base_name() == "Id")
                    current_node = next_node;

                // Add vertices as nodes. connect the current and next vertices with edges from the connections map
                // (-1 indicates no match and should use a dummy node)

                const auto & current_lines = current->lines();
                size_t current_len = current_lines.size();
                // loop over internal lines
                for (const auto &line: link->int_lines_) {

                    // initialize edge label
                    std::string edge_label = line.label_;

                    // check if line is in bra
                    bool is_bra = false;
                    auto it = std::find(current_lines.begin(), current_lines.end(), line);
                    size_t dist = std::distance(current_lines.begin(), it);

                    if (dist < current_len / 2) is_bra = true;

                    // determine direction of edge
                    bool right_directed = is_bra == !line.o_;

                    std::string direction;
                    if (right_directed)
                         direction = current_node + " -> " + next_node;
                    else direction = next_node + " -> " + current_node;

                    // write edge
                    os << padding << direction << " [label=\"" << edge_label << "\"," + int_edge_style + "];\n";
                }
            }
        }


        /// plot external lines
        for (size_t i = 0; i < vertices.size(); i++) {

            // initialize current node
            const VertexPtr &current = vertices[i];
            std::string l_id = std::to_string(i) + to_string(op_id);

            if (current->base_name() == "Id")
                continue; // this is a self contraction. No external lines

            if (vertices[i]->base_name().empty())
                continue;

            std::string current_node = current->base_name() + "_" + l_id;

            /// link all vertices to external lines
            // loop over left external lines
            size_t ext_count = 0;
            const auto & current_lines = current->lines();
            size_t current_len = current_lines.size();
            for (const auto &line: this->lines_) {

                // initialize dummy node name
                std::string null = "null" + std::to_string(dummy_count) + line.label_ + std::to_string(ext_count++);
                null_nodes.insert(null);

                // make edge label
                std::string edge_label = line.label_;

                // find line in this vertex
                auto it = std::find(current_lines.begin(), current_lines.end(), line);
                if (it == current->lines().end()) continue; // line not found

                bool is_bra = false;
                size_t dist = std::distance(current_lines.begin(), it);

                if (dist < current_len / 2) is_bra = true;

                // determine direction of edge
                bool right_directed = is_bra == !line.o_;

                std::string direction;
                if (right_directed)
                     direction = current_node + " -> " + null;
                else direction = null + " -> " + current_node;

                // write edge
                os << padding << direction << " [label=\"" << edge_label << "\"," + ext_edge_style + "];\n";

            }
        }


        /// declare nodes

        // loop over vertices and declare nodes
        for (size_t i = 0; i < vertices.size(); i++) {
            // initialize current node
            const VertexPtr &current = vertices[i];
            std::string l_id = std::to_string(i) + to_string(op_id);

            if (current->base_name() == "Id")
                continue; // this is a self contraction. No external lines

            if (vertices[i]->base_name().empty())
                continue;

            std::string current_node = current->base_name() + "_" + l_id;
            std::string node_signature = padding + current_node + " [label=\"" + current->base_name() + "\", ";

            if (current->base_name().empty())
                node_signature += null_node_style + "];\n";
            else
                node_signature += node_style + "];\n";
            node_names.insert(node_signature);
        }

        // print nodes name
        for (const auto &node_name : node_names)
            os << node_name;

        // make dummy nodes invisible
        for (const auto &dummy_node : null_nodes)
            os << padding << dummy_node << " [label=\"\", " + null_node_style + "];\n";

        return os;
    }
} // pdaggerq
