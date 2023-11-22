#ifndef PDAGGERQ_LINKAGE_SET_HPP
#define PDAGGERQ_LINKAGE_SET_HPP
#include "linkage.h"
#include <functional>
#include <unordered_set>

using std::string;
using std::hash;

namespace pdaggerq {

    /**
    * hash function class for linkages
    */
    struct LinkageHash {
        size_t operator()(const LinkagePtr &linkage_ptr) const {
            const Linkage &linkage = *linkage_ptr;
            string hash_string = linkage.name(); // this alone would usually be sufficient
            hash_string += linkage.flop_scale().str();
            hash_string += linkage.mem_scale().str();

            hash_string += "int";
            for (const auto &[leftidx, rightidx] : linkage.connections()) {
                hash_string += to_string(leftidx);
                hash_string += "->";
                hash_string += to_string(rightidx);
                hash_string += ",";
            }

            hash_string += "lext";
            for (const auto & leftidx : linkage.l_ext_idx()){
                hash_string += to_string(leftidx);
                hash_string += ",";
            }

            hash_string += "rext";
            for (const auto & rightidx : linkage.r_ext_idx()){
                hash_string += to_string(rightidx);
                hash_string += ",";
            }

            return hash<string>()(hash_string);
        }
    }; // struct linkage_hash

    struct LinkagePred {
        bool operator()(const LinkagePtr &lhs, const LinkagePtr &rhs) const {
            return *lhs == *rhs;
        }
    }; // struct linkage_pred

    class linkage_set{

        unordered_set<LinkagePtr, LinkageHash, LinkagePred> linkages_; // set of linkages

    public:
        /**
         * constructor
         */
        linkage_set() : linkages_(256) {}

        /**
         * constructor with initial bucket n_ops
         * @param size initial n_ops of the set
         */
        explicit linkage_set(size_t size) : linkages_(size) {}

        /**
         * copy constructor
         * @param other linkage set to copy
         */
        linkage_set(const linkage_set &other){
            linkages_ = other.linkages_;
        }

        /**
         * move constructor
         * @param other linkage set to move
         */
        linkage_set(linkage_set &&other) noexcept {
            linkages_ = std::move(other.linkages_);
        }

        /**
         * copy assignment operator
         * @param other linkage set to copy
         * @return reference to this
         */
        linkage_set &operator=(const linkage_set &other){
            linkages_ = other.linkages_;
            return *this;
        };

        /**
         * move assignment operator
         * @param other linkage set to move
         * @return reference to this
         */
        linkage_set &operator=(linkage_set &&other) noexcept{
            linkages_ = std::move(other.linkages_);
            return *this;
        }

        /**
         * destructor
         */
        ~linkage_set() = default;

        /**
         * insert a linkage into the set
         * @param linkage linkage to insert
         */
        auto insert(const LinkagePtr &linkage) {
            return linkages_.insert(linkage);
        }

        /**
         * check if a linkage is in the set
         * @param linkage linkage to check
         * @return true if linkage is in set
         */
        bool contains(const LinkagePtr &linkage) const { return linkages_.find(linkage) != linkages_.end(); }

        /**
         * get the number of linkages in the set
         * @return number of linkages
         */
        size_t size() const { return linkages_.size(); }

        /**
         * get the set of linkages
         * @return set of linkages
         */
        const unordered_set<LinkagePtr, LinkageHash, LinkagePred> &linkages() const { return linkages_; }

        /**
         * clear the set of linkages
         */
        void clear() { linkages_.clear(); }

        /**
         * test if the set is empty
         * @return true if the set is empty
         */
        bool empty() const { return linkages_.empty(); }

        /**
         * begin iterator for set of linkages
         */
        auto begin() const { return linkages_.begin(); }

        /**
         * end iterator for set of linkages
         */
        auto end() const { return linkages_.end(); }

        /**
         * find a linkage in the set
         * @param linkage linkage to find
         * @return iterator to linkage in set
         */
        auto find(const LinkagePtr &linkage) const { return linkages_.find(linkage); }

        /**
         * const overload [] operator
         * @param i index of linkage
         * @return const reference to linkage
         */
        const LinkagePtr &operator[](size_t i) const { return *next(linkages_.begin(), (long) i); }


        /**
         * get reference to linkage in set by value from [] operator
         * @param linkage linkage to get reference to
         * @return reference to linkage
         */
        const LinkagePtr &operator[](const LinkagePtr &linkage) const { return *linkages_.find(linkage); }

        /**
         * overload + operator
         * @param other linkage set to add
         * @return new linkage set
         */
        linkage_set operator+(const linkage_set &other) const {
            linkage_set new_set = *this; // new linkage set
            for (const auto &linkage: other.linkages_) new_set.insert(linkage); // insert other set
            return new_set; // return new linkage set
        }

        /**
         * overload - operator
         * @param other linkage set to remove from this
         * @return new linkage set
         */
        linkage_set operator-(const linkage_set &other) const {
            linkage_set new_set = *this; // new linkage set
            for (const auto &linkage: other.linkages_) new_set.linkages_.erase(linkage); // remove other set
            return new_set; // return new linkage set
        }


        /**
         * overload += operator
         * @param other linkage set to add
         * @return reference to this
         */
        linkage_set &operator+=(const linkage_set &other) {

            for (const auto &linkage: other.linkages_) insert(linkage); // insert the other set
            return *this; // return this
        }

        /**
         * overload -= operator
         * @param other linkage set to remove from this
         * @return reference to this
         */
        linkage_set &operator-=(const linkage_set &other) {
            for (const auto &linkage: other.linkages_)
                linkages_.erase(linkage); // remove other set
            return *this; // return this
        }

        /**
         * erase a linkage from the set by index
         * @param i index of linkage to erase
         */
        void erase(size_t i) { linkages_.erase(next(linkages_.begin(), i)); }

        /**
         * erase a linkage from the set by value
         * @param linkage linkage to erase
         */
        void erase(const LinkagePtr &linkage) { linkages_.erase(linkage); }

    }; // class linkage_set

} // namespace pdaggerq


#endif //PDAGGERQ_LINKAGE_SET_HPP