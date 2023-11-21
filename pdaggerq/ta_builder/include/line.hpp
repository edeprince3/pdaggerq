#include <utility>
#include <stdexcept>
#include <set>

#ifndef PDAGGERQ_LINE_HPP
#define PDAGGERQ_LINE_HPP

using namespace std;
namespace pdaggerq {

    /**
     * A line is a single index in an operator.
     * It is defined by its position in the tensor (idx_), whether it is occupied, virtual, alpha, or beta, and its name
     */
    struct Line {
        string label_; // name of the line

        bool o_ = false; // whether the line is occupied (true) or virtual (false)
        bool a_ = true; // whether the line is alpha/active (true) or beta/external (false)
        char blk_type_ = '\0'; // type of blocking (s: spin, r: range, '\0': none)
        bool sig_ = false; // whether the line is an excited state index
        bool den_ = false; // whether the line is for density fitting
        
        static inline set<char> occ_labels_ = { // names of occupied lines
            'i', 'j', 'k', 'l', 'm', 'n', 'o',
                      'K', 'L', 'M', 'N', 'O'
        };
        static inline set<char> virt_labels_ = { // names of virtual lines
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'            
        };
        static inline set<char> sig_labels_ = { // names of excited state lines
            'I', 'J' //TODO: replace with 'S'  for state
        };
        static inline set<char> den_labels_ = { // names of density fitting lines
            'Q'
        };

        Line() = default;

        /**
         * Constructor
         * @param index index of the line position in the operator
         * @param name name of the line
         * @param blk whether the line has blocking
         */
        inline explicit Line(string name, char blk = '\0') : label_(std::move(name)){

            char line_char = label_[0];
              o_ = occ_labels_.find(line_char) != occ_labels_.end();
            sig_ = sig_labels_.find(line_char) != sig_labels_.end();
            den_ = den_labels_.find(line_char) != den_labels_.end();

//            if (!o_ & !sig_ && !den_){
//                if (virt_labels_.find(line_char) == virt_labels_.end())
//                    throw runtime_error("Invalid line name " + label_);
//            }

            if (blk == 'a' || blk == 'b') blk_type_ = 's';
            else if (blk == '0' || blk == '1') blk_type_ = 'r';
            else if (blk != '\0')
                throw runtime_error("Invalid blk " + string(1, blk));

            if (blk_type_ == 's') a_ = blk == 'a';
            else if (blk_type_ == 'r') a_ = blk == '1';
            else a_ = false;

        }

        Line(const Line &other) = default; // copy constructor
        Line(Line &&other) noexcept = default; // move constructor
        Line &operator=(const Line &other) = default; // copy assignment
        Line &operator=(Line &&other) noexcept = default; // move assignment



        /// *** Comparison rhs *** ///
        /// all comparison rhs are defined in terms of name and properties. Index is not used.

        bool operator==(const Line& other) const {
            return label_ == other.label_ &&
                       o_ == other.o_     &&
                       a_ == other.a_     &&
                     sig_ == other.sig_   &&
                     den_ == other.den_;
        }



        bool equivalent(const Line& other) const {
            return   o_ == other.o_   &&
                     a_ == other.a_   &&
                   sig_ == other.sig_ &&
                   den_ == other.den_;
        }

        bool operator!=(const Line& other) const {
            return !(*this == other);
        }

        bool operator<(const Line& other) const {
            bool is_equiv = equivalent(other);
            if ( is_equiv ) return label_ < other.label_; // if same properties, compare names
            if ( sig_ && !other.sig_ ) return true; // if this is excited and other is not, this is less
            if ( den_ && !other.den_ ) return true; // if this is density fitting and other is not, this is less
            if (  !o_ &&  other.o_ ) return true; // if this is virtual and other is occupied, this is less
            if (   a_ && !other.a_ ) return true; // if this is alpha and other is beta, this is less
            return false; // otherwise, this is greater
        }

        bool operator>(const Line& other) const {
            return other < *this;
        }

        bool operator<=(const Line& other) const {
            return *this < other || *this == other;
        }

        bool operator>=(const Line& other) const {
            return *this > other || *this == other;
        }

        inline char blk() const {
            if (blk_type_ == 's') return a_ ? 'a' : 'b';
            if (blk_type_ == 'r') return a_ ? '1' : '0';
            return '\0';
        }

        inline bool has_blk() const { return blk_type_ != '\0'; }

        inline char ov() const {
            if (sig_) return 'L';
            if (den_) return 'Q';
            return o_ ? 'o' : 'v';
        }

        inline bool empty() const {
            return label_.empty();
        }
    };

    // define hash function for Line
    struct LineHash {
        size_t operator()(const Line &line) const {
            string blk{
                line.o_ ? 'o' : 'v',
                line.a_ ? 'a' : 'b',
                line.sig_ ? 'L' : 'N',
                line.den_ ? 'Q' : 'N'
            };

            return hash<string>()(line.label_ + blk);
        }
    }; // struct LineHash

    struct LineEqual {
        bool operator()(const Line &lhs, const Line &rhs) const {
            return lhs == rhs;
        }
    }; // struct LineEqual

    struct LinePtrHash {
        size_t operator()(const Line *line) const {
            string blk{
                    line->o_ ? 'o' : 'v',
                    line->a_ ? 'a' : 'b',
                    line->sig_ ? 'L' : 'N',
                    line->den_ ? 'Q' : 'N'
            };

            return hash<string>()(line->label_ + blk);
        }
    }; // struct LineHash

    struct LinePtrEqual {
        bool operator()(const Line *lhs, const Line *rhs) const {
            return *lhs == *rhs;
        }
    }; // struct LinePtrEqual

} // pdaggerq

#endif //PDAGGERQ_LINE_HPP
