#ifndef AHAT_HELPER_H
#define AHAT_HELPER_H

#include "ahat.h"
#include "data.h"

namespace pdaggerq {

class ahat_helper {

  private:

    std::vector< std::shared_ptr<ahat> > ordered;

    // strings, tensors, etc.
    std::shared_ptr<StringData> data;

  public:

    ahat_helper();
    ~ahat_helper();

    /// set a string of creation / annihilation operators
    void set_string(std::vector<std::string> in);

    /// set labels for a one- or two-body tensor
    void set_tensor(std::vector<std::string> in);

    /// set labels for t1 or t2 amplitudes
    void set_amplitudes(std::vector<std::string> in);

    /// set a numerical factor
    void set_factor(double in);

    /// add new completed string / tensor / amplitudes / factor
    void add_new_string();

    /// add new complete string as a product of operators (i.e., {'h(pq)','t1(ai)'} )
    void set_operator_product(double factor, std::vector<std::string> in);

    /// cancel terms, if possible
    void simplify();

    /// clear strings
    void clear();

    /// print strings
    void print();

    /// print one-body strings
    void print_one_body();

    /// print two-body strings
    void print_two_body();

};

}

#endif
