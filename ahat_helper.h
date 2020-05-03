#ifndef AHAT_HELPER_H
#define AHAT_HELPER_H

#include "ahat.h"
#include "data.h"

namespace pdaggerq {

class ahat_helper {

  private:

    std::vector< std::shared_ptr<ahat> > ordered;

    /// strings, tensors, etc.
    std::shared_ptr<StringData> data;

    /// vacuum (fermi or true)
    std::string vacuum;

    /// bra (vacuum, singles, or doubles)
    std::string bra;

  public:

    ahat_helper(std::string vacuum_type);
    ~ahat_helper();

    /// when bringing to normal order, does the bra involve any operators?
    void set_bra(std::string ket_type);

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

    /// add new completed string / tensor / amplitudes / factor (assuming normal order is definied relative to the true vacuum
    void add_new_string_true_vacuum();

    /// add new completed string / tensor / amplitudes / factor (assuming normal order is definied relative to the fermi vacuum
    void add_new_string_fermi_vacuum();

    /// add new complete string as a product of operators (i.e., {'h(pq)','t1(ai)'} )
    void add_operator_product(double factor, std::vector<std::string> in);

    /// add commutator of two operators
    void add_commutator(double factor, std::vector<std::string> in);

    /// add double commutator involving three operators
    void add_double_commutator(double factor, std::vector<std::string> in);

    /// add triple commutator involving four operators
    void add_triple_commutator(double factor, std::vector<std::string> in);

    /// add quadruple commutator involving five operators
    void add_quadruple_commutator(double factor, std::vector<std::string> in);

    /// cancel terms, if possible
    void simplify();

    /// clear strings
    void clear();

    /// print strings
    void print();

    /// print fully-contracted strings
    void print_fully_contracted();

    /// print one-body strings
    void print_one_body();

    /// print two-body strings
    void print_two_body();

};

}

#endif
