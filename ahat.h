#ifndef SQE_H
#define SQE_H

#include "data.h"

namespace pdaggerq {

class ahat {

  private:

    bool is_normal_order();

  public:

    ahat(std::string vacuum_type);
    ~ahat();

    std::string vacuum;

    bool skip     = false;
    int sign      = 1;

    void shallow_copy(void * copy_me);
    void copy(void * copy_me);

    std::vector<std::string> symbol;
    std::vector<bool> is_dagger;
    std::vector<bool> is_dagger_fermi;
    std::vector<std::string> delta1;
    std::vector<std::string> delta2;

    std::shared_ptr<StringData> data;

    void print();
    void check_spin();
    void check_occ_vir();

    void normal_order(std::vector<std::shared_ptr<ahat> > &ordered);
    void normal_order_fermi_vacuum(std::vector<std::shared_ptr<ahat> > &ordered);
    void normal_order_true_vacuum(std::vector<std::shared_ptr<ahat> > &ordered);

    void alphabetize(std::vector<std::shared_ptr<ahat> > &ordered);
    void cleanup(std::vector<std::shared_ptr<ahat> > &ordered);

    bool is_occ(char idx);
    bool is_vir(char idx);
};

}

#endif
