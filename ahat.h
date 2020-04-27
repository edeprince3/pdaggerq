#ifndef SQE_H
#define SQE_H

#include "data.h"

namespace pdaggerq {

class ahat {

  private:

    bool is_normal_order();

  public:

    ahat();
    ~ahat();

    bool skip     = false;
    int sign      = 1;

    std::vector<std::string> used_labels;

    void shallow_copy(void * copy_me);
    void copy(void * copy_me);

    std::vector<std::string> symbol;
    std::vector<bool> is_dagger;
    std::vector<std::string> delta1;
    std::vector<std::string> delta2;

    std::shared_ptr<StringData> data;

    void print();
    void check_spin();
    void check_occ_vir();
    void normal_order(std::vector<std::shared_ptr<ahat> > &ordered);
    void alphabetize(std::vector<std::shared_ptr<ahat> > &ordered);
    void cleanup(std::vector<std::shared_ptr<ahat> > &ordered);

};

}

#endif
