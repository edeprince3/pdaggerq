#ifndef SQE_H
#define SQE_H

#include "data.h"

namespace pdaggerq {

class ahat {

  private:

    bool is_normal_order();

    bool index_in_tensor(std::string idx);
    bool index_in_amplitudes(std::string idx);

    void replace_index_in_tensor(std::string old_idx, std::string new_idx);
    void replace_index_in_amplitudes(std::string old_idx, std::string new_idx);

    bool compare_strings(std::shared_ptr<ahat> ordered_1, std::shared_ptr<ahat> ordered_2, int & n_permute);
    void update_summation_labels();
    void update_bra_labels();

    void swap_two_labels(std::string label1, std::string label2);


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
    void gobble_deltas();
    void use_conventional_labels();

    void normal_order(std::vector<std::shared_ptr<ahat> > &ordered);
    void normal_order_fermi_vacuum(std::vector<std::shared_ptr<ahat> > &ordered);
    void normal_order_true_vacuum(std::vector<std::shared_ptr<ahat> > &ordered);

    void alphabetize(std::vector<std::shared_ptr<ahat> > &ordered);
    void cleanup(std::vector<std::shared_ptr<ahat> > &ordered);

    bool is_occ(std::string idx);
    bool is_vir(std::string idx);
};

}

#endif
