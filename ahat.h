#ifndef SQE_H
#define SQE_H

namespace psi{ namespace pdaggerq {

class ahat {

  private:

    bool is_normal_order();

  public:

    ahat();
    ~ahat();

    bool skip     = false;
    int sign      = 1;
    double factor = 1.0;

    std::vector<std::string> symbol;
    std::vector<bool> is_dagger;
    std::vector<std::string> delta1;
    std::vector<std::string> delta2;
    std::vector<std::string> tensor;

    void print();
    void print_code(Options& options);
    void check_spin();
    void check_occ_vir();
    void normal_order(std::vector<ahat *> &ordered);
    void alphabetize(std::vector<ahat *> &ordered);
    void cleanup(std::vector<ahat *> &ordered);

};

}}

#endif
