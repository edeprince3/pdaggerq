#ifndef AHAT_HELPER_H
#define AHAT_HELPER_H

namespace psi{ namespace pdaggerq {

class ahat_helper {

  private:


  public:

    ahat_helper();
    ~ahat_helper();

    void add_new_string(Options& options,std::vector< ahat* > &ordered,std::string stringnum);
    void finalize(std::vector<ahat *> &in);

};

}}

#endif
