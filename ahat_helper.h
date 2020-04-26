#ifndef AHAT_HELPER_H
#define AHAT_HELPER_H

#include "ahat.h"

namespace psi{ namespace pdaggerq {

class ahat_helper {

  private:

    std::vector< ahat* > ordered;

  public:

    ahat_helper();
    ~ahat_helper();

    void add_new_string(Options& options,std::string stringnum);
    void finalize();

};

}}

#endif
