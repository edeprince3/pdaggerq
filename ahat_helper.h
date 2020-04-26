#ifndef AHAT_HELPER_H
#define AHAT_HELPER_H

#include "ahat.h"
#include "data.h"

namespace psi{ namespace pdaggerq {

class ahat_helper {

  private:

    std::vector< ahat* > ordered;

  public:

    // strings, tensors, etc.
    std::shared_ptr<StringData> data;

    ahat_helper();
    ~ahat_helper();

    void set_string(std::vector<std::string> in);
    void set_tensor(std::vector<std::string> in);
    void set_amplitudes(std::vector<std::string> in);
    void set_factor(double in);
    void normal_ordered_string();

    void add_new_string(Options& options,std::string stringnum);
    void finalize();

};

}}

#endif
