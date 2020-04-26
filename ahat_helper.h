#ifndef AHAT_HELPER_H
#define AHAT_HELPER_H

#include "ahat.h"
#include "data.h"

namespace pdaggerq {

class ahat_helper {

  private:

    std::vector< ahat* > ordered;

    // strings, tensors, etc.
    std::shared_ptr<StringData> data;

  public:

    ahat_helper();
    ~ahat_helper();

    void set_string(std::vector<std::string> in);
    void set_tensor(std::vector<std::string> in);
    void set_amplitudes(std::vector<std::string> in);
    void set_factor(double in);
    void add_new_string();
    void bring_to_normal_order();

};

}

#endif
