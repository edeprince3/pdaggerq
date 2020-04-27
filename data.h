#ifndef DATA_H
#define DATA_H

namespace pdaggerq {

class StringData {

  private:


  public:

    StringData(){
    };
    ~StringData(){};

    double factor = 1.0;
    std::vector<std::string> string;
    std::vector<std::string> tensor;
    std::vector<std::string> amplitudes1;
    std::vector<std::string> amplitudes2;

};

}

#endif
