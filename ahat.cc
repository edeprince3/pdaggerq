#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include <math.h>

#include "ahat.h"

namespace pdaggerq {

ahat::ahat(std::string vacuum_type) {

  vacuum = vacuum_type;
  skip = false;
  data = (std::shared_ptr<StringData>)(new StringData());

}

ahat::~ahat() {
}

bool ahat::is_occ(char idx) {
    if ( idx == 'I' || idx == 'i') {
        return true;
    }else if ( idx == 'J' || idx == 'j') {
        return true;
    }else if ( idx == 'K' || idx == 'k') {
        return true;
    }else if ( idx == 'L' || idx == 'l') {
        return true;
    }else if ( idx == 'M' || idx == 'm') {
        return true;
    }else if ( idx == 'N' || idx == 'n') {
        return true;
    }else if ( idx == 'O' || idx == 'o') {
        return true;
    }else if ( idx == 'T' || idx == 't') {
        return true;
    }else if ( idx == 'U' || idx == 'u') {
        return true;
    }else if ( idx == 'V' || idx == 'v') {
        return true;
    }
    return false;
}

bool ahat::is_vir(char idx) {
    if ( idx == 'A' || idx == 'a') {
        return true;
    }else if ( idx == 'B' || idx == 'b') {
        return true;
    }else if ( idx == 'C' || idx == 'c') {
        return true;
    }else if ( idx == 'D' || idx == 'd') {
        return true;
    }else if ( idx == 'E' || idx == 'e') {
        return true;
    }else if ( idx == 'F' || idx == 'f') {
        return true;
    }else if ( idx == 'W' || idx == 'w') {
        return true;
    }else if ( idx == 'X' || idx == 'x') {
        return true;
    }else if ( idx == 'Y' || idx == 'y') {
        return true;
    }else if ( idx == 'Z' || idx == 'z') {
        return true;
    }
    return false;
}

void ahat::check_occ_vir() {

   // OCC: I,J,K,L,M,N
   // VIR: A,B,C,D,E,F
   // GEN: P,Q,R,S,T,U,V,W

   for (int i = 0; i < (int)delta1.size(); i++ ) {
       bool first_is_occ = false;
       if ( is_occ(delta1[i].at(0)) ){
           first_is_occ = true;
       }else if ( is_vir(delta1[i].at(0)) ) {
           first_is_occ = false;
       }else {
           continue;
       }

       bool second_is_occ = false;
       if ( is_occ(delta2[i].at(0)) ){
           second_is_occ = true;
       }else if ( is_vir(delta2[i].at(0)) ) {
           second_is_occ = false;
       }else {
           continue;
       }

       if ( first_is_occ != second_is_occ ) {
           skip = true;
       }

   }

}

void ahat::check_spin() {

    // check A/B in delta functions
    for (int j = 0; j < (int)delta1.size(); j++) {
        if ( delta1[j].length() == 2 ) {
            if ( delta1[j].at(1) == 'A' && delta2[j].at(1) == 'B' ) {
                skip = true;
                break;
            }else if ( delta1[j].at(1) == 'B' && delta2[j].at(1) == 'A' ) {
                skip = true;
                break;
            }
        }
    }

    // check A/B in two-index data->tensors
    if ( (int)data->tensor.size() == 2 ) {
        if ( data->tensor[0].length() == 2 ) {
            if ( data->tensor[1].length() == 2 ) {

                if ( data->tensor[0].at(1) == 'A' && data->tensor[1].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( data->tensor[0].at(1) == 'B' && data->tensor[1].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }
    }

    // check A/B in four-index data->tensors
    if ( (int)data->tensor.size() == 4 ) {
        // check bra
        if ( data->tensor[0].length() == 2 ) {
            if ( data->tensor[1].length() == 2 ) {

                if ( data->tensor[0].at(1) == 'A' && data->tensor[1].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( data->tensor[0].at(1) == 'B' && data->tensor[1].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }
        // check ket
        if ( data->tensor[2].length() == 2 ) {
            if ( data->tensor[3].length() == 2 ) {

                if ( data->tensor[2].at(1) == 'A' && data->tensor[3].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( data->tensor[2].at(1) == 'B' && data->tensor[3].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }

    }


}

void ahat::print() {
    if ( skip ) return;

    if ( vacuum == "FERMI" && (int)symbol.size() > 0 ) {
        // check if stings should be zero or not
        bool is_dagger_right = is_dagger_fermi[(int)symbol.size() - 1];
        bool is_dagger_left  = is_dagger_fermi[0];
        if ( !is_dagger_right || is_dagger_left ) {
            //return;
        }
    }

    //for (int i = 0; i < (int)symbol.size(); i++) {
    //    printf("%5i\n",(int)is_dagger_fermi[i]);
    //}

    printf("    ");
    printf("//     ");
    printf("%c", sign > 0 ? '+' : '-');
    printf(" ");
    printf("%7.5lf", fabs(data->factor));
    printf(" ");
    for (int i = 0; i < (int)symbol.size(); i++) {
        printf("%s",symbol[i].c_str());
        if ( is_dagger[i] ) {
            printf("%c",'*');
        }
        printf(" ");
    }
    for (int i = 0; i < (int)delta1.size(); i++) {
        printf("d(%s%s)",delta1[i].c_str(),delta2[i].c_str());
        printf(" ");
    }
    if ( (int)data->tensor.size() > 0 ) {
        // two-electron integrals
        if ( (int)data->tensor.size() == 4 ) {
            printf("g(");
            for (int i = 0; i < 2; i++) {
                printf("%s",data->tensor[i].c_str());
            }
            //printf("|");
            for (int i = 2; i < 4; i++) {
                printf("%s",data->tensor[i].c_str());
            }
            printf(")");
        }
        // one-electron integrals
        if ( (int)data->tensor.size() == 2 ) {
            printf("h(");
            for (int i = 0; i < 2; i++) {
                printf("%s",data->tensor[i].c_str());
            }
            printf(")");
        }
        printf(" ");
    }

    // amplitudes
    if ( (int)data->amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->amplitudes.size(); i++) {
           
            if ( (int)data->amplitudes[i].size() > 0 ) {
                // t1
                if ( (int)data->amplitudes[i].size() == 2 ) {
                    printf("t1(");
                    for (int j = 0; j < 2; j++) {
                        printf("%s",data->amplitudes[i][j].c_str());
                    }
                    printf(")");
                }
                // t2
                if ( (int)data->amplitudes[i].size() == 4 ) {
                    printf("t2(");
                    for (int j = 0; j < 4; j++) {
                        printf("%s",data->amplitudes[i][j].c_str());
                    }
                    printf(")");
                }
                printf(" ");
            } 
        }
    }
    printf("\n");
}

bool ahat::is_normal_order() {

    // don't bother bringing to normal order if we're going to skip this string
    if (skip) return true;

    if ( vacuum == "TRUE" ) {
        for (int i = 0; i < (int)symbol.size()-1; i++) {
            if ( !is_dagger[i] && is_dagger[i+1] ) {
                return false;
            }
        }
    }else {
        for (int i = 0; i < (int)symbol.size()-1; i++) {
            // check if stings should be zero or not
            bool is_dagger_right = is_dagger_fermi[(int)symbol.size() - 1];
            bool is_dagger_left  = is_dagger_fermi[0];
            if ( !is_dagger_right || is_dagger_left ) {
                return true;
            }
            if ( !is_dagger_fermi[i] && is_dagger_fermi[i+1] ) {
                return false;
            }
        }
    }
    return true;
}

// in order to compare strings, the creation and annihilation 
// operators should be ordered in some consistent way.
// alphabetically seems reasonable enough
void ahat::alphabetize(std::vector<std::shared_ptr<ahat> > &ordered) {

    // alphabetize string
    for (int i = 0; i < (int)ordered.size(); i++) {

        // creation
        bool not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (int j = 0; j < (int)ordered[i]->symbol.size(); j++) {
                if ( ordered[i]->is_dagger[j] ) ndagger++;
            }
            for (int j = 0; j < ndagger-1; j++) {
                int val1 = ordered[i]->symbol[j].c_str()[0];
                int val2 = ordered[i]->symbol[j+1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = ordered[i]->symbol[j];
                    ordered[i]->symbol[j] = ordered[i]->symbol[j+1];
                    ordered[i]->symbol[j+1] = dum;
                    ordered[i]->sign = -ordered[i]->sign;
                    not_alphabetized = true;
                    j = (int)ordered[i]->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
        // annihilation
        not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (int j = 0; j < (int)ordered[i]->symbol.size(); j++) {
                if ( ordered[i]->is_dagger[j] ) ndagger++;
            }
            for (int j = ndagger; j < (int)ordered[i]->symbol.size()-1; j++) {
                int val1 = ordered[i]->symbol[j].c_str()[0];
                int val2 = ordered[i]->symbol[j+1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = ordered[i]->symbol[j];
                    ordered[i]->symbol[j] = ordered[i]->symbol[j+1];
                    ordered[i]->symbol[j+1] = dum;
                    ordered[i]->sign = -ordered[i]->sign;
                    not_alphabetized = true;
                    j = (int)ordered[i]->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
    }

    // alphabetize deltas
    for (int i = 0; i < (int)ordered.size(); i++) {
        for (int j = 0; j < (int)ordered[i]->delta1.size(); j++) {
            int val1 = ordered[i]->delta1[j].c_str()[0];
            int val2 = ordered[i]->delta2[j].c_str()[0];
            if ( val2 < val1 ) {
                std::string dum = ordered[i]->delta1[j];
                ordered[i]->delta1[j] = ordered[i]->delta2[j];
                ordered[i]->delta2[j] = dum;
            }
        }
    }
}

// once strings are alphabetized, we can compare them
// and remove terms that cancel. 

void ahat::cleanup(std::vector<std::shared_ptr<ahat> > &ordered) {

    // cancel like terms
    for (int i = 0; i < (int)ordered.size(); i++) {

// this is done below now, but i don't want to delete the code yet.
break;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming

        if ( vacuum == "FERMI" && ordered[i]->symbol.size() != 0 ) continue;

        if ( ordered[i]->skip ) continue;

        for (int j = i+1; j < (int)ordered.size(); j++) {

            // for normal order relative to fermi vacuum, i doubt anyone will care 
            // about terms that aren't fully contracted. so, skip those because this
            // function is time consuming

            if ( vacuum == "FERMI" && ordered[j]->symbol.size() != 0 ) continue;

            if ( ordered[j]->skip ) continue;

            // are factors same?
            if ( ordered[i]->data->factor != ordered[j]->data->factor ) continue;

            // are signs opposite?
            if ( ordered[i]->sign == ordered[j]->sign ) continue;

            // are strings same?
            if ( ordered[i]->symbol.size() != ordered[j]->symbol.size() ) continue;
            int nsame_s = 0;
            for (int k = 0; k < (int)ordered[i]->symbol.size(); k++) {
                if ( ordered[i]->symbol[k] == ordered[j]->symbol[k] ) {
                    nsame_s++;
                }
            }
            if ( nsame_s != ordered[i]->symbol.size() ) continue;
            // are tensors same?
            if ( ordered[i]->data->tensor.size() != ordered[j]->data->tensor.size() ) continue;
            int nsame_t = 0;
            for (int k = 0; k < (int)ordered[i]->data->tensor.size(); k++) {
                if ( ordered[i]->data->tensor[k] == ordered[j]->data->tensor[k] ) {
                    nsame_t++;
                }
            }
            if ( nsame_t != ordered[i]->data->tensor.size() ) continue;

            // same delta functions (recall these aren't sorted in any way)
            int nsame_d = 0;
            for (int k = 0; k < (int)ordered[i]->delta1.size(); k++) {
                for (int l = 0; l < (int)ordered[j]->delta1.size(); l++) {
                    if ( ordered[i]->delta1[k] == ordered[j]->delta1[l] && ordered[i]->delta2[k] == ordered[j]->delta2[l] ) {
                        nsame_d++;
                        //break;
                    }else if ( ordered[i]->delta2[k] == ordered[j]->delta1[l] && ordered[i]->delta1[k] == ordered[j]->delta2[l] ) {
                        nsame_d++;
                        //break;
                    }
                }
            }
            if ( nsame_d != (int)ordered[i]->delta1.size() ) continue;

            // amplitudes, which can be complicated since they aren't sorted

            // same number of amplitudes?
            if ( ordered[i]->data->amplitudes.size() != ordered[j]->data->amplitudes.size() ) continue;
         
            int nsame_amps = 0;
            for (int ii = 0; ii < (int)ordered[i]->data->amplitudes.size(); ii++) {
                for (int jj = 0; jj < (int)ordered[j]->data->amplitudes.size(); jj++) {

                    // t1 vs t2?
                    if ( ordered[i]->data->amplitudes[ii].size() != ordered[j]->data->amplitudes[jj].size() ) continue;

                    // indices?
                    int nsame_idx = 0;
                    for (int iii = 0; iii < (int)ordered[i]->data->amplitudes[ii].size(); iii++) {
                        for (int jjj = 0; jjj < (int)ordered[j]->data->amplitudes[jj].size(); jjj++) {
                            if ( ordered[i]->data->amplitudes[ii][iii] == ordered[j]->data->amplitudes[jj][jjj] ) {
                                nsame_idx++;
                                break;
                            }
                        }
                    }
                    // if all indices are the same, the amplitudes must be the same
                    if ( nsame_idx == (int)ordered[i]->data->amplitudes[ii].size() ) {
                        nsame_amps++;
                        break;
                    }
                }
            }
            if ( nsame_amps != (int)ordered[i]->data->amplitudes.size() ) continue;

            // well, i guess they cancel
            ordered[i]->skip = true;
            ordered[j]->skip = true;

            // break j so we don't cancel any other terms j with i
            break;
            
        }

    }

    // prioritize summation labels as i > j > k > l and a > b > c > d.
    // this means that j, k, or l should not arise in a term if i is not
    // already present.
    for (int i = 0; i < (int)ordered.size(); i++) {

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming

        if ( vacuum == "FERMI" && ordered[i]->symbol.size() != 0 ) continue;

        if ( ordered[i]->skip ) continue;

        if ( !ordered[i]->index_in_tensor("i") && !ordered[i]->index_in_amplitudes("i") ) {
            if ( ordered[i]->index_in_tensor("j") ) {
               ordered[i]->replace_index_in_tensor("j","i");
               ordered[i]->replace_index_in_tensor("j","i");
               ordered[i]->replace_index_in_tensor("j","i");
               ordered[i]->replace_index_in_tensor("j","i");
               ordered[i]->replace_index_in_amplitudes("j","i");
               ordered[i]->replace_index_in_amplitudes("j","i");
               ordered[i]->replace_index_in_amplitudes("j","i");
               ordered[i]->replace_index_in_amplitudes("j","i");
            }else {
            }
        }
        if ( !ordered[i]->index_in_tensor("a") && !ordered[i]->index_in_tensor("a") ) {
            if ( ordered[i]->index_in_tensor("b") ) {
               ordered[i]->replace_index_in_tensor("b","a");
               ordered[i]->replace_index_in_tensor("b","a");
               ordered[i]->replace_index_in_tensor("b","a");
               ordered[i]->replace_index_in_tensor("b","a");
               ordered[i]->replace_index_in_amplitudes("b","a");
               ordered[i]->replace_index_in_amplitudes("b","a");
               ordered[i]->replace_index_in_amplitudes("b","a");
               ordered[i]->replace_index_in_amplitudes("b","i");
            }else {
            }
        }

    }

    // consolidate terms, including those that differ only by symmetric quantities [i.e., g(iajb) and g(jbia)]
    for (int i = 0; i < (int)ordered.size(); i++) {

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming

        if ( vacuum == "FERMI" && ordered[i]->symbol.size() != 0 ) continue;

        if ( ordered[i]->skip ) continue;

        for (int j = i+1; j < (int)ordered.size(); j++) {

            // for normal order relative to fermi vacuum, i doubt anyone will care 
            // about terms that aren't fully contracted. so, skip those because this
            // function is time consuming

            if ( vacuum == "FERMI" && ordered[j]->symbol.size() != 0 ) continue;

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            if ( !strings_same ) continue;

            //printf("same tensors\n");

            // are factors same?
            //if ( ordered[i]->data->factor != ordered[j]->data->factor ) continue;

            // are signs same?
            //if ( ordered[i]->sign != ordered[j]->sign ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                //printf("skipping\n");
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms

            //printf("combining\n");
            // well, i guess the are the same term
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

            // break j because i'm not yet sure the best way to combine multiple terms.
            //break;
            
        }

    }

}

bool ahat::compare_strings(std::shared_ptr<ahat> ordered_1, std::shared_ptr<ahat> ordered_2, int & n_permute) {

    //printf("ok, how about these\n");
    //ordered[i]->print();
    //ordered[j]->print();

    // are strings same?
    if ( ordered_1->symbol.size() != ordered_2->symbol.size() ) return false;
    int nsame_s = 0;
    for (int k = 0; k < (int)ordered_1->symbol.size(); k++) {
        if ( ordered_1->symbol[k] == ordered_2->symbol[k] ) {
            nsame_s++;
        }
    }
    if ( nsame_s != ordered_1->symbol.size() ) return false;
    //printf("same strings\n");

    // same delta functions (recall these aren't sorted in any way)
    int nsame_d = 0;
    for (int k = 0; k < (int)ordered_1->delta1.size(); k++) {
        for (int l = 0; l < (int)ordered_2->delta1.size(); l++) {
            if ( ordered_1->delta1[k] == ordered_2->delta1[l] && ordered_1->delta2[k] == ordered_2->delta2[l] ) {
                nsame_d++;
                //break;
            }else if ( ordered_1->delta2[k] == ordered_2->delta1[l] && ordered_1->delta1[k] == ordered_2->delta2[l] ) {
                nsame_d++;
                //break;
            }
        }
    }
    if ( nsame_d != (int)ordered_1->delta1.size() ) return false;
    //printf("same deltas\n");

    // amplitudes, which can be complicated since they aren't sorted

    // same number of amplitudes?
    if ( ordered_1->data->amplitudes.size() != ordered_2->data->amplitudes.size() ) return false;
    
    int nsame_amps = 0;
    n_permute = 0;
    for (int ii = 0; ii < (int)ordered_1->data->amplitudes.size(); ii++) {
        for (int jj = 0; jj < (int)ordered_2->data->amplitudes.size(); jj++) {

            // t1 vs t2?
            if ( ordered_1->data->amplitudes[ii].size() != ordered_2->data->amplitudes[jj].size() ) continue;

            // indices?
            int nsame_idx = 0;
            for (int iii = 0; iii < (int)ordered_1->data->amplitudes[ii].size(); iii++) {
                for (int jjj = 0; jjj < (int)ordered_2->data->amplitudes[jj].size(); jjj++) {
                    if ( ordered_1->data->amplitudes[ii][iii] == ordered_2->data->amplitudes[jj][jjj] ) {
                        if ( (iii - jjj) % 2 != 0  && iii < jjj ) n_permute++;
                        nsame_idx++;
                        break;
                    }
                }
            }
            // if all indices are the same, the amplitudes must be the same, but we need to be careful of permutations
            if ( nsame_idx == (int)ordered_1->data->amplitudes[ii].size() ) {
                nsame_amps++;
                break;
            }
        }
    }
    if ( nsame_amps != (int)ordered_1->data->amplitudes.size() ) return false;

    //printf("same amps\n");
    //if ( (n_permute % 2) != 0 ) continue;

    // are tensors same?
    if ( ordered_1->data->tensor.size() != ordered_2->data->tensor.size() ) return false;
    int nsame_t = 0;
    for (int k = 0; k < (int)ordered_1->data->tensor.size(); k++) {
        if ( ordered_1->data->tensor[k] == ordered_2->data->tensor[k] ) {
            nsame_t++;
        }
    }
    // if not the same, check bras againt kets
    if ( nsame_t != ordered_1->data->tensor.size() ) {

        // let's just limit ourselves to four-index tensors for now
        if ( ordered_1->data->tensor.size() == 4 ) {

            int nsame_t_swap = 0;
            if ( ordered_1->data->tensor[0] != ordered_2->data->tensor[2] ||
                 ordered_1->data->tensor[1] != ordered_2->data->tensor[3] ||
                 ordered_1->data->tensor[2] != ordered_2->data->tensor[0] ||
                 ordered_1->data->tensor[3] != ordered_2->data->tensor[1]) {
                 return false;
            }

        }
    }
    return true;
}

// copy all data, except symbols and daggers. 

void ahat::shallow_copy(void * copy_me) { 

    ahat * in = reinterpret_cast<ahat * >(copy_me);

    // skip string?
    skip   = in->skip;
    
    // sign
    sign   = in->sign;
    
    // factor
    data->factor = in->data->factor;

    // temporary delta functions
    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;

    // data->tensor
    for (int i = 0; i < (int)in->data->tensor.size(); i++) {
        data->tensor.push_back(in->data->tensor[i]);
    }

    // delta1, delta2
    for (int i = 0; i < (int)in->delta1.size(); i++) {
        delta1.push_back(in->delta1[i]);
        delta2.push_back(in->delta2[i]);
    }

    // amplitudes
    for (int i = 0; i < (int)in->data->amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in->data->amplitudes[i].size(); j++) {
            tmp.push_back(in->data->amplitudes[i][j]);
        }
        data->amplitudes.push_back(tmp);
    }

}

bool ahat::index_in_tensor(std::string idx) {

    for (int i = 0; i < (int)data->tensor.size(); i++) {
        if ( data->tensor[i] == idx ) {
            return true;
        }
    }
    return false;

}

bool ahat::index_in_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->amplitudes[i].size(); j++) {
            if ( data->amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

void ahat::replace_index_in_tensor(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->tensor.size(); i++) {
        if ( data->tensor[i] == old_idx ) {
            data->tensor[i] = new_idx;
            return;
        }
    }

}

void ahat::replace_index_in_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->amplitudes[i].size(); j++) {
            if ( data->amplitudes[i][j] == old_idx ) {
                data->amplitudes[i][j] = new_idx;
                return; 
            }
        }
    }

}

// find and replace any funny labels in tensors with conventional ones. i.e., t -> i ,w -> a
void ahat::use_conventional_labels() {

    // occupied first:
    std::vector<std::string> occ_in{"o","t","u","v"};
    std::vector<std::string> occ_out{"i","j","k","l"};

    for (int i = 0; i < (int)occ_in.size(); i++) {

        if ( index_in_tensor(occ_in[i]) ) {

            for (int j = 0; j < (int)occ_out.size(); j++) {

                if ( !index_in_tensor(occ_out[j]) ) {

                    replace_index_in_tensor(occ_in[i],occ_out[j]);
                    replace_index_in_tensor(occ_in[i],occ_out[j]);
                    replace_index_in_tensor(occ_in[i],occ_out[j]);
                    replace_index_in_tensor(occ_in[i],occ_out[j]);
                    break;
                }
            }
        }
    }

    // now virtual
    std::vector<std::string> vir_in{"w","x","y","z"};
    std::vector<std::string> vir_out{"a","b","c","d"};

    for (int i = 0; i < (int)vir_in.size(); i++) {

        if ( index_in_tensor(vir_in[i]) ) {

            for (int j = 0; j < (int)vir_out.size(); j++) {

                if ( !index_in_tensor(vir_out[j]) ) {

                    replace_index_in_tensor(vir_in[i],vir_out[j]);
                    replace_index_in_tensor(vir_in[i],vir_out[j]);
                    replace_index_in_tensor(vir_in[i],vir_out[j]);
                    replace_index_in_tensor(vir_in[i],vir_out[j]);
                    break;
                }
            }
        }
    }


}

void ahat::gobble_deltas() {

    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;

    for (int i = 0; i < (int)delta1.size(); i++) {

        bool delta1_in_tensor     = index_in_tensor( delta1[i] );
        bool delta2_in_tensor     = index_in_tensor( delta2[i] );
        bool delta1_in_amplitudes = index_in_amplitudes( delta1[i] );
        bool delta2_in_amplitudes = index_in_amplitudes( delta2[i] );

        if ( delta1_in_tensor ) {

            if ( delta2_in_tensor ) {

                replace_index_in_tensor( delta1[i], delta2[i] );

                continue;

            }else if ( delta2_in_amplitudes) {

                // replace index in tensor
                replace_index_in_tensor( delta1[i], delta2[i] );

                continue;

            }else {

                // index two must come from the bra, so replace index one in tensor
                replace_index_in_tensor( delta1[i], delta2[i] );

                continue;

            }

        }else if ( delta2_in_tensor ) {

            if ( delta1_in_amplitudes) {

                // replace index in tensor
                replace_index_in_tensor( delta2[i], delta1[i] );

                continue;

            }else {

                // index one must come from the bra, so replace index two in tensor
                replace_index_in_tensor( delta2[i], delta1[i] );

                continue;

            }

        }else if ( delta1_in_amplitudes ) {

            if ( delta2_in_amplitudes) {

                replace_index_in_amplitudes( delta1[i], delta2[i] );

                continue;

            }else {

                // index two must come from the bra, so replace index one in amplitudes
                replace_index_in_amplitudes( delta1[i], delta1[i] );

                continue;

            }

        }else if ( delta2_in_amplitudes) {

            // index one must come from the bra, so replace index two in amplitudes
            replace_index_in_amplitudes( delta2[i], delta1[i] );

            continue;

        }

        // at this point, it is safe to assume the delta function must remain
        tmp_delta1.push_back(delta1[i]);
        tmp_delta2.push_back(delta2[i]);

    }

    delta1.clear();
    delta2.clear();

    for (int i = 0; i < (int)tmp_delta1.size(); i++) {
        delta1.push_back(tmp_delta1[i]);
        delta2.push_back(tmp_delta2[i]);
    }


/*
    std::vector<std::string> tmp_tensor;

    for (int j = 0; j < (int)data->tensor.size(); j++) {

        // does data->tensor index show up in a delta function? 
        bool skipme = false;
        for (int k = 0; k < (int)delta1.size(); k++) {
            if ( data->tensor[j] == delta1[k] ) {
                tmp_tensor.push_back(delta2[k]);
                skipme = true;
                break;
            }
            if ( data->tensor[j] == delta2[k] ) {
                tmp_tensor.push_back(delta1[k]);
                skipme = true;
                break;
            }
        }
        if ( skipme ) continue;

        tmp_tensor.push_back(data->tensor[j]);
    }

    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;
    for (int j = 0; j < (int)delta1.size(); j++) {
        bool skipme = false;
        for (int k = 0; k < (int)data->tensor.size(); k++) {
            if ( data->tensor[k] == delta1[j] ) {
                skipme = true;
                break;
            }
            if ( data->tensor[k] == delta2[j] ) {
                skipme = true;
                break;
            }
        }
        if ( skipme ) continue;
    
        tmp_delta1.push_back(delta1[j]);
        tmp_delta2.push_back(delta2[j]);
    }

    data->tensor.clear();
    for (int j = 0; j < (int)tmp_tensor.size(); j++) {
        data->tensor.push_back(tmp_tensor[j]);
    }
    delta1.clear();
    delta1.clear();
    for (int j = 0; j < (int)tmp_delta1.size(); j++) {
        delta1.push_back(tmp_delta1[j]);
        delta2.push_back(tmp_delta2[j]);
    }

    // amplitudes
    std::vector<std::vector<std::string>> tmp_amplitudes;
    for (int i = 0; i < (int)data->amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->amplitudes[i].size(); j++) {

            // does data->amplitude index show up in a delta function?
            bool skipme = false;
            for (int k = 0; k < (int)delta1.size(); k++) {
                if ( data->amplitudes[i][j] == delta1[k] ) {
                    tmp.push_back(delta2[k]);
                    skipme = true;
                    break;
                }
                if ( data->amplitudes[i][j] == delta2[k] ) {
                    tmp.push_back(delta1[k]);
                    skipme = true;
                    break;
                }
            }
            if ( skipme ) continue;

            tmp.push_back(data->amplitudes[i][j]);
        }
        tmp_amplitudes.push_back(tmp);

        // now, remove delta functions from list that were gobbled up above
        std::vector<std::string> tmp_delta1;
        std::vector<std::string> tmp_delta2;

        for (int k = 0; k < (int)delta1.size(); k++) {
            bool skipme = false;
            for (int l = 0; l < (int)data->amplitudes[i].size(); l++) {
                if ( data->amplitudes[i][l] == delta1[k] ) {
                    skipme = true;
                    break;
                }
                if ( data->amplitudes[i][l] == delta2[k] ) {
                    skipme = true;
                    break;
                }
            }
            if ( skipme ) continue;
        
            tmp_delta1.push_back(delta1[k]);
            tmp_delta2.push_back(delta2[k]);
        }

        // update deltas
        delta1.clear();
        delta2.clear();
        for (int k = 0; k < (int)tmp_delta1.size(); k++) {
            delta1.push_back(tmp_delta1[k]);
            delta2.push_back(tmp_delta2[k]);
        }


    }

    // update amplitudes
    data->amplitudes.clear();
    for (int i = 0; i < (int)tmp_amplitudes.size(); i++) {
        data->amplitudes.push_back(tmp_amplitudes[i]);
    }
*/
    

}

// copy all data, including symbols and daggers
void ahat::copy(void * copy_me) { 

    shallow_copy(copy_me);

    ahat * in = reinterpret_cast<ahat * >(copy_me);


    // operators
    for (int j = 0; j < (int)in->symbol.size(); j++) {
        symbol.push_back(in->symbol[j]);

        // dagger?
        is_dagger.push_back(in->is_dagger[j]);

        // dagger (relative to fermi vacuum)?
        if ( vacuum == "FERMI" ) {
            is_dagger_fermi.push_back(in->is_dagger_fermi[j]);
        }
    }
    
}

void ahat::normal_order_true_vacuum(std::vector<std::shared_ptr<ahat> > &ordered) {
    if ( skip ) return;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        std::shared_ptr<ahat> newguy (new ahat(vacuum));

        newguy->copy((void*)this);

        ordered.push_back(newguy);

        return;
    }

    // new strings
    std::shared_ptr<ahat> s1 ( new ahat(vacuum) );
    std::shared_ptr<ahat> s2 ( new ahat(vacuum) );

    // copy data common to both new strings
    s1->shallow_copy((void*)this);
    s2->shallow_copy((void*)this);

    // rearrange operators
    for (int i = 0; i < (int)symbol.size()-1; i++) {

        bool swap = ( !is_dagger[i] && is_dagger[i+1] );

        if ( swap ) {

            s1->delta1.push_back(symbol[i]);
            s1->delta2.push_back(symbol[i+1]);

            // check spin in delta functions
            for (int j = 0; j < (int)delta1.size(); j++) {
                if ( s1->delta1[j].length() != 2 ) {
                    //throw PsiException("be sure to specify spin as second character in labels",__FILE__,__LINE__);
                    break;
                }
                if ( s1->delta1[j].at(1) == 'A' && s1->delta2[j].at(1) == 'B' ) {
                    s1->skip = true;
                }else if ( s1->delta1[j].at(1) == 'B' && s1->delta2[j].at(1) == 'A' ) {
                    s1->skip = true;
                }
            }

            s2->sign = -s2->sign;
            s2->symbol.push_back(symbol[i+1]);
            s2->symbol.push_back(symbol[i]);
            s2->is_dagger.push_back(is_dagger[i+1]);
            s2->is_dagger.push_back(is_dagger[i]);

            for (int j = i+2; j < (int)symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);
                s2->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);
                s2->is_dagger.push_back(is_dagger[j]);

            }
            break;

        }else {

            s1->symbol.push_back(symbol[i]);
            s2->symbol.push_back(symbol[i]);

            s1->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger.push_back(is_dagger[i]);

        }
    }

    s1->normal_order_true_vacuum(ordered);
    s2->normal_order_true_vacuum(ordered);

}

void ahat::normal_order_fermi_vacuum(std::vector<std::shared_ptr<ahat> > &ordered) {
    if ( skip ) return;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        std::shared_ptr<ahat> newguy (new ahat(vacuum));

        newguy->copy((void*)this);

        ordered.push_back(newguy);

        return;
    }

    // new strings
    std::shared_ptr<ahat> s1 ( new ahat(vacuum) );
    std::shared_ptr<ahat> s2 ( new ahat(vacuum) );

    // copy data common to both new strings
    s1->shallow_copy((void*)this);
    s2->shallow_copy((void*)this);

    // rearrange operators

    int n_new_strings = 0;

    for (int i = 0; i < (int)symbol.size()-1; i++) {

        bool swap = ( !is_dagger_fermi[i] && is_dagger_fermi[i+1] );

        // four cases: **, --, *-, -*
        // **, --: change sign, swap labels
        // *-, -*: standard swap

        bool daggers_differ = ( is_dagger[i] != is_dagger[i+1] );

        if ( swap && daggers_differ ) {

            // we're going to have two new strings
            n_new_strings = 2;

            s1->delta1.push_back(symbol[i]);
            s1->delta2.push_back(symbol[i+1]);

            // check spin in delta functions
            for (int j = 0; j < (int)delta1.size(); j++) {
                if ( s1->delta1[j].length() != 2 ) {
                    //throw PsiException("be sure to specify spin as second character in labels",__FILE__,__LINE__);
                    break;
                }
                if ( s1->delta1[j].at(1) == 'A' && s1->delta2[j].at(1) == 'B' ) {
                    s1->skip = true;
                }else if ( s1->delta1[j].at(1) == 'B' && s1->delta2[j].at(1) == 'A' ) {
                    s1->skip = true;
                }
            }

            s2->sign = -s2->sign;
            s2->symbol.push_back(symbol[i+1]);
            s2->symbol.push_back(symbol[i]);
            s2->is_dagger.push_back(is_dagger[i+1]);
            s2->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            for (int j = i+2; j < (int)symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);
                s2->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);
                s2->is_dagger.push_back(is_dagger[j]);

                s1->is_dagger_fermi.push_back(is_dagger_fermi[j]);
                s2->is_dagger_fermi.push_back(is_dagger_fermi[j]);

            }
            break;

        }else if ( swap && !daggers_differ )  {

            // we're only going to have one new string, with a different sign
            n_new_strings = 1;

            s1->sign = -s1->sign;
            s1->symbol.push_back(symbol[i+1]);
            s1->symbol.push_back(symbol[i]);
            s1->is_dagger.push_back(is_dagger[i+1]);
            s1->is_dagger.push_back(is_dagger[i]);
            s1->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            s1->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            //s2->sign = -s2->sign;
            //s2->symbol.push_back(symbol[i+1]);
            //s2->symbol.push_back(symbol[i]);
            //s2->is_dagger.push_back(is_dagger[i+1]);
            //s2->is_dagger.push_back(is_dagger[i]);
            //s2->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            //s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            for (int j = i+2; j < (int)symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);
                //s2->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);
                //s2->is_dagger.push_back(is_dagger[j]);

                s1->is_dagger_fermi.push_back(is_dagger_fermi[j]);
                //s2->is_dagger_fermi.push_back(is_dagger_fermi[j]);

            }
            break;

        }else {

            s1->symbol.push_back(symbol[i]);
            s2->symbol.push_back(symbol[i]);

            s1->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger.push_back(is_dagger[i]);

            s1->is_dagger_fermi.push_back(is_dagger_fermi[i]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

        }
    }


    if ( n_new_strings == 1 ) {
        s1->normal_order_fermi_vacuum(ordered);
    }else if ( n_new_strings == 2 ) {
        s1->normal_order_fermi_vacuum(ordered);
        s2->normal_order_fermi_vacuum(ordered);
    }

}

void ahat::normal_order(std::vector<std::shared_ptr<ahat> > &ordered) {
    if ( vacuum == "TRUE" ) {
        normal_order_true_vacuum(ordered);
    }else {
        normal_order_fermi_vacuum(ordered);
    }
}


} // End namespaces

