# T1 residual equations for CCSDt

import pdaggerq

def main():

    pq = pdaggerq.pq_helper("fermi")

    # set bra/ket
    pq.set_left_operators([['e1(i,a)']])    
    pq.set_right_operators([['1']])    

    # add operators
    pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
    pq.add_st_operator(1.0,['v'],['t1','t2','t3'])
   
    pq.simplify()
    
    print('')
    print('#    0 = <0|i*a e(-T) h e(T)|0> ... (a = act, i = act)')
    print('')
    
    label_ranges = {
        't3' : ['act', 'act', 'all', 'act', 'act', 'all'],
        't2' : ['all', 'all', 'all', 'all'],
        't1' : ['all', 'all'],
        'a' : ['act'],
        'i' : ['act']
    }

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings(label_ranges = label_ranges)
    for my_term in terms:
        print(my_term)
    
    
    print('')
    print('#    0 = <0|i*a e(-T) h e(T)|0> ... (a = act, i = ext)')
    print('')
    
    label_ranges = {
        't3' : ['act', 'act', 'all', 'act', 'act', 'all'],
        't2' : ['all', 'all', 'all', 'all'],
        't1' : ['all', 'all'],
        'a' : ['act'],
        'i' : ['ext']
    }

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings(label_ranges = label_ranges)
    for my_term in terms:
        print(my_term)
    
    
    print('')
    print('#    0 = <0|i*a e(-T) h e(T)|0> ... (a = ext, i = act)')
    print('')
    
    label_ranges = {
        't3' : ['act', 'act', 'all', 'act', 'act', 'all'],
        't2' : ['all', 'all', 'all', 'all'],
        't1' : ['all', 'all'],
        'a' : ['ext'],
        'i' : ['act']
    }

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings(label_ranges = label_ranges)
    for my_term in terms:
        print(my_term)
    
    
    print('')
    print('#    0 = <0|i*a e(-T) h e(T)|0> ... (a = ext, i = ext)')
    print('')
    
    label_ranges = {
        't3' : ['act', 'act', 'all', 'act', 'act', 'all'],
        't2' : ['all', 'all', 'all', 'all'],
        't1' : ['all', 'all'],
        'a' : ['ext'],
        'i' : ['ext']
    }

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings(label_ranges = label_ranges)
    for my_term in terms:
        print(my_term)
    
    pq.clear()
    
if __name__ == "__main__":
    main()
