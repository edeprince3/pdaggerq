
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms

def main():
    pq = pdaggerq.pq_helper("fermi")

    # set right and left-hand operators
    pq.set_left_operators([['1']])
    pq.set_right_operators([['1']])

    print('')
    print('#    H(0;0) = <0| e(-T) H e(T) |0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    H00 = pq.strings()
    H00 = contracted_strings_to_tensor_terms(H00)
    for my_term in H00:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='H00'))
        print()

    pq.clear()

    # set right and left-hand operators
    pq.set_left_operators([['e1(m,e)']])
    pq.set_right_operators([['1']])

    print('')
    print('#    H(m,e;0) = <0|e1(m,e) e(-T) H e(T) |0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    Hs0 = pq.strings()
    Hs0 = contracted_strings_to_tensor_terms(Hs0)
    for my_term in Hs0:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='Hs0',
                                    output_variables=('e', 'm')))
        print()

    pq.clear()

    # set right and left-hand operators
    pq.set_left_operators([['1']])
    pq.set_right_operators([['e1(a,i)']])

    print('')
    print('#    H(0;i,a) = <0| e(-T) H e(T) e1(a,i)|0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    H0s = pq.strings()
    H0s = contracted_strings_to_tensor_terms(H0s)
    for my_term in H0s:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='H0s',
                                    output_variables=('a', 'i')))
        print()

    pq.clear()

    # set right and left-hand operators
    pq.set_left_operators([['e2(m,n,f,e)']])
    pq.set_right_operators([['1']])

    print('')
    print('#    H(m,n,e,f;0) = <0|e2(m,n,f,e) e(-T) H e(T) |0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    Hd0 = pq.strings()
    Hd0 = contracted_strings_to_tensor_terms(Hd0)
    for my_term in Hd0:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='Hd0',
                                    output_variables=('e', 'f', 'm', 'n')))
        print()

    pq.clear()
    
    # set right and left-hand operators
    pq.set_left_operators([['1']])
    pq.set_right_operators([['e2(a,b,j,i)']])
    
    print('')
    print('#    H(0;i,j,a,b) = <0| e(-T) H e(T) e2(a,b,j,i)|0>')
    print('')
    
    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])
    
    pq.simplify()

    H0d = pq.strings()
    H0d = contracted_strings_to_tensor_terms(H0d)
    for my_term in H0d:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='H0d',
                                    output_variables=('a', 'b', 'i', 'j')))
        print()

    pq.clear()

    # set right and left-hand operators
    pq.set_left_operators([['e1(m,e)']])
    pq.set_right_operators([['e1(a,i)']])

    print('')
    print('#    H(m,e;i,a) = <0|e1(m,e) e(-T) H e(T) e1(a,i)|0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    Hss = pq.strings()
    Hss = contracted_strings_to_tensor_terms(Hss)
    for my_term in Hss:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='Hss',
                                    output_variables=('e', 'm', 'a', 'i')))
        print()

    pq.clear()
    
    # set right and left-hand operators
    pq.set_left_operators([['e1(m,e)']])
    pq.set_right_operators([['e2(a,b,j,i)']])
    
    print('')
    print('#    H(m,e;i,j,a,b) = <0|e1(m,e) e(-T) H e(T) e2(a,b,j,i)|0>')
    print('')
    
    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])
    
    pq.simplify()

    Hsd = pq.strings()
    Hsd = contracted_strings_to_tensor_terms(Hsd)
    for my_term in Hsd:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='Hsd',
                                    output_variables=('e', 'm', 'a', 'b', 'i', 'j')))
        print()

    pq.clear()
    
    # set right and left-hand operators
    pq.set_left_operators([['e2(m,n,f,e)']])
    pq.set_right_operators([['e1(a,i)']])
    
    print('')
    print('#    H(m,n,e,f;i,a) = <0|e2(m,n,f,e) e(-T) H e(T) e1(a,i)|0>')
    print('')
    
    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])
    
    pq.simplify()

    Hds = pq.strings()
    Hds = contracted_strings_to_tensor_terms(Hds)
    for my_term in Hds:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='Hds',
                                    output_variables=('e', 'f', 'm', 'n', 'a', 'i')))
        print()

    pq.clear()
    
    # set right and left-hand operators
    pq.set_left_operators([['e2(m,n,f,e)']])
    pq.set_right_operators([['e2(a,b,j,i)']])
    
    print('')
    print('#    H(m,n,e,f;i,j,a,b) = <0|e2(m,n,f,e) e(-T) H e(T) e2(a,b,j,i)|0>')
    print('')
    
    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])
    
    pq.simplify()

    Hdd = pq.strings()
    Hdd = contracted_strings_to_tensor_terms(Hdd)
    for my_term in Hdd:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='Hdd',
                                    output_variables=('e', 'f', 'm', 'n', 'a', 'b', 'i', 'j')))
        print()

    pq.clear()

if __name__ == "__main__":
    main()
