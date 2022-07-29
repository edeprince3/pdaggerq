
import pdaggerq
    
from pdaggerq.parser import contracted_strings_to_tensor_terms

def main():    
    pq = pdaggerq.pq_helper("fermi")
    
    # set right and left-hand operators
    pq.set_right_operators_type('IP')
    pq.set_left_operators([['a*(m)']])
    pq.set_right_operators([['r0'],['r1'],['r2']])
    
    print('')
    print('    sigma(m) = <0|m* e(-T) H e(T) (r0 + r1 + r2)|0>')
    print('')
    
    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])
    
    pq.simplify()
    singles_residual_terms = pq.fully_contracted_strings()
    for my_term in singles_residual_terms:
        print(my_term)

    #singles_residual_terms = contracted_strings_to_tensor_terms(singles_residual_terms)

    #for my_term in singles_residual_terms:
    #    print("#\t", my_term)
    #    print(my_term.einsum_string(update_val='energy'))
    #    print()
    pq.clear()
    
    # set right and left-hand operators
    pq.set_left_operators_type('IP')
    pq.set_left_operators([['a*(m)','a*(n)','a(e)']])
    pq.set_right_operators([['r0'],['r1'],['r2']])
    
    print('')
    print('    sigma(e,m,n) = <0|m*n*e e(-T) H e(T) (r0 + r1 + r2)|0>')
    print('')
    
    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])
    
    pq.simplify()
    # grab list of fully-contracted strings, then print
    doubles_residual_terms = pq.fully_contracted_strings()
    for my_term in doubles_residual_terms:
        print(my_term)
    pq.clear()


if __name__ == "__main__":
    main()
