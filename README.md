<img src="https://render.githubusercontent.com/render/math?math=p^{\dagger}q">

A code for bringing strings of creation / annihilation operators to normal order.

**Notes**

1. Normal order may be defined relative to the true vacuum or the fermi vacuum. This selection is made when creating the ahat_helper class:

        # true vacuum
        ahat = pdaggerq.ahat_helper("")
    or    
        # true vacuum
        ahat = pdaggerq.ahat_helper("true")
    or    
        # fermi vacuum
        ahat = pdaggerq.ahat_helper("fermi")
        
    Note than full functionality is not yet available for manual string specification when normal order is defined relative to the fermi vacuum.  In this case, it is better to use the functions defined below that add complete strings in a single command.

2. We follow the usual convention for labeling orbitals: i, j, k, l, m, and n represent occupied orbitals and a, b, c, d, e, and f represent virtual orbitals. Additionally, any label starting with i or a will be considered occupied or virtual, respectively (e.g., i_1 or a2). All other labels are considered general labels. Delta functions involving occupied / virtual combinations will be set to zero. When normal order is defined relative to the fermi vacuum, sums involving general labels are split into sums involving occupied and virtual orbitals using internal labels o1, o2, o3, and o4 (occupied) or v1, v2, v3, and v4 (virtual). So, we recommend avoiding using these labels when specifying any other components of your strings.

3. Orbital labels refer to spin orbitals. In principle, one could explicitly specify spin labels with labels such as ia, ib, etc., but no checks on delta functions involving alpha / beta spin components defined in this way are performed.

4. Strings are defined in Python using the ahat_helper class, which has the following functions:

    #### add_operator_product: 
    
    set strings corresponding to a product of operators. 
    
        add_operator_product( 0.5, ['h(p,q)','t1(a,i)','t2(c,d,k,l)'])
    
    Currently supported operators include 
    
    a general one-body operator
    
        'h(p,q)' 
    
    an antisymetrized two-body operator
    
        'g(p,q,r,s)' 
    
    a pair of creation/annihilation operators, e.g., p*q
    
        'e(p,q)' 
    
    singles and doubles t-amplitudes 
    
        't1(a,i)'
        't2(a,b,i,j)' 
    
    reference, singles, and doubles left-hand amplitudes 
    
        'l0'
        'l1(i,a)'  
        'l2(i,j,a,b)'   
        
    reference, singles, and doubles right-hand amplitudes 
    
        'r0'
        'r1(a,i)'  
        'r2(a,b,i,j)'   
    
    Note that the factor of 1/4 associated with t2, l2, r2, and g are handled internally.
     
    #### add_commutator: 
    
    set strings corresponding to a commutator of two operators. If one of the operators is t2, l2, r2, or g, recall that the factors of 1/4 associated with these operators are handled internally.
    
        add_commutator(1.0, ['h(p,q)','t2(a,b,i,j)'])
  
    #### add_double_commutator: 
    
    set strings corresponding to a double commutator involving three operators. If any of the operators is t2, l2, r2, or g, recall that the factors of 1/4 associated with these operators are handled internally.
    
        add_double_commutator(1.0/2.0, ['h(p,q)','t2(a,b,i,j)','t1(c,k)'])
        
    #### add_triple_commutator: 
    
    set strings corresponding to a triple commutator involving four operators. If any of the operators are t2, l2, r2, or g, recall that the factors of 1/4 associated with these operators are handled internally.
    
        add_triple_commutator(1.0/6.0, ['h(p,q)','t2(a,b,i,j)','t1(c,k)', 't1(d,l)'])
        
    #### add_quadruple_commutator: 
    
    set strings corresponding to a quadruple commutator involving five operators. If any of the operators is t2, l2, r2, or g, recall that the factors of 1/4 associated with these operators are handled internally.
    
        add_quadruple_commutator(1.0/24.0, ['h(p,q)','t2(a,b,i,j)','t1(c,k)', 't1(d,l)', 't1(e,m)'])

    #### set_print_level: 
    
    control the amount of output.  any value greater than the default value of 0 will cause the code to print starting strings.
    
        set_print_level(0)

    #### set_bra: 
    
    set a bra state to include in the operator string. possible bra states include "vacuum", "singles" (m* e), and "doubles" (m* n* f e)
    
        set_bra("doubles")
    
        
    #### set_ket: 
    
    set a ket state to include in the operator string. possible ket states include "vacuum", "singles" (e* m), and "doubles" (e* f* n m)
    
        set_ket("doubles")
    
    #### simplify: 
    
    consolidate/cancel terms and zero any delta functions that involve occupied / virtual combinations.
    
        simplify()
        
    #### print: 
    
    print current list of strings.
        
        print()

    #### print_fully_contracted: 
    
    print strings involving no creation / annihilation operators
    
        print_fully_contracted()
    
    #### print_one_body: 
    
    print strings involving only one-body operators.
    
        print_one_body()
        
    #### print_two_body: 
    
    print strings involving only two-body operators.
    
        print_two_body()
        
    #### clear: 
    
    clear the current set of strings
    
        clear()

5. Strings may also be specified manually using the following commands, but some of these don't yet work correctly when normal order is defined relative to the fermi vacuum. Most use cases would probably be best treated using the functions defined in the previous bullet.

    #### set_string: 
    
    set the string of creation and annihiliation operators.
    
        set_string(['p*','q','a*','i'])
        
    #### set_tensor: 
    
    define a one- or two-body tensor to accompany the string. Note that only one tensor can accompany the string.
    
        set_tensor(['p','q'])
        
        or
        
        set_tensor(['p','q','r','s'])
        
    #### set_t_amplitudes: 
    
    define t1 or t2 amplitudes to accompany the string. Note that an arbitrary number of amplitudes can be set.

        set_t_amplitudes(['a','i'])
        
        or 
        
        set_t_amplitudes(['a','b','i','j'])
        
    #### set_left_amplitudes: 
    
    define l1 or l2 amplitudes to accompany the string. 

        set_left_amplitudes(['i','a'])
        
        or 
        
        set_left_amplitudes(['i','j','a','b'])
        
    #### set_right_amplitudes: 
    
    define r1 or r2 amplitudes to accompany the string. 

        set_right_amplitudes(['a','i'])
        
        or 
        
        set_right_amplitudes(['a','b','i','j'])
        
    #### set_factor: 
    
    define a numerical factor to accompany the string. The default value is 1.0.
    
        set_factor(0.5)

    #### add_new_string: 
    
    bring string to normal order and add to existing list of strings.
    
        add_new_string()
                
        
**Usage**

The following code evaluates the energy for coupled cluster with single and double excitations. For the sake of readability, I have excluded commutators that will evaluate to zero.

Python:

    import pdaggerq

    ahat = pdaggerq.ahat_helper("fermi")

    ahat.set_bra("")
    ahat.set_print_level(0)

    print('')
    print('    < 0 | e(-T) H e(T) | 0> :')
    print('')

    # one-electron part: 
    
    # h
    ahat.add_operator_product(1.0,['h(p,q)'])

    # [h, T1]
    ahat.add_commutator(1.0,['h(p,q)','t1(a,i)'])

    # [h, T2]
    ahat.add_commutator(1.0,['h(p,q)','t2(a,b,i,j)'])

    # two-electron part: 

    # g
    ahat.add_operator_product(1.0,['g(p,r,q,s)'])

    # [g, T1]
    ahat.add_commutator(1.0,['g(p,r,q,s)','t1(a,i)'])

    # [g, T2]
    ahat.add_commutator(1.0,['g(p,r,q,s)','t2(a,b,i,j)'])

    # [[g, T1], T1]]
    ahat.add_double_commutator(0.5, ['g(p,r,q,s)','t1(a,i)','t1(b,j)'])

    ahat.simplify()
    ahat.print_fully_contracted()
    ahat.clear()

Output:

    < 0 | e(-T) H e(T) | 0> :


    // fully-contracted strings:
    //     + 1.00000 h(i,i) 
    //     + 1.00000 h(i,a) t1(a,i) 
    //     + 0.50000 <i,j||i,j> 
    //     + 1.00000 <i,j||a,j> t1(a,i) 
    //     + 0.25000 <i,j||a,b> t2(a,b,i,j) 
    //     + 0.50000 <i,j||a,b> t1(a,i) t1(b,j) 
