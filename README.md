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

2. We follow the usual convention for labeling orbitals: i, j, k, l, m, and n represent occupied orbitals; a, b, c, d, e, and f represent virtual orbitals; and p, q, r, and s represent general orbital labels. Delta functions involving occupied / virtual combinations will be set to zero. When normal order is defined relative to the fermi vacuum, sums involving general labels are split into sums involving occupied and virtual orbitals using the labels o, t, u and v for occupied orbials and w, x, y, and z for virtual orbitals.  So, we recommend avoiding using these labels when specifying any other components of your strings.

3. Orbital labels refer to spin orbitals. Explicit spin labels may be included by adding "A" or "B" to an orbital label, but this is not necessary, and this functionality has not been tested for all use cases (user beware!). Note also that these spin labels must be capitalized in the current version of the code. Delta functions involving alpha / beta combinations will be set to zero. Our recommendation is to avoid explicit spin labels.

4. Strings are defined in Python using the ahat_helper class, which has the following functions:

    add_operator_product: set strings corresponding to a product of operators. Currently supported operators include general one-body operators ('h(pq)'), singles amplitudes ('t1(ai)'), and doubles amplitudes ('t2(abij)'). Note that the factor of 1/4 associated with t2 will be handled internally.
    
        set_operator_product( 0.5, ['h(pq)','t1(ai)','t1(ck)'])
        
    add_commutator: set strings corresponding to a commutator of two operators. If one of the operators is t2, note that the factor of 1/4 associated with this operator will be handled internally.
    
        add_commutator(1.0, ['h(pq)','t2(abij)'])
  
    add_double_commutator: set strings corresponding to a double commutator involving three operators. If any of the operators are t2, note that the factors of 1/4 will be handled internally.
    
        add_double_commutator(0.5, ['h(pq)','t2(abij)','t1(ck)'])
        
    add_triple_commutator: set strings corresponding to a triple commutator involving four operators. If any of the operators are t2, note that the factors of 1/4 will be handled internally.
    
        add_triple_commutator(0.1667, ['h(pq)','t2(abij)','t1(ck)', 't1(dl)'])
        
    add_quadruple_commutator: set strings corresponding to a quadruple commutator involving five operators. If any of the operators are t2, note that the factors of 1/4 will be handled internally.
    
        add_quadruple_commutator(0.0417, ['h(pq)','t2(abij)','t1(ck)', 't1(dl)', 't1(em)'])

    simplify: consolidate/cancel terms and zero any delta functions that involve occupied / virtual or alpha / beta combinations.
    
        simplify()
        
    print: print current list of strings.
        
        print()

    print_fully_contracted: print strings involving no operators
    
        print_fully_contracted()
    
    print_one_body: print strings involving only one-body operators.
    
        print_one_body()
        
    print_two_body: print strings involving only two-body operators.
    
        print_two_body()
        
    clear: clear the current set of strings
    
        clear()

5. Strings may also be specified manually using the following commands, but some of these aren't yet working correctly when normal order is defined relative to the fermi vacuum.

    set_string: set the string of creation and annihiliation operators.
    
        set_string(['p*','q','a*','i'])
        
    set tensor: define a one- or two-body tensor to accompany the string. Note that only one tensor can accompany the string.
    
        set_tensor(['p','q'])
        
        or
        
        set_tensor(['p','q','r','s'])
        
    set amplitudes: define t1 or t2 amplitudes to accompany the string. Note that an arbitrary number of amplitudes can be set.

        set_amplitudes(['a','i'])
        
        or 
        
        set_amplitudes(['a','b','i','j'])
        
    set_factor: define a numerical factor to accompany the string. The default value is 1.0.
    
        set_factor(0.5)

    add_new_string: bring string to normal order and add to existing list of strings.
    
        add_new_string()
                
    add_operator_product: set strings corresponding to a product of operators. Currently supported operators include general one-body operators ('h(pq)'), singles amplitudes ('t1(ai)'), and doubles amplitudes ('t2(abij)'). Note that the factor of 1/4 associated with t2 will be handled internally.
    
        set_operator_product( 0.5, ['h(pq)','t1(ai)','t1(ck)'])
        
    add_commutator: set strings corresponding to a commutator of two operators. If one of the operators is t2, note that the factor of 1/4 associated with this operator will be handled internally.
    
        add_commutator(1.0, ['h(pq)','t2(abij)'])
  
    add_double_commutator: set strings corresponding to a double commutator involving three operators. If any of the operators are t2, note that the factors of 1/4 will be handled internally.
    
        add_double_commutator(0.5, ['h(pq)','t2(abij)','t1(ck)'])
        
    add_triple_commutator: set strings corresponding to a triple commutator involving four operators. If any of the operators are t2, note that the factors of 1/4 will be handled internally.
    
        add_triple_commutator(0.1667, ['h(pq)','t2(abij)','t1(ck)', 't1(dl)'])
        
    add_quadruple_commutator: set strings corresponding to a quadruple commutator involving five operators. If any of the operators are t2, note that the factors of 1/4 will be handled internally.
    
        add_quadruple_commutator(0.0417, ['h(pq)','t2(abij)','t1(ck)', 't1(dl)', 't1(em)'])
        
**Usage**

The following code evaluates the double commutator 0.5 [[h, T1], T1], where h is a one-body operator and normal-order is defined relative to the true vacuum

Python:

    import pdaggerq
    
    ahat = pdaggerq.ahat_helper("")

    ahat.set_string(['p*','q','a*','i','c*','k'])
    ahat.set_tensor(['p','q'])
    ahat.set_amplitudes(['a','i'])
    ahat.set_amplitudes(['c','k'])
    ahat.set_factor(0.5)
    ahat.add_new_string()

    ahat.set_string(['a*','i','p*','q','c*','k'])
    ahat.set_tensor(['p','q'])
    ahat.set_amplitudes(['a','i'])
    ahat.set_amplitudes(['c','k'])
    ahat.set_factor(-0.5)
    ahat.add_new_string()

    ahat.set_string(['c*','k','p*','q','a*','i'])
    ahat.set_tensor(['p','q'])
    ahat.set_amplitudes(['a','i'])
    ahat.set_amplitudes(['c','k'])
    ahat.set_factor(-0.5)
    ahat.add_new_string()

    ahat.set_string(['c*','k','a*','i','p*','q'])
    ahat.set_tensor(['p','q'])
    ahat.set_amplitudes(['a','i'])
    ahat.set_amplitudes(['c','k'])
    ahat.set_factor(0.5)
    ahat.add_new_string()

    ahat.simplify()
    ahat.print()

Output:

    // starting string:
    //     + 0.50000 p* q a* i c* k h(pq) t1(ai) t1(ck)

    // starting string:
    //     - 0.50000 a* i p* q c* k h(pq) t1(ai) t1(ck)

    // starting string:
    //     - 0.50000 c* k p* q a* i h(pq) t1(ai) t1(ck)

    // starting string:
    //     + 0.50000 c* k a* i p* q h(pq) t1(ai) t1(ck)

    // normal-ordered strings:
    //     - 0.50000 a* k h(ic) t1(ai) t1(ck)
    //     - 0.50000 c* i h(ka) t1(ai) t1(ck)

The same output can be generated using the add_operator_product function:

    import pdaggerq
    
    ahat = pdaggerq.ahat_helper("")

    ahat.add_operator_product( 0.5, ['h(pq)','t1(ai)','t1(ck)'])
    ahat.add_operator_product(-0.5, ['t1(ai)','h(pq)','t1(ck)'])
    ahat.add_operator_product(-0.5, ['t1(ck)','h(pq)','t1(ai)'])
    ahat.add_operator_product( 0.5, ['t1(ck)','t1(ai)','h(pq)'])

    ahat.simplify()
    ahat.print()

or the add_double_commutator function:

    import pdaggerq
    
    ahat = pdaggerq.ahat_helper("")

    ahat.add_double_commutator( 0.5, ['h(pq)','t1(ai)','t1(ck)'])

    ahat.simplify()
    ahat.print()
