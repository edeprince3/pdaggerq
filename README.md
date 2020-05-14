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

    add_operator_product: set strings corresponding to a product of operators. Currently supported operators include general one-body operators ('h(p,q)'), antissymetrized two-body operators ('g(p,q,r,s)'), singles amplitudes ('t1(a,i)'), and doubles amplitudes ('t2(a,b,i,j)'). Note that the factor of 1/4 associated with t2 and g are handled internally.
    
        set_operator_product( 0.5, ['h(p,q)','t1(a,i)','t2(c,d,k,l)'])
        
    add_commutator: set strings corresponding to a commutator of two operators. If one of the operators is t2 or g, recall that the factors of 1/4 associated with these operators are handled internally.
    
        add_commutator(1.0, ['h(pq)','t2(abij)'])
  
    add_double_commutator: set strings corresponding to a double commutator involving three operators. If any of the operators is t2 or g, recall that the factors of 1/4 associated with these operators are handled internally.
    
        add_double_commutator(1.0/2.0, ['h(p,q)','t2(a,b,i,j)','t1(c,k)'])
        
    add_triple_commutator: set strings corresponding to a triple commutator involving four operators. If any of the operators are t2, note that the factors of 1/4 will be handled internally.
    
        add_triple_commutator(1.0/6.0, ['h(pq)','t2(abij)','t1(ck)', 't1(dl)'])
        
    add_quadruple_commutator: set strings corresponding to a quadruple commutator involving five operators. If any of the operators is t2 or g, recall that the factors of 1/4 associated with these operators are handled internally.
    
        add_quadruple_commutator(1.0/24.0, ['h(p,q)','t2(a,b,i,j)','t1(c,k)', 't1(d,l)', 't1(e,m)'])

    simplify: consolidate/cancel terms and zero any delta functions that involve occupied / virtual or alpha / beta combinations.
    
        simplify()
        
    print: print current list of strings.
        
        print()

    print_fully_contracted: print strings involving no creation / annihilation operators
    
        print_fully_contracted()
    
    print_one_body: print strings involving only one-body operators.
    
        print_one_body()
        
    print_two_body: print strings involving only two-body operators.
    
        print_two_body()
        
    clear: clear the current set of strings
    
        clear()

5. Strings may also be specified manually using the following commands, but some of these don't yet work correctly when normal order is defined relative to the fermi vacuum.

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
                
        
**Usage**

The following code evaluates the energy for coupled cluster with single and double excitations. For the sake of readability, I have excluded commutators that will evaluate to zero.

Python:

    import pdaggerq

    ahat = pdaggerq.ahat_helper("fermi")

    ahat.set_bra("")

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

    // starting string:
    //     + 1.00000 o1* o2 h(o1,o2) 

    // starting string:
    //     + 1.00000 o1* v2 h(o1,v2) 

    // starting string:
    //     + 1.00000 v1* o2 h(v1,o2) 

    // starting string:
    //     + 1.00000 v1* v2 h(v1,v2) 

    // starting string:
    //     + 1.00000 o1* o2 a* i h(o1,o2) t1(a,i) 

    // starting string:
    //     + 1.00000 o1* v2 a* i h(o1,v2) t1(a,i) 

    // starting string:
    //     + 1.00000 v1* o2 a* i h(v1,o2) t1(a,i) 

    // starting string:
    //     + 1.00000 v1* v2 a* i h(v1,v2) t1(a,i) 

    // starting string:
    //     - 1.00000 a* i o1* o2 h(o1,o2) t1(a,i) 

    // starting string:
    //     - 1.00000 a* i o1* v2 h(o1,v2) t1(a,i) 

    // starting string:
    //     - 1.00000 a* i v1* o2 h(v1,o2) t1(a,i) 

    // starting string:
    //     - 1.00000 a* i v1* v2 h(v1,v2) t1(a,i) 

    // starting string:
    //     + 0.25000 o1* o2 a* b* j i h(o1,o2) t2(a,b,i,j) 

    // starting string:
    //     + 0.25000 o1* v2 a* b* j i h(o1,v2) t2(a,b,i,j) 

    // starting string:
    //     + 0.25000 v1* o2 a* b* j i h(v1,o2) t2(a,b,i,j) 

    // starting string:
    //     + 0.25000 v1* v2 a* b* j i h(v1,v2) t2(a,b,i,j) 

    // starting string:
    //     - 0.25000 a* b* j i o1* o2 h(o1,o2) t2(a,b,i,j) 

    // starting string:
    //     - 0.25000 a* b* j i o1* v2 h(o1,v2) t2(a,b,i,j) 

    // starting string:
    //     - 0.25000 a* b* j i v1* o2 h(v1,o2) t2(a,b,i,j) 

    // starting string:
    //     - 0.25000 a* b* j i v1* v2 h(v1,v2) t2(a,b,i,j) 

    // starting string:
    //     + 0.25000 o1* o2* o3 o4 <o1,o2||o4,o3> 

    // starting string:
    //     + 0.25000 o1* o2* o3 v4 <o1,o2||v4,o3> 

    // starting string:
    //     + 0.25000 o1* o2* v3 o4 <o1,o2||o4,v3> 

    // starting string:
    //     + 0.25000 o1* o2* v3 v4 <o1,o2||v4,v3> 

    // starting string:
    //     + 0.25000 o1* v2* o3 o4 <o1,v2||o4,o3> 

    // starting string:
    //     + 0.25000 o1* v2* o3 v4 <o1,v2||v4,o3> 

    // starting string:
    //     + 0.25000 o1* v2* v3 o4 <o1,v2||o4,v3> 

    // starting string:
    //     + 0.25000 o1* v2* v3 v4 <o1,v2||v4,v3> 

    // starting string:
    //     + 0.25000 v1* o2* o3 o4 <v1,o2||o4,o3> 

    // starting string:
    //     + 0.25000 v1* o2* o3 v4 <v1,o2||v4,o3> 

    // starting string:
    //     + 0.25000 v1* o2* v3 o4 <v1,o2||o4,v3> 

    // starting string:
    //     + 0.25000 v1* o2* v3 v4 <v1,o2||v4,v3> 

    // starting string:
    //     + 0.25000 v1* v2* o3 o4 <v1,v2||o4,o3> 

    // starting string:
    //     + 0.25000 v1* v2* o3 v4 <v1,v2||v4,o3> 

    // starting string:
    //     + 0.25000 v1* v2* v3 o4 <v1,v2||o4,v3> 

    // starting string:
    //     + 0.25000 v1* v2* v3 v4 <v1,v2||v4,v3> 

    // starting string:
    //     + 0.25000 o1* o2* o3 o4 a* i <o1,o2||o4,o3> t1(a,i) 

    // starting string:
    //     + 0.25000 o1* o2* o3 v4 a* i <o1,o2||v4,o3> t1(a,i) 

    // starting string:
    //     + 0.25000 o1* o2* v3 o4 a* i <o1,o2||o4,v3> t1(a,i) 

    // starting string:
    //     + 0.25000 o1* o2* v3 v4 a* i <o1,o2||v4,v3> t1(a,i) 

    // starting string:
    //     + 0.25000 o1* v2* o3 o4 a* i <o1,v2||o4,o3> t1(a,i) 

    // starting string:
    //     + 0.25000 o1* v2* o3 v4 a* i <o1,v2||v4,o3> t1(a,i) 

    // starting string:
    //     + 0.25000 o1* v2* v3 o4 a* i <o1,v2||o4,v3> t1(a,i) 

    // starting string:
    //     + 0.25000 o1* v2* v3 v4 a* i <o1,v2||v4,v3> t1(a,i) 

    // starting string:
    //     + 0.25000 v1* o2* o3 o4 a* i <v1,o2||o4,o3> t1(a,i) 

    // starting string:
    //     + 0.25000 v1* o2* o3 v4 a* i <v1,o2||v4,o3> t1(a,i) 

    // starting string:
    //     + 0.25000 v1* o2* v3 o4 a* i <v1,o2||o4,v3> t1(a,i) 

    // starting string:
    //     + 0.25000 v1* o2* v3 v4 a* i <v1,o2||v4,v3> t1(a,i) 

    // starting string:
    //     + 0.25000 v1* v2* o3 o4 a* i <v1,v2||o4,o3> t1(a,i) 

    // starting string:
    //     + 0.25000 v1* v2* o3 v4 a* i <v1,v2||v4,o3> t1(a,i) 

    // starting string:
    //     + 0.25000 v1* v2* v3 o4 a* i <v1,v2||o4,v3> t1(a,i) 

    // starting string:
    //     + 0.25000 v1* v2* v3 v4 a* i <v1,v2||v4,v3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i o1* o2* o3 o4 <o1,o2||o4,o3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i o1* o2* o3 v4 <o1,o2||v4,o3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i o1* o2* v3 o4 <o1,o2||o4,v3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i o1* o2* v3 v4 <o1,o2||v4,v3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i o1* v2* o3 o4 <o1,v2||o4,o3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i o1* v2* o3 v4 <o1,v2||v4,o3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i o1* v2* v3 o4 <o1,v2||o4,v3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i o1* v2* v3 v4 <o1,v2||v4,v3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i v1* o2* o3 o4 <v1,o2||o4,o3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i v1* o2* o3 v4 <v1,o2||v4,o3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i v1* o2* v3 o4 <v1,o2||o4,v3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i v1* o2* v3 v4 <v1,o2||v4,v3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i v1* v2* o3 o4 <v1,v2||o4,o3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i v1* v2* o3 v4 <v1,v2||v4,o3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i v1* v2* v3 o4 <v1,v2||o4,v3> t1(a,i) 

    // starting string:
    //     - 0.25000 a* i v1* v2* v3 v4 <v1,v2||v4,v3> t1(a,i) 

    // starting string:
    //     + 0.06250 o1* o2* o3 o4 a* b* j i <o1,o2||o4,o3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 o1* o2* o3 v4 a* b* j i <o1,o2||v4,o3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 o1* o2* v3 o4 a* b* j i <o1,o2||o4,v3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 o1* o2* v3 v4 a* b* j i <o1,o2||v4,v3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 o1* v2* o3 o4 a* b* j i <o1,v2||o4,o3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 o1* v2* o3 v4 a* b* j i <o1,v2||v4,o3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 o1* v2* v3 o4 a* b* j i <o1,v2||o4,v3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 o1* v2* v3 v4 a* b* j i <o1,v2||v4,v3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 v1* o2* o3 o4 a* b* j i <v1,o2||o4,o3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 v1* o2* o3 v4 a* b* j i <v1,o2||v4,o3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 v1* o2* v3 o4 a* b* j i <v1,o2||o4,v3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 v1* o2* v3 v4 a* b* j i <v1,o2||v4,v3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 v1* v2* o3 o4 a* b* j i <v1,v2||o4,o3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 v1* v2* o3 v4 a* b* j i <v1,v2||v4,o3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 v1* v2* v3 o4 a* b* j i <v1,v2||o4,v3> t2(a,b,i,j) 

    // starting string:
    //     + 0.06250 v1* v2* v3 v4 a* b* j i <v1,v2||v4,v3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i o1* o2* o3 o4 <o1,o2||o4,o3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i o1* o2* o3 v4 <o1,o2||v4,o3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i o1* o2* v3 o4 <o1,o2||o4,v3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i o1* o2* v3 v4 <o1,o2||v4,v3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i o1* v2* o3 o4 <o1,v2||o4,o3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i o1* v2* o3 v4 <o1,v2||v4,o3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i o1* v2* v3 o4 <o1,v2||o4,v3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i o1* v2* v3 v4 <o1,v2||v4,v3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i v1* o2* o3 o4 <v1,o2||o4,o3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i v1* o2* o3 v4 <v1,o2||v4,o3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i v1* o2* v3 o4 <v1,o2||o4,v3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i v1* o2* v3 v4 <v1,o2||v4,v3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i v1* v2* o3 o4 <v1,v2||o4,o3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i v1* v2* o3 v4 <v1,v2||v4,o3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i v1* v2* v3 o4 <v1,v2||o4,v3> t2(a,b,i,j) 

    // starting string:
    //     - 0.06250 a* b* j i v1* v2* v3 v4 <v1,v2||v4,v3> t2(a,b,i,j) 

    // starting string:
    //     + 0.12500 o1* o2* o3 o4 a* i b* j <o1,o2||o4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 o1* o2* o3 v4 a* i b* j <o1,o2||v4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 o1* o2* v3 o4 a* i b* j <o1,o2||o4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 o1* o2* v3 v4 a* i b* j <o1,o2||v4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 o1* v2* o3 o4 a* i b* j <o1,v2||o4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 o1* v2* o3 v4 a* i b* j <o1,v2||v4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 o1* v2* v3 o4 a* i b* j <o1,v2||o4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 o1* v2* v3 v4 a* i b* j <o1,v2||v4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 v1* o2* o3 o4 a* i b* j <v1,o2||o4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 v1* o2* o3 v4 a* i b* j <v1,o2||v4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 v1* o2* v3 o4 a* i b* j <v1,o2||o4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 v1* o2* v3 v4 a* i b* j <v1,o2||v4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 v1* v2* o3 o4 a* i b* j <v1,v2||o4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 v1* v2* o3 v4 a* i b* j <v1,v2||v4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 v1* v2* v3 o4 a* i b* j <v1,v2||o4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     + 0.12500 v1* v2* v3 v4 a* i b* j <v1,v2||v4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i o1* o2* o3 o4 b* j <o1,o2||o4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i o1* o2* o3 v4 b* j <o1,o2||v4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i o1* o2* v3 o4 b* j <o1,o2||o4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i o1* o2* v3 v4 b* j <o1,o2||v4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i o1* v2* o3 o4 b* j <o1,v2||o4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i o1* v2* o3 v4 b* j <o1,v2||v4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i o1* v2* v3 o4 b* j <o1,v2||o4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i o1* v2* v3 v4 b* j <o1,v2||v4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i v1* o2* o3 o4 b* j <v1,o2||o4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i v1* o2* o3 v4 b* j <v1,o2||v4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i v1* o2* v3 o4 b* j <v1,o2||o4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i v1* o2* v3 v4 b* j <v1,o2||v4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i v1* v2* o3 o4 b* j <v1,v2||o4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i v1* v2* o3 v4 b* j <v1,v2||v4,o3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i v1* v2* v3 o4 b* j <v1,v2||o4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 a* i v1* v2* v3 v4 b* j <v1,v2||v4,v3> t1(a,i) t1(b,j) 

    // starting string:
    //     - 0.12500 b* j o1* o2* o3 o4 a* i <o1,o2||o4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j o1* o2* o3 v4 a* i <o1,o2||v4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j o1* o2* v3 o4 a* i <o1,o2||o4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j o1* o2* v3 v4 a* i <o1,o2||v4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j o1* v2* o3 o4 a* i <o1,v2||o4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j o1* v2* o3 v4 a* i <o1,v2||v4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j o1* v2* v3 o4 a* i <o1,v2||o4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j o1* v2* v3 v4 a* i <o1,v2||v4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j v1* o2* o3 o4 a* i <v1,o2||o4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j v1* o2* o3 v4 a* i <v1,o2||v4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j v1* o2* v3 o4 a* i <v1,o2||o4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j v1* o2* v3 v4 a* i <v1,o2||v4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j v1* v2* o3 o4 a* i <v1,v2||o4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j v1* v2* o3 v4 a* i <v1,v2||v4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j v1* v2* v3 o4 a* i <v1,v2||o4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     - 0.12500 b* j v1* v2* v3 v4 a* i <v1,v2||v4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i o1* o2* o3 o4 <o1,o2||o4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i o1* o2* o3 v4 <o1,o2||v4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i o1* o2* v3 o4 <o1,o2||o4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i o1* o2* v3 v4 <o1,o2||v4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i o1* v2* o3 o4 <o1,v2||o4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i o1* v2* o3 v4 <o1,v2||v4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i o1* v2* v3 o4 <o1,v2||o4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i o1* v2* v3 v4 <o1,v2||v4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i v1* o2* o3 o4 <v1,o2||o4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i v1* o2* o3 v4 <v1,o2||v4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i v1* o2* v3 o4 <v1,o2||o4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i v1* o2* v3 v4 <v1,o2||v4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i v1* v2* o3 o4 <v1,v2||o4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i v1* v2* o3 v4 <v1,v2||v4,o3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i v1* v2* v3 o4 <v1,v2||o4,v3> t1(b,j) t1(a,i) 

    // starting string:
    //     + 0.12500 b* j a* i v1* v2* v3 v4 <v1,v2||v4,v3> t1(b,j) t1(a,i) 

    // fully-contracted strings::
    //     + 1.00000 h(i,i) 
    //     + 1.00000 h(i,a) t1(a,i) 
    //     + 0.50000 <i,j||i,j> 
    //     + 1.00000 <i,j||a,j> t1(a,i) 
    //     + 0.25000 <i,j||a,b> t2(a,b,i,j) 
    //     + 0.50000 <i,j||a,b> t1(a,i) t1(b,j) 
