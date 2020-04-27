<img src="https://render.githubusercontent.com/render/math?math=p^{\dagger}q">

A code for bringing strings of creation / annihilation operators to normal order.

**Notes**

1. We follow the usual convention for labeling orbitals: i, j, k, l, m, and n represent occupied orbitals; a, b, c, d, e, and f represent virtual orbitals; and p, q, r, and s (and any other label not included in the occupied or virtua sets) represent general orbital labels. Delta functions involving occupied / virtual combinations will be set to zero.

2. Orbital labels refer to spin orbitals. Spin labels may be included by adding "A" or "B" to an orbital label, but this is not necessary. Note that these spin labels must be capitalized in the current version of the code. Delta functions involving alpha / beta combinations will be set to zero.

3. Strings are defined in Python using the ahat_helper class, which has the following functions:

    set_string (required): set the string of creation and annihiliation operators
    
        set_string(['p*','q','a*','i'])
        
    set tensor (optional): define a one- or two-body tensor to accompany the string
    
        set_tensor(['p','q'])
        
    set amplitudes (optional): define t1 or t2 amplitudes to accompany the string. Note that up to four sets of amplitudes can be added to a given string.

        set_tensor(['a','i'])
        
    set_factor (optional): define a numerical factor to accompany the string. The default value is 1.0.
    
        set_factor(0.5)

    add_new_string (required): add string to list of strings to be brought to normal order
    
        add_new_string()
        
    bring_to_normal_order (required): bring all strings to normal order, consolidate/cancel terms, and zero any delta functions that involve occupied / virtual or alpha / beta combinations.
    
        bring_to_normal_order()

**Usage**

The following code evaluates the commutator 0.5 [[h, T1], T1], where h is a one-body operator

Python:

    import pdaggerq
    
    ahat = pdaggerq.ahat_helper()

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

    ahat.bring_to_normal_order()

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


