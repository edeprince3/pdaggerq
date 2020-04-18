<img src="https://render.githubusercontent.com/render/math?math=p^{\dagger}q">

A code for bringing strings of creation / annihilation operators to normal order.

**Use cases (all with all alpha labels)**

1.  A single operator string:<img src="https://render.githubusercontent.com/render/math?math=0.5\ i^{\dagger}j l^{\dagger}k ">

Input:

    set sqstring  [ia*,ja,la*,ka]

    energy('pdaggerq')

Output:

    // starting string:
    //     + 1.00 IA* JA LA* KA 

    // normal-ordered strings:
    //     + 1.00 IA* KA d(JALA) 
    //     - 1.00 IA* LA* JA KA 

2.  a string involving a four-index tensor: <img src="https://render.githubusercontent.com/render/math?math=0.5\ i^{\dagger}j p^{\dagger}q^{\dagger}s r (pr|qs)">

Input:

    set sqfactor  0.5
    set sqtensor  [pa,ra,qa,sa]
    set sqstring  [ia*,ja,pa*,qa*,sa,ra]

    energy('pdaggerq')

Output:

    // starting string:
    //     + 0.50 IA* JA PA* QA* SA RA (PARA|QASA)

    // normal-ordered strings:
    //     - 0.50 IA* QA* RA SA (JARA|QASA)
    //     + 0.50 IA* PA* RA SA (PARA|JASA)
    //     - 0.50 IA* PA* QA* JA RA SA (PARA|QASA)

Note that the delta functions get gobbled up in this case and the relevant indices in the four-index tensor are modified. Note also that you can modify the constant factor (0.5 here; the default value is 1.0, as you can see in the first example).

3.  a string involving a two-index tensor: <img src="https://render.githubusercontent.com/render/math?math=0.5\ i^{\dagger}j p^{\dagger}q h_{pq}"> 

Input: 

    set sqfactor  0.5
    set sqtensor  [pa,qa]
    set sqstring  [ia*,ja,pa*,qa]

energy('pdaggerq')

Output:

    // starting string:
    //     + 0.50 IA* JA PA* QA h(PAQA)

    // normal-ordered strings:
    //     + 0.50 IA* QA h(JAQA)
    //     - 0.50 IA* PA* JA QA h(PAQA)

4. Multiple terms: <img src="https://render.githubusercontent.com/render/math?math=0.5\ [ k^{\dagger} l , [ p^{\dagger}q^{\dagger}s r, j^{\dagger}i] ] (pr|qs)"> 

Input:

    set sqfactor  0.5
    set sqtensor  [pa,ra,qa,sa]
    set sqstring  [ka*,la,pa*,qa*,sa,ra,ja*,ia]

    set sqfactor2 -0.5
    set sqtensor2 [pa,ra,qa,sa]
    set sqstring2 [ka*,la,ja*,ia,pa*,qa*,sa,ra]

    set sqfactor3 -0.5
    set sqtensor3 [pa,ra,qa,sa]
    set sqstring3 [pa*,qa*,sa,ra,ja*,ia,ka*,la]

    set sqfactor4 0.5
    set sqtensor4 [pa,ra,qa,sa]
    set sqstring4 [ja*,ia,pa*,qa*,sa,ra,ka*,la]

    energy('pdaggerq')

Output:


    // starting string:
    //     + 0.50 KA* LA PA* QA* SA RA JA* IA (PARA|QASA)

    // starting string:
    //     - 0.50 KA* LA JA* IA PA* QA* SA RA (PARA|QASA)

    // starting string:
    //     - 0.50 PA* QA* SA RA JA* IA KA* LA (PARA|QASA)

    // starting string:
    //     + 0.50 JA* IA PA* QA* SA RA KA* LA (PARA|QASA)

    // normal-ordered strings:
    //     - 0.50 KA* QA* IA SA (LAJA|QASA)
    //     + 0.50 KA* QA* IA RA (LARA|QAJA)
    //     + 0.50 KA* PA* IA SA (PAJA|LASA)
    //     - 0.50 KA* PA* IA RA (PARA|LAJA)
    //     + 0.50 KA* QA* RA SA d(JALA) (IARA|QASA)
    //     - 0.50 KA* PA* RA SA d(JALA) (PARA|IASA)
    //     + 0.50 JA* KA* RA SA (IARA|LASA)
    //     - 0.50 JA* KA* RA SA (LARA|IASA)
    //     + 0.50 PA* QA* LA SA d(IAKA) (PAJA|QASA)
    //     + 0.50 PA* QA* IA LA (PAJA|QAKA)
    //     - 0.50 PA* QA* LA RA d(IAKA) (PARA|QAJA)
    //     - 0.50 PA* QA* IA LA (PAKA|QAJA)
    //     - 0.50 JA* QA* LA SA (IAKA|QASA)
    //     + 0.50 JA* QA* LA RA (IARA|QAKA)
    //     + 0.50 JA* PA* LA SA (PAKA|IASA)
    //     - 0.50 JA* PA* LA RA (PARA|IAKA)

Note that all of the terms with > four operators canceled, so they are not printed
