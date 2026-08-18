
import pdaggerq

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
    for my_term in H00:
        print(my_term)

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
    for my_term in Hs0:
        print(my_term)

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
    for my_term in H0s:
        print(my_term)

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
    for my_term in Hd0:
        print(my_term)

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
    for my_term in H0d:
        print(my_term)

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
    for my_term in Hss:
        print(my_term)

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
    for my_term in Hsd:
        print(my_term)

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
    for my_term in Hds:
        print(my_term)

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
    for my_term in Hdd:
        print(my_term)

    pq.clear()

if __name__ == "__main__":
    main()
