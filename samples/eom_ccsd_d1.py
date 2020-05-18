import sys
sys.path.insert(0, './..')

# elements of the one-electron reduced density matrix 
# at the EOM-CCSD level of theory: D(pq) = <0|(l0 + l1 + l2) e(-T) p*q e(T) (r0 + r1 + r2)|0> 

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("vacuum")
ahat.set_print_level(0)

left_hand_operators  = ['l0','l1(i1,a1)','l2(i1,i2,a1,a2)']
right_hand_operators = ['r0','r1(a3,i3)','r2(a3,a4,i3,i4)']

print('')
print('    D1(m,n) = <0|(l0 + l1 + l2) e(-T) e(m,n) e(T) (r0 + r1 + r2)|0>')
print('')

Dop = 'e(m,n)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
        # L [h, T2] R
        t2 = 't2(a5,a6,i5,i6)'
        ahat.add_operator_product( 1.0,[L,Dop,t2,R])
        ahat.add_operator_product(-1.0,[L,t2,Dop,R])
        
        # L [h, T1] R
        t1 = 't1(a5,i5)'
        ahat.add_operator_product( 1.0,[L,Dop,t1,R])
        ahat.add_operator_product(-1.0,[L,t1,Dop,R])

        # L [[e, T2], T2] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't2(a5,a6,i5,i6)'
        C = 't2(a7,a8,i7,i8)'
        ahat.add_operator_product( 1.0/2.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0/2.0,[L,C,B,A,R]) 

        # L [[e, T1], T2] + [[e, T2], T1] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't1(a5,i5)'
        C = 't2(a7,a8,i7,i8)'
        ahat.add_operator_product( 1.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0,[L,C,B,A,R]) 

        # L [[e, T1], T1] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't1(a5,i5)'
        C = 't1(a7,i7)'
        ahat.add_operator_product( 1.0/2.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0/2.0,[L,C,B,A,R]) 

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(e,f) = <0|(l0 + l1 + l2) e(-T) e(e,f) e(T) (r0 + r1 + r2)|0>')
print('')

Dop = 'e(e,f)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
        # L [h, T2] R
        t2 = 't2(a5,a6,i5,i6)'
        ahat.add_operator_product( 1.0,[L,Dop,t2,R])
        ahat.add_operator_product(-1.0,[L,t2,Dop,R])
        
        # L [h, T1] R
        t1 = 't1(a5,i5)'
        ahat.add_operator_product( 1.0,[L,Dop,t1,R])
        ahat.add_operator_product(-1.0,[L,t1,Dop,R])

        # L [[e, T2], T2] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't2(a5,a6,i5,i6)'
        C = 't2(a7,a8,i7,i8)'
        ahat.add_operator_product( 1.0/2.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0/2.0,[L,C,B,A,R]) 

        # L [[e, T1], T2] + [[e, T2], T1] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't1(a5,i5)'
        C = 't2(a7,a8,i7,i8)'
        ahat.add_operator_product( 1.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0,[L,C,B,A,R]) 

        # L [[e, T1], T1] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't1(a5,i5)'
        C = 't1(a7,i7)'
        ahat.add_operator_product( 1.0/2.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0/2.0,[L,C,B,A,R]) 

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(m,e) = <0|(l0 + l1 + l2) e(-T) e(m,e) e(T) (r0 + r1 + r2)|0>')
print('')

Dop = 'e(m,e)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
        # L [h, T2] R
        t2 = 't2(a5,a6,i5,i6)'
        ahat.add_operator_product( 1.0,[L,Dop,t2,R])
        ahat.add_operator_product(-1.0,[L,t2,Dop,R])
        
        # L [h, T1] R
        t1 = 't1(a5,i5)'
        ahat.add_operator_product( 1.0,[L,Dop,t1,R])
        ahat.add_operator_product(-1.0,[L,t1,Dop,R])

        # L [[e, T2], T2] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't2(a5,a6,i5,i6)'
        C = 't2(a7,a8,i7,i8)'
        ahat.add_operator_product( 1.0/2.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0/2.0,[L,C,B,A,R]) 

        # L [[e, T1], T2] + [[e, T2], T1] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't1(a5,i5)'
        C = 't2(a7,a8,i7,i8)'
        ahat.add_operator_product( 1.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0,[L,C,B,A,R]) 

        # L [[e, T1], T1] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't1(a5,i5)'
        C = 't1(a7,i7)'
        ahat.add_operator_product( 1.0/2.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0/2.0,[L,C,B,A,R]) 

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(e,m) = <0|(l0 + l1 + l2) e(-T) e(e,m) e(T) (r0 + r1 + r2)|0>')
print('')

Dop = 'e(e,m)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
        # L [h, T2] R
        t2 = 't2(a5,a6,i5,i6)'
        ahat.add_operator_product( 1.0,[L,Dop,t2,R])
        ahat.add_operator_product(-1.0,[L,t2,Dop,R])
        
        # L [h, T1] R
        t1 = 't1(a5,i5)'
        ahat.add_operator_product( 1.0,[L,Dop,t1,R])
        ahat.add_operator_product(-1.0,[L,t1,Dop,R])

        # L [[e, T2], T2] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't2(a5,a6,i5,i6)'
        C = 't2(a7,a8,i7,i8)'
        ahat.add_operator_product( 1.0/2.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0/2.0,[L,C,B,A,R]) 

        # L [[e, T1], T2] + [[e, T2], T1] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't1(a5,i5)'
        C = 't2(a7,a8,i7,i8)'
        ahat.add_operator_product( 1.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0,[L,C,B,A,R]) 

        # L [[e, T1], T1] R
        #[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
        A = Dop
        B = 't1(a5,i5)'
        C = 't1(a7,i7)'
        ahat.add_operator_product( 1.0/2.0,[L,A,B,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,B,A,C,R]) 
        ahat.add_operator_product(-1.0/2.0,[L,C,A,B,R]) 
        ahat.add_operator_product( 1.0/2.0,[L,C,B,A,R]) 

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

