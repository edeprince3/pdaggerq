
import sys
sys.path.insert(0, './..')

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

