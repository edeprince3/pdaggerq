import sys
sys.path.insert(0, './..')

# ccsd lambda equations
# L = <0| (1+L) e(-T) H e(T) |0>
# dL/dtu = <0| e(-T) H e(T) |u> + <0| L e(-T) H e(T) |u> - <0| L tu e(-T) H e(T) |0>

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")
ahat.set_print_level(0)

print('')
print('    0 = <0| e(-T) H e*m e(T)|0> + <0| L e(-T) [H, e*m] e(T)|0>')
print('')

#  <0| e(-T) H e*m e(T)|0>

ahat.set_left_operators(['1'])
ahat.set_right_operators(['1'])

ahat.add_st_operator(1.0,['h','e1(e,m)'],['t1','t2'])
ahat.add_st_operator(1.0,['g','e1(e,m)'],['t1','t2'])

# <0| L e(-T) [H,e*m] e(T)|0>

ahat.set_left_operators(['l1','l2'])

ahat.add_st_operator( 1.0,['h','e1(e,m)'],['t1','t2'])
ahat.add_st_operator( 1.0,['g','e1(e,m)'],['t1','t2'])

ahat.add_st_operator(-1.0,['e1(e,m)','h'],['t1','t2'])
ahat.add_st_operator(-1.0,['e1(e,m)','g'],['t1','t2'])

ahat.simplify()

# grab list of fully-contracted strings, then print
singles_residual_terms = ahat.fully_contracted_strings()
for my_term in singles_residual_terms:
    print(my_term)

ahat.clear()

