"""
Multicomponent (nuclear-electronic orbital, NEO) CCD with pdaggerq.

A second *fermionic* particle species (e.g. a quantum proton) is supported
alongside the electrons. The two species share one fermionic algebra over
disjoint orbital spaces: operators of different species never contract and
commute (the standard NEO/tensor-product convention).

Notation
--------
Nuclear orbital indices carry the prefix 'n':
    electron occ  i,j,...   electron vir  a,b,...
    nuclear  occ  ni,nj,..  nuclear  vir  na,nb,..

Operators (electron operators are unchanged):
    fp                 nuclear (proton) Fock operator
    gep                electron-nuclear two-body Coulomb (non-antisymmetrized)
    tp<n>              nuclear cluster amplitude (n nuclear pairs)
    tep<ne><np>        mixed amplitude (ne electron pairs, np nuclear pairs)
Projections take explicit indices, so a mixed/nuclear bra is just e.g.
    e2(i,ni,a,na)      mixed electron-nuclear double
    e2(ni,nj,na,nb)    nuclear double

Generated tensors are distinguished by species:
    t2 / t2_n / t2_ep   electron / nuclear / mixed doubles
    f(i,j) / f(ni,nj)   electron / nuclear Fock
    g(i,ni,a,na)        electron-nuclear integral

Below: the two residual blocks of single-proton NEO-CCD,
H = f (electron Fock) + gep (e-p), cluster T = t2 (ee) + tep11 (ep).
(Add 'fp','v' to the Hamiltonian and 'tp2' to T for the general case; add the
nuclear-nuclear operator 'vp' only for two or more quantum nuclei.)
"""
import pdaggerq

H = ["f", "gep"]
T = ["t2", "tep11"]

for name, proj in [("electron doubles  R2(a,b,i,j)", "e2(i,j,a,b)"),
                   ("mixed e-p doubles  R(a,A,i,I)", "e2(i,ni,a,na)")]:
    pq = pdaggerq.pq_helper("fermi")
    pq.set_left_operators([[proj]])
    # similarity-transform each term of H separately (a multi-element targets
    # list is an operator *product*, not a sum); max_order=2 is exact for a
    # doubles residual and keeps generation fast for the full Hamiltonian.
    for h in H:
        pq.add_st_operator(1.0, [h], T, True, 2)
    pq.simplify()
    print(f"\n# {name}   ( <{proj}| e^-T H e^T |0> )")
    for term in pq.strings():
        print("   ", " ".join(term))
