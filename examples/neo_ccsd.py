"""
Multicomponent (nuclear-electronic orbital, NEO) CCSD with pdaggerq.

A second *fermionic* particle species (e.g. a quantum proton) is treated
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
    tp<n> / lp<n>      nuclear cluster / lambda amplitude (n nuclear pairs)
    tep<ne><np>        mixed amplitude (ne electron pairs, np nuclear pairs)

Generated tensors are distinguished by species, e.g.
    t1 / t2            electron singles / doubles
    tp1                nuclear (proton) single
    tep11              mixed: one electron pair + one nuclear pair

Below: the full single-proton NEO-CCSD energy and amplitude residuals,
    H = f + v (electron Fock + ee)  +  fp + gep (proton Fock + e-p Coulomb)
    T = t1 + t2 (electron)  +  tp1 (proton single)  +  tep11 (mixed e-p).

A single quantum proton has one occupied nuclear orbital, so it cannot be
doubly excited: there is no tp2 and no nuclear-doubles residual.  For two or
more quantum nuclei, add the nuclear-nuclear operator 'vp' to H, 'tp2' to T,
and project the nuclear double e2(ni,nj,na,nb) as well.
"""
import pdaggerq

H = ["f", "v", "fp", "gep"]
T = ["t1", "t2", "tp1", "tep11"]

# CCSD needs the full 4th-order BCH: with singles present the two-body
# operators generate up to quadruple commutators (CCD, with a doubles-only
# cluster, is already exact at 2nd order).  The similarity-transformed
# two-body Hamiltonian truncates exactly at the 4-fold commutator, so
# max_order=4 is exact for CCSD.
MAXORD = 4

blocks = [
    ("CCSD energy                  E", "1"),
    ("electron singles    R1(a,i)",     "e1(i,a)"),
    ("electron doubles    R2(a,b,i,j)", "e2(i,j,a,b)"),
    ("proton   singles    R1(A,I)",     "e1(ni,na)"),
    ("mixed e-p           R(a,A,i,I)",  "e2(i,ni,a,na)"),
]

for name, proj in blocks:
    pq = pdaggerq.pq_helper("fermi")
    pq.set_left_operators([[proj]])
    # similarity-transform each term of H separately (a multi-element targets
    # list is an operator *product*, not a sum).
    for h in H:
        pq.add_st_operator(1.0, [h], T, True, MAXORD)
    pq.simplify()
    print(f"\n# {name}   ( <{proj}| e^-T H e^T |0> )")
    for term in pq.strings():
        print("   ", " ".join(term))
