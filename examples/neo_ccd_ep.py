"""
Multicomponent (nuclear-electronic orbital, NEO) CCD(ep) with pdaggerq.

CCD(ep) is the minimal electron-proton correlation method: the ONLY cluster
amplitude is the mixed electron-proton double t^{aA}_{iI} (tep11). Electrons and
protons each stay at the mean-field (NEO-HF) level; all correlation lives in the
e-p double. The residual is at most quadratic in tep11, so the method is very
cheap -- a natural correlation model to sit inside an orbital optimization.

    T = t^{aA}_{iI} a^{iI}_{aA}        (tep11 only -- no t2, tp2, or singles)

Notation (nuclear orbitals carry the 'n' prefix):
    electron  occ i,j   vir a,b        nuclear  occ ni,nj   vir na,nb
    g(i,ni,a,na)  electron-nuclear (attractive) two-body integral
    t2_ep / l2_ep / D2_ep   mixed amplitude / lambda / two-particle RDM

This script prints the four objects an orbital-optimized CCD(ep) needs:
    1. the correlation energy
    2. the amplitude residual            (solve for tep11)
    3. the lambda residual               (solve for lep11 -> relaxed densities)
    4. the mixed two-particle RDM D2_ep  (orbital-gradient ingredient)

Orbital optimization: the orbital gradient is the species-resolved generalized
Fock built from the relaxed RDMs (electron D1/D2, nuclear D1_n/D2_n, mixed
D2_ep). The mixed D2_ep enters BOTH the electron and the nuclear orbital
gradients, so the e-p correlation relaxes both sets of orbitals simultaneously --
which is exactly what makes OO-CCD(ep) attractive for proton densities.
"""
import pdaggerq

H = ["f", "v", "fp", "gep"]   # single quantum proton: electron Fock+ee, proton Fock, e-p (no vp)
T = ["tep11"]                 # the only amplitude
P = "e2(i,ni,a,na)"           # mixed e-p double projection (e-occ, p-occ, e-vir, p-vir)


def show(title, pq):
    pq.simplify()
    print(f"\n# {title}   ({len(pq.strings())} terms)")
    for term in pq.strings():
        print("   ", " ".join(term))


# 1. correlation energy:  <0| e^-T H e^T |0>
pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([["1"]])
for h in H:
    pq.add_st_operator(1.0, [h], T)
show("CCD(ep) energy", pq)

# 2. amplitude residual:  <aA,iI| e^-T H e^T |0> = 0  -> solve tep11
pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([[P]])
for h in H:
    pq.add_st_operator(1.0, [h], T)
show("CCD(ep) amplitude residual  R^{aA}_{iI}", pq)

# 3. lambda residual:  <0|(1+L)[ Hbar, e2(a,na,i,ni) ]|0> = 0  -> solve lep11
Pex = "e2(a,na,i,ni)"         # the conjugate excitation (vir,vir,occ,occ)
pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([["1"], ["lep11"]])
for h in H:
    pq.add_st_operator( 1.0, [h, Pex], T)
    pq.add_st_operator(-1.0, [Pex, h], T)
show("CCD(ep) lambda residual  L^{aA}_{iI}", pq)

# 4. mixed two-particle RDM:  D2_ep = <0|(1+L) e^-T a^iI_aA e^T|0>  (orbital-gradient ingredient)
pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([["1"], ["lep11"]])
pq.add_st_operator(1.0, [P], T)
show("CCD(ep) mixed 2-RDM  D2_ep", pq)
