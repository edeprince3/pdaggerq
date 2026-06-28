#!/usr/bin/env python
"""
Numerical test for density fitting of the cross-species electron-nuclear (gep) integral.

pq_graph density-fits same-species eri natively; this guards the NEO extension that also
factorizes the e-n integral g(p,P,q,Q) = (pq|PQ) = sum_Q' B_e(Q'|pq) B_p(Q'|PQ) with a
shared auxiliary basis (no exchange term -- different species do not antisymmetrize).

We build the integrals *exactly* from random DF factors, then evaluate the full mixed e-p
doubles residual with the DF codegen (3-index B only) and the non-DF codegen (4-index
eri/gep), and require they agree to machine precision.

Run: python neo_df_test.py
"""
import re
import numpy as np
import pdaggerq
from numpy import einsum

NAUX, NMOE, NMOP = 14, 4, 3
NOE = NVE = 2
NOP, NVP = 1, 2
S = {"o": [0, 1], "v": [2, 3], "O": [0], "V": [1, 2]}     # electron occ/vir, nuclear occ/vir


def _gen(df):
    pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([["e2(i,ni,a,na)"]])
    for h in ["v", "gep"]:
        pq.add_st_operator(1.0, [h], ["t2", "tep11"], True)
    pq.simplify()
    g = pdaggerq.pq_graph({"opt_level": 0, "density_fitting": df}); g.add(pq, "R"); g.optimize()
    return "\n".join(g.to_strings("python"))


def _run(code, df, Be, Bp, ERI, GEP, t2, t2ep, f, fp):
    ns = dict(np=np, einsum=einsum, scalars_={}, tmps_={}, perm_tmps={}, t2=t2, t2_ep=t2ep)
    ns["Id"] = {k: np.eye(len(S[k[0]])) for k in set(re.findall(r'Id\["([a-zA-Z]+)"\]', code))}
    ns["f"] = {k: (f if k[0] in "ov" else fp)[np.ix_(S[k[0]], S[k[1]])]
               for k in set(re.findall(r'f\["([a-zA-Z]+)"\]', code))}
    if df:
        ns["B"] = {k: (Be if k[1] in "ov" else Bp)[np.ix_(range(NAUX), S[k[1]], S[k[2]])]
                   for k in set(re.findall(r'B\["([A-Za-z]+)"\]', code))}
    else:
        ns["eri"] = {k: ERI[np.ix_(*[S[c] for c in k])] for k in set(re.findall(r'eri\["([a-zA-Z]+)"\]', code))}
        ns["g"] = {k: GEP[np.ix_(*[S[c] for c in k])] for k in set(re.findall(r'g\["([a-zA-Z]+)"\]', code))}
    body = "\n".join(l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#"))
    exec(body, ns)
    return ns["R"]


def test_gep_density_fitting():
    rng = np.random.default_rng(5)
    Be = rng.standard_normal((NAUX, NMOE, NMOE)); Be = Be + Be.transpose(0, 2, 1)
    Bp = rng.standard_normal((NAUX, NMOP, NMOP)); Bp = Bp + Bp.transpose(0, 2, 1)
    ERI = einsum("Qpr,Qqs->pqrs", Be, Be) - einsum("Qps,Qqr->pqrs", Be, Be)   # <pq||rs>
    GEP = einsum("Qpq,QPR->pPqR", Be, Bp)                                     # g(p,P,q,Q)
    t2 = rng.standard_normal((NVE, NVE, NOE, NOE)); t2 = t2 - t2.transpose(1, 0, 2, 3); t2 = t2 - t2.transpose(0, 1, 3, 2)
    t2ep = rng.standard_normal((NVE, NVP, NOP, NOE))
    f = np.diag(rng.standard_normal(NMOE)); fp = np.diag(rng.standard_normal(NMOP))
    R_nodf = _run(_gen(False), False, Be, Bp, ERI, GEP, t2, t2ep, f, fp)
    R_df = _run(_gen(True), True, Be, Bp, ERI, GEP, t2, t2ep, f, fp)
    err = np.abs(R_df - R_nodf).max()
    assert np.allclose(R_df, R_nodf, atol=1e-10), "gep DF != non-DF (max|diff|=%.2e)" % err
    print("OK  gep density fitting matches the 4-index integral to %.1e" % err)


if __name__ == "__main__":
    test_gep_density_fitting()
    print("PASS: NEO density fitting (cross-species gep)")
