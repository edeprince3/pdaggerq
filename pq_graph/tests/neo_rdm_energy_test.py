#!/usr/bin/env python
"""
Numerical RDM<->energy consistency test for multicomponent (nuclear-electronic /
NEO) coupled cluster.

The reduced density matrices (built from the e1/e2 density operators) must
reconstruct the energy obtained directly from the t-amplitudes.  pdaggerq's own
use_rdms machinery fixes the contraction convention:

    v   :  E = -<p,i||q,i> D1(p,q)         + 1/4 <p,q||s,r> D2(p,q,s,r)
    vp  :  E = -<nP,nI||nQ,nI> D1_n(nP,nQ) + 1/4 <nP,nQ||nS,nR> D2_n(nP,nQ,nS,nR)
    gep :  E = +g(p,nP,q,nQ) D2_ep(nP,p,q,nQ)

For each two-body piece we evaluate the energy <0|(1+L) e^-T h e^T|0> from the
amplitudes and, independently, assemble the RDM blocks and contract them with
the integrals.  A correct set of RDMs reproduces the energy on ANY inputs.

Conventions that make this work (all verified):
  * same-species doubles amplitudes MUST be antisymmetric -- simplify() exploits
    that to combine terms (mixed t2_ep has no same-species antisymmetry);
  * the antisymmetrized eri carries full <pq||rs> = <rs||pq> symmetry;
  * e2(A,B,C,D) = <A^ B^ C D>, so D2(p,q,r,s)=<p^q^sr> comes from e2(p,q,s,r);
  * the e2 mixed 2-RDM orders cross-species creators <nP^ p^ ..> = -<p^ nP^ ..>,
    i.e. it is minus the use_rdms D2_ep (hence the overall sign on the gep term).
"""
import re
import itertools
import numpy as np
import pdaggerq

d = 2                                            # dimension of each orbital block
slc = {"o": slice(0, d), "v": slice(d, 2 * d), "O": slice(0, d), "V": slice(d, 2 * d)}
LAM = [["1"], ["l2"], ["lp2"], ["lep11"]]        # (1 + Lambda) for the energy functional
T = ["t2", "tp2", "tep11"]                        # NEO-CCD cluster


# ----- tiny interpreter for pdaggerq's contracted strings -------------------
def _is_nuc(l):  return len(l) > 1 and l[0] == "n"
def _base(l):    return l[1] if _is_nuc(l) else l[0]
def _space(l):
    occ = _base(l) in "ijklmno"
    return ("O" if occ else "V") if _is_nuc(l) else ("o" if occ else "v")


def _make_tensors(rng):
    def make_eri(n):                              # <pq||rs> with full physical symmetry
        c = rng.standard_normal((n, n, n, n))
        c = c + c.transpose(1, 0, 2, 3); c = c + c.transpose(0, 1, 3, 2)
        c = c + c.transpose(2, 3, 0, 1)
        phys = c.transpose(0, 2, 1, 3)
        return phys - phys.transpose(0, 1, 3, 2)
    def antisym(n):                               # doubles amplitude: antisym (vir,vir)&(occ,occ)
        a = rng.standard_normal((n, n, n, n))
        a = a - a.transpose(1, 0, 2, 3); return a - a.transpose(0, 1, 3, 2)
    eri_e, eri_p = make_eri(2 * d), make_eri(2 * d)
    gep = rng.standard_normal((2 * d,) * 4); gep = gep + gep.transpose(2, 3, 0, 1)
    amps = {n: antisym(d) for n in ("t2", "t2_n", "l2", "l2_n")}
    amps.update({n: rng.standard_normal((d, d, d, d)) for n in ("t2_ep", "l2_ep")})
    return eri_e, eri_p, gep, amps


def _evaluator(eri_e, eri_p, gep, amps):
    def tensor(name, idx):
        if name == "ERI":
            src = eri_p if _is_nuc(idx[0]) else eri_e
            return src[tuple(slc[_space(i)] for i in idx)]
        if name == "g":
            return gep[tuple(slc[_space(i)] for i in idx)]
        if name == "d":
            return np.eye(d)
        return amps[name]                         # block-shaped (d,d,d,d) amplitudes

    def parse(s):
        if s.startswith("<"):
            return "ERI", s[1:-1].replace("||", ",").split(",")
        m = re.match(r"([A-Za-z0-9_+\-]+)\(([^)]*)\)", s)
        return m.group(1), m.group(2).split(",")

    def eval_term(term, output):
        coeff = float(term[0]); perms = []; factors = []
        for f in term[1:]:
            (perms if f.startswith("P(") else factors).append(parse(f))
        perms = [p[1] for p in perms]
        letters = {}
        def lett(ix):
            return letters.setdefault(ix, chr(ord("a") + len(letters)))
        subs, ops = [], []
        for name, idx in factors:
            ops.append(tensor(name, idx)); subs.append("".join(lett(i) for i in idx))
        out = "".join(lett(i) for i in output)
        res = np.einsum(",".join(subs) + "->" + out, *ops, optimize=True) if factors else np.array(1.0)
        res = coeff * res
        for (i, j) in perms:                      # P(i,j): A -> A - A(i<->j) over output indices
            if i in output and j in output:
                ax = list(range(len(output))); pi, pj = output.index(i), output.index(j)
                ax[pi], ax[pj] = pj, pi
                res = res - res.transpose(ax)
        return res

    def evaluate(strings, output):
        total = np.zeros((d,) * len(output)) if output else 0.0
        for term in strings:
            total = total + eval_term(term, output)
        return total
    return evaluate


# ----- derivation helpers ---------------------------------------------------
def _gen(op):
    pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators(LAM)
    pq.add_st_operator(1.0, [op], T, True, 2); pq.simplify()
    return pq.strings()


def _labels(spaces):
    occ, vir = iter("ijkl"), iter("abcd")
    return [("n" if s in "OV" else "") + (next(occ) if s in "oO" else next(vir)) for s in spaces]


def _rdm(evaluate, rank, spaces):
    """Assemble D1[p,q] (rank 2) or D2[p,q,r,s] (rank 4, standard <p^q^sr>)."""
    D = np.zeros((2 * d,) * rank)
    for combo in itertools.product(*spaces):
        lab = _labels(combo)
        if rank == 2:
            op, out = "e1(%s,%s)" % tuple(lab), lab
        else:                                     # D2(p,q,r,s) via e2(p,q,s,r)
            op, out = "e2(%s,%s,%s,%s)" % (lab[0], lab[1], lab[3], lab[2]), lab
        st = _gen(op)
        if st:
            D[tuple(slc[c] for c in combo)] = evaluate(st, out)
    return D


# ----- the tests ------------------------------------------------------------
def _same_species(evaluate, op, eri, sp):
    E = float(evaluate(_gen(op), []))
    D1 = _rdm(evaluate, 2, [sp] * 2); D2 = _rdm(evaluate, 4, [sp] * 4)
    E_rdm = -np.einsum("piqi,pq->", eri[:, :d, :, :d], D1, optimize=True) \
            + 0.25 * np.einsum("pqsr,pqsr->", eri, D2, optimize=True)
    return E, E_rdm


def test_electron_rdm_energy():
    eri_e, eri_p, gep, amps = _make_tensors(np.random.default_rng(2026))
    ev = _evaluator(eri_e, eri_p, gep, amps)
    E, E_rdm = _same_species(ev, "v", eri_e, "ov")
    assert np.isclose(E, E_rdm), (E, E_rdm)
    print("OK  electron: E_v == -<p,i||q,i>D1 + 1/4<pq||sr>D2  (%.6f)" % E)


def test_nuclear_rdm_energy():
    eri_e, eri_p, gep, amps = _make_tensors(np.random.default_rng(2027))
    ev = _evaluator(eri_e, eri_p, gep, amps)
    E, E_rdm = _same_species(ev, "vp", eri_p, "OV")
    assert np.isclose(E, E_rdm), (E, E_rdm)
    print("OK  nuclear : E_vp == -<nP,nI||nQ,nI>D1_n + 1/4<..>D2_n  (%.6f)" % E)


def test_mixed_rdm_energy():
    eri_e, eri_p, gep, amps = _make_tensors(np.random.default_rng(2028))
    ev = _evaluator(eri_e, eri_p, gep, amps)
    E = float(ev(_gen("gep"), []))
    Dep = np.zeros((2 * d,) * 4)                  # Dep[nP,p,q,nQ] = <nP^ p^ nQ q> via e2(nP,p,nQ,q)
    for Pn, pe, qe, Qn in itertools.product("OV", "ov", "ov", "OV"):
        lab = _labels([Pn, pe, qe, Qn]); st = _gen("e2(%s,%s,%s,%s)" % (lab[0], lab[1], lab[3], lab[2]))
        if st:
            Dep[slc[Pn], slc[pe], slc[qe], slc[Qn]] = ev(st, lab)
    E_rdm = -np.einsum("pmqn,mpqn->", gep, Dep, optimize=True)   # minus: e2-density cross-species sign
    assert np.isclose(E, E_rdm), (E, E_rdm)
    print("OK  mixed   : E_gep == g(p,nP,q,nQ) D2_ep(nP,p,q,nQ)  (%.6f)" % E)


if __name__ == "__main__":
    test_electron_rdm_energy()
    test_nuclear_rdm_energy()
    test_mixed_rdm_energy()
    print("PASS: NEO RDMs reconstruct the energy (electron, nuclear, mixed)")
