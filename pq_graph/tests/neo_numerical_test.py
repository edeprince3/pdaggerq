#!/usr/bin/env python
"""
Numerical correctness test for multicomponent (nuclear-electronic / NEO) coupled
cluster support in pdaggerq + pq_graph.

It derives a mixed electron-nuclear doubles residual, optimizes it at opt_level 0
(faithful transcription) and opt_level 5 (substitution + merging), evaluates both
on random tensors, and checks they agree to floating-point precision -- two
correct factorizations of one symbolic expression must agree on ANY inputs, so a
nonzero gap would be a mis-factorization in the nuclear-aware optimizer.

All four orbital spaces (electron occ/vir, nuclear occ/vir) are given equal
dimension so every tensor block is the same shape; the factorization is an
algebraic identity, so random (non-symmetric) inputs are a valid probe.
"""
import re
import numpy as np
from numpy import einsum
import pdaggerq

def derive(opt_level, projection):
    pq = pdaggerq.pq_helper("fermi")
    pq.set_left_operators([[projection]])
    # NEO-CCD: similarity-transform each piece of H = f + fp + v + vp + gep
    # separately (a multi-element targets list would be an operator *product*).
    # max_order=2 is exact for a doubles residual and keeps generation tractable.
    for h in ["f", "fp", "v", "vp", "gep"]:
        pq.add_st_operator(1.0, [h], ["t2", "tp2", "tep11"], True, 2)
    pq.simplify()
    g = pdaggerq.pq_graph({"opt_level": opt_level})
    g.add(pq, "rep")
    g.optimize()
    return "\n".join(g.to_strings("python"))

def evaluate(code, tensors, d):
    sl = lambda c: slice(0, d) if c in "oO" else slice(d, 2 * d)
    f_full, g_full, eri_full = tensors["f_full"], tensors["g_full"], tensors["eri_full"]
    f   = {k: f_full[sl(k[0]), sl(k[1])] for k in set(re.findall(r'f\["([a-zA-Z]+)"\]', code))}
    g   = {k: g_full[sl(k[0]), sl(k[1]), sl(k[2]), sl(k[3])] for k in set(re.findall(r'g\["([a-zA-Z]+)"\]', code))}
    eri = {k: eri_full[sl(k[0]), sl(k[1]), sl(k[2]), sl(k[3])] for k in set(re.findall(r'eri\["([a-zA-Z]+)"\]', code))}
    ns = dict(np=np, einsum=einsum, f=f, g=g, eri=eri,
              Id={"oo": np.eye(d), "OO": np.eye(d), "vv": np.eye(d), "VV": np.eye(d)},
              t2=tensors["t2"], t2_n=tensors["t2_n"], t2_ep=tensors["t2_ep"],
              scalars_={}, tmps_={}, perm_tmps={})
    body = "\n".join(l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#"))
    exec(body, ns)
    return ns["rep"]

def main():
    d = 4
    rng = np.random.default_rng(20260627)
    tensors = dict(
        f_full=rng.standard_normal((2 * d, 2 * d)),
        eri_full=rng.standard_normal((2 * d, 2 * d, 2 * d, 2 * d)),  # same-species two-body (v, vp)
        g_full=rng.standard_normal((2 * d, 2 * d, 2 * d, 2 * d)),    # electron-nuclear two-body (gep)
        t2=rng.standard_normal((d, d, d, d)),
        t2_n=rng.standard_normal((d, d, d, d)),
        t2_ep=rng.standard_normal((d, d, d, d)),
    )
    blocks = {
        "electron doubles  <ee|": "e2(i,j,a,b)",
        "mixed e-p doubles  <ep|": "e2(i,ni,a,na)",
    }
    ok = True
    for name, proj in blocks.items():
        r0 = evaluate(derive(0, proj), tensors, d)
        r5 = evaluate(derive(5, proj), tensors, d)
        err = float(np.abs(r0 - r5).max())
        match = np.allclose(r0, r5)
        ok = ok and match
        print(f"{name}: opt0 vs opt5 max|diff| = {err:.2e}  {'OK' if match else 'FAIL'}")
    assert ok, "NEO optimizer changed the equations"
    print("PASS: nuclear-aware optimization preserves the NEO equations (both residual blocks)")

if __name__ == "__main__":
    main()
