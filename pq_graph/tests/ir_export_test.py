"""Regression for the pq_graph "ir" export (g.to_strings("ir")).

Two layers:

1. structural -- every statement parses, contractions are well-formed (free
   indices appear in operands, contracted indices >= 2x, classes match indices),
   no operand has an empty name, no unexpanded additions, and the orbital-class
   chars are species-aware (electron o/v, proton O/V, DF aux Q).

2. NUMERICAL EQUIVALENCE (the real coverage) -- interpret the IR with numpy and
   compare to pq_graph's own, FCI-validated, python backend on identical random
   tensors. Both come from the same optimize(), so any infidelity in the IR
   (a dropped/empty operand, a mis-routed intermediate, a lost permute) shows up
   as a mismatch. Run for BOTH df=True and df=False: the non-DF path is the one
   that produces inline (A+B) operands, which the IR must hoist into named temps.

Run with pdaggerq on the PYTHONPATH:  python ir_export_test.py
"""
import json
from collections import Counter

import numpy as np

import pdaggerq

SIZES = {"o": 2, "v": 3, "O": 4, "V": 5, "Q": 6, "L": 2}  # distinct -> catch transposes


def build(projection, hamiltonian, cluster, label, df):
    pq = pdaggerq.pq_helper("fermi")
    pq.set_left_operators([[projection]])
    for h in hamiltonian:
        pq.add_st_operator(1.0, [h], cluster)
    pq.simplify()
    g = pdaggerq.pq_graph({"opt_level": 6, "density_fitting": df})
    g.add(pq, label)
    g.optimize()
    ir = [json.loads(l.strip()) for l in g.to_strings("ir") if l.strip().startswith("{")]
    return ir, g.to_strings("python")


# ----------------------------------------------------------------------- structural
def check(stmts, label, expect_classes):
    bad = 0
    for o in stmts:
        tgt = o["target"]["indices"]
        op_idx = Counter()
        for op in o["operands"]:
            op_idx.update(op["indices"])
            if op["name"] == "":                 # the inline-linkage bug: must be hoisted
                print(f"  [{label}] empty-name operand on {o['target']['name']}"); bad += 1
        for ix in tgt:
            if op_idx[ix] < 1:
                print(f"  [{label}] free index {ix} absent from operands"); bad += 1
        for ix, c in op_idx.items():
            if ix not in tgt and c < 2:
                print(f"  [{label}] dangling contracted index {ix} (count {c})"); bad += 1
        for v in [o["target"], *o["operands"]]:
            if len(v["classes"]) != len(v["indices"]):
                print(f"  [{label}] classes/indices length mismatch on {v['name']}"); bad += 1
        if o.get("addition"):
            print(f"  [{label}] unexpanded addition on {o['target']['name']}"); bad += 1
    cls = sorted({c for o in stmts for v in [o["target"], *o["operands"]] for c in v["classes"]})
    missing = set(expect_classes) - set(cls)
    if missing:
        print(f"  [{label}] missing expected class chars {missing}"); bad += 1
    print(f"[{label}] {len(stmts)} statements, {bad} structural problems; classes: {cls}")
    return bad


# ------------------------------------------------------------------------ numerical
def random_inputs(ir, seed=7):
    rng = np.random.default_rng(seed)
    flat, nested = {}, {}
    for st in ir:
        for op in st["operands"]:
            if op["is_intermediate"] or op["name"] in flat:
                continue
            arr = rng.standard_normal(tuple(SIZES[c] for c in op["classes"]))
            flat[op["name"]] = arr
            if '["' in op["name"]:
                base, block = op["name"].split('["')
                nested.setdefault(base, {})[block.rstrip('"]')] = arr
            else:
                nested[op["name"]] = arr
    return flat, nested


def interp(ir, flat, label):
    store, sizes = {}, {}
    for st in ir:
        for op in st["operands"]:
            if not op["is_intermediate"]:
                for c, d in zip(op["classes"], flat[op["name"]].shape):
                    sizes.setdefault(c, d)
    for st in ir:
        t = st["target"]
        if st["is_assignment"] or t["name"] not in store:
            store[t["name"]] = np.zeros(tuple(sizes[c] for c in t["classes"]))
        subs = ",".join("".join(op["indices"]) for op in st["operands"])
        out = "".join(t["indices"])
        arrs = [store[op["name"]] if op["is_intermediate"] else flat[op["name"]]
                for op in st["operands"]]
        store[t["name"]] = store[t["name"]] + st["coeff"] * np.einsum(f"{subs}->{out}", *arrs, optimize=True)
    return store[label]


def run_python(py, nested, label):
    env = {"np": np, "einsum": np.einsum, "contract": np.einsum,
           "tmps_": {}, "perm_tmps": {}, "scalars_": {}, "reused_": {}}
    env.update(nested)
    keep = (label + " ", label + "=", "tmps_", "perm_tmps", "scalars_", "reused_", "del ")
    for raw in py:
        s = raw.strip()
        if s.startswith(keep):
            exec(s, env)
    return env[label]


def equivalence(ir, py, label):
    flat, nested = random_inputs(ir)
    err = float(np.max(np.abs(interp(ir, flat, label) - run_python(py, nested, label))))
    ok = err < 1e-12
    print(f"[{label}] IR-vs-python max error {err:.2e} {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- main
CASES = [
    ("e-CCD R2",  "e2(i,j,a,b)",   ["f", "v"],              ["t2"],    "R2", "ov"),
    ("NEO R_ep",  "e2(i,ni,a,na)", ["f", "v", "fp", "gep"], ["tep11"], "R",  "ovOV"),
]


def main():
    problems = 0
    for name, proj, H, T, label, classes in CASES:
        for df in (True, False):                          # non-DF exercises operand hoisting
            tag = f"{name} df={df}"
            ir, py = build(proj, H, T, label, df)
            problems += check(ir, tag, classes)
            problems += equivalence(ir, py, label)
    assert problems == 0, f"IR export had {problems} problems"
    print("\nIR export OK")


if __name__ == "__main__":
    main()
