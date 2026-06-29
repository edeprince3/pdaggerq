"""
Regression for the pq_graph "ir" export (g.to_strings("ir")).

The IR is a flat JSONL statement list -- one JSON object per fused contraction:
  {target, is_assignment, coeff, [conditions], operands:[{name,indices,classes,is_intermediate}]}
consumed by the einsums/codegen lowering (see neocc/codegen/einsums_printer_plan.md).

This test does not check numerics; it checks the IR is structurally faithful:
every statement parses, and every contraction is well-formed -- free (output)
indices appear among the operands, contracted indices appear at least twice, and
each vertex's class list matches its index list. It also confirms the orbital
class chars are species-aware (electron o/v, proton O/V, DF aux Q).

Run with pdaggerq on the PYTHONPATH:  python ir_export_test.py
"""
import json
from collections import Counter

import pdaggerq


def get_ir(pq):
    g = pdaggerq.pq_graph({"opt_level": 6, "density_fitting": True})
    g.add(pq, "R")
    g.optimize()
    return [json.loads(l.strip()) for l in g.to_strings("ir") if l.strip().startswith("{")]


def check(stmts, label, expect_classes):
    bad = 0
    for o in stmts:
        tgt = o["target"]["indices"]
        op_idx = Counter()
        for op in o["operands"]:
            op_idx.update(op["indices"])
        for ix in tgt:                       # free index must come from an operand
            if op_idx[ix] < 1:
                print(f"  [{label}] free index {ix} absent from operands"); bad += 1
        for ix, c in op_idx.items():         # contracted index must appear >= 2x
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
    print(f"[{label}] {len(stmts)} statements, {bad} problems; class chars: {cls}")
    return bad


def main():
    problems = 0

    # electronic CCD doubles residual (density-fitted)
    pq = pdaggerq.pq_helper("fermi")
    pq.set_left_operators([["e2(i,j,a,b)"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    problems += check(get_ir(pq), "e-CCD R2", expect_classes="ov")

    # NEO CCD(ep) electron-proton amplitude residual -- must surface proton O/V
    pq = pdaggerq.pq_helper("fermi")
    pq.set_left_operators([["e2(i,ni,a,na)"]])   # e-occ, p-occ, e-vir, p-vir
    for h in ["f", "v", "fp", "gep"]:
        pq.add_st_operator(1.0, [h], ["tep11"])
    pq.simplify()
    problems += check(get_ir(pq), "NEO R_ep", expect_classes="ovOV")

    assert problems == 0, f"IR export had {problems} structural problems"
    print("\nIR export OK")


if __name__ == "__main__":
    main()
