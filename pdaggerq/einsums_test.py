"""Unit tests for pdaggerq.einsums (the einsums dispatch over to_strings("ir")).

Pure-Python: feeds hand-built IR statements to the lowering, so it needs neither
the compiled _pdaggerq nor an einsums build. Run: python -m pdaggerq.einsums_test
"""

from pdaggerq import einsums


def _operand_cxx(name):
    if name in ("R", "t2_ep"):
        return ("param", name)
    if name.startswith("tmps_"):
        # tmps_["k"] -> tmp_k
        key = name.split('["')[1].rstrip('"]')
        return ("temp", "tmp_" + einsums.sanitize(key))
    base, key = name.split('["')
    return ("input", f'{base}.at("{key.rstrip(chr(34) + "]")}")')


DIM = {"o": "no", "v": "nv", "O": "nO", "V": "nV", "Q": "naux"}


def test_binary_leftfold():
    # one three-operand contraction: R[a,i] = 2 * B["Qov"][Q,a,i] B["Qoo"][Q,j,j?]...
    # use a clean 3-operand term:  R[a,i] += B[Q,a,b] * t[Q,b,c] * u[c,i]
    stmt = {
        "target": {"name": "R", "indices": ["a", "i"], "classes": ["v", "o"],
                   "is_intermediate": False},
        "is_assignment": False, "coeff": 2.0,
        "operands": [
            {"name": 'B["Qvv"]', "indices": ["Q", "a", "b"], "classes": ["Q", "v", "v"],
             "is_intermediate": False},
            {"name": 't["Qvv"]', "indices": ["Q", "b", "c"], "classes": ["Q", "v", "v"],
             "is_intermediate": False},
            {"name": 'u["vo"]', "indices": ["c", "i"], "classes": ["v", "o"],
             "is_intermediate": False},
        ],
    }
    lines = einsums.lower([stmt], _operand_cxx, DIM)
    einsum_calls = [l for l in lines if "einsum(" in l]
    # n=3 operands -> n-1 = 2 binary einsums
    assert len(einsum_calls) == 2, lines
    # every einsum is binary: exactly three Indices{...} groups (out, A, B)
    for c in einsum_calls:
        assert c.count("Indices{") == 3, c
    # one intermediate materialised for the fold, plus the final into &R
    assert any("create_zero_tensor" in l for l in lines)
    assert any("&R" in l for l in einsum_calls)
    # final coefficient applied once, on the last einsum
    assert "2.0" in einsum_calls[-1]
    print("test_binary_leftfold OK")


def test_split_repeats_diagonal():
    # proton trace: T[] = B["QOO"][Q,J,J] * Id["OO"][J,J]  (J summed, repeated)
    stmt = {
        "target": {"name": 'tmps_["s"]', "indices": [], "classes": [],
                   "is_intermediate": True},
        "is_assignment": True, "coeff": 1.0,
        "operands": [
            {"name": 'B["QOO"]', "indices": ["Q", "J", "J"], "classes": ["Q", "O", "O"],
             "is_intermediate": False},
            {"name": 'Id["OO"]', "indices": ["J", "J"], "classes": ["O", "O"],
             "is_intermediate": False},
        ],
    }
    ns, lab2cls = einsums.split_repeats(stmt)
    # the repeated summed label J is split into two distinct labels per operand
    b_idx = ns["operands"][0]["indices"]
    id_idx = ns["operands"][1]["indices"]
    assert b_idx[1] != b_idx[2], b_idx          # B[Q,J,K]
    assert id_idx[0] != id_idx[1], id_idx        # Id[J,K]
    assert b_idx[1:] == id_idx, (b_idx, id_idx)  # contraction labels line up
    # fresh label inherits the proton-occ class
    assert lab2cls[b_idx[2]] == "O"
    print("test_split_repeats_diagonal OK")


def test_variadic_is_not_emitted():
    # guard against regressing to a 3-operand variadic einsum (einsums is binary)
    stmt = {
        "target": {"name": "R", "indices": ["a"], "classes": ["v"], "is_intermediate": False},
        "is_assignment": True, "coeff": 1.0,
        "operands": [
            {"name": 'x["vv"]', "indices": ["a", "b"], "classes": ["v", "v"], "is_intermediate": False},
            {"name": 'y["vv"]', "indices": ["b", "c"], "classes": ["v", "v"], "is_intermediate": False},
            {"name": 'z["v"]', "indices": ["c"], "classes": ["v"], "is_intermediate": False},
        ],
    }
    for c in (l for l in einsums.lower([stmt], _operand_cxx, DIM) if "einsum(" in l):
        assert c.count("Indices{") == 3, f"non-binary einsum emitted: {c}"
    print("test_variadic_is_not_emitted OK")


def test_ttgt_ladder():
    # the fused B*B*t ladder: einsums would fall to the generic scalar path on the
    # second contraction; TTGT permute-aligns it so every einsum is a GEMM.
    #   R[a,i,A,I] += -1 * B["QVV"][Q,A,B] * B["Qvv"][Q,a,b] * t2_ep[b,B,I,i]
    stmt = {
        "target": {"name": "R", "indices": ["a", "i", "A", "I"],
                   "classes": ["v", "o", "V", "O"], "is_intermediate": False},
        "is_assignment": False, "coeff": -1.0,
        "operands": [
            {"name": 'B["QVV"]', "indices": ["Q", "A", "B"], "classes": ["Q", "V", "V"],
             "is_intermediate": False},
            {"name": 'B["Qvv"]', "indices": ["Q", "a", "b"], "classes": ["Q", "v", "v"],
             "is_intermediate": False},
            {"name": "t2_ep", "indices": ["b", "B", "I", "i"], "classes": ["v", "V", "O", "o"],
             "is_intermediate": False},
        ],
    }
    lines = einsums.lower([stmt], _operand_cxx, DIM)
    einsum_calls = [l for l in lines if "einsum(" in l]
    permutes = [l for l in lines if l.strip().startswith("permute(")]

    # two binary GEMMs (one per fold step), each strictly binary (3 Indices groups)
    assert len(einsum_calls) == 2, lines
    for c in einsum_calls:
        assert c.count("Indices{") == 3, c
    # first contraction (Q) needs no operand permute; the second contracts the
    # non-contiguous {b,B} -> at least one operand-align permute is emitted
    assert any("&a" in p or "&b" in p for p in permutes), lines
    # the result lands in GEMM order [.,.,I,i] != R's [a,i,A,I] -> a final
    # accumulating permute into R carries the coefficient
    last = lines[-1].strip()
    assert last.startswith("permute(") and "&R" in last and "-1.0" in last, last
    # the slow generic 2-operand contraction directly into R must NOT appear
    assert not any("&R" in c for c in einsum_calls), "ladder still writes R via generic einsum"
    print("test_ttgt_ladder OK")


def test_target_shape():
    # the residual R's rank + index classes are read straight from the IR, so a
    # consumer needn't hard-code t_rank / r_classes.
    stmts = [{
        "target": {"name": "R", "indices": ["a", "i", "A", "I"],
                   "classes": ["v", "o", "V", "O"], "is_intermediate": False},
        "is_assignment": True, "coeff": 1.0,
        "operands": [
            {"name": 'B["Qvo"]', "indices": ["Q", "a", "i"], "classes": ["Q", "v", "o"],
             "is_intermediate": False},
            {"name": 'B["QVO"]', "indices": ["Q", "A", "I"], "classes": ["Q", "V", "O"],
             "is_intermediate": False},
        ],
    }]
    rank, classes = einsums.target_shape(stmts, "R")
    assert rank == 4 and classes == ["v", "o", "V", "O"], (rank, classes)
    try:
        einsums.target_shape(stmts, "nope")
        assert False, "expected ValueError for a missing target"
    except ValueError:
        pass
    print("test_target_shape OK")


def test_single_operand():
    # bare-integral assignment R[a,i] = f["vo"][a,i] -> overwriting permute (cpref 0)
    s1 = {"target": {"name": "R", "indices": ["a", "i"], "classes": ["v", "o"],
                     "is_intermediate": False},
          "is_assignment": True, "coeff": 1.0,
          "operands": [{"name": 'f["vo"]', "indices": ["a", "i"], "classes": ["v", "o"],
                        "is_intermediate": False}]}
    code = lambda st: [l for l in einsums.lower([st], _operand_cxx, DIM)
                       if not l.strip().startswith("//")]   # drop the // [i] comment
    l1 = code(s1)
    assert len(l1) == 1 and l1[0].strip().startswith("permute(0.0,"), l1
    assert "&R" in l1[0] and "1.0" in l1[0]

    # antisymmetrizer permuted accumulate R[a,b,i,j] += -1 * P[b,a,i,j]
    s2 = {"target": {"name": "R", "indices": ["a", "b", "i", "j"],
                     "classes": ["v", "v", "o", "o"], "is_intermediate": False},
          "is_assignment": False, "coeff": -1.0,
          "operands": [{"name": 'tmps_["p"]', "indices": ["b", "a", "i", "j"],
                        "classes": ["v", "v", "o", "o"], "is_intermediate": True}]}
    l2 = code(s2)
    assert l2[0].strip().startswith("permute(1.0,"), l2   # cpref 1.0 = accumulate
    assert "-1.0" in l2[0]
    # no einsum/axpy for a unary op -- it's a permute
    assert not any("einsum(" in x for x in l1 + l2)
    print("test_single_operand OK")


def test_outer_product():
    # 2-RDM reference R[i,j,k,l] = Id[i,k] * Id[j,l]: a pure outer product (no
    # contracted index). einsums only treats it as an outer product when each
    # operand's indices are contiguous in the einsum output, so it must be emitted
    # into concatenated [i,k,j,l] order + a permute to the interleaved target
    # [i,j,k,l] -- NOT a single einsum straight into [i,j,k,l] (which einsums would
    # mis-handle: right only when a factor is 1x1).
    st = {"target": {"name": "R", "indices": ["i", "j", "k", "l"],
                     "classes": ["o", "o", "o", "o"], "is_intermediate": False},
          "is_assignment": True, "coeff": 1.0,
          "operands": [{"name": 'Id["oo"]', "indices": ["i", "k"], "classes": ["o", "o"],
                        "is_intermediate": False},
                       {"name": 'Id["oo"]', "indices": ["j", "l"], "classes": ["o", "o"],
                        "is_intermediate": False}]}
    lines = einsums.lower([st], _operand_cxx, DIM)
    einsum_calls = [l for l in lines if "einsum(" in l]
    permutes = [l for l in lines if l.strip().startswith("permute(")]
    assert len(einsum_calls) == 1, lines
    out_idx = einsum_calls[0].split("Indices{")[1].split("}")[0]
    assert out_idx == "index::i, index::k, index::j, index::l", out_idx   # A-block then B-block
    assert any("index::i, index::j, index::k, index::l" in p for p in permutes), permutes
    print("test_outer_product OK")


if __name__ == "__main__":
    test_binary_leftfold()
    test_split_repeats_diagonal()
    test_variadic_is_not_emitted()
    test_ttgt_ladder()
    test_target_shape()
    test_single_operand()
    test_outer_product()
    print("\nall einsums dispatch tests passed")
