"""Regression test for the process-global spin/range blocking-state leak.

``block_by_spin`` / ``block_by_range`` set static flags on ``pq_string``
(``is_spin_blocked`` / ``is_range_blocked``) that were previously only reset by
``clear()``. A freshly constructed ``pq_helper`` must start unblocked -- otherwise a
prior blocking call on a *different* helper leaks in through the shared statics and
``strings()`` returns this helper's (empty) ``ordered_blocked`` instead of its real
equations, surfacing as "Empty terms for Equation". The ``pq_helper`` constructor now
resets both flags. Run: python -m pdaggerq.blocking_leak_test
"""
import pdaggerq


def _energy():
    """A small closed-shell equation with only summed labels (works for both
    spin and range blocking)."""
    pq = pdaggerq.pq_helper("fermi")
    pq.set_left_operators([["1"]])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    return pq


def test_spin_block_does_not_leak():
    a = _energy()
    a.block_by_spin({"i": "a", "j": "a", "a": "a", "b": "a"})
    assert len(a.strings()) > 0
    fresh = _energy()                                  # never blocked
    assert len(fresh.strings()) > 0, "is_spin_blocked leaked from a prior helper"
    print("test_spin_block_does_not_leak OK")


def test_range_block_does_not_leak():
    a = _energy()
    a.block_by_range({"i": ["all"], "j": ["all"], "a": ["all"], "b": ["all"]})
    assert len(a.strings()) > 0
    fresh = _energy()
    assert len(fresh.strings()) > 0, "is_range_blocked leaked from a prior helper"
    print("test_range_block_does_not_leak OK")


def test_fresh_after_block_matches_never_blocked():
    # the fresh (post-block) equation must equal the never-blocked one exactly
    ref = sorted(" ".join(t) for t in _energy().strings())
    poison = _energy()
    poison.block_by_spin({"i": "a", "j": "a", "a": "a", "b": "a"})
    poison.strings()
    after = sorted(" ".join(t) for t in _energy().strings())
    assert after == ref, (len(after), len(ref))
    print("test_fresh_after_block_matches_never_blocked OK")


if __name__ == "__main__":
    test_spin_block_does_not_leak()
    test_range_block_does_not_leak()
    test_fresh_after_block_matches_never_blocked()
    print("\nall blocking-leak tests passed")
