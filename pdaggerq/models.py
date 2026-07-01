"""Coupled-cluster model library: the backend-agnostic *input* to code generation.

A model names its Hamiltonian, cluster amplitudes, and the conjugate projection
per amplitude. The builders turn a model into optimized ``pq_graph`` objects
(correlation energy and per-amplitude residuals) ready for ``to_strings("ir")`` /
``"c++"`` / ``"python"`` / latex -- the same model drives every backend. This is
the one canonical place each CC method's generation input lives; consumers
(e.g. neocc) select a model and add only their backend glue.

Excitation conventions
----------------------
Amplitudes are named by electron/proton excitation rank::

    t1..t4   electron 1..4-fold          tp1      proton single
    tep11    1 electron + 1 proton (the mixed "ep" double)
    tep21    2e1p  ("eep" triple)        tep31    3e1p  ("eeep" quadruple)

Truncation is by rank in the *combined* electron+proton Fock space. With a single
quantum proton the protonic rank is capped at one, so tp2 / tep12 / ... vanish and
are omitted.

Method families
---------------
* traditional electronic: ``ccd``, ``ccsd``, ``ccsdt``, ``ccsdtq``  (H = f, v)
* full NEO: ``neo-ccd``, ``neo-ccsd``, ``neo-ccsdt``, ``neo-ccsdtq`` -- electron CC
  + proton single + the full mixed hierarchy through that rank  (H = f, v, fp, gep)
* hybrid NEO: ``neo-ccd(ep)``, ``neo-ccsdt(eep)``, ``neo-ccsdtq(eeep)`` -- add one
  targeted mixed excitation (ep / eep / eeep) on top of a lower base *without* the
  matching pure-electron excitation. ``neo-ccsdt(eep)`` is the published Pavoševic
  cluster (no electron t3). ``neo-ccd(ep)`` is the minimal e-p model (tep11 only).

The runnable tutorial counterparts (raw pdaggerq API, with derivations) live in
``examples/`` -- e.g. ``ccsd.py``, ``ccsdt.py``, ``ccsdtq.py``, ``neo_ccd.py``,
``neo_ccsd.py``, ``neo_ccd_ep.py``. This module is their importable form, so a
consumer can ``from pdaggerq.models import residual_ir`` instead of scraping a
script; ``neo-ccsdt``/``neo-ccsdtq`` and the ``(eep)``/``(eeep)`` hybrids are
defined here only.
"""

import json

from ._pdaggerq import pq_helper, pq_graph
from . import einsums
from .spin import get_spin_labels

__all__ = [
    "Model", "MODELS", "PROJECTION", "EXCITATION", "H_ELEC", "H_NEO",
    "model", "lambda_amps",
    "energy_graph",
    "residual_graph", "residual_ir", "spin_cases", "residual_blocks",
    "lambda_graph", "lambda_ir",
    "gradient_graph", "gradient_ir",
    "rdm_graph", "rdm_ir",
    "energy_from_rdm_graph", "energy_from_rdm_ir",
    "orbital_gradient_graph", "orbital_gradient_ir",
    "orbital_hessian_graph", "orbital_hessian_ir", "orbital_hessian_diag_ir",
]

# Conjugate (de-excitation) projection per amplitude: all-occ then all-vir,
# electrons before the proton within each group (the proton carries pq_helper's
# nuclear 'n' prefix: ni/nj occ, na/nb vir).
PROJECTION = {
    "t1":    "e1(i,a)",
    "t2":    "e2(i,j,a,b)",
    "t3":    "e3(i,j,k,a,b,c)",
    "t4":    "e4(i,j,k,l,a,b,c,d)",
    "tp1":   "e1(ni,na)",
    "tep11": "e2(i,ni,a,na)",
    "tep21": "e3(i,j,ni,a,b,na)",
    "tep31": "e4(i,j,k,ni,a,b,c,na)",
}

# The excitation operator tau for each amplitude -- its projection with the occ and
# vir index halves swapped. Used to build Lambda residuals and orbital gradients.
EXCITATION = {
    "t1":    "e1(a,i)",
    "t2":    "e2(a,b,i,j)",
    "t3":    "e3(a,b,c,i,j,k)",
    "t4":    "e4(a,b,c,d,i,j,k,l)",
    "tp1":   "e1(na,ni)",
    "tep11": "e2(a,na,i,ni)",
    "tep21": "e3(a,b,na,i,j,ni)",
    "tep31": "e4(a,b,c,na,i,j,k,ni)",
}

H_ELEC = ("f", "v")
H_NEO = ("f", "v", "fp", "gep")  # single proton: nuclear-nuclear vp is a constant


class Model:
    """A CC model: Hamiltonian ``H`` and cluster amplitudes ``T`` (name tuples)."""

    def __init__(self, name, H, T):
        self.name = name
        self.H = tuple(H)
        self.T = tuple(T)

    def __repr__(self):
        return f"Model({self.name!r}, T={list(self.T)})"


def _m(name, H, T):
    return name, Model(name, H, T)


MODELS = dict([
    # --- traditional electronic CC ---
    _m("ccd",    H_ELEC, ["t2"]),
    _m("ccsd",   H_ELEC, ["t1", "t2"]),
    _m("ccsdt",  H_ELEC, ["t1", "t2", "t3"]),
    _m("ccsdtq", H_ELEC, ["t1", "t2", "t3", "t4"]),
    # --- full NEO CC: electron CC + proton single + full mixed hierarchy ---
    _m("neo-ccd",    H_NEO, ["t2", "tep11"]),
    _m("neo-ccsd",   H_NEO, ["t1", "t2", "tp1", "tep11"]),
    _m("neo-ccsdt",  H_NEO, ["t1", "t2", "t3", "tp1", "tep11", "tep21"]),
    _m("neo-ccsdtq", H_NEO, ["t1", "t2", "t3", "t4", "tp1", "tep11", "tep21", "tep31"]),
    # --- hybrid NEO: a targeted mixed excitation without the pure-electron one ---
    _m("neo-ccd(ep)",      H_NEO, ["tep11"]),
    _m("neo-ccsdt(eep)",   H_NEO, ["t1", "t2", "tp1", "tep11", "tep21"]),
    _m("neo-ccsdtq(eeep)", H_NEO, ["t1", "t2", "tp1", "tep11", "tep21", "tep31"]),
])


def model(name):
    """Look up a model by name (raises KeyError listing the valid names)."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")


def lambda_amps(name):
    """The de-excitation (Lambda) amplitude names of a model: leading t -> l
    (t2 -> l2, tp1 -> lp1, tep11 -> lep11, ...)."""
    return [a.replace("t", "l", 1) for a in model(name).T]


def _optimized(pq, label, df, opt_level):
    g = pq_graph({"opt_level": opt_level, "density_fitting": df})
    g.add(pq, label)
    g.optimize()
    return g


def energy_graph(name, df=True, opt_level=6):
    """Optimized pq_graph for the correlation energy ``<0| e^-T H e^T |0>``."""
    m = model(name)
    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]])
    for h in m.H:
        pq.add_st_operator(1.0, [h], list(m.T))
    pq.simplify()
    return _optimized(pq, "energy", df, opt_level)


def residual_graph(name, amplitude, df=True, opt_level=6, label="R",
                   spin_case=None, nuclear_spin="high-spin"):
    """Optimized pq_graph for the amplitude residual
    ``<proj(amplitude)| e^-T H e^T |0> = 0``.

    spin_case : None -> spin-orbital (no blocking, the default). Otherwise a spin
                block name from :func:`spin_cases` (e.g. "abab", or NEO "aa_n") --
                the equation is restricted to that block via ``block_by_spin``.
    nuclear_spin : "high-spin" (single nuclear channel) or "full" -- see
                :mod:`pdaggerq.spin`.
    """
    m = model(name)
    if amplitude not in m.T:
        raise ValueError(f"model {name!r} has no amplitude {amplitude!r}; T={list(m.T)}")
    if amplitude not in PROJECTION:
        raise KeyError(f"no projection defined for amplitude {amplitude!r}")
    pq = pq_helper("fermi")
    pq.set_left_operators([[PROJECTION[amplitude]]])
    for h in m.H:
        pq.add_st_operator(1.0, [h], list(m.T))
    pq.simplify()
    if spin_case is not None:
        cases = get_spin_labels([[PROJECTION[amplitude]]], nuclear_spin)
        if spin_case not in cases:
            raise ValueError(f"unknown spin_case {spin_case!r} for {amplitude!r}; "
                             f"choose from {sorted(cases)}")
        pq.block_by_spin(cases[spin_case])
    return _optimized(pq, label, df, opt_level)


def residual_ir(name, amplitude, df=True, opt_level=6, label="R",
                spin_case=None, nuclear_spin="high-spin"):
    """The amplitude residual as ``to_strings("ir")`` JSONL lines."""
    g = residual_graph(name, amplitude, df=df, opt_level=opt_level, label=label,
                       spin_case=spin_case, nuclear_spin=nuclear_spin)
    return g.to_strings("ir")


def spin_cases(amplitude, nuclear_spin="high-spin"):
    """The spin-block case names for an amplitude's residual, e.g. t2 ->
    ['aaaa','abab','bbbb']; NEO tep11 high-spin -> ['aa_n','bb_n']."""
    if amplitude not in PROJECTION:
        raise KeyError(f"no projection defined for amplitude {amplitude!r}")
    return sorted(get_spin_labels([[PROJECTION[amplitude]]], nuclear_spin))


def residual_blocks(name, amplitude, df=True, opt_level=6, label="R",
                    nuclear_spin="high-spin"):
    """``{spin_case: ir_lines}`` for every spin block of the amplitude's residual
    (the full unrestricted set). Spin-orbital is ``residual_ir(..., spin_case=None)``;
    a restricted (closed-shell) implementation uses the closed-shell subset of
    these blocks with the per-block Integrals factors supplied by the consumer."""
    return {c: residual_ir(name, amplitude, df=df, opt_level=opt_level, label=label,
                           spin_case=c, nuclear_spin=nuclear_spin)
            for c in spin_cases(amplitude, nuclear_spin)}


def lambda_graph(name, amplitude, df=True, opt_level=6, label="R"):
    """Optimized pq_graph for the Lambda residual ``<(1+L) [Hbar, tau_amplitude]>``,
    the equation whose root is the de-excitation amplitude for ``amplitude``."""
    m = model(name)
    if amplitude not in m.T:
        raise ValueError(f"model {name!r} has no amplitude {amplitude!r}; T={list(m.T)}")
    if amplitude not in EXCITATION:
        raise KeyError(f"no excitation operator defined for amplitude {amplitude!r}")
    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]] + [[l] for l in lambda_amps(name)])   # (1 + Lambda)
    tau = EXCITATION[amplitude]
    for h in m.H:
        pq.add_st_operator(1.0, [h, tau], list(m.T))                    # [Hbar, tau]
        pq.add_st_operator(-1.0, [tau, h], list(m.T))
    pq.simplify()
    return _optimized(pq, label, df, opt_level)


def lambda_ir(name, amplitude, df=True, opt_level=6, label="R"):
    """The Lambda residual as ``to_strings("ir")`` JSONL lines."""
    return lambda_graph(name, amplitude, df=df, opt_level=opt_level, label=label).to_strings("ir")


def gradient_graph(name, species, df=True, opt_level=6, label="R"):
    """Optimized pq_graph for the per-species orbital-rotation gradient
    ``<(1+L) [Hbar, E_ai - E_ia]>`` (species = "electron" or "proton"). The
    generalized-Fock orbital gradient neocc rotates the orbitals with."""
    if species not in ("electron", "proton"):
        raise ValueError(f"species must be 'electron' or 'proton', not {species!r}")
    m = model(name)
    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]] + [[l] for l in lambda_amps(name)])
    ai, ia = (("e1(a,i)", "e1(i,a)") if species == "electron"
              else ("e1(na,ni)", "e1(ni,na)"))
    for h in m.H:
        pq.add_st_operator(1.0, [h, ai], list(m.T))
        pq.add_st_operator(-1.0, [ai, h], list(m.T))
        pq.add_st_operator(-1.0, [h, ia], list(m.T))
        pq.add_st_operator(1.0, [ia, h], list(m.T))
    pq.simplify()
    return _optimized(pq, label, df, opt_level)


def gradient_ir(name, species, df=True, opt_level=6, label="R"):
    """The per-species orbital-rotation gradient as ``to_strings("ir")`` lines."""
    return gradient_graph(name, species, df=df, opt_level=opt_level, label=label).to_strings("ir")


def rdm_graph(name, operator, df=True, opt_level=6, label="D"):
    """Optimized pq_graph for a reduced-density-matrix block
    ``<(1+L) e^-T operator e^T>``.

    operator : a density-operator string -- ``e1(p,q)`` for a 1-RDM block,
               ``e2(p,q,s,r)`` for a 2-RDM block (note the last index pair is
               swapped, as in examples/ccsd_d2.py). The index letters pick occ/vir
               (o/v); an 'n' prefix picks the proton classes (O/V), same convention
               as gradient_graph's e1(na,ni). Examples: "e1(i,j)" (D_oo),
               "e1(a,b)" (D_vv), "e2(a,b,i,j)" (D_vvoo), "e1(ni,nj)" (proton D_OO),
               "e2(a,na,ni,i)" (mixed e-p 2-RDM block)."""
    m = model(name)
    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]] + [[l] for l in lambda_amps(name)])   # (1 + Lambda)
    pq.add_st_operator(1.0, [operator], list(m.T))
    pq.simplify()
    return _optimized(pq, label, df, opt_level)


def rdm_ir(name, operator, df=True, opt_level=6, label="D"):
    """An RDM block as ``to_strings("ir")`` JSONL lines."""
    return rdm_graph(name, operator, df=df, opt_level=opt_level, label=label).to_strings("ir")


def energy_from_rdm_graph(name, df=True, opt_level=6, label="E"):
    """Optimized pq_graph for the ground-state energy ``<H>`` written as the trace
    of the integrals against the RDMs -- ``E = h.D1 + (1/4)<pq||rs>.D2`` (electronic;
    pdaggerq expresses ``h`` as ``f`` minus its mean-field), plus ``f.D1_n`` and
    ``g.D2_ep`` for NEO. Built on the true vacuum with ``use_rdms`` so the density
    operators are left as the RDM tensors (``D1``, ``D2``, ``D1_n``, ``D2_ep``) in
    pdaggerq's own e1/e2 convention; neocc supplies those RDMs and the integrals and
    evaluates the contraction. The D2 index-order / sign / factor convention thus
    lives entirely in pdaggerq and matches :func:`rdm_graph`.

    Uses the model's Hamiltonian (``model(name).H``) so the energy is consistent
    with the residual/RDM equations. Returns the total ``<H>``; subtract the
    reference energy for the correlation contribution.
    """
    m = model(name)
    pq = pq_helper("true")
    pq.set_use_rdms(True)
    for h in m.H:
        pq.add_operator_product(1.0, [h])
    pq.simplify()
    return _optimized(pq, label, df, opt_level)


def energy_from_rdm_ir(name, df=True, opt_level=6, label="E"):
    """The energy-from-RDM contraction as ``to_strings("ir")`` JSONL lines."""
    return energy_from_rdm_graph(name, df=df, opt_level=opt_level, label=label).to_strings("ir")


# vir-occ rotation generators E_ai - E_ia per species. Row and column use distinct
# label sets so a Hessian block H_(row),(col) keeps four open indices.
_ROT_ROW = {"electron": ("e1(a,i)", "e1(i,a)"), "proton": ("e1(na,ni)", "e1(ni,na)")}
_ROT_COL = {"electron": ("e1(b,j)", "e1(j,b)"), "proton": ("e1(nb,nj)", "e1(nj,nb)")}


def orbital_gradient_graph(name, species="electron", df=True, opt_level=6, label="g"):
    """Fixed-RDM orbital-rotation gradient ``g_pq = <[H, E_pq - E_qp]>`` over the
    vir-occ block of ``species`` ("electron" or "proton") -- the antisymmetrized
    generalized Fock ``2(F_pq - F_qp)``. The RDMs are treated as given tensors (true
    vacuum + use_rdms) and contracted with the integrals; neocc supplies D1/D2 (and
    D1_n/D2_ep for NEO) and evaluates. For NEO the electron-proton coupling enters
    automatically through gep, so the electron gradient sees the mixed RDM and vice
    versa. Same e1/e2 convention as rdm_graph/energy_from_rdm."""
    if species not in _ROT_ROW:
        raise ValueError(f"species must be 'electron' or 'proton', not {species!r}")
    m = model(name)
    pq = pq_helper("true")
    pq.set_use_rdms(True)
    ai, ia = _ROT_ROW[species]
    for h in m.H:
        pq.add_commutator(1.0, [h], [ai])
        pq.add_commutator(-1.0, [h], [ia])
    pq.simplify()
    return _optimized(pq, label, df, opt_level)


def orbital_gradient_ir(name, species="electron", df=True, opt_level=6, label="g"):
    """The fixed-RDM orbital gradient as ``to_strings("ir")`` JSONL lines."""
    return orbital_gradient_graph(name, species, df=df, opt_level=opt_level, label=label).to_strings("ir")


def orbital_hessian_graph(name, row_species="electron", col_species=None,
                          df=True, opt_level=6, label="H"):
    """Fixed-RDM orbital Hessian block
    ``H_pq,rs = <[[H, E_pq - E_qp], E_rs - E_sr]>`` coupling ``row_species``
    rotations (indices p,q) to ``col_species`` rotations (r,s); ``col_species``
    defaults to ``row_species``. The RDMs are given tensors contracted with the
    integrals. For NEO, ``row_species != col_species`` is the electron-proton
    off-diagonal block. The diagonal preconditioner is the (p,q,p,q) slice of the
    same-species block -- neocc extracts it (pdaggerq sums repeated labels, so it
    cannot emit the rank-2 diagonal directly)."""
    if col_species is None:
        col_species = row_species
    if row_species not in _ROT_ROW or col_species not in _ROT_COL:
        raise ValueError("row/col species must be 'electron' or 'proton'")
    m = model(name)
    pq = pq_helper("true")
    pq.set_use_rdms(True)
    ai, ia = _ROT_ROW[row_species]
    bj, jb = _ROT_COL[col_species]
    for h in m.H:
        for x, sx in ((ai, 1.0), (ia, -1.0)):
            for y, sy in ((bj, 1.0), (jb, -1.0)):
                pq.add_double_commutator(sx * sy, [h], [x], [y])
    pq.simplify()
    return _optimized(pq, label, df, opt_level)


def orbital_hessian_ir(name, row_species="electron", col_species=None,
                       df=True, opt_level=6, label="H"):
    """The fixed-RDM orbital Hessian block as ``to_strings("ir")`` JSONL lines."""
    return orbital_hessian_graph(name, row_species, col_species,
                                 df=df, opt_level=opt_level, label=label).to_strings("ir")


# einsum-char relabel taking the column rotation indices onto the row (the diagonal)
_HESS_DIAG_RELABEL = {"electron": {"b": "a", "j": "i"}, "proton": {"B": "A", "J": "I"}}


def orbital_hessian_diag_ir(name, species="electron", df=True, label="h"):
    """The diagonal orbital-Hessian preconditioner ``h_pq = H_pq,pq`` over the
    vir-occ block of ``species``, as JSONL IR.

    pdaggerq sums repeated generator labels, so it cannot emit the rank-2 diagonal
    directly. Instead this generates the UNFUSED same-species Hessian (opt_level 0,
    one statement per term) and relabels the column rotation indices onto the row
    (electron b->a, j->i; proton B->A, J->I): the diagonal of a sum is the sum of
    the per-term diagonals, and each term drops to rank-2. Verified to reproduce
    diag(orbital_hessian_ir) to ~1e-14. The result is unfused (fine for a
    preconditioner built once per macro-iteration); neocc supplies D1/D2 + integrals."""
    if species not in _HESS_DIAG_RELABEL:
        raise ValueError(f"species must be 'electron' or 'proton', not {species!r}")
    colrow = _HESS_DIAG_RELABEL[species]
    full = einsums.parse_ir(
        orbital_hessian_ir(name, species, species, df=df, opt_level=0, label=label))
    out = []
    for st in full:
        st = dict(st)
        st["operands"] = [{**o, "indices": [colrow.get(l, l) for l in o["indices"]]}
                          for o in st["operands"]]
        t = st["target"]
        seen, cls = [], []
        for lab, c in zip((colrow.get(l, l) for l in t["indices"]), t["classes"]):
            if lab not in seen:            # dedupe [a,a,i,i] -> [a,i]
                seen.append(lab)
                cls.append(c)
        st["target"] = {**t, "indices": seen, "classes": cls}
        out.append(json.dumps(st))
    return out
