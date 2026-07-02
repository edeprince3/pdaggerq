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
    "energy_from_rdm_ir",
    "orbital_gradient_ir",
    "orbital_hessian_ir", "orbital_hessian_diag_ir",
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


# --- explicit occ/vir-block RDM contractions -----------------------------------
# The true-vacuum use_rdms trace E = h.D1 + 1/2 g.D2 has *general* orbital indices
# (all MOs). pdaggerq has no "general" orbital class, so pq_graph collapses them to
# the virtual block and silently drops the occupied contributions. Instead we emit
# the trace explicitly, enumerating every orbital index over its species' occ/vir
# blocks: the sum over blocks *is* the full general sum, exactly (validated to
# machine precision against h.D1 + 1/2 g.D2). Each RDM/integral block carries proper
# o/v (electron) / O/V (proton) classes, so neocc sizes and slices them directly;
# the D2 index-order matches :func:`rdm_graph` (D2["pqsr"] = <p+ q+ r s>).
_RDM_LAB = {("e", "o"): "ijkl", ("e", "v"): "abcd", ("p", "o"): "IJKL", ("p", "v"): "ABCD"}
_RDM_CLS = {("e", "o"): "o", ("e", "v"): "v", ("p", "o"): "O", ("p", "v"): "V"}


def _emit_block_terms(terms, target):
    """Enumerate each contraction term over occ/vir blocks of its slots.

    ``terms``  : list of ``(coeff, species_per_slot, [(tensor, [slot_idx, ...]), ...])``
                 where ``species_per_slot`` is a string of ``e``/``p`` (one per slot)
                 and each operand references shared slot indices.
    ``target`` : ``(tensor, [slot_idx, ...])`` for the LHS (``[]`` for a scalar); its
                 slots are the open indices, block-enumerated so every block is emitted.
    """
    import itertools
    out = []
    seen_assign = set()
    for coeff, species, operands in terms:
        for combo in itertools.product("ov", repeat=len(species)):
            def vertex(nm, slots, intermediate=False):
                blk = "".join(_RDM_CLS[(species[s], combo[s])] for s in slots)
                lab = [_RDM_LAB[(species[s], combo[s])][s] for s in slots]
                cls = [_RDM_CLS[(species[s], combo[s])] for s in slots]
                name = f'{nm}["{blk}"]' if slots else nm
                return {"name": name, "indices": lab, "classes": cls, "is_intermediate": intermediate}
            tname, tslots = target
            tgt = vertex(tname, tslots) if tslots else \
                {"name": tname, "indices": [], "classes": [], "is_intermediate": False}
            key = tgt["name"]
            out.append(json.dumps({
                "target": tgt, "is_assignment": key not in seen_assign, "coeff": coeff,
                "operands": [vertex(nm, sl) for nm, sl in operands]}))
            seen_assign.add(key)
    return out


# E = h.D1 + 1/2 g.D2 (electron) [+ hp.D1_n proton one-body + gep.D2_ep e-p, for NEO].
# hp is the *bare* proton core (not the proton Fock) so no electron mean-field is
# double-counted; the e-p coupling is carried entirely by gep.D2_ep (factor 1).
_ENERGY_ELEC = [(1.0, "ee", [("h", [0, 1]), ("D1", [0, 1])]),
                (0.5, "eeee", [("g", [0, 1, 2, 3]), ("D2", [0, 1, 2, 3])])]
_ENERGY_NEO = _ENERGY_ELEC + [(1.0, "pp", [("hp", [0, 1]), ("D1_n", [0, 1])]),
                              (1.0, "epep", [("gep", [0, 1, 2, 3]), ("D2_ep", [1, 0, 2, 3])])]


def energy_from_rdm_ir(name, label="E"):
    """The ground-state energy ``<H> = h.D1 + 1/2 g.D2`` (+ NEO proton/e-p terms) as
    explicit occ/vir-block JSONL IR (see the block-contraction note above). Returns
    the total ``<H>``; subtract the reference energy for the correlation part."""
    is_neo = any(op in model(name).H for op in ("fp", "gep"))
    return _emit_block_terms(_ENERGY_NEO if is_neo else _ENERGY_ELEC, (label, []))


# vir-occ rotation generators E_ai - E_ia per species. Row and column use distinct
# label sets so a Hessian block H_(row),(col) keeps four open indices.
_ROT_ROW = {"electron": ("e1(a,i)", "e1(i,a)"), "proton": ("e1(na,ni)", "e1(ni,na)")}
_ROT_COL = {"electron": ("e1(b,j)", "e1(j,b)"), "proton": ("e1(nb,nj)", "e1(nj,nb)")}

# The orbital gradient/Hessian are, like the energy, integrals contracted with the
# RDMs -- but with the *general* internal indices from those contractions. We
# generate the bare-form commutators (H = h + 1/2 g, the mean-field-free operator
# equal to f+v), then block-enumerate each general internal index over occ/vir
# exactly as the energy does. The external rotation indices a,i (b,j) stay vir/occ,
# so the target is the physical vir-occ block. Validated block-sum == full
# contraction to ~1e-14. Pure-electron only: the NEO cross-species (gep) commutator
# mis-slots the e/p integral indices in pdaggerq, so NEO OO is not yet supported.
_OCC_LET, _VIR_LET = set("ijklmno"), set("abcdefgh")


def _classify_letter(letter):
    """(species, fixed-class) for a pdaggerq index letter; general -> fixed None."""
    species = "p" if letter.startswith("n") else "e"
    core = letter[1:] if species == "p" else letter
    fixed = "o" if core in _OCC_LET else "v" if core in _VIR_LET else None
    if species == "p" and fixed:
        fixed = fixed.upper()
    return species, fixed


# The gep (electron-proton) contribution to the ELECTRON gradient, hand-derived
# because pdaggerq's cross-species commutator mis-slots the integral/RDM indices.
# From <[gep, E_ai - E_ia]> with gep = sum gep(E,P,E',P') E_{E,E'} b+_P b_P' and the
# energy's convention D2_ep(P,E,E',P') = <a+_E a_E' b+_P b_P'>, four well-formed
# 2e2p terms (external a vir, i occ; internal electron p,q and proton np,nq).
# Validated to finite differences of E_ep = gep.D2_ep to ~1e-9.
_GEP_GRAD_TERMS = [
    "+1.0 g(p,np,a,nq) D2_ep(np,p,i,nq)",
    "-1.0 g(i,np,q,nq) D2_ep(np,a,q,nq)",
    "-1.0 g(p,np,i,nq) D2_ep(np,p,a,nq)",
    "+1.0 g(a,np,q,nq) D2_ep(np,i,q,nq)",
]

# The gep contribution to the electron-electron OO Hessian, hand-derived (like the
# gradient) as d^2 E_ep/dkappa^2 -- the fixed-RDM energy Hessian, symmetric by
# construction -- from the 2nd-order Taylor expansion of the rotated gep. Eight
# well-formed 2e2p "cross" terms plus eight Kronecker-delta terms (electron index
# summed, coeff -1/2). Validated: closed form == Taylor Hessian to ~1e-15, Taylor ==
# finite differences to ~1e-7.
_GEP_HESS_TERMS = [
    "+1.0 g(a,np,b,nq) D2_ep(np,i,j,nq)",
    "-1.0 g(a,np,j,nq) D2_ep(np,i,b,nq)",
    "-1.0 g(i,np,b,nq) D2_ep(np,a,j,nq)",
    "+1.0 g(i,np,j,nq) D2_ep(np,a,b,nq)",
    "+1.0 g(b,np,a,nq) D2_ep(np,j,i,nq)",
    "-1.0 g(b,np,i,nq) D2_ep(np,j,a,nq)",
    "-1.0 g(j,np,a,nq) D2_ep(np,b,i,nq)",
    "+1.0 g(j,np,i,nq) D2_ep(np,b,a,nq)",
    "-0.5 d(i,j) g(a,np,p,nq) D2_ep(np,b,p,nq)",
    "-0.5 d(i,j) g(b,np,p,nq) D2_ep(np,a,p,nq)",
    "-0.5 d(i,j) g(p,np,a,nq) D2_ep(np,p,b,nq)",
    "-0.5 d(i,j) g(p,np,b,nq) D2_ep(np,p,a,nq)",
    "-0.5 d(a,b) g(i,np,p,nq) D2_ep(np,j,p,nq)",
    "-0.5 d(a,b) g(j,np,p,nq) D2_ep(np,i,p,nq)",
    "-0.5 d(a,b) g(p,np,i,nq) D2_ep(np,p,j,nq)",
    "-0.5 d(a,b) g(p,np,j,nq) D2_ep(np,p,i,nq)",
]

# PROTON-row orbital gradient. Two pieces, both hand-derived (pdaggerq has no bare
# proton core, and its cross-species commutator mis-slots): (i) the proton
# generalized Fock <[h_p, E^p_naNi - E^p_niNa]>, structurally identical to the
# electron h gradient with electron->proton labels (h->hp, D1->D1_n); (ii) the gep
# piece rotating gep's PROTON slots, the proton analog of _GEP_GRAD_TERMS. Validated
# to finite differences to ~1e-9. External na (proton vir), ni (proton occ).
_HP_GRAD_TERMS = [
    "+1.0 h(np,na) D1(np,ni)",
    "-1.0 h(ni,np) D1(na,np)",
    "-1.0 h(np,ni) D1(np,na)",
    "+1.0 h(na,np) D1(ni,np)",
]
_GEP_PROTON_GRAD_TERMS = [
    "+1.0 g(p,np,q,na) D2_ep(np,p,q,ni)",
    "-1.0 g(p,ni,q,nq) D2_ep(na,p,q,nq)",
    "-1.0 g(p,np,q,ni) D2_ep(np,p,q,na)",
    "+1.0 g(p,na,q,nq) D2_ep(ni,p,q,nq)",
]

# gep contribution to the PROTON-proton OO Hessian (proton analog of _GEP_HESS_TERMS,
# rotating gep's proton slots). 8 cross terms + 8 delta terms; validated to ~1e-15
# vs the Taylor Hessian and ~1e-7 vs finite differences.
_GEP_PROTON_HESS_TERMS = [
    "+1.0 g(p,na,q,nb) D2_ep(ni,p,q,nj)",
    "-1.0 g(p,na,q,nj) D2_ep(ni,p,q,nb)",
    "-1.0 g(p,ni,q,nb) D2_ep(na,p,q,nj)",
    "+1.0 g(p,ni,q,nj) D2_ep(na,p,q,nb)",
    "+1.0 g(p,nb,q,na) D2_ep(nj,p,q,ni)",
    "-1.0 g(p,nb,q,ni) D2_ep(nj,p,q,na)",
    "-1.0 g(p,nj,q,na) D2_ep(nb,p,q,ni)",
    "+1.0 g(p,nj,q,ni) D2_ep(nb,p,q,na)",
    "-0.5 d(ni,nj) g(p,na,q,nq) D2_ep(nb,p,q,nq)",
    "-0.5 d(ni,nj) g(p,nb,q,nq) D2_ep(na,p,q,nq)",
    "-0.5 d(ni,nj) g(p,np,q,na) D2_ep(np,p,q,nb)",
    "-0.5 d(ni,nj) g(p,np,q,nb) D2_ep(np,p,q,na)",
    "-0.5 d(na,nb) g(p,ni,q,nq) D2_ep(nj,p,q,nq)",
    "-0.5 d(na,nb) g(p,nj,q,nq) D2_ep(ni,p,q,nq)",
    "-0.5 d(na,nb) g(p,np,q,ni) D2_ep(np,p,q,nj)",
    "-0.5 d(na,nb) g(p,np,q,nj) D2_ep(np,p,q,ni)",
]


def _relabel_e2p(term):
    """A pq.strings() electron term -> proton by prefixing every index with 'n'
    (a->na, i->ni, p->np, ...). Tensor names are unchanged; _block_resolve then tags
    the all-proton h/D1 as hp/D1_n and the delta as Id. Used for the proton core
    gradient/Hessian, whose 1-body algebra is identical to the electron h piece."""
    coeff, tensors = _parse_rdm_term(term)
    body = " ".join(f"{nm}({','.join('n' + l for l in idx)})" for nm, idx in tensors)
    return f"{coeff} {body}"


def _bare_H(name, species):
    """(operator, coeff) list for the mean-field-free Hamiltonian entering the
    ``species`` orbital-rotation commutator. Operators that commute with the rotation
    contribute nothing, so they may be omitted. The NEO gep contribution to the
    electron rotation is added separately (see ``_GEP_GRAD_TERMS``)."""
    if species == "electron":
        return [("h", 1.0), ("g", 0.5)]
    if species == "proton":
        raise NotImplementedError(
            "proton-species orbital gradient/Hessian needs the bare proton core "
            "(pdaggerq's 'fp' is the proton Fock and would double-count the e-p "
            "mean-field); electron-species is supported")
    raise ValueError(f"species must be 'electron' or 'proton', not {species!r}")


def _block_resolve(terms, target_name, target_letters):
    """Enumerate each bare-form term over occ/vir blocks of its general indices,
    keeping the fixed (external rotation) indices. Kronecker deltas ``d`` become the
    identity ``Id`` and vanish when their two indices land in different blocks. The
    einsum label is the pdaggerq letter (proton ``nX`` -> uppercase ``X``)."""
    import itertools
    out, seen = [], set()
    for term in terms:
        coeff, tensors = _parse_rdm_term(term)
        letters = {}
        for _, idx in tensors:
            for l in idx:
                letters.setdefault(l, _classify_letter(l))
        gen = [l for l, (sp, fx) in letters.items() if fx is None]
        for combo in itertools.product("ov", repeat=len(gen)):
            cls = {l: (fx if fx else (combo[gen.index(l)].upper() if sp == "p" else combo[gen.index(l)]))
                   for l, (sp, fx) in letters.items()}
            if any(nm == "d" and cls[idx[0]] != cls[idx[1]] for nm, idx in tensors):
                continue
            lab = {l: (l if len(l) == 1 else l[1:].upper()) for l in letters}

            def vtx(nm, idx):                                          # operand: block-suffixed
                # tag every integral/RDM by the species of its indices so neocc's
                # distinct tensors don't collide: all-electron keeps the base name,
                # all-proton -> hp/fp/gpp/D1_n/D2_n, mixed e-p -> gep/D2_ep. (pdaggerq
                # already tags D1_n/D2_ep; strip any suffix first, then re-tag.)
                if nm == "d":
                    out_nm = "Id"
                else:
                    base = nm[:-2] if nm.endswith("_n") else nm[:-3] if nm.endswith("_ep") else nm
                    if base in ("h", "f", "g", "D1", "D2"):
                        sp = {("p" if l.startswith("n") else "e") for l in idx}
                        out_nm = base if sp == {"e"} else \
                            {"h": "hp", "f": "fp", "g": "gpp", "D1": "D1_n", "D2": "D2_n"}[base] \
                            if sp == {"p"} else {"g": "gep", "D2": "D2_ep"}[base]
                    else:
                        out_nm = nm
                return {"name": f'{out_nm}["{"".join(cls[l] for l in idx)}"]',
                        "indices": [lab[l] for l in idx], "classes": [cls[l] for l in idx],
                        "is_intermediate": False}
            tgt = {"name": target_name, "indices": [lab[l] for l in target_letters],   # single block
                   "classes": [cls[l] for l in target_letters], "is_intermediate": False}
            out.append(json.dumps({
                "target": tgt, "is_assignment": tgt["name"] not in seen, "coeff": coeff,
                "operands": [vtx(nm, idx) for nm, idx in tensors]}))
            seen.add(tgt["name"])
    return out


def _parse_rdm_term(term):
    """A pq.strings() line -> (coeff, [(tensor, [index, ...]), ...])."""
    import re
    toks = term.split()
    return float(toks[0]), [(m.group(1), m.group(2).split(","))
                            for m in re.finditer(r"([A-Za-z_0-9]+)\(([^)]+)\)", " ".join(toks[1:]))]


def orbital_gradient_ir(name, species="electron", label="grad"):
    """Fixed-RDM orbital-rotation gradient ``g_ai = <[H, E_ai - E_ia]>`` over the
    vir-occ block of ``species`` -- the antisymmetrized generalized Fock -- as
    explicit occ/vir-block JSONL IR. The RDMs are given tensors (true vacuum +
    use_rdms) contracted with the integrals; neocc supplies D1/D2 (and gep/D2_ep for
    NEO) and evaluates. Electron species is supported for both electronic and NEO
    models (the e-p coupling gep is added via the hand-derived _GEP_GRAD_TERMS, since
    pdaggerq's cross-species commutator mis-slots the indices); proton rows are the
    follow-up. Same e1/e2 convention as rdm_graph/energy_from_rdm."""
    if species not in _ROT_ROW:
        raise ValueError(f"species must be 'electron' or 'proton', not {species!r}")
    is_neo = any(op in model(name).H for op in ("fp", "gep"))
    if species == "proton":                            # fully hand-derived (see terms)
        if not is_neo:
            raise ValueError("proton-species orbital gradient requires a NEO model")
        return _block_resolve(_HP_GRAD_TERMS + _GEP_PROTON_GRAD_TERMS, label, ["na", "ni"])
    ai, ia = _ROT_ROW["electron"]
    pq = pq_helper("true")
    pq.set_use_rdms(True)
    for op, c in _bare_H(name, "electron"):
        pq.add_commutator(c, [op], [ai])
        pq.add_commutator(-c, [op], [ia])
    pq.simplify()
    terms = [" ".join(t) for t in pq.strings()]
    if is_neo:
        terms += _GEP_GRAD_TERMS                        # NEO: hand-derived e-p coupling
    return _block_resolve(terms, label, ["a", "i"])


def orbital_hessian_ir(name, row_species="electron", col_species=None, label="H"):
    """Fixed-RDM orbital Hessian block
    ``H_ai,bj = <[[H, E_ai - E_ia], E_bj - E_jb]>`` (rows a,i; columns b,j) as
    explicit occ/vir-block JSONL IR. ``col_species`` defaults to ``row_species``.
    Electron same-species is supported for electronic and NEO models (the e-p coupling
    gep is added via the hand-derived _GEP_HESS_TERMS); the e-p cross block and proton
    rows are the follow-up (bare proton core)."""
    if col_species is None:
        col_species = row_species
    if row_species not in _ROT_ROW or col_species not in _ROT_COL:
        raise ValueError("row/col species must be 'electron' or 'proton'")
    if row_species != col_species:
        raise NotImplementedError("electron-proton cross Hessian block is the follow-up")
    is_neo = any(op in model(name).H for op in ("fp", "gep"))
    if row_species == "proton":                        # fully hand-derived (see terms)
        if not is_neo:
            raise ValueError("proton-species orbital Hessian requires a NEO model")
        ai, ia = _ROT_ROW["electron"]                  # h_p Hessian = electron h-only, relabeled
        bj, jb = _ROT_COL["electron"]
        pq = pq_helper("true")
        pq.set_use_rdms(True)
        for x, sx in ((ai, 1.0), (ia, -1.0)):
            for y, sy in ((bj, 1.0), (jb, -1.0)):
                pq.add_double_commutator(sx * sy, ["h"], [x], [y])
        pq.simplify()
        terms = [_relabel_e2p(" ".join(t)) for t in pq.strings()] + _GEP_PROTON_HESS_TERMS
        return _block_resolve(terms, label, ["na", "nb", "ni", "nj"])
    ai, ia = _ROT_ROW["electron"]
    bj, jb = _ROT_COL["electron"]
    pq = pq_helper("true")
    pq.set_use_rdms(True)
    for op, c in _bare_H(name, "electron"):
        for x, sx in ((ai, 1.0), (ia, -1.0)):
            for y, sy in ((bj, 1.0), (jb, -1.0)):
                pq.add_double_commutator(c * sx * sy, [op], [x], [y])
    pq.simplify()
    terms = [" ".join(t) for t in pq.strings()]
    if is_neo:
        terms += _GEP_HESS_TERMS                        # NEO: hand-derived e-p coupling
    return _block_resolve(terms, label, ["a", "b", "i", "j"])


# einsum-char relabel taking the column rotation indices onto the row (the diagonal)
_HESS_DIAG_RELABEL = {"electron": {"b": "a", "j": "i"}, "proton": {"B": "A", "J": "I"}}


def orbital_hessian_diag_ir(name, species="electron", label="hdiag"):
    """The diagonal orbital-Hessian preconditioner ``h_ai = H_ai,ai`` over the
    vir-occ block of ``species``, as explicit occ/vir-block JSONL IR.

    pdaggerq sums repeated generator labels, so it cannot emit the rank-2 diagonal
    directly. Instead this relabels the column rotation indices of the (unfused,
    block-resolved) same-species Hessian onto the row (electron b->a, j->i): the
    diagonal of a sum is the sum of the per-term diagonals, and each term drops to
    rank-2. Verified against diag(orbital_hessian_ir) to ~1e-14."""
    colrow = _HESS_DIAG_RELABEL[species]
    full = einsums.parse_ir(orbital_hessian_ir(name, species, species, label=label))
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
