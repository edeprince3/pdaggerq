"""Coupled-cluster model library: the backend-agnostic *input* to code generation.

A model names its Hamiltonian, cluster amplitudes, and the conjugate projection
per amplitude. The builders turn a model into optimized ``pq_graph`` objects
(correlation energy and per-amplitude residuals) ready for ``to_strings("ir")`` /
``"c++"`` / ``"python"`` / latex -- the same model drives every backend. This is
the one canonical place each CC method's generation input lives; consumers
(e.g. neocc) select a model and add only their backend glue.

Excitation conventions
----------------------
Amplitudes are named by electron/proton excitation rank -- ``tp<M>`` is a pure
M-proton excitation, ``tep<N><M>`` is a mixed N-electron/M-proton one::

    t1..t4   electron 1..4-fold          tp1..tp4  proton 1..4-fold
    tep11    1e1p  ("ep" double)         tep21     2e1p  ("eep" triple)
    tep12    1e2p  ("epp" triple)        tep31     3e1p  ("eeep" quadruple)
    tep22    2e2p  ("eepp" quadruple)    tep13     1e3p  ("eppp" quadruple)

Truncation is by rank in the *combined* electron+proton Fock space, and the models
are **general in the number of quantum protons**: the proton doubles/triples/... and
the proton-proton fluctuation ``vp`` are included wherever the combined rank allows.

For a single quantum proton the proton *correlation* vanishes identically -- every
proton-rank >= 2 amplitude is antisymmetric in the one occupied proton, and ``vp``'s
two-body part annihilates the lone proton (two proton annihilators on a one-proton
state). ``vp`` still contributes its one-body mean-field fold, which cancels the
proton-proton mean field carried by the dressed proton Fock. So the single-proton
limit is the correct self-interaction-free NEO-CC **provided ``fp`` is the fully-
dressed Fock** -- it must include the proton-proton mean field that ``vp``'s fold
cancels, the same dressed-Fock convention already required for ``f`` (electron ``v``)
and the ``gep`` traces. (A consumer must therefore supply the p-p mean field in ``fp``
before adding ``vp``; the two must be kept consistent, exactly as for ``f``/``v``.)

Method families
---------------
* traditional electronic: ``ccd``, ``ccsd``, ``ccsdt``, ``ccsdtq``  (H = f, v)
  (the "proton" naming below is historical -- see *Charge convention*: the second
  quantized species may equally be a positron or a negative muon)
* full NEO: ``neo-ccd``, ``neo-ccsd``, ``neo-ccsdt``, ``neo-ccsdtq`` -- electron CC
  + the complete proton and mixed hierarchy through that combined rank
  (H = f, v, fp, gep, vp)
* hybrid NEO: ``neo-ccd(ep)``, ``neo-ccsdt(eep)``, ``neo-ccsdtq(eeep)`` -- a complete
  doubles base plus one targeted higher mixed excitation (ep / eep / eeep), *without*
  the matching pure-electron excitation. ``neo-ccsdt(eep)`` is the Pavoševic-style
  cluster (no electron t3). ``neo-ccd(ep)`` stays the minimal single-proton e-p model
  (tep11 only, no proton correlation).
* perturbative triples and quadruples: ``ccsd(t)``, ``neo-ccsd(t)``, ``ccsdt(q)``,
  ``neo-ccsdt(q)`` -- the CCSD (resp. CCSDT) cluster plus a non-iterative correction over
  every block one rank up. The perturbative blocks are listed in the model's ``T_pt``
  (never in ``T``): eee for ``ccsd(t)``, and eee/eep/epp/ppp for ``neo-ccsd(t)``; eeee
  for ``ccsdt(q)``, and eeee/eeep/eepp/eppp/pppp for ``neo-ccsdt(q)``. A block with n
  protons needs n quantum protons to be nonzero, so the ``-1p`` reductions keep only the
  0- and 1-proton blocks; a consumer gates the rest on its own particle count. Built by
  :func:`pt_amplitude_graph` and :func:`pt_energy_graph`, not the residual builders.
* single-proton NEO: every ``vp`` model has a ``<name>-1p`` counterpart (e.g.
  ``neo-ccsd-1p``) auto-derived by dropping ``vp`` and the >=2-proton amplitudes -- the
  exact equations for one quantum proton, and cheaper. Having no ``vp`` it takes the
  plain SI-free proton Fock (no dressing). A consumer dispatches on the proton count:
  1 -> ``-1p``, >=2 -> the full ``vp`` model.

  A consumer that gates amplitude blocks at runtime by particle count still cannot
  replace these by gating alone -- but it can with ``operators=`` (see below). The one- and many-proton problems have
  genuinely different Hamiltonians: with a single quantum proton there is no
  proton-proton interaction, so ``vp`` is absent from H entirely. Deleting the
  >=2-proton amplitudes from a ``vp`` model leaves every ``vp`` term standing, and those
  terms are a proton interacting with itself. Concretely, for ``neo-ccsd`` the proton
  singles residual with the >=2-proton amplitudes deleted still has 24 terms against
  ``neo-ccsd-1p``'s 20; the four extra are pure proton-proton, e.g.
  ``<nj,na||nb,ni> t1_n(nb,nj)``. The reduction is a change of Hamiltonian, not a
  truncation of the cluster (see :func:`_single_proton`).

  If you would rather gate everything at runtime than carry ``-1p`` variants, generate
  the Hamiltonian in parts instead: ``residual_ir(..., operators=(...))`` restricts H to
  a subset, and the parts are exactly additive. Emit ``vp`` separately, load it only at
  >=2 quantum protons, and one generated set with the full amplitude set covers every
  particle count. See :func:`residual_graph`.

Charge convention
-----------------
The derivation is **charge-independent**: nothing in the equations knows the second
species' charge or mass. Both enter only through the integrals the consumer supplies,
so the same generated code serves protons, positrons and negative muons.

With ``q_e = -1`` and ``Z_x`` the second species' charge in units of e::

    gep = q_e q_x V_ex = -Z_x V_ex     cross-species two-body (V = bare positive Coulomb)
    vp  = Z_x^2 V_xx                    same-species two-body (always repulsive)
    fp  = kinetic(m_x) + Z_x * (nuclear attraction) + mean fields

``gep`` carries **no built-in sign** -- it *is* the signed interaction. So a proton or a
positron (``Z_x = +1``) is fed ``gep = -V_ex`` (attractive), a negative muon
(``Z_x = -1``) is fed ``gep = +V_ex`` (repulsive), and ``|Z_x| != 1`` just scales. The
energy is ``E_ep = +gep.D2_ep`` (see :func:`energy_from_rdm_ir`), which also agrees in
sign with the hand-derived OO gradient/Hessian gep terms.

(Historically pq_helper's ``gep`` multiplied by ``-1``, hardcoding ``Z_x = +1``; that has
been removed, so a consumer that used to feed the bare positive Coulomb ``V_ex`` must now
feed ``-V_ex`` for a proton/positron.)

The runnable tutorial counterparts (raw pdaggerq API, with derivations) live in
``examples/`` -- e.g. ``ccsd.py``, ``ccsdt.py``, ``ccsdtq.py``, ``neo_ccd.py``,
``neo_ccsd.py``, ``neo_ccd_ep.py``. This module is their importable form, so a
consumer can ``from pdaggerq.models import residual_ir`` instead of scraping a
script; ``neo-ccsdt``/``neo-ccsdtq`` and the ``(eep)``/``(eeep)`` hybrids are
defined here only.
"""

import json
from fractions import Fraction

from ._pdaggerq import pq_helper, pq_graph
from . import einsums
from .spin import get_spin_labels

__all__ = [
    "Model", "MODELS", "PROJECTION", "EXCITATION", "H_ELEC", "H_NEO", "H_NEO_PP",
    "H_FLUCTUATION",
    "model", "lambda_amps", "pt_amps",
    "energy_graph",
    "residual_graph", "residual_ir", "spin_cases", "residual_blocks",
    "pt_amplitude_graph", "pt_amplitude_ir", "pt_energy_graph", "pt_energy_ir",
    "pt_lambda_numerator_graph", "pt_lambda_numerator_ir", "pt_lambda_source_ir",
    "lambda_graph", "lambda_ir",
    "gradient_graph", "gradient_ir",
    "hessian_graph", "hessian_ir",
    "rdm_graph", "rdm_ir", "rdm_block_ir", "pt_rdm_block_ir", "pt_rdm_graph",
    "pt_gradient_rdm_block_ir",
    "energy_from_rdm_ir", "rdm_energy_reference",
    "equations_graph", "equations_ir",
    "orbital_gradient_ir",
    "orbital_hessian_ir", "orbital_hessian_diag_ir", "orbital_sigma_ir",
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
    "tp2":   "e2(ni,nj,na,nb)",
    "tp3":   "e3(ni,nj,nk,na,nb,nc)",
    "tp4":   "e4(ni,nj,nk,nl,na,nb,nc,nd)",
    "tep11": "e2(i,ni,a,na)",
    "tep21": "e3(i,j,ni,a,b,na)",
    "tep31": "e4(i,j,k,ni,a,b,c,na)",
    "tep12": "e3(i,ni,nj,a,na,nb)",
    "tep22": "e4(i,j,ni,nj,a,b,na,nb)",
    "tep13": "e4(i,ni,nj,nk,a,na,nb,nc)",
}

# The excitation operator tau for each amplitude -- its projection with the occ and
# vir index halves swapped. Used to build Lambda residuals and orbital gradients.
EXCITATION = {
    "t1":    "e1(a,i)",
    "t2":    "e2(a,b,i,j)",
    "t3":    "e3(a,b,c,i,j,k)",
    "t4":    "e4(a,b,c,d,i,j,k,l)",
    "tp1":   "e1(na,ni)",
    "tp2":   "e2(na,nb,ni,nj)",
    "tp3":   "e3(na,nb,nc,ni,nj,nk)",
    "tp4":   "e4(na,nb,nc,nd,ni,nj,nk,nl)",
    "tep11": "e2(a,na,i,ni)",
    "tep21": "e3(a,b,na,i,j,ni)",
    "tep31": "e4(a,b,c,na,i,j,k,ni)",
    "tep12": "e3(a,na,nb,i,ni,nj)",
    "tep22": "e4(a,b,na,nb,i,j,ni,nj)",
    "tep13": "e4(a,na,nb,nc,i,ni,nj,nk)",
}

H_ELEC = ("f", "v")
# The one- and many-proton Hamiltonians are genuinely different operators, not a
# truncation of one another: a single quantum proton has no proton-proton interaction, so
# vp is absent rather than merely inactive. See _single_proton.
H_NEO    = ("f", "v", "fp", "gep")        # single-proton NEO (no proton-proton term)
H_NEO_PP = ("f", "v", "fp", "gep", "vp")  # + proton-proton fluctuation (multi-proton)


#: the two-body fluctuation operators -- the perturbation W of the (T) construction.
#: Everything else in a model's H (f, fp) is one-body and defines the zeroth order.
H_FLUCTUATION = ("v", "gep", "vp")


class Model:
    """A CC model: Hamiltonian ``H``, cluster amplitudes ``T``, and optionally the
    *perturbative* amplitudes ``T_pt`` (name tuples).

    ``T`` is solved iteratively. ``T_pt`` is not part of the cluster operator at all:
    each such amplitude is built once, non-iteratively, from the converged ``T`` and
    added to the energy -- the ``(T)`` of CCSD(T). See :func:`pt_amplitude_graph`."""

    def __init__(self, name, H, T, T_pt=()):
        self.name = name
        self.H = tuple(H)
        self.T = tuple(T)
        self.T_pt = tuple(T_pt)

    def __repr__(self):
        pt = f", T_pt={list(self.T_pt)}" if self.T_pt else ""
        return f"Model({self.name!r}, T={list(self.T)}{pt})"


def _m(name, H, T, T_pt=()):
    return name, Model(name, H, T, T_pt)


MODELS = dict([
    # --- traditional electronic CC ---
    _m("ccd",    H_ELEC, ["t2"]),
    _m("ccsd",   H_ELEC, ["t1", "t2"]),
    _m("ccsdt",  H_ELEC, ["t1", "t2", "t3"]),
    _m("ccsdtq", H_ELEC, ["t1", "t2", "t3", "t4"]),
    # --- full NEO CC: the complete electron + proton + mixed hierarchy through the
    #     combined excitation rank, general in the proton count. The proton doubles/
    #     triples/... and the proton-proton fluctuation vp vanish for a single proton,
    #     so single-proton results are unchanged. ---
    _m("neo-ccd",    H_NEO_PP, ["t2", "tp2", "tep11"]),
    _m("neo-ccsd",   H_NEO_PP, ["t1", "t2", "tp1", "tp2", "tep11"]),
    _m("neo-ccsdt",  H_NEO_PP, ["t1", "t2", "t3",
                                "tp1", "tp2", "tp3",
                                "tep11", "tep21", "tep12"]),
    _m("neo-ccsdtq", H_NEO_PP, ["t1", "t2", "t3", "t4",
                                "tp1", "tp2", "tp3", "tp4",
                                "tep11", "tep21", "tep31", "tep12", "tep22", "tep13"]),
    # --- hybrid NEO: complete doubles base + one targeted higher mixed excitation
    #     (eep / eeep), without the matching pure-electron excitation. neo-ccd(ep) is
    #     the minimal single-proton e-p model (tep11 only). ---
    _m("neo-ccd(ep)",      H_NEO,    ["tep11"]),
    _m("neo-ccsdt(eep)",   H_NEO_PP, ["t1", "t2", "tp1", "tp2", "tep11", "tep21"]),
    _m("neo-ccsdtq(eeep)", H_NEO_PP, ["t1", "t2", "tp1", "tp2", "tep11", "tep21", "tep31"]),
    # --- perturbative triples: the CCSD cluster plus a non-iterative triples
    #     correction over every rank-3 block -- eee (t3), eep (tep21), epp (tep12) and
    #     ppp (tp3). ppp needs three quantum protons to be nonzero and is expected to be
    #     numerically tiny (the protonic basis is small, and same-species proton
    #     correlation is weak), but it costs nothing to carry: it is dropped outright
    #     from the -1p reduction, and elsewhere it is the smallest block of the four.
    #     Leaving it out would be the only unbalanced omission in the rank-3 set. ---
    _m("ccsd(t)",     H_ELEC,   ["t1", "t2"], ["t3"]),
    _m("neo-ccsd(t)", H_NEO_PP, ["t1", "t2", "tp1", "tp2", "tep11"],
                                ["t3", "tep21", "tep12", "tp3"]),
    # --- perturbative quadruples: the same construction one rank up -- the CCSDT
    #     cluster plus a non-iterative correction over every rank-4 block: eeee (t4),
    #     eeep (tep31), eepp (tep22), eppp (tep13) and pppp (tp4). The many-proton
    #     blocks need that many quantum protons to be nonzero and are expected to be
    #     negligible for protons, but the multicomponent machinery is not
    #     proton-specific -- for a heavier or more numerous second species they need not
    #     be -- so the set is carried complete and a consumer gates each block on its
    #     own particle count (the -1p reduction already does exactly that, keeping only
    #     eeee and eeep). ---
    _m("ccsdt(q)",     H_ELEC,   ["t1", "t2", "t3"], ["t4"]),
    _m("neo-ccsdt(q)", H_NEO_PP, ["t1", "t2", "t3",
                                  "tp1", "tp2", "tp3",
                                  "tep11", "tep21", "tep12"],
                                 ["t4", "tep31", "tep22", "tep13", "tp4"]),
])


def _proton_count(amp):
    """Proton excitation rank encoded in an amplitude name: ``tp<M>`` -> M,
    ``tep<N><M>`` -> M, pure-electron ``t<n>`` -> 0."""
    if amp.startswith("tep"):
        return int(amp[4:])          # tep<N><M>: N at index 3, M is the remainder
    if amp.startswith("tp"):
        return int(amp[2:])
    return 0


def _single_proton(m):
    """The one-quantum-proton reduction of a NEO model: drop ``vp`` and every amplitude
    that needs >=2 protons.

    A proton-rank>=2 amplitude is antisymmetric in the one occupied proton and so
    vanishes, but ``vp`` is a different matter, and it is why this reduction exists as a
    separate model rather than as something a consumer can do by itself.

    **The Hamiltonian genuinely differs.** With one quantum proton there is no
    proton-proton interaction at all, so ``vp`` is not part of H. It is not enough to
    drop the >=2-proton amplitudes and keep the many-proton H: the ``vp`` terms do not
    follow the amplitudes out. They contract with the surviving proton singles and
    doubles and remain in the equations, where they describe the lone proton interacting
    with itself. For ``neo-ccsd``, deleting the >=2-proton amplitudes from the full model
    leaves the proton singles residual with 24 terms where ``neo-ccsd-1p`` has 20; the
    four extra are pure proton-proton (``<nj,na||nb,ni> t1_n(nb,nj)`` and friends).
    ``models_test.test_single_proton_models`` pins this.

    So a downstream code that detects the particle count cannot reach these equations by
    gating amplitudes -- it has to select the ``-1p`` model, which drops ``vp`` from H.
    Without ``vp`` the model also wants the plain SI-free proton Fock rather than the
    dressed one."""
    H = tuple(h for h in m.H if h != "vp")
    T = [a for a in m.T if _proton_count(a) <= 1]
    T_pt = [a for a in m.T_pt if _proton_count(a) <= 1]
    return Model(m.name + "-1p", H, T, T_pt)


# Register a "<name>-1p" single-proton counterpart for every model that actually carries
# vp (the multi-proton content). neo-ccd(ep) and the electronic models are already at or
# below one proton, so they gain nothing and are skipped.
for _mp in [m for m in list(MODELS.values()) if "vp" in m.H]:
    _sp = _single_proton(_mp)
    MODELS[_sp.name] = _sp


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


def pt_amps(name):
    """The perturbative (non-iterative) amplitude names of a model -- the ``(T)`` blocks.
    Empty for every model without a perturbative correction."""
    return list(model(name).T_pt)


#: default pq_graph optimization level for every generated equation. Full opt_level 6
#: (reordering / substitution / separation / pruning / merging / intermediate fusion).
#: History: the fusion pass used to be nondeterministic (hash-ordered LinkMerger
#: decisions) and its constant-scalar vertices were mis-emitted by the IR export, which
#: made opt6 output unreproducible and made consumers read wrong numbers -- generation
#: was capped at 5 while that stood (see edeprince3/pdaggerq#114). Both defects are
#: fixed (canonical fusion ordering in fusion.cc; constant folding in ir_emit), opt6 is
#: byte-reproducible and validated against opt_level 0 numerically, and
#: ``models_test.test_opt_level_safe_default`` guards exactly that -- if it trips,
#: re-cap to 5 here.
_SAFE_OPT_LEVEL = 6


def _opt_level_for(name, opt_level):
    """Resolve the pq_graph optimization level for a model. ``opt_level=None`` (the
    default for every generated equation) returns :data:`_SAFE_OPT_LEVEL`. An explicit
    ``opt_level`` always wins."""
    return _SAFE_OPT_LEVEL if opt_level is None else opt_level


#: Representative line-class sizes for the dimension-aware optimizer cost model
#: (pq_graph option "dims"). pq_graph's default metric counts every summation line
#: equally, so it mis-ranks NEO candidates badly: a nuclear-occupied line (O = number of
#: quantum protons, often 1!) is scored like an electronic virtual line. With dims set,
#: candidates are ranked by numeric flop estimates at these sizes instead, so the
#: contraction order removes the largest indices first.
#:
#: The line classes (pq_graph ``Line::type()``) are::
#:
#:     o   electron occupied           O   nuclear occupied  (per-model, see _dims_for)
#:     v   electron virtual            V   nuclear virtual
#:     L   excited-state / trial-vector index (sigma builds)
#:     Q   density-fitting auxiliary
#:
#: The values are taken from a representative NEO target -- FHF- with the electrons in
#: aug-cc-pVTZ and the quantum proton in PB4-F1::
#:
#:     electrons  2x46 (F) + 23 (H) = 115 spatial; 20 electrons
#:                -> o = 20 spin orbitals, v = 2*115 - 20 = 210
#:     proton     PB4-F1 = 4s3p2d1f = 30 functions, one proton
#:                -> O = 1, V = 29
#:     DF aux     aug-cc-pVTZ-RI, ~3.5x the orbital basis -> ~400
#:
#: so the ratios that matter are v/o ~ 10 and V/v ~ 0.14: the second species' basis is
#: MUCH smaller than the electronic virtual space, and smaller still for muons and
#: positrons. Codegen is frozen per model, so the decisions are optimal near these
#: ratios; a consumer with different ones should pass its own via the ``dims`` argument
#: (see :func:`_dims_for`).
#:
#: The values are distinct primes ON PURPOSE. Equal sizes make the metric blind: with the
#: previous table V and o were both 10, so any two candidates differing only by swapping
#: an electron-occupied line for a nuclear-virtual one scored identically and the
#: optimizer fell back to arbitrary tie-breaking. Distinct primes go further -- by unique
#: factorization, two different multisets of index classes cannot share a cost -- so a
#: tie in this metric means a genuine tie. L was also 1, which made the trial-vector
#: dimension multiplicatively free and dropped it out of sigma-build ranking entirely.
#:
#: "O" is filled in per-model by :func:`_dims_for` (the model's proton rank), so it is
#: physical rather than prime and can coincide with another class; it is the one place
#: an accidental tie remains possible. Electron-only models keep pq_graph's scale-safe
#: default metric (no dims) unless a caller passes dims explicitly.
#: KNOWN LIMITATION: o and V are both 10, so the metric cannot tell an electron-occupied
#: line from a nuclear-virtual one and any two candidates differing only by that swap tie
#: exactly. L = 1 likewise makes the trial-vector dimension multiplicatively free, so it
#: drops out of sigma-build ranking. Both are real defects in principle. Neither was worth
#: changing in practice: at the FHF- sizes above, breaking the tie (V = 11) moved the
#: neo-ccsd residual set by 0.04%, and a table anchored to the true ratios above was
#: neutral-to-worse (it also made cross-equation CSE marginally worse for neo-ccd(ep)).
#: The optimizer is a greedy search, so a more accurate metric does not monotonically
#: produce better code -- feeding it the exact FHF- dimensions made the tep11 residual
#: 28% MORE expensive by binary-contraction flop count than these defaults do.
#: So: if the ratios matter for your system, measure with your own ``dims`` rather than
#: trusting either these defaults or a "more realistic" table.
DIMS = {"o": 10.0, "v": 40.0, "V": 10.0, "L": 1.0, "Q": 120.0}


def _dims_for(name, dims=None):
    """Dimension table for the optimizer cost model: :data:`DIMS` with the nuclear
    occupied size set to the model's proton count, or None (dimension-blind legacy
    metric) for electron-only models.

    ``dims`` overrides entries of the table, e.g. ``{"V": 4.0}`` for a small protonic
    basis or ``{"v": 400.0}`` for a large electronic one. Partial dicts are merged over
    the defaults. Passing ``dims`` for an electron-only model turns the cost model on for
    it as well.

    Why it matters: the second species' basis is typically far smaller than the
    electronic one -- markedly so for muons and positrons -- so the cheapest contraction
    order removes the large electronic indices first. The optimizer can only see that if
    it is told the sizes. The defaults here are representative, not yours; supplying the
    real ones is worth a further few percent, and more as the ratio gets more extreme."""
    m = model(name)
    protons = max((_proton_count(a) for a in m.T + m.T_pt), default=0)
    if protons == 0 and dims is None:
        return None
    table = {**DIMS, "O": float(protons)} if protons else dict(DIMS)
    if dims is not None:
        unknown = [k for k in dims if k not in DIMS and k != "O"]
        if unknown:
            raise ValueError(f"unknown dimension class(es) {unknown}; "
                             f"known: {sorted(set(DIMS) | {'O'})}")
        table.update({k: float(v) for k, v in dims.items()})
    return table


def _optimized(pq, label, df, opt_level, dims=None, gep_traces=True):
    return _optimized_multi([(label, pq)], df, opt_level, dims, gep_traces)


def _optimized_multi(labeled_pqs, df, opt_level, dims=None, gep_traces=True):
    # nthreads=-1: run the optimizer on all available cores. Its output used to depend on
    # thread count -- structurally-equal candidate intermediates differ only in their generic
    # labels, and the dedup kept a thread-order-dependent representative, so multithreaded
    # codegen was not byte-reproducible (neocc freezes it), which forced a single-thread pin.
    # That is now fixed at the source (linkage_set keeps the canonical representative; the
    # printer backend is pinned during optimization so candidate ordering does not depend on
    # the last-emitted format), so codegen is byte-identical at any thread count -- verified
    # by models_test.test_opt_level_safe_default. The optimizer is ~85% of codegen time, so
    # this is the dominant speedup.
    options = {"opt_level": opt_level, "density_fitting": df, "nthreads": -1}
    if dims is not None:
        options["dims"] = dims  # dimension-aware candidate ranking (see DIMS)
    g = pq_graph(options)
    for label, pq in labeled_pqs:
        # Normal-order gep: the NEO integral dumps carry the dressed NEO-HF Fock (f/fp
        # include the e-p mean field), so the one-body reference traces of gep must not
        # appear explicitly in the equations or they double-count that field (a nonzero
        # singles residual at t=0). No-op for non-NEO.
        #
        # NOT for the orbital-rotation gradient (gep_traces=False; see gradient_graph):
        # dropping trace-carrying terms does NOT commute with taking the commutator --
        # removing them FROM <[H, E-]> is not the same as forming <[H - T, E-]> -- so it
        # does not yield the derivative of anything. With the removal the NEO gradient
        # disagreed with the finite-difference-verified orbital_gradient_ir (electron rel
        # 0.41, proton 0.73); without it the two routes agree to ~5e-16 for BOTH species.
        # The gradient therefore lives in the same (no-removal) convention as the RDM
        # energy that energy_from_rdm_ir traces and that orbital_gradient_ir differentiates.
        if gep_traces:
            pq.remove_gep_reference_traces()
        g.add(pq, label)
    g.optimize()
    return g


def _hamiltonian(m, operators):
    """Validate and resolve an operator subset against a model's Hamiltonian."""
    if operators is None:
        return tuple(m.H)
    ops = tuple(operators)
    unknown = [h for h in ops if h not in m.H]
    if unknown:
        raise ValueError(f"model {m.name!r} has no Hamiltonian operator(s) {unknown}; "
                         f"H={list(m.H)}")
    return ops


def _projected_pq(m, left, operators=None):
    """pq_helper holding ``<left| e^-T H e^T |0>`` for a model, simplified.

    ``operators`` restricts H to a subset of the model's. The similarity transform is
    applied to each operator separately and is linear in H, so the contributions are
    exactly additive: generating a subset and its complement and summing the two
    reproduces the full equation term for term."""
    pq = pq_helper("fermi")
    pq.set_left_operators([left])
    for h in _hamiltonian(m, operators):
        pq.add_st_operator(1.0, [h], list(m.T))
    pq.simplify()
    return pq


def energy_graph(name, df=True, opt_level=None, operators=None, dims=None):
    """Optimized pq_graph for the correlation energy ``<0| e^-T H e^T |0>``.

    ``operators`` restricts H to a subset -- see :func:`residual_graph`."""
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    pq = _projected_pq(m, ["1"], operators)
    return _optimized(pq, "energy", df, opt_level, _dims_for(name, dims))


def residual_graph(name, amplitude, df=True, opt_level=None, label="R",
                   spin_case=None, nuclear_spin="high-spin", operators=None, dims=None):
    """Optimized pq_graph for the amplitude residual
    ``<proj(amplitude)| e^-T H e^T |0> = 0``.

    spin_case : None -> spin-orbital (no blocking, the default). Otherwise a spin
                block name from :func:`spin_cases` (e.g. "abab", or NEO "aa_n") --
                the equation is restricted to that block via ``block_by_spin``.
    nuclear_spin : "high-spin" (single nuclear channel) or "full" -- see
                :mod:`pdaggerq.spin`.
    operators : restrict H to a subset of the model's (default: all of it).

    **Splitting the Hamiltonian.** The similarity transform is applied to each operator
    separately and is linear in H, so operator contributions are exactly additive:
    generating a subset and its complement and summing the two reproduces the full
    equation. Each part is optimized on its own, so its intermediates are private to it
    and a consumer can load or drop a part wholesale -- no provenance tracking needed.

    The motivating case is ``vp``. A consumer that gates amplitude blocks at runtime by
    particle count still cannot drop the proton-proton terms that way: ``vp`` is part of
    H, not of the cluster, so it survives every amplitude gate (this is exactly why the
    ``-1p`` models exist -- see :func:`_single_proton`). Nor is it identifiable in the
    emitted code: density-fitted ``vp`` is proton-B x proton-B where ``gep`` is
    electron-B x proton-B, so no rule over tensor names separates them once the optimizer
    has folded them into intermediates. Generating the two parts separately does::

        base = residual_ir("neo-ccsd", "tp1", operators=("f", "v", "fp", "gep"))
        vp   = residual_ir("neo-ccsd", "tp1", operators=("vp",))

    Load ``base`` always and ``vp`` only at >=2 quantum protons, and one generated set
    serves every particle count -- with the full amplitude set, gated at runtime.
    """
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    if amplitude not in m.T:
        raise ValueError(f"model {name!r} has no amplitude {amplitude!r}; T={list(m.T)}")
    if amplitude not in PROJECTION:
        raise KeyError(f"no projection defined for amplitude {amplitude!r}")
    pq = _projected_pq(m, [PROJECTION[amplitude]], operators)
    if spin_case is not None:
        cases = get_spin_labels([[PROJECTION[amplitude]]], nuclear_spin)
        if spin_case not in cases:
            raise ValueError(f"unknown spin_case {spin_case!r} for {amplitude!r}; "
                             f"choose from {sorted(cases)}")
        pq.block_by_spin(cases[spin_case])
    return _optimized(pq, label, df, opt_level, _dims_for(name, dims))


def residual_ir(name, amplitude, df=True, opt_level=None, label="R",
                spin_case=None, nuclear_spin="high-spin", operators=None, dims=None):
    """The amplitude residual as ``to_strings("ir")`` JSONL lines."""
    g = residual_graph(name, amplitude, df=df, opt_level=opt_level, label=label,
                       spin_case=spin_case, nuclear_spin=nuclear_spin, operators=operators,
                       dims=dims)
    return g.to_strings("ir")


def equations_graph(name, df=True, opt_level=None):
    """One optimized pq_graph holding the correlation energy AND every amplitude
    residual of a model. pq_graph's subexpression elimination scores candidate
    intermediates across all equations it holds, so intermediates common to several
    residuals (dressed one-body contractions, shared ladders, ...) are built once and
    reused -- unlike the per-equation builders above, which re-derive them in every
    equation. Use this to generate a model's full ground-state iteration workload.

    Equation labels (the IR target names): ``energy`` for the correlation energy and
    ``R_<amp>`` for each amplitude residual (e.g. ``R_t2``, ``R_tep11``). Spin-orbital
    only (no spin blocking)."""
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    labeled = [("energy", _projected_pq(m, ["1"]))]
    for amp in m.T:
        if amp not in PROJECTION:
            raise KeyError(f"no projection defined for amplitude {amp!r}")
        labeled.append((f"R_{amp}", _projected_pq(m, [PROJECTION[amp]])))
    return _optimized_multi(labeled, df, opt_level, _dims_for(name))


def equations_ir(name, df=True, opt_level=None):
    """The full ground-state equation set (energy + every residual, intermediates
    shared across equations) as ``to_strings("ir")`` JSONL lines. Targets are named
    ``energy`` and ``R_<amp>`` -- see :func:`equations_graph`."""
    return equations_graph(name, df=df, opt_level=opt_level).to_strings("ir")


def spin_cases(amplitude, nuclear_spin="high-spin"):
    """The spin-block case names for an amplitude's residual, e.g. t2 ->
    ['aaaa','abab','bbbb']; NEO tep11 high-spin -> ['aa_n','bb_n']."""
    if amplitude not in PROJECTION:
        raise KeyError(f"no projection defined for amplitude {amplitude!r}")
    return sorted(get_spin_labels([[PROJECTION[amplitude]]], nuclear_spin))


def residual_blocks(name, amplitude, df=True, opt_level=None, label="R",
                    nuclear_spin="high-spin"):
    """``{spin_case: ir_lines}`` for every spin block of the amplitude's residual
    (the full unrestricted set). Spin-orbital is ``residual_ir(..., spin_case=None)``;
    a restricted (closed-shell) implementation uses the closed-shell subset of
    these blocks with the per-block Integrals factors supplied by the consumer."""
    return {c: residual_ir(name, amplitude, df=df, opt_level=opt_level, label=label,
                           spin_case=c, nuclear_spin=nuclear_spin)
            for c in spin_cases(amplitude, nuclear_spin)}


def _fluctuation(m):
    """The two-body (perturbation) part of a model's Hamiltonian, in H order."""
    return [h for h in m.H if h in H_FLUCTUATION]


def _pt_pq(m, left, amps):
    """pq_helper holding ``<left| [W, amps] |0>``, first order in the two-body
    fluctuation W and in ``amps``. Both perturbative equations are this shape."""
    pq = pq_helper("fermi")
    pq.set_left_operators(left)
    for w in _fluctuation(m):
        for a in amps:
            pq.add_commutator(1.0, [w], [a])
    pq.simplify()
    return pq


def pt_amplitude_graph(name, amplitude, df=True, opt_level=None, label="R"):
    """Optimized pq_graph for the NUMERATOR of a perturbative amplitude -- the ``(T)``
    driver ``<proj(amplitude)| [W, T] |0>``, first order in the two-body fluctuation
    ``W`` (:data:`H_FLUCTUATION`) and in the converged cluster ``T``.

    The amplitude is that numerator over the orbital-energy denominator. The sign is
    RANK-DEPENDENT -- do not carry the rank-3 form over to rank 4::

        t = -rsign * R / D        D = sum(e_vir) - sum(e_occ)   (positive)

    with ``rsign`` the PROJECTION reversal parity ``(-1)^sum_species floor(rank_s / 2)``,
    the same parity the cluster amplitudes carry. It is -1 for EVERY rank-3 block, which
    is why ``(T)`` reads ``t = R / D`` throughout, and that simple form is rank-3 ONLY.
    At rank 4 the blocks whose excitation rank is even in each species flip to +1::

        rsign = -1   eee, eep, epp, ppp      (all of (T))   ->  t = +R/D
        rsign = -1   eeep (tep31), eppp (tep13)             ->  t = +R/D
        rsign = +1   eeee (t4), eepp (tep22), pppp (tp4)    ->  t = -R/D

    each index contributing the diagonal Fock element of *its own* species, so the
    NEO blocks read (lower case = electron, upper case = proton)::

        eee  (t3)     e_a + e_b + e_c - e_i - e_j - e_k
        eep  (tep21)  e_a + e_b + E_A - e_i - e_j - E_I
        epp  (tep12)  e_a + E_A + E_B - e_i - E_I - E_J
        ppp  (tp3)    E_A + E_B + E_C - E_I - E_J - E_K

    and the rank-4 blocks of ``(Q)`` continue the same rule::

        eeee (t4)     e_a + e_b + e_c + e_d - e_i - e_j - e_k - e_l
        eeep (tep31)  e_a + e_b + e_c + E_A - e_i - e_j - e_k - E_I
        eepp (tep22)  e_a + e_b + E_A + E_B - e_i - e_j - E_I - E_J
        eppp (tep13)  e_a + E_A + E_B + E_C - e_i - E_I - E_J - E_K
        pppp (tp4)    E_A + E_B + E_C + E_D - E_I - E_J - E_K - E_L

    That is the same denominator convention this library's *doubles* residuals use
    (``<proj|[F, T]|0> = -D t``, so ``R = 0`` gives ``t = rest/D``), and it is verified
    against a pdaggerq-derived ``[F, T_pt]`` in
    ``models_test.test_perturbative_triples`` rather than assumed.

    The denominator is the only thing not emitted: like every ``(T)``, this presumes a
    **(semi)canonical reference**, where the off-diagonal ``[f, T_pt]`` coupling that
    would make the equation implicit in the triples is absent. That is what makes the
    correction non-iterative, and it is what lets a consumer never store the triples --
    see :func:`pt_energy_graph`.

    Low-rank amplitudes are passed to the commutator for uniformity but cannot reach the
    projection (a two-body ``W`` plus a rank-n cluster amplitude reaches rank n+2 at
    most, so ``(T)`` sees nothing below doubles and ``(Q)`` nothing below triples); they
    contribute no terms and the derivation drops them."""
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    if amplitude not in m.T_pt:
        raise ValueError(f"model {name!r} has no perturbative amplitude {amplitude!r}; "
                         f"T_pt={list(m.T_pt)}")
    if amplitude not in PROJECTION:
        raise KeyError(f"no projection defined for amplitude {amplitude!r}")
    pq = _pt_pq(m, [[PROJECTION[amplitude]]], m.T)
    return _optimized(pq, label, df, opt_level, _dims_for(name))


def pt_amplitude_ir(name, amplitude, df=True, opt_level=None, label="R"):
    """The perturbative-amplitude numerator as ``to_strings("ir")`` JSONL lines."""
    return pt_amplitude_graph(name, amplitude, df=df, opt_level=opt_level,
                              label=label).to_strings("ir")


def pt_energy_graph(name, amplitude=None, df=True, opt_level=None, label="energy"):
    """Optimized pq_graph for the perturbative-triples energy ``<0| L [W, T_pt] |0>``,
    with ``L`` the de-excitation operators of the model (``lambda_amps``). This is the
    standard ``(T)``: the multipliers are *not* solved for, the consumer feeds
    ``l1 = t1^dagger``, ``l2 = t2^dagger``, ... The ``L1`` term is what makes it ``(T)``
    rather than ``[T]``.

    ``amplitude=None`` (the default) sums every perturbative block into one equation.
    Passing one block name emits only that block's contribution -- the energy is linear
    in ``T_pt``, so the blocks sum to the same total.

    **Per-block is what an on-the-fly implementation wants.** The triples are never
    stored: for one occupied index tuple the consumer builds that slice of the numerator
    (:func:`pt_amplitude_graph`), divides by the denominator, contracts it straight into
    this energy expression, and discards it. Both equations for a block carry the same
    external indices in the same order (the block's :data:`PROJECTION`), so the slice
    built by one is the slice consumed by the other -- ``t3`` here is that slice, not a
    global tensor. Generating the blocks separately keeps those three loops independent.

    The pairing between the two equations is exact and fixed::

        E = (rsign / w) sum(R * t)  =  -(1/w) sum(R^2 / D)   < 0

    over the block's full (unrestricted) index range, with ``rsign`` the reversal parity
    of :func:`pt_amplitude_graph` and ``w`` the product of the factorials of the
    index-group sizes -- equivalently ``(n_e!)^2 (n_p!)^2`` for a block with ``n_e``
    electron and ``n_p`` proton excitations::

        (T)  eee/ppp   36     eep/epp     4
        (Q)  eeee/pppp 576    eeep/eppp  36    eepp  16

    The energy is negative at every rank -- but note that the FIRST form carries
    ``rsign`` too. Both halves of the rank-3 special case (``t = R/D`` together with
    ``E = -(1/w) sum(R*t)``) hide the same sign and it cancels between them, so carrying
    the pair over to rank 4 by analogy yields a correction of the WRONG SIGN and no other
    symptom. Use the ``rsign`` forms.

    **Which multipliers pair.** A perturbative block of rank ``n`` pairs directly with the
    rank ``n-1`` multipliers; the lower-rank ones are the extra terms that make the
    correction ``(T)``/``(Q)`` rather than the bracketed ``[T]``/``[Q]``. So ``(T)`` keeps
    ``L2`` and its extra term is ``L1``, while ``(Q)`` keeps ``L3`` and its extra terms are
    ``L1`` AND ``L2`` -- the same over-generalization trap as the sign. A mixed block can
    pair with more than one multiplier at once: the ``(Q)`` eepp energy contracts both
    mixed rank-3 multipliers (see the naming caveat below).

    All of this is checked numerically, per block, in
    ``models_test.test_perturbative_triples`` and ``..._quadruples``.

    Naming: mixed blocks from rank 3 up spell out their electron/proton split, because a
    single equation can reference two of them at once -- the ``(Q)`` eepp energy contracts
    BOTH mixed rank-3 multipliers. So eep is ``t3_ep21`` and epp is ``t3_ep12`` (likewise
    ``l3_ep21``/``l3_ep12``, and ``t4_ep31``/``t4_ep22``/``t4_ep13`` at rank 4), and the
    names are unique within an equation. Rank 2 has only the 1+1 split and keeps the plain
    ``t2_ep``/``l2_ep``.

    Index classes still carry the shape (``vvVOoo`` for eep, ``vVVOOo`` for epp), and
    keying by (name, classes) remains the most robust thing a consumer can do -- but with
    distinct names it is no longer *required* to tell two physically different blocks
    apart."""
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    if not m.T_pt:
        raise ValueError(f"model {name!r} has no perturbative correction (T_pt is empty)")
    if amplitude is None:
        amps = list(m.T_pt)
    elif amplitude in m.T_pt:
        amps = [amplitude]
    else:
        raise ValueError(f"model {name!r} has no perturbative amplitude {amplitude!r}; "
                         f"T_pt={list(m.T_pt)}")
    pq = _pt_pq(m, [[l] for l in lambda_amps(name)], amps)
    return _optimized(pq, label, df, opt_level, _dims_for(name))


def pt_energy_ir(name, amplitude=None, df=True, opt_level=None, label="energy"):
    """The perturbative-triples energy as ``to_strings("ir")`` JSONL lines."""
    return pt_energy_graph(name, amplitude, df=df, opt_level=opt_level,
                           label=label).to_strings("ir")


def _pt_lambda_numerator_pq(m, amplitude):
    """``pq_helper`` for the numerator of the perturbative LAMBDA amplitude,
    ``<proj(amplitude)| W T |0>`` -- a plain operator PRODUCT, not a commutator.

    This is the one place the two perturbative triples differ. The amplitude numerator
    (:func:`pt_amplitude_graph`) is the CONNECTED ``<proj|[W, T]|0>``: a connected
    ``[W, T1]`` is a two-body operator at most, so it cannot reach a rank-3 projection and
    the singles drop out entirely -- ``R`` is a function of ``T2`` alone. The product keeps
    the disconnected ``W T1`` piece as well, and that piece is exactly what makes the
    correction ``(T)`` rather than ``[T]``: it is the term the ``L1`` multiplier pairs with
    in :func:`pt_energy_graph`. So the two numerators differ by precisely the singles
    contribution, and both go over the SAME denominator."""
    pq = pq_helper("fermi")
    pq.set_left_operators([[PROJECTION[amplitude]]])
    for w in _fluctuation(m):
        for a in m.T:
            pq.add_operator_product(1.0, [w, a])
    pq.simplify()
    return pq


def pt_lambda_numerator_graph(name, amplitude, df=True, opt_level=None, label="R"):
    """Optimized pq_graph for the numerator of the perturbative LAMBDA amplitude.

    The gradient of a ``(T)`` model needs TWO perturbative arrays per block, not one. Both
    are a numerator over the same orbital-energy denominator, and both are built and
    discarded one occupied subset at a time exactly as :func:`pt_amplitude_graph` is::

        t_pt      = -rsign * R      / D      R      from pt_amplitude_graph   (connected)
        l_pt(lam) = -rsign * R_lam  / D      R_lam  from HERE                 (product)

    with the SAME ``rsign`` and the SAME ``D`` documented on :func:`pt_amplitude_graph` --
    nothing new to get right, one extra numerator build per subset.

    ``t_pt`` is the amplitude of the ``(T)``-corrected wavefunction; ``l_pt(lam)`` is the
    multiplier of the ``(T)`` energy's stationarity condition. They are NOT the same array
    and are not interchangeable: they differ by the singles (disconnected) contribution to
    the numerator, which is zero only at ``t1 = 0``. The wavefunction density
    (:func:`pt_rdm_block_ir`) takes ``t_pt`` in both slots; every gradient quantity here
    takes one of each. In the CCSD(T) gradient literature this is the array written
    ``(w + v)/D`` against the amplitude's ``w/D``."""
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    if amplitude not in m.T_pt:
        raise ValueError(f"model {name!r} has no perturbative amplitude {amplitude!r}; "
                         f"T_pt={list(m.T_pt)}")
    if amplitude not in PROJECTION:
        raise KeyError(f"no projection defined for amplitude {amplitude!r}")
    return _optimized(_pt_lambda_numerator_pq(m, amplitude), label, df, opt_level,
                      _dims_for(name))


def pt_lambda_numerator_ir(name, amplitude, df=True, opt_level=None, label="R"):
    """The perturbative LAMBDA numerator as ``to_strings("ir")`` JSONL lines."""
    return pt_lambda_numerator_graph(name, amplitude, df=df, opt_level=opt_level,
                                     label=label).to_strings("ir")


def _pt_lambda_source_pqs(m, amplitude, cluster):
    """The two halves of ``dE_pt/dt_cluster``: ``<0| L_pt W tau |0>`` (pairs with the
    perturbative AMPLITUDE) and ``<0| L_pt [W, tau] |0>`` (pairs with the perturbative
    MULTIPLIER). Either may be empty for a given block."""
    lpt = "l" + amplitude[1:]
    tau = EXCITATION[cluster]
    out = []
    for connected in (False, True):
        pq = pq_helper("fermi")
        pq.set_left_operators([[lpt]])
        for w in _fluctuation(m):
            if connected:
                pq.add_commutator(1.0, [w], [tau])
            else:
                pq.add_operator_product(1.0, [w, tau])
        pq.simplify()
        out.append(pq)
    return out


def pt_lambda_source_ir(name, amplitude, cluster, label="R"):
    """JSONL IR for the ``(T)`` SOURCE TERM in the Lambda equation for ``cluster`` --
    the inhomogeneity that makes the cluster multipliers response multipliers.

    WHY IT EXISTS. A gradient comes from a Lagrangian stationary in every amplitude::

        L = <0|(1 + Lambda) e^-T H e^T|0> + E_pt[t, f, W]

    Plain CCSD already satisfies ``dL/dt = 0``: Lambda solving :func:`lambda_ir` IS the
    response multiplier, so the CCSD density is already the response density. Adding
    ``E_pt`` breaks that -- the amplitudes were converged for the parent's functional -- and
    stationarity becomes

        <0|(1 + Lambda) [Hbar, tau_mu]|0>  +  dE_pt/dt_mu  =  0

    This emits the second term, in EXACTLY the units of the first: ``lambda_ir`` and this
    function both carry pq_helper's ``w / rsign`` weighting for ``cluster`` (``w`` the
    product of index-group factorials, ``rsign`` the PROJECTION reversal parity), so the
    consumer ADDS the two arrays slot for slot, with coefficient one, and solves the
    augmented equations with the machinery it already runs. There is no convention to
    apply and none to invent -- verified by finite difference against ``L`` itself, not
    assumed.

    THE TWO OPERANDS. Writing ``E_pt = rsign/w sum(R_lam * t_pt)`` with
    ``t_pt = -rsign R/D`` and differentiating through BOTH the amplitude and the
    multipliers (in ``(T)`` the cluster multipliers are ``t^dagger``, so ``t`` appears in
    both slots) gives a symmetric pair::

        dE_pt/dt_mu  =  sum  t_pt      * d(R_lam)/dt_mu        <- the PRODUCT half
                     +  sum  l_pt(lam) * d(R)/dt_mu            <- the COMMUTATOR half

    so the emitted equation carries two ordinary operands, named for the array to feed::

        l3        reverse-index-order of  t_pt       (pt_amplitude_ir      / D)
        l3_lam    reverse-index-order of  l_pt(lam)  (pt_lambda_numerator_ir / D)

    (``l3_ep21`` / ``l3_ep21_lam`` and so on for a mixed NEO block.) They are DIFFERENT
    arrays -- see :func:`pt_lambda_numerator_graph` -- and the equation is wrong if the
    same one is fed to both. A block where one half vanishes simply carries no statements
    for that operand: the singles source is pure product (a connected ``[W, tau_1]``
    cannot reach the projection), and the doubles source happens to have the two halves
    algebraically equal, which is why the sum is often written with a single ``2 t + l``
    array in the literature. Emitting them separately keeps that a fact about a block
    rather than a convention the consumer has to know.

    SLICING. Every statement carries EXACTLY ONE perturbative operand and no
    intermediates, so this is strictly simpler to drive than :func:`pt_rdm_block_ir`: there
    is no ``t_pt``-against-``l_pt`` term, hence no pair of subsets straddling ``r-1``
    shared indices, hence no second loop. Each statement sums over occupied subsets exactly
    as the energy does."""
    m = model(name)
    if amplitude not in m.T_pt:
        raise ValueError(f"model {name!r} has no perturbative amplitude {amplitude!r}; "
                         f"T_pt={list(m.T_pt)}")
    if cluster not in m.T:
        raise ValueError(f"model {name!r} has no amplitude {cluster!r}; T={list(m.T)}")
    if cluster not in EXCITATION:
        raise KeyError(f"no excitation operator defined for amplitude {cluster!r}")
    spec = _excitation_spec(cluster)
    lpt = "l" + amplitude[1:]
    prod, comm = _pt_lambda_source_pqs(m, amplitude, cluster)
    out = _block_ir_from_strings(name, None, spec, label, pq=prod)
    out += _block_ir_from_strings(name, None, spec, label, pq=comm, lam=True,
                                  assign=not out)
    return out


def lambda_graph(name, amplitude, df=True, opt_level=None, label="R"):
    """Optimized pq_graph for the Lambda residual ``<(1+L) [Hbar, tau_amplitude]>``,
    the equation whose root is the de-excitation amplitude for ``amplitude``."""
    opt_level = _opt_level_for(name, opt_level)
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
    return _optimized(pq, label, df, opt_level, _dims_for(name))


def lambda_ir(name, amplitude, df=True, opt_level=None, label="R"):
    """The Lambda residual as ``to_strings("ir")`` JSONL lines."""
    return lambda_graph(name, amplitude, df=df, opt_level=opt_level, label=label).to_strings("ir")


def gradient_graph(name, species, df=True, opt_level=None, label="R"):
    """Optimized pq_graph for the per-species orbital-rotation gradient
    ``<(1+L) [Hbar, E_ai - E_ia]>`` (species = "electron" or "proton"), in AMPLITUDE
    form: the integrals are contracted with t/Lambda directly, so no RDMs need to be
    materialised (unlike the fixed-RDM :func:`orbital_gradient_ir`). Both routes give
    the same gradient -- verified to 4e-16 for electronic models -- but this one lets
    pq_graph factorise the whole contraction and avoids ever forming D2.

    ONE-BODY INPUTS -- these DIFFER from the residual/energy equations, because the gep
    reference traces are retained here (``gep_traces=False``, see below). The traces supply
    the e-p mean field, so it must NOT also be carried by the Fock matrices::

        f  = hcore_e + mf_ee          (electron mean field YES, e-p mean field NO)
        fp = hcore_p                  (BARE proton core -- no mean field at all)

    equivalently, starting from the dressed NEO-HF Focks, strip ONLY the e-p mean field::

        f  = f_dressed  - mf_ep       mf_ep[p,q] = sum_{I in occ_p} gep[p,I,q,I]
        fp = fp_dressed - mf_pe       mf_pe[P,Q] = sum_{i in occ_e} gep[i,P,i,Q]

    The asymmetry (f keeps mf_ee, fp keeps nothing) is not a typo: pdaggerq's Fermi-vacuum
    normal ordering already subtracts the ELECTRONIC mean field internally -- that is the
    ``h = f - mf_ee`` relation in the RDM identity -- while the proton one-body has no such
    subtraction (``hp = fp`` there). Feeding the fully dressed ``f``/``fp`` double-counts the
    e-p mean field: the sign is right but the magnitude is off by a few percent.

    Summary of the three conventions in this module:

    ==========================================  =========================================
    residual / energy / Lambda (gep_traces=True)  fully dressed f, fp
    gradient_ir (gep_traces=False)                f = f_dressed - mf_ep, fp = fp_dressed - mf_pe
    energy_from_rdm_ir / orbital_gradient_ir      BARE cores h = hcore_e, hp = hcore_p
    ==========================================  =========================================

    (The gradient's f differs from the RDM route's h by exactly mf_ee, as the identity
    ``h = f - mf_ee`` requires; both routes agree to ~5e-16 -- see
    models_test.test_gradient_ir_matches_orbital_gradient.)

    **The gep reference traces are NOT removed here** (``gep_traces=False``), unlike every
    other generated equation. Term-dropping does not commute with taking a commutator:
    removing trace-carrying terms FROM ``<[H, E-]>`` is not the same as forming
    ``<[H - T, E-]>``, so the removal does not yield the derivative of anything. It was
    being applied blanket-style by ``_optimized`` and made the NEO gradient disagree with
    the finite-difference-verified :func:`orbital_gradient_ir` (electron rel 0.41, proton
    rel 0.73) -- a consumer saw a gradient that failed its FD check and, because the error
    rode on gep, flipped when gep's charge sign flipped. Without the removal the two
    routes agree to ~5e-16 for BOTH species, so the gradient lives in the same convention
    as the RDM energy that :func:`energy_from_rdm_ir` traces.

    Guarded by models_test.test_gradient_ir_matches_orbital_gradient (both routes, both
    species, electronic and NEO)."""
    opt_level = _opt_level_for(name, opt_level)
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
    # gep_traces=False: see the docstring -- the removal does not commute with the
    # commutator, and applying it here broke the NEO gradient.
    return _optimized(pq, label, df, opt_level, _dims_for(name), gep_traces=False)


def gradient_ir(name, species, df=True, opt_level=None, label="R"):
    """The per-species orbital-rotation gradient as ``to_strings("ir")`` lines."""
    return gradient_graph(name, species, df=df, opt_level=opt_level, label=label).to_strings("ir")


_ROT_GEN = {"electron": (("e1(a,i)", +1.0), ("e1(i,a)", -1.0)),
            "proton":   (("e1(na,ni)", +1.0), ("e1(ni,na)", -1.0))}
_ROT_GEN_COL = {"electron": (("e1(b,j)", +1.0), ("e1(j,b)", -1.0)),
                "proton":   (("e1(nb,nj)", +1.0), ("e1(nj,nb)", -1.0))}


def hessian_graph(name, row_species="electron", col_species=None, df=True,
                  opt_level=None, label="H"):
    """Optimized pq_graph for the orbital-rotation HESSIAN in AMPLITUDE-CONTRACTED (T,Lambda)
    form -- the second derivative of the CC Lagrangian at fixed (t, Lambda)::

        H_(ai),(bj) = 1/2 <(1+L) e^-T ( [[H, A], B] + [[H, B], A] ) e^T |0>
        A = E_ai - E_ia   (row, `row_species`)      B = E_bj - E_jb   (col, `col_species`)

    This is the amplitude-route counterpart of the fixed-RDM :func:`orbital_hessian_ir`, the
    same way :func:`gradient_ir` is the counterpart of :func:`orbital_gradient_ir`. The two
    routes are contraction paths to the SAME number (the RDMs are built from the same
    t/Lambda) and agree elementwise to ~6e-16 -- guarded by
    models_test.test_hessian_ir_matches_orbital_hessian.

    **Why it exists: the fixed-RDM route has an unavoidable v^4 wall.** Its Hessian/diag/sigma
    contract the 2-RDM block ``D2[vvvv]``, which is a WAVEFUNCTION quantity -- density fitting
    cannot factorize it (it factorizes integrals, not RDMs). At v=400 spin-orbitals D2[vvvv] is
    ~191 GB and the route is simply unavailable. This one never forms D2 at all: it contracts
    the integrals against t/Lambda directly, so with ``df=True`` (the default) it is DF-native
    -- B factors, no ``g[vvvv]``, no ``D2[vvvv]`` -- and its largest object is t2 (o^2 v^2).

    **H is exactly the shape of t2** -- (vir,occ,vir,occ) -- so it can never be the binding
    constraint: if H does not fit, neither does t2, and then there is no coupled-cluster
    calculation to orbital-optimize in the first place. (Consumer-measured, FHF-/aug-cc-pVTZ:
    t2 = 139.8 MB, all three H blocks together = 140.7 MB, D2[vvvv] = 15.6 GB. With one quantum
    particle the proton-proton and cross blocks are noise, since O = 1.) A consumer therefore
    stores H and takes the Hessian DIAGONAL (preconditioner) and the SIGMA product
    ``sigma_ai = sum_bj H_(ai),(bj) kappa_bj`` from it directly -- the latter an o^2 v^2 matvec
    that lands on GEMM. H is built ONCE per macro-iteration and reused across all 10-50 sigma
    builds of a Newton solve, which strictly beats a matrix-free sigma repeating the O(N^6)
    contraction every Krylov step. There is therefore no separate T,Lambda diag/sigma emitter,
    and none is needed.

    NB the double commutator is expanded into operator PRODUCTS
    (``H A B - B H A - A H B + B A H``) fed to ``add_st_operator``, NOT built with
    pq_helper's ``add_double_commutator``, whose two-body piece is wrong (see the note on
    orbital_hessian_ir). It is symmetrized at the operator level, so the emitted H is
    symmetric under (a,i)<->(b,j) by construction.

    One-body inputs are the same as :func:`gradient_graph` (``gep_traces=False``):
    ``f = f_dressed - mf_ep`` and ``fp = fp_dressed - mf_pe``. See that docstring."""
    if col_species is None:
        col_species = row_species
    for sp in (row_species, col_species):
        if sp not in _ROT_GEN:
            raise ValueError(f"species must be 'electron' or 'proton', not {sp!r}")
    m = model(name)
    is_neo = any(op in m.H for op in ("fp", "gep"))
    if not is_neo and "proton" in (row_species, col_species):
        raise ValueError("proton-species orbital Hessian requires a NEO model")
    opt_level = _opt_level_for(name, opt_level)

    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]] + [[l] for l in lambda_amps(name)])
    T = list(m.T)
    for h in m.H:
        for a_op, sa in _ROT_GEN[row_species]:
            for b_op, sb in _ROT_GEN_COL[col_species]:
                # 1/2 ( [[H,A],B] + [[H,B],A] ): the two rotation generators do not commute,
                # so the second derivative of exp(kappa) is the SYMMETRIZED double derivative.
                for x, y in ((a_op, b_op), (b_op, a_op)):
                    c = 0.5 * sa * sb
                    pq.add_st_operator(+c, [h, x, y], T)     #  H X Y
                    pq.add_st_operator(-c, [y, h, x], T)     # -Y H X
                    pq.add_st_operator(-c, [x, h, y], T)     # -X H Y
                    pq.add_st_operator(+c, [y, x, h], T)     # +Y X H
    pq.simplify()
    # gep_traces=False, for the same reason as the gradient: term-dropping does not commute
    # with taking a commutator, so the removal does not yield the derivative of anything.
    return _optimized(pq, label, df, opt_level, _dims_for(name), gep_traces=False)


def hessian_ir(name, row_species="electron", col_species=None, df=True,
               opt_level=None, label="H"):
    """The amplitude-contracted (T,Lambda) orbital Hessian as ``to_strings("ir")`` lines.
    DF-native and free of any v^4 object -- see :func:`hessian_graph`."""
    return hessian_graph(name, row_species, col_species, df=df, opt_level=opt_level,
                         label=label).to_strings("ir")


def rdm_graph(name, operator, df=True, opt_level=None, label="D"):
    """Optimized pq_graph for a reduced-density-matrix block
    ``<(1+L) e^-T operator e^T>``.

    operator : a density-operator string -- ``e1(p,q)`` for a 1-RDM block,
               ``e2(p,q,s,r)`` for a 2-RDM block (note the last index pair is
               swapped, as in examples/ccsd_d2.py). The index letters pick occ/vir
               (o/v); an 'n' prefix picks the proton classes (O/V), same convention
               as gradient_graph's e1(na,ni). Examples: "e1(i,j)" (D_oo),
               "e1(a,b)" (D_vv), "e2(a,b,i,j)" (D_vvoo), "e1(ni,nj)" (proton D_OO),
               "e2(a,na,ni,i)" (mixed e-p 2-RDM block)."""
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]] + [[l] for l in lambda_amps(name)])   # (1 + Lambda)
    pq.add_st_operator(1.0, [operator], list(m.T))
    pq.simplify()
    return _optimized(pq, label, df, opt_level, _dims_for(name))


def rdm_ir(name, operator, df=True, opt_level=None, label="D"):
    """An RDM block as ``to_strings("ir")`` JSONL lines."""
    return rdm_graph(name, operator, df=df, opt_level=opt_level, label=label).to_strings("ir")


# distinct pdaggerq index letters per (species, class) -- one per slot, so a block's
# slots stay individually trackable through rdm_graph's relabelling (needed to permute
# the mixed e-p RDM into the consumer layout even when classes repeat, e.g. "OooO").
_BLK_LETTERS = {("e", "o"): "ijkl", ("e", "v"): "abcd",
                ("p", "o"): "IJKL", ("p", "v"): "ABCD"}


def _rdm_block_spec(tensor, block):
    """(operator string, consumer-order [(pq_label, class), ...]) for an RDM block.

    ``tensor`` in {D1, D1_n, D2, D2_n, D2_ep}; ``block`` is the class string in the exact
    order the consumers (energy_from_rdm_ir / orbital_*_ir) annotate the operand (e.g.
    "ov", "OO", "ovvo", "OovV"). Each slot gets a distinct letter so it stays trackable;
    proton labels carry pq_helper's nuclear ``n`` prefix. The D2 index order and the
    D2_ep (P,E,E',P') layout match the energy convention in :func:`energy_from_rdm_ir`."""
    used = {"e": {"o": 0, "v": 0}, "p": {"o": 0, "v": 0}}

    def take(sp, cls):                                   # next distinct letter of (sp,cls)
        low = "o" if cls in "oO" else "v"
        L = _BLK_LETTERS[(sp, low)][used[sp][low]]
        used[sp][low] += 1
        return ("n" + L) if sp == "p" else L

    if tensor in ("D1", "D1_n"):
        sp = "e" if tensor == "D1" else "p"
        c0, c1 = block[0], block[1]
        L0, L1 = take(sp, c0), take(sp, c1)
        return f"e1({L0},{L1})", [(L0, c0), (L1, c1)]    # D1[pq] = <p+ q>, direct order

    if tensor in ("D2", "D2_n"):
        sp = "e" if tensor == "D2" else "p"
        cs = list(block)
        Ls = [take(sp, c) for c in cs]
        op = f"e2({Ls[0]},{Ls[1]},{Ls[2]},{Ls[3]})"     # D2[p,q,s,r] = <p+ q+ r s>
        return op, list(zip(Ls, cs))

    if tensor == "D2_ep":
        # consumer layout D2_ep(P, E, E', P'); operator e2(P, E, P', E') -- the convention
        # validated in tests/neo_rdm_energy_test.py. The e-p energy is
        # E_ep = +gep.D2_ep (gep carries the charge sign; see energy_from_rdm_ir).
        cP, cE, cE2, cP2 = block[0], block[1], block[2], block[3]
        LE = take("e", cE); LP = take("p", cP); LP2 = take("p", cP2); LE2 = take("e", cE2)
        return f"e2({LP},{LE},{LP2},{LE2})", [(LP, cP), (LE, cE), (LE2, cE2), (LP2, cP2)]

    raise ValueError(f"unknown RDM tensor {tensor!r}; choose D1/D1_n/D2/D2_n/D2_ep")


def _pt_rdm_pq(name, op, amplitude):
    """``pq_helper`` for the PERTURBATIVE contribution of ``amplitude`` to a density block.

    The ``(T)``/``(Q)``-corrected wavefunction carries the perturbative amplitude in the
    exponential and its multiplier in the bra::

        D = <0| (1 + L + L_pt) e^-(T + T_pt) O e^(T + T_pt) |0>

    linearized in ``T_pt`` (it is first order, so ``T_pt^2`` is beyond the correction) and
    with the cluster part subtracted off -- what is left is what this emits. ``T`` and
    ``T_pt`` are both excitation operators and so commute, which lets the ``T_pt`` term be
    written as a similarity transform of a bare commutator::

        [e^-T O e^T, T_pt] = e^-T [O, T_pt] e^T

    so the three surviving pieces are

        <0| (1 + L) e^-T [O, T_pt] e^T |0>     the amplitude entering the density
        <0| L_pt    e^-T  O         e^T |0>     the multiplier's response
        <0| L_pt    e^-T [O, T_pt] e^T |0>     their cross term

    The last is second order in the perturbative pair but is where the diagonal blocks
    live -- it is what produces the familiar ``-1/12 t3 l3`` in ``D1_oo`` and ``+1/12`` in
    ``D1_vv`` -- so dropping it would leave those blocks empty."""
    m = model(name)
    lpt = "l" + amplitude[1:]
    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]] + [[l] for l in lambda_amps(name)] + [[lpt]])
    pq.add_st_operator( 1.0, [op, amplitude], list(m.T))     # e^-T [O, T_pt] e^T
    pq.add_st_operator(-1.0, [amplitude, op], list(m.T))
    pq.set_left_operators([[lpt]])
    pq.add_st_operator( 1.0, [op], list(m.T))                 # <L_pt| e^-T O e^T
    pq.simplify()
    return pq


#: Hamiltonian operands are emitted BLOCKED (``eri["oovv"]``, ``g["oOvV"]``, ``f["oo"]``)
#: -- the same names and index order pq_graph gives them, so equations emitted from
#: pq.strings() and equations emitted through pq_graph present one contract to a consumer.
#: Amplitudes and multipliers stay bare (``t2``, ``l3_ep21``). ``eri`` covers both
#: same-species antisymmetrized blocks; the case of the class string says which species.
_INTEGRAL_NAMES = {"eri", "f", "fp", "g", "gep", "vp"}


def _exact_coeff(token):
    """Exact value of a pq_helper coefficient token.

    ``pq.strings()`` prints coefficients to seven decimals, so ``float()`` turns 1/12 into
    0.0833333 -- a 4e-7 RELATIVE error that then rides into every emitted statement. It is
    invisible in any check that compares two pdaggerq-derived quantities (both carry it)
    and shows up only against an independent value, which is how it was found: a
    finite-difference of the (T) energy disagreed with the density that is supposed to be
    its derivative by exactly 1 - 4e-7 in every element. 50 of the 304 statements of
    ccsd's and ccsd(t)'s D1/D2 blocks carry an affected coefficient (every 1/6 and 1/12).

    Recover the rational whenever one reproduces the printed digits; fall back to the
    float when none does, which leaves such a term exactly as accurate as before."""
    x = float(token)
    f = Fraction(x).limit_denominator(2000)
    return float(f) if abs(float(f) - x) <= 1e-7 else x


def _cls_of(lab):
    """Line class of a pq_helper index label. Occupancy is case-insensitive so the same
    rule covers open and internal indices; proton labels carry the nuclear ``n`` prefix
    (open ``nI``/``nA``, internal ``ni``/``na``)."""
    nuc = len(lab) > 1 and lab[0] == "n"
    b = lab[1] if nuc else lab[0]
    occ = b.lower() in "ijklmno"
    return ("O" if occ else "V") if nuc else ("o" if occ else "v")


def _excitation_spec(amplitude):
    """Consumer-order ``[(pq_label, class), ...]`` for the open indices of an amplitude's
    EXCITATION operator, in the operator's own argument order -- which is the order
    :func:`lambda_ir` gives its target, so a source term emitted against this spec adds
    onto that residual slot for slot."""
    op = EXCITATION[amplitude]
    labels = op[op.index("(") + 1:-1].split(",")
    spec = [(l, _cls_of(l)) for l in labels]
    # pq_graph groups a mixed block's target indices BY SPECIES -- tep11's excitation
    # operator reads e2(a,nA,i,nI) but lambda_ir's target is (a,i,A,I) -- keeping each
    # species' internal order. Match that, or a NEO source term would land on the right
    # block with its axes transposed, which no shape check would catch.
    return sorted(spec, key=lambda lc: lc[1] in "OV")


def _block_ir_from_strings(name, op, consumer, tgt, pq=None, lam=False, assign=True,
                          drop_traces=False, scale=1.0):
    """Emit one RDM block's IR straight from ``pq.strings()``, bypassing ``pq_graph``.

    pq_graph's nuclear-index relabelling collapses a block's *internal* proton indices onto
    its *open* ones -- it cannot keep two same-class proton labels distinct -- silently
    restricting the contraction. That corrupts every block carrying both open and internal
    proton indices (``D1_n``, ``D2_n``, ``D2_ep``); the pure-electron blocks are unaffected,
    which is why it went unnoticed. Emitting from the simplified strings keeps every index
    distinct, so all tensors go through here.

    Kronecker deltas ``d(p,q)`` become ``Id["cc"]`` (the consumer supplies an identity);
    ``P(i,j)`` antisymmetrizers are expanded into signed statements over the open indices.

    ``drop_traces`` skips terms carrying a Kronecker delta between two OPEN indices.
    Those are traces of the density operator against itself, and they are exactly the
    difference between the BARE operator this builds from and pq_helper's NORMAL-ORDERED
    Hamiltonian. An expectation value wants them (the RDM is a true-vacuum object); a
    derivative with respect to the normal-ordered fluctuation must not have them, or the
    mean-field part of the integral is counted twice -- once here and once in the Fock
    response. Verified by Euler's theorem: the perturbative energy is linear in W in the
    slot this differentiates, so the emitted density contracted back with the integral
    must return the energy exactly, and it does so only with the traces dropped.

    ``lam`` suffixes ``_lam`` onto every de-excitation operand. In the halves that use it
    the ONLY such operand is the perturbative multiplier -- the bra is ``L_pt`` alone and
    the cluster contributes amplitudes, not multipliers -- so the rule is exact and,
    unlike a name-keyed map, survives pq_helper renaming its operators on the way out: the
    operator fed in as ``lep21`` prints as ``l3_ep21``, and a map keyed on the input name
    silently matches nothing. That is how the two halves of a mixed NEO block came to
    share one operand name, with no symptom except a wrong density.

    ``assign=False`` makes every statement accumulate rather than letting the first one
    assign. Together they let two
    separately-built pq_helpers emit into ONE target while keeping their operands
    distinguishable -- which is how :func:`pt_lambda_source_ir` puts the perturbative
    amplitude and the perturbative multiplier, two different arrays, in one equation.
    """
    if pq is None:
        pq = pq_helper("fermi")
        pq.set_left_operators([["1"]] + [[l] for l in lambda_amps(name)])
        pq.add_st_operator(1.0, [op], list(model(name).T))
        pq.simplify()
    strings = pq.strings()
    if not strings:
        return []                                        # model cannot populate this block

    out_pq = [L for L, _ in consumer]
    out_cls = [c for _, c in consumer]
    LET = {"o": "ijklmno", "v": "abcdefg", "O": "IJKLMNO", "V": "ABCDEFG"}

    cls_of = _cls_of

    base_used = {"o": 0, "v": 0, "O": 0, "V": 0}
    base_map = {}
    for lab in out_pq:                                   # open indices get fixed leading letters
        c = cls_of(lab); base_map[lab] = LET[c][base_used[c]]; base_used[c] += 1
    out_idx = [base_map[lab] for lab in out_pq]

    stmts = []
    for term in strings:
        coeff = _exact_coeff(term[0])
        perms, factors = [], []
        for tok in term[1:]:
            if tok.startswith("P("):
                perms.append(tok[2:-1].split(","))
            elif tok.startswith("<") or "(" in tok:
                factors.append(tok)
        if drop_traces and any(
                tok.startswith("d(") and
                all(x in base_map for x in tok[2:-1].split(","))
                for tok in factors):
            continue
        used = dict(base_used); m = dict(base_map)

        def letter(lab):
            if lab not in m:
                c = cls_of(lab); m[lab] = LET[c][used[c]]; used[c] += 1
            return m[lab]

        operands = []
        for tok in factors:
            if tok.startswith("<"):                      # <p,q||r,s> antisymmetrized 2-body
                nm = "eri"; idx = tok[1:-1].replace("||", ",").split(",")
            else:
                nm = tok[:tok.index("(")]; idx = tok[tok.index("(") + 1:-1].split(",")
            letters = [letter(i) for i in idx]; classes = [cls_of(i) for i in idx]
            if lam and nm.startswith("l"):
                nm = nm + "_lam"
            if nm == "d":                                # Kronecker delta -> identity block
                nm = f'Id["{classes[0]}{classes[1]}"]'
            elif nm in _INTEGRAL_NAMES:                  # blocked exactly as pq_graph names
                nm = f'{nm}["{"".join(classes)}"]'
            operands.append({"name": nm, "indices": letters, "classes": classes,
                             "is_intermediate": False})
        if not operands:
            raise ValueError(f"{tgt}: constant term with no factors: {term}")

        variants = [(1.0, list(out_idx))]                 # expand P(i,j): A -> A - A(i<->j)
        for a, b in perms:
            la, lb = base_map.get(a), base_map.get(b)
            nxt = []
            for sgn, idxs in variants:
                nxt.append((sgn, idxs))
                nxt.append((-sgn, [lb if x == la else (la if x == lb else x) for x in idxs]))
            variants = nxt

        for sgn, idxs in variants:
            stmts.append(json.dumps({
                "target": {"name": tgt, "indices": idxs, "classes": out_cls,
                           "is_intermediate": False},
                "is_assignment": assign and not stmts, "coeff": scale * sgn * coeff,
                "operands": operands}))
    return stmts


def rdm_block_ir(name, tensor, block, df=True, opt_level=None):
    """JSONL IR for one reduced-density-matrix block, with the TARGET named exactly
    ``tensor["block"]`` and indexed in the order :func:`energy_from_rdm_ir` and the
    ``orbital_*_ir`` builders consume it -- so a neocc driver can pipe rdm-block IR
    (inputs: amplitudes + Id) straight into the energy/gradient/sigma IR through one
    contraction engine, with zero convention knowledge outside pdaggerq.

    ``tensor`` in {"D1", "D1_n", "D2", "D2_n", "D2_ep"}; ``block`` is the class string in
    the consumer's index order (e.g. ``rdm_block_ir("neo-ccsd", "D2_ep", "OovV")``). A
    block the model cannot populate (e.g. ``D1["ov"]`` for a singles-free model) returns an
    empty list -- the consumer zero-fills.

    Every block is emitted by :func:`_block_ir_from_strings` (pq_graph is bypassed -- see
    that function for why), so ``df``/``opt_level`` are accepted for API compatibility and
    ignored: an RDM block is a handful of amplitude contractions, not worth optimizing.
    """
    op, consumer = _rdm_block_spec(tensor, block)
    return _block_ir_from_strings(name, op, consumer, f'{tensor}["{block}"]')


def pt_rdm_block_ir(name, amplitude, tensor, block, df=True, opt_level=None):
    """JSONL IR for the PERTURBATIVE contribution of one ``T_pt`` block to one RDM block.

    Same shape and target naming as :func:`rdm_block_ir` -- target ``tensor["block"]``,
    indexed in the order :func:`energy_from_rdm_ir` and the ``orbital_*_ir`` builders
    consume it -- so a consumer adds this on top of the cluster density with no convention
    knowledge. Returns ``[]`` for a block the correction cannot populate.

    The perturbative amplitude and its multiplier appear as ORDINARY operands (``t3`` and
    ``l3`` for ``ccsd(t)``, ``t3_ep21``/``l3_ep21`` for a mixed NEO block, and so on), so
    a driver that builds ``T_pt`` one occupied subset at a time can contract it straight
    into the density and discard it, exactly as it does for :func:`pt_energy_ir` -- the
    energy case with an array instead of a scalar on the left.

    ``l_pt`` is the consumer's to supply. At the order of the correction the perturbative
    multiplier is the amplitude's own conjugate -- the multiplier of the constraint
    ``R + rsign*D*t_pt = 0`` comes out proportional to ``t_pt`` itself, because the
    numerator/energy pairing makes the Lambda-side numerator proportional to ``R`` -- so
    ``l_pt`` is ``t_pt`` with its index order reversed, the same relation
    :func:`lambda_ir` uses for the cluster multipliers.

    SCOPE -- read before using this for a gradient. This is the density of the
    ``(T)``-corrected wavefunction: it answers "what is <p+q> for this wavefunction". A
    full analytic-gradient CCSD(T) density is a different object -- it additionally carries
    orbital relaxation and the response of the perturbative denominator to the
    perturbation, neither of which is a property of the wavefunction and neither of which
    is emitted here. For expectation values of one- and two-body operators (dipoles,
    populations, and the like) this is the density you want; for forces it is not the
    whole story."""
    m = model(name)
    if amplitude not in m.T_pt:
        raise ValueError(f"model {name!r} has no perturbative amplitude {amplitude!r}; "
                         f"T_pt={list(m.T_pt)}")
    op, consumer = _rdm_block_spec(tensor, block)
    return _block_ir_from_strings(name, op, consumer, f'{tensor}["{block}"]',
                                  pq=_pt_rdm_pq(name, op, amplitude))


def _pt_gradient_rdm_halves(m, op, amplitude, one_body):
    """``[(pq_helper, lam), ...]`` for the EXPLICIT integral derivatives of the
    perturbative energy -- what ``E_pt`` contributes to the density in its own right,
    over and above what reaches it through the Lambda source terms.

    ``E_pt = rsign/w sum(R_lam * t_pt)`` with ``t_pt = -rsign R/D``. Differentiating the
    two-body fluctuation gives the same symmetric pair the Lambda source has, with the
    density operator in place of ``W``::

        dE_pt/dW  =  <0| L     [O, T_pt] |0>    explicit -> pairs with t_pt
                  +  <0| L_pt  [O, T]     |0>    response -> pairs with l_pt(lam)

    Differentiating the DENOMINATOR is the other half of the story and is a genuinely
    different object. ``t_pt`` is fixed by ``R + <mu|[F, T_pt]|0> = 0``; call that linear
    map ``A`` (``A = rsign * D`` when the reference is canonical). Then
    ``dt/df = -A^-1 (dA/df) t``, and with ``l_pt(lam)`` defined by the SAME equation
    (``A l + R_lam = 0``, i.e. ``A^-1 R_lam = -l``) the symmetry of ``A`` collapses it to

        dE_pt/df  =  rsign/w sum  l_pt(lam) * <mu_pt| [E_pq, T_pt] |0>

    -- a one-body density operator commuted with the perturbative amplitude, with NO
    reference to the denominator itself. That is what makes it emittable: it never assumes
    ``D = sum(e_vir) - sum(e_occ)``, so it carries a non-diagonal Fock correctly. See
    :func:`pt_gradient_rdm_block_ir` for what that does and does not buy at a
    non-canonical reference."""
    lpt = "l" + amplitude[1:]
    if one_body:                                          # dE_pt/df: the denominator response
        pq = pq_helper("fermi")
        pq.set_left_operators([[lpt]])
        pq.add_commutator(1.0, [op], [amplitude])
        pq.simplify()
        return [(pq, True)]
    # dE_pt/dW. The EXPLICIT half differentiates W where it stands in the energy,
    # <0|L [W, T_pt]|0>, giving the cluster multipliers against the perturbative
    # AMPLITUDE. It is NOT the adjoint form <0|L_pt O T|0>: that rewriting of the energy
    # relies on W = W^dagger, which a single density operator is not, so the two agree
    # only after summing over all blocks -- not block by block, which is how this is
    # consumed. The RESPONSE half is dR/dW carried through t_pt, and pairs with the
    # perturbative MULTIPLIER.
    explicit = pq_helper("fermi")
    explicit.set_left_operators([[l] for l in lambda_amps(m.name)])
    explicit.add_commutator(1.0, [op], [amplitude])
    explicit.simplify()
    response = pq_helper("fermi")
    response.set_left_operators([[lpt]])
    for a in m.T:
        response.add_commutator(1.0, [op], [a])
    response.simplify()
    return [(explicit, False), (response, True)]


def pt_gradient_rdm_block_ir(name, amplitude, tensor, block, df=True, opt_level=None):
    """JSONL IR for the EXPLICIT perturbative contribution to one GRADIENT-density block.

    Same target naming and index order as :func:`rdm_block_ir` and
    :func:`pt_rdm_block_ir`, so a consumer adds it on top of the others with no convention
    knowledge. Returns ``[]`` for a block the correction cannot populate.

    THIS IS NOT :func:`pt_rdm_block_ir`. That one is the density of the ``(T)``-corrected
    WAVEFUNCTION -- the answer to "what is <p+q>". This one is a derivative of the
    ``(T)`` energy with respect to the integrals, which is a different object and is not
    an expectation value of anything. Two visible differences make them hard to confuse:
    this one pairs the perturbative amplitude with the perturbative MULTIPLIER
    (``l3_lam``, from :func:`pt_lambda_numerator_ir`, NOT ``t_pt`` reversed), and it
    populates ``D1`` from the denominator response, which no wavefunction density has.

    ASSEMBLING THE GRADIENT DENSITY. Three pieces, of which this is one::

        1. rdm_block_ir(parent, ...)        with Lambda solving the AUGMENTED equations
                                            (lambda_ir + pt_lambda_source_ir) -- that
                                            substitution alone turns the parent's density
                                            into a response density; no new equation.
        2. this function                    the explicit dE_pt/dW (-> D2) and dE_pt/df
                                            (-> D1) at fixed amplitudes.
        3. the Z-vector / orbital relaxation, which is the consumer's: it needs the
           orbital Hessian, not the cluster algebra.

    WHICH MULTIPLIERS TO FEED. The ``D2`` blocks carry the cluster multipliers ``l1``,
    ``l2``, ... as operands, and in this family -- as in :func:`pt_energy_ir`, which
    defines ``E_pt`` -- those mean the ``(T)`` PRESCRIPTION ``l = t^dagger``, NOT the
    solved Lambda. The solved (augmented) Lambda belongs in :func:`rdm_block_ir`, piece 1
    above. The two densities are added together and use the same operand names for
    different arrays, so this is the one thing here that can go wrong without any symptom
    except a wrong gradient; ``models_test.test_perturbative_gradient`` builds both and
    keeps them apart deliberately.

    ``D1`` here is exactly ``dE_pt/dh`` -- ``f = h + mean field`` and ``df/dh = 1``, so it
    adds to ``D1`` with no fold. ``D2`` here is only the DIRECT ``dE_pt/dg``; because ``f``
    also depends on ``g``, the consumer must add the mean-field image of the ``D1`` piece,
    ``sum_pq (dE_pt/df)_pq df_pq/dg``, which is the same fold it already applies to turn
    any normal-ordered one-body density into a true-vacuum two-body one. That fold is left
    to the consumer deliberately: for NEO it runs through the DRESSED ``f``/``fp``, whose
    definition (which mean fields are folded in, and with which charge sign for ``gep``)
    is the consumer's, not this library's -- see :func:`energy_from_rdm_ir`.

    SLICING. The ``D2`` blocks carry exactly ONE perturbative operand per statement, so
    they sum over occupied subsets exactly as the energy does -- the cheap shape. The
    ``D1`` blocks pair ``t_pt`` against ``l3_lam`` and so have the same two-subset,
    ``r-1``-shared-index shape that ``pt_rdm_block_ir``'s diagonal blocks already have.
    No statement anywhere carries a perturbative pair differing by more than one index,
    and there are no intermediates."""
    m = model(name)
    if amplitude not in m.T_pt:
        raise ValueError(f"model {name!r} has no perturbative amplitude {amplitude!r}; "
                         f"T_pt={list(m.T_pt)}")
    op, consumer = _rdm_block_spec(tensor, block)
    one_body = tensor in ("D1", "D1_n")
    if tensor in ("D2", "D2_n"):
        # SAME-SPECIES two-body only. _rdm_block_spec hands back the operator in RDM
        # order: with its arguments named (p,q,s,r) it is <p+ q+ r s>, which is what an
        # expectation value wants. A derivative with respect to <pq||rs> needs the
        # operator standing next to that integral in the Hamiltonian, p+ q+ s r -- the
        # same operator with its last two arguments swapped. The TARGET keeps the RDM
        # index order, so the emitted block still adds straight onto rdm_block_ir's.
        #
        # NOT for D2_ep. The e-p term is g(e,p,e',p') {e+ e'}{p+ p'}, and operators of
        # different species commute, so that is e+ p+ e' p' = e2(P,E,P2,E2) -- exactly
        # what _rdm_block_spec already returns. Swapping there gives a block that is
        # wrong by no constant factor at all; finite difference against a NEO (T) energy
        # is what caught it.
        a = op[op.index("(") + 1:-1].split(",")
        op = f"e2({a[0]},{a[1]},{a[3]},{a[2]})"
    tgt = f'{tensor}["{block}"]'
    out = []
    for pq, lam in _pt_gradient_rdm_halves(m, op, amplitude, one_body):
        # sign, per tensor, every one fixed by finite difference against E_pt itself:
        #   D1 / D1_n  +1  dE_pt/df adds to D1 as it stands (f = h + mean field, df/dh = 1)
        #   D2 / D2_n  -1  same-species, and carries the 1/4 of W's definition through the
        #                  RDM convention together with the operator swap above
        #   D2_ep      +1  no 1/4 and no swap -- gep multiplies a plain product of one-body
        #                  operators, so it lands on the other sign
        out += _block_ir_from_strings(name, op, consumer, tgt, pq=pq, lam=lam,
                                      assign=not out, drop_traces=True,
                                      scale=-1.0 if tensor in ("D2", "D2_n") else 1.0)
    return out


def pt_rdm_graph(name, amplitude, operator, df=True, opt_level=None, label="D"):
    """Optimized pq_graph for the perturbative contribution to an RDM block, taking a raw
    density-operator string (``e1(i,a)``, ``e2(a,b,i,j)``, ...) like :func:`rdm_graph`.

    Prefer :func:`pt_rdm_block_ir`, which names and orders the target the way the energy
    and orbital builders consume it. See :func:`_pt_rdm_pq` for the construction and
    :func:`pt_rdm_block_ir` for what this density is and is not."""
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    if amplitude not in m.T_pt:
        raise ValueError(f"model {name!r} has no perturbative amplitude {amplitude!r}; "
                         f"T_pt={list(m.T_pt)}")
    return _optimized(_pt_rdm_pq(name, operator, amplitude), label, df, opt_level,
                      _dims_for(name))


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
# double-counted; the e-p coupling is carried entirely by E_ep = +gep.D2_ep.
#
# CHARGE CONVENTION: gep carries no built-in sign (pq_helper does not negate it), so the
# consumer supplies the SIGNED interaction integral gep = -Z_x V_ex (Z_x = second species'
# charge in units of e; V_ex = bare positive Coulomb). Attractive for protons/positrons
# (Z=+1), repulsive for negative muons (Z=-1) -- see the gep block in pq_helper.cc. This
# also makes E_ep here agree in sign with the hand-derived OO gradient/Hessian terms below,
# which were derived as d^n(gep.D2_ep). Guarded by the full-energy identity in
# models_test.test_energy_from_rdm.
# NOTE on the g.D2 pairing: rdm_block_ir builds D2["pqsr"] from e2(p,q,s,r), i.e.
# D2[p,q,c,d] = <p+ q+ d c>. The two-electron energy is E_2e = 1/2 sum <pq|rs> <p+ q+ s r>,
# so g's LAST TWO slots must pair with D2's last two *swapped* -- ("D2", [0,1,3,2]). Pairing
# them slot-for-slot instead would silently require the consumer to hand in an ERI whose last
# two indices are transposed (Coulomb <-> exchange scrambled); with [0,1,3,2] a consumer feeds
# the natural plain <pq|rs>. Guarded by the full-energy identity in test_energy_from_rdm.
_ENERGY_ELEC = [(1.0, "ee", [("h", [0, 1]), ("D1", [0, 1])]),
                (0.5, "eeee", [("g", [0, 1, 2, 3]), ("D2", [0, 1, 3, 2])])]
_ENERGY_NEO = _ENERGY_ELEC + [(1.0, "pp", [("hp", [0, 1]), ("D1_n", [0, 1])]),
                              (1.0, "epep", [("gep", [0, 1, 2, 3]), ("D2_ep", [1, 0, 2, 3])])]
# proton-proton two-body, present only when the model's H carries vp (>=2 quantum protons).
# Same-species, so it mirrors the electron g.D2 term exactly (plain vp, coeff 1/2, and the
# D2_n last-two slots swapped against vp's).
_ENERGY_PP = [(0.5, "pppp", [("vp", [0, 1, 2, 3]), ("D2_n", [0, 1, 3, 2])])]


def energy_from_rdm_ir(name, label="E"):
    """The ground-state energy ``<H> = h.D1 + 1/2 g.D2`` (+ NEO proton/e-p terms) as
    explicit occ/vir-block JSONL IR (see the block-contraction note above). Returns
    the total ``<H>``; subtract the reference energy for the correlation part.

    CONSUMER CONTRACT -- how to build the integral inputs from raw MO quantities.
    Guarded by ``models_test.test_energy_from_rdm``, which checks BOTH the algebraic
    identity and the PHYSICAL one (E_rdm at zero amplitudes == the NEO-HF reference
    energy). All formulas are SPIN-ORBITAL; a spatial-orbital (restricted) consumer
    must insert its own spin factors (electron occupied sums become
    ``2 * sum_i(spatial)`` for closed shells).

    **The one-body operators are the BARE cores.** This is the whole point of the RDM
    form: every mean field emerges from the TWO-body terms contracted with the RDMs, so
    feeding a dressed Fock double-counts it::

        E = h.D1 + 1/2 g.D2 + hp.D1_n + gep.D2_ep          (the TOTAL energy)

    * ``h  = hcore_e``  -- the BARE electron core (kinetic + nuclear attraction).
      NOT the Fock matrix, and NOT ``f - mf_ee``: subtracting only the electronic mean
      field leaves ``hcore_e + mf_ep`` behind, which together with ``hp`` and
      ``gep.D2_ep`` counts the e-p mean field THREE times. If you start from the dressed
      NEO-HF Fock, you must remove BOTH mean fields:
      ``h = f - mf_ee - mf_ep`` with
      ``mf_ee[p,q] = sum_{i in occ_e} <pi||qi>``  (ANTISYMMETRIZED -- exchange included)
      ``mf_ep[p,q] = sum_{I in occ_p} gep[p,I,q,I]``  (signed gep; no exchange between
      distinguishable species).
    * ``hp = hcore_p`` -- the BARE proton core, i.e. ``fp - mf_pe`` with
      ``mf_pe[P,Q] = sum_{i in occ_e} gep[i,P,i,Q]`` (all occupied electron
      spin-orbitals). NOT the dressed ``fp``.
    * Sanity check a consumer can run in one line: at ZERO amplitudes this E must equal
      your NEO-HF reference energy exactly. If it is too high by ``2 * sum_{i,I}
      gep(i,I,i,I)`` you have fed dressed Focks.
    * ``g = plain physicist <pq|rs>`` (NOT antisymmetrized; the antisymmetry lives
      in D2, hence the 1/2). Slot pairing as emitted: ``g[abcd] . D2[abdc]``
      (D2's last two slots swapped), with ``D2[pqsr] = <p+ q+ r s>``.

    NB the *generated equations* (residual/energy/Lambda, via ``_optimized``) DO take the
    dressed NEO-HF Fock -- ``remove_gep_reference_traces`` strips gep's mean-field traces
    precisely because f/fp already carry them. The two routes therefore take DIFFERENT
    one-body inputs, and their totals differ by the constant reference e-p energy
    ``sum_{i,I} gep(i,I,i,I)`` (dropped by the trace removal), so they agree on E_corr.
    Do not feed the same one-body matrices to both.
    * ``gep`` is the SAME tensor fed to the residual/energy equations, in slot
      order ``(e, P, e', P')`` -- pdaggerq attaches no charge factor to it (the
      equations are charge-agnostic), so the physical sign is the consumer's:
      for electron-proton, ``gep[p,P,q,Q] = -(pq|PQ)`` (attractive; chemist
      notation), i.e. ``-<pP|qQ>`` physicist; ``+`` for a negative muon. Whatever
      sign convention the ground-state equations were converged with MUST be used
      here too, including inside ``mf_ep``/``mf_pe`` above. ``E_ep`` enters as
      ``+ gep . D2_ep`` with the pairing encoded in the emitted statement indices
      (D2_ep consumer layout is ``(P, E, E', P')``).
    """
    H = model(name).H
    is_neo = any(op in H for op in ("fp", "gep"))
    terms = _ENERGY_NEO if is_neo else _ENERGY_ELEC
    if "vp" in H:                                     # >=2 quantum protons: p-p two-body
        terms = terms + _ENERGY_PP
    return _emit_block_terms(terms, (label, []))


def rdm_energy_reference(name, seed=17, no=2, nv=3, nO=1, nV=4):
    # Every extent is DISTINCT on purpose (no != nv, nO != nV, and both pairs differ across
    # species). With no == nv a mis-ordered index -- e.g. the Hessian's [a,b,i,j] read as
    # [a,i,b,j] -- is SHAPE-COMPATIBLE, so a wrong transpose compares silently and the test
    # cannot catch it, ever. With distinct extents the same mistake is a shape error and
    # numpy raises. This bit twice here (the fixed-RDM Hessian, then the T,Lambda one) and
    # once in the consumer; do not "simplify" these back to equal dims.
    """Numeric byte-check reference for the RDM->energy consumer contract (see
    :func:`energy_from_rdm_ir`). Builds random symmetric integrals and arbitrary
    (antisymmetrized) amplitudes at tiny dimensions, evaluates every
    :func:`rdm_block_ir` block and the :func:`energy_from_rdm_ir` trace, and
    independently evaluates the CC Lagrangian ``<(1+Lambda) H>`` from the raw
    ``pq.strings`` -- the two must agree to ~1e-9.

    Returns a dict with every array a consumer needs to verify its own
    construction step by step: ``f``, ``fp``, ``eri`` (antisym ``<pq||rs>``),
    ``g`` (plain physicist ``<pq|rs>``), ``gep`` (``(e,P,e',P')``), ``mf_ee``,
    ``h``, ``hp``, ``amps`` ({name: array}), ``rdm`` ({(base, block): array}),
    and the scalars ``E_rdm`` / ``E_lagrangian``. Deterministic in ``seed``.
    Note: electron-only models ignore ``nO``/``nV`` and the proton arrays are
    absent from the result."""
    import itertools, json
    import numpy as np
    from collections import defaultdict

    m = model(name)
    is_neo = any(op in m.H for op in ("fp", "gep"))
    ne, npr = no + nv, nO + nV
    sle = {"o": slice(0, no), "v": slice(no, ne)}
    slp = {"O": slice(0, nO), "V": slice(nO, npr)}
    D = {"o": no, "v": nv, "O": nO, "V": nV}
    rg = np.random.default_rng(seed)

    # BARE cores, then the PHYSICALLY CONSISTENT dressed Focks built from the same
    # two-body integrals -- so a consumer can check either construction against the other.
    hcore = rg.standard_normal((ne, ne)); hcore = hcore + hcore.T
    hcore_p = rg.standard_normal((npr, npr)); hcore_p = hcore_p + hcore_p.T
    cq = rg.standard_normal((ne,) * 4)                     # chemist (pq|rs) symmetries
    cq = cq + cq.transpose(1, 0, 2, 3); cq = cq + cq.transpose(0, 1, 3, 2)
    cq = cq + cq.transpose(2, 3, 0, 1)
    g = cq.transpose(0, 2, 1, 3)                           # physicist <pq|rs>
    eri = g - g.transpose(0, 1, 3, 2)                      # antisym <pq||rs>
    gep = rg.standard_normal((ne, npr, ne, npr))
    gep = gep + gep.transpose(2, 3, 0, 1)                  # hermitian e-p tensor
    # proton-proton two-body (>=2 quantum protons only; identically absent otherwise).
    # Plain physicist <PQ|RS> for the RDM energy (antisymmetry lives in D2_n, hence the
    # 1/2), antisymmetrized <PQ||RS> for the Lagrangian strings -- exactly as g/eri for
    # electrons.
    has_vp = "vp" in m.H
    if has_vp:
        cp = rg.standard_normal((npr,) * 4)
        cp = cp + cp.transpose(1, 0, 2, 3); cp = cp + cp.transpose(0, 1, 3, 2)
        cp = cp + cp.transpose(2, 3, 0, 1)
        vp = cp.transpose(0, 2, 1, 3)                      # plain <PQ|RS>
        vp_anti = vp - vp.transpose(0, 1, 3, 2)            # antisym <PQ||RS>
        mf_pp = np.einsum("PIQI->PQ", vp_anti[:, slp["O"], :, slp["O"]])   # antisym p-p
    else:
        vp = vp_anti = None
        mf_pp = np.zeros((npr, npr))

    mf_ee = np.einsum("piqi->pq", eri[:, sle["o"], :, sle["o"]])      # antisym e-e
    mf_ep = np.einsum("pIqI->pq", gep[:, slp["O"], :, slp["O"]])      # e-p on the electron
    mf_pe = np.einsum("iPiQ->PQ", gep[sle["o"], :, sle["o"], :])      # e-p on the proton

    # The RDM energy takes the BARE cores -- every mean field must come from the TWO-body
    # terms contracted with the RDMs, or it is counted twice (see energy_from_rdm_ir).
    h, hp = hcore, hcore_p
    # The raw-H Lagrangian's one-body operators, chosen so that the algebraic identity
    # (h = f - mf_ee, hp = fp - mf_pp) and the PHYSICAL one (bare cores -> E_rdm(0) = E_HF)
    # hold simultaneously: each Fock carries only its OWN species' mean field (which the
    # Fermi-vacuum normal ordering subtracts back out), never the cross-species one.
    f, fp = hcore + mf_ee, hcore_p + mf_pp
    # What the GENERATED equations take instead: the fully dressed NEO-HF Focks (that is
    # why _optimized strips gep's mean-field traces). Exposed so a consumer can see that
    # the two routes take DIFFERENT one-body inputs.
    f_dressed, fp_dressed = hcore + mf_ee + mf_ep, hcore_p + mf_pp + mf_pe

    def spc(l):
        nuc = len(l) > 1 and l[0] == "n"
        b = l[1] if nuc else l[0]
        occ = b.lower() in "ijklmno"
        return ("O" if occ else "V") if nuc else ("o" if occ else "v")

    def sl(l):
        return slp[spc(l)] if spc(l) in "OV" else sle[spc(l)]

    def asym(a, cl):
        out = a.copy(); grp = defaultdict(list)
        for ax, c in enumerate(cl): grp[c].append(ax)
        for c, axes in grp.items():
            if len(axes) >= 2:
                P = list(itertools.permutations(range(len(axes)))); acc = np.zeros_like(out)
                for pm in P:
                    par = sum(1 for i in range(len(pm)) for j in range(i + 1, len(pm))
                              if pm[i] > pm[j]) & 1
                    src = list(range(out.ndim))
                    for k, ax in enumerate(axes): src[ax] = axes[pm[k]]
                    acc += (-1 if par else 1) * np.transpose(out, src)
                out = acc / len(P)
        return out

    amps = {}
    def amp(nm, cls):
        if nm not in amps:
            amps[nm] = asym(rg.standard_normal(tuple(D[c] for c in cls)), list(cls))
        return amps[nm]

    def interp(ir, tgt):
        prod = {s["target"]["name"] for s in ir}; st = {}
        def val(op):
            nm = op["name"]
            if nm in prod and nm in st: return st[nm]
            if nm.startswith("Id["): return np.eye(D[op["classes"][0]])
            return amp(nm, op["classes"])
        for s in ir:
            oi = "".join(s["target"]["indices"])
            sub = ",".join("".join(x["indices"]) for x in s["operands"])
            cc = s["coeff"] * np.einsum(sub + "->" + oi,
                                        *[val(x) for x in s["operands"]], optimize=True)
            st[s["target"]["name"]] = cc.copy() if s["is_assignment"] else st[s["target"]["name"]] + cc
        return st[tgt]

    # left side: <(1+Lambda) H> from the RAW strings (gep traces NOT removed)
    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]] + [[l] for l in lambda_amps(name)])
    for oper in m.H:
        pq.add_st_operator(1.0, [oper], list(m.T))
    pq.simplify()
    L = 0.0
    for term in pq.strings():
        cf = float(term[0]); ops = []; sub = []; lts = {}
        for tok in term[1:]:
            if tok.startswith("<"):
                # antisymmetrized two-body: <pq||rs> is the ELECTRON eri, <nP,nQ||nR,nS>
                # the PROTON one (vp's fluctuation potential). Dispatch on the species.
                idx = tok[1:-1].replace("||", ",").split(",")
                src = vp_anti if all(spc(i) in "OV" for i in idx) else eri
                arr = src[tuple(sl(i) for i in idx)]
            elif "(" in tok:
                nm = tok[:tok.index("(")]; idx = tok[tok.index("(") + 1:-1].split(",")
                if nm == "f":   arr = (fp if spc(idx[0]) in "OV" else f)[tuple(sl(i) for i in idx)]
                elif nm == "g": arr = gep[tuple(sl(i) for i in idx)]
                elif nm == "d": arr = np.eye(D[spc(idx[0])])
                else:           arr = amp(nm, [spc(i) for i in idx])
            else:
                continue
            ops.append(arr); sub.append("".join(lts.setdefault(i, chr(65 + len(lts))) for i in idx))
        L += cf * (np.einsum(",".join(sub) + "->", *ops, optimize=True) if ops else 1.0)

    # right side: every rdm_block_ir block traced through energy_from_rdm_ir
    from . import einsums as _einsums
    rdm = {}
    def getD(base, blk, cls):
        if (base, blk) not in rdm:
            ir = _einsums.parse_ir(rdm_block_ir(name, base, blk))
            rdm[(base, blk)] = interp(ir, f'{base}["{blk}"]') if ir \
                else np.zeros(tuple(D[c] for c in cls))
        return rdm[(base, blk)]

    E_rdm = 0.0
    for line in energy_from_rdm_ir(name):
        st = json.loads(line); ts = []
        for op in st["operands"]:
            base = op["name"].split('["')[0]; blk = op["name"].split('"')[1]
            if base == "h":     ts.append(h[sle[blk[0]], sle[blk[1]]])
            elif base == "hp":  ts.append(hp[slp[blk[0]], slp[blk[1]]])
            elif base == "g":   ts.append(g[tuple(sle[c] for c in blk)])
            elif base == "gep": ts.append(gep[sle[blk[0]], slp[blk[1]], sle[blk[2]], slp[blk[3]]])
            elif base == "vp":  ts.append(vp[tuple(slp[c] for c in blk)])   # plain <PQ|RS>
            else:               ts.append(getD(base, blk, op["classes"]))
        sub = ",".join("".join(op["indices"]) for op in st["operands"])
        E_rdm += st["coeff"] * float(np.einsum(sub + "->", *ts, optimize=True))

    # PHYSICAL reference energy: the RDM trace at ZERO amplitudes must reproduce the
    # (NEO-)HF energy built from the BARE cores. This is the check that catches a dressed
    # Fock being fed to the RDM route (it comes out high by 2*sum_iI gep(i,I,i,I)).
    o_, O_ = sle["o"], slp["O"]
    E_hf = float(np.trace(hcore[o_, o_]) + 0.5 * np.einsum("ijij->", eri[o_, o_, o_, o_]))
    if is_neo:
        E_hf += float(np.trace(hcore_p[O_, O_]) + np.einsum("iIiI->", gep[o_, O_, o_, O_]))
    if has_vp:                                  # proton-proton reference energy (>=2 protons)
        E_hf += float(0.5 * np.einsum("IJIJ->", vp_anti[O_, O_, O_, O_]))

    out = {"f": f, "eri": eri, "g": g, "mf_ee": mf_ee, "h": h, "hcore": hcore,
           "f_dressed": f_dressed, "E_hf": E_hf,
           "amps": amps, "rdm": rdm, "E_rdm": E_rdm, "E_lagrangian": L,
           "dims": {"o": no, "v": nv, "O": nO, "V": nV}}
    if is_neo:
        out.update({"fp": fp, "gep": gep, "hp": hp, "hcore_p": hcore_p,
                    "fp_dressed": fp_dressed, "mf_ep": mf_ep, "mf_pe": mf_pe})
    if has_vp:
        out.update({"vp": vp, "vp_anti": vp_anti, "mf_pp": mf_pp})
    return out


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













def _rdm_base(nm):
    return nm[:-2] if nm.endswith("_n") else nm[:-3] if nm.endswith("_ep") else nm


def _block_resolve(terms, target_name, target_letters, ext_classes=None, drop_inactive_rdm=False):
    """Enumerate each bare-form term over occ/vir blocks of its general indices,
    keeping the fixed (external rotation) indices. Kronecker deltas ``d`` become the
    identity ``Id`` and vanish when their two indices land in different blocks. The
    einsum label is the pdaggerq letter (proton ``nX`` -> uppercase ``X``).

    ``ext_classes`` (letter -> class) overrides the class of external rotation letters
    for the active-space OO split -- e.g. row ``a`` -> ``x`` (inactive-virtual) or
    ``c`` (core); those letters are then not enumerated. ``drop_inactive_rdm`` drops
    any term where an inactive index (c/x/C/X) lands in an RDM (which is active-only,
    so that block does not exist) -- leaving every surviving term with the inactive
    rotation index appearing only in an integral (one free inactive-virtual index)."""
    import itertools
    ext_classes = ext_classes or {}
    out, seen = [], set()
    for term in terms:
        coeff, tensors = _parse_rdm_term(term)
        letters = {}
        for _, idx in tensors:
            for l in idx:
                letters.setdefault(l, _classify_letter(l))
        gen = [l for l, (sp, fx) in letters.items() if fx is None and l not in ext_classes]
        for combo in itertools.product("ov", repeat=len(gen)):
            cls = {}
            for l, (sp, fx) in letters.items():
                cls[l] = ext_classes[l] if l in ext_classes else \
                    fx if fx else (combo[gen.index(l)].upper() if sp == "p" else combo[gen.index(l)])
            if any(nm == "d" and cls[idx[0]] != cls[idx[1]] for nm, idx in tensors):
                continue
            if drop_inactive_rdm and any(
                    _rdm_base(nm) in ("D1", "D2") and any(cls[l] in ("c", "x", "C", "X") for l in idx)
                    for nm, idx in tensors):
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
                        # all-proton two-body is "vp" -- the SAME name energy_from_rdm_ir
                        # emits (_ENERGY_PP). It was "gpp" here, which no consumer knows.
                        out_nm = base if sp == {"e"} else \
                            {"h": "hp", "f": "fp", "g": "vp", "D1": "D1_n", "D2": "D2_n"}[base] \
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




# orbital classes ordered by occupation for the active-space OO rotation split:
# core (inactive-occ) < active-occ < active-vir < inactive-vir (external). Electron
# classes are lowercase, proton uppercase; rotation_classes is given lowercase and
# mapped to the species' case.
_CLASS_LEVEL = {"c": 0, "o": 1, "v": 2, "x": 3, "C": 0, "O": 1, "V": 2, "X": 3}


def _rotation_blocks(rotation_classes, species="electron"):
    """Non-redundant rotation blocks (row, col) -- row 'higher' than col -- from the
    given (lowercase) classes, cased for the species. ('o','v') -> [('v','o')] electron
    or [('V','O')] proton (today's active-active block)."""
    cs = [c.upper() if species == "proton" else c for c in rotation_classes]
    return [(hi, lo) for hi in cs for lo in cs if _CLASS_LEVEL[hi] > _CLASS_LEVEL[lo]]




def orbital_gradient_ir(name, species="electron", rotation_classes=("o", "v"),
                        internal="active", factorize_inactive_virtual=True, label="grad"):
    """Fixed-RDM orbital-rotation gradient ``g_pq = <[H, E_pq - E_qp]>`` (the
    antisymmetrized generalized Fock) as explicit block JSONL IR.

    Active-space OO: the **rotation** (row/col) indices range over ``rotation_classes``
    -- core ``c``, active-occ ``o``, active-vir ``v``, inactive-vir/external ``x`` --
    while the **internal** (RDM-contracted) indices stay active (``o/v``), since the
    correlation RDM is active-only. Every non-redundant block (row 'higher' than col)
    is emitted as its own target ``grad["<row><col>"]`` (e.g. ``grad["xo"]``); the
    default ``("o","v")`` keeps the single active-active block as bare ``grad``
    (byte-identical to before). Terms where an inactive (c/x) rotation index would land
    in an RDM are dropped (that block is zero), so each surviving inactive-virtual index
    appears only in an integral -- one free ``x`` per term, J/K-factorizable by neocc.

    neocc supplies D1/D2 (active, and gep/D2_ep for NEO) plus the integral blocks
    (with the ``x``/``c`` rows) and evaluates the ``x`` contributions density-driven."""
    if species not in _ROT_ROW:
        raise ValueError(f"species must be 'electron' or 'proton', not {species!r}")
    if internal != "active":
        raise ValueError("internal indices are RDM-contracted and must be 'active'")
    is_neo = any(op in model(name).H for op in ("fp", "gep"))
    single = tuple(rotation_classes) == ("o", "v")
    # ONE rule for both species: the gradient is the first rotation-derivative of the
    # fixed-RDM energy (the Hessian is the second; see _same_species_hessian). This
    # replaces the pq_helper commutator + hand-derived _HP_GRAD_TERMS/_GEP_*_GRAD_TERMS,
    # and -- crucially -- it picks up EVERY operator in the energy automatically, which is
    # how the proton-proton two-body (vp) had gone missing from the OO quantities.
    if species == "proton":
        if not is_neo:
            raise ValueError("proton-species orbital gradient requires a NEO model")
        terms, letters, sp = _rot_deriv(_energy_bare_terms(name), "na", "ni", "p"), ["na", "ni"], "proton"
    else:
        terms, letters, sp = _rot_deriv(_energy_bare_terms(name), "a", "i", "e"), ["a", "i"], "electron"
    out = []
    for hi, lo in _rotation_blocks(rotation_classes, sp):
        suffix = "" if single else f'["{hi}{lo}"]'
        out += _block_resolve(terms, label + suffix, letters,
                              ext_classes={letters[0]: hi, letters[1]: lo}, drop_inactive_rdm=True)
    return out


def orbital_hessian_ir(name, row_species="electron", col_species=None, label="H"):
    """Fixed-RDM orbital Hessian block ``H_ai,bj = d2E/dkappa_ai dkappa_bj`` (rows a,i;
    columns b,j) as explicit occ/vir-block JSONL IR. ``col_species`` defaults to
    ``row_species``. All three blocks are emitted for NEO -- electron-electron,
    proton-proton, and the electron-proton cross block -- and all are
    finite-difference-verified.

    The Hessian is the SECOND derivative of the fixed-RDM energy, obtained by rotation-
    differentiating it twice (see :func:`_same_species_hessian`) -- NOT from pq_helper's
    double commutator, whose two-body piece is wrong (the single commutator is fine) and
    which emitted the UNSYMMETRIZED ``<[[H,A],B]>``: that is not symmetric under
    (a,i)<->(b,j), while the true fixed-RDM Hessian is exactly symmetric. Finite-difference
    verified (models_test.test_orbital_gradient_finite_difference), as are the diagonal and
    the sigma product, which share this term source."""
    if col_species is None:
        col_species = row_species
    if row_species not in _ROT_ROW or col_species not in _ROT_COL:
        raise ValueError("row/col species must be 'electron' or 'proton'")
    is_neo = any(op in model(name).H for op in ("fp", "gep"))
    if row_species != col_species:                     # e-p cross block (gep only)
        if not is_neo:
            raise ValueError("the electron-proton cross Hessian requires a NEO model")
        if (row_species, col_species) != ("electron", "proton"):
            raise NotImplementedError(
                "only the (electron, proton) cross block is emitted; the "
                "(proton, electron) block is its transpose")
        return _block_resolve(_cross_hessian_terms(name), label, ["a", "nb", "i", "nj"])
    if row_species == "proton":
        if not is_neo:
            raise ValueError("proton-species orbital Hessian requires a NEO model")
        return _block_resolve(_proton_hessian_terms(name), label, ["na", "nb", "ni", "nj"])
    return _block_resolve(_electron_hessian_terms(name), label, ["a", "b", "i", "j"])


# einsum-char relabel taking the column rotation indices onto the row (the diagonal)
_HESS_DIAG_RELABEL = {"electron": {"b": "a", "j": "i"}, "proton": {"B": "A", "J": "I"}}


def orbital_hessian_diag_ir(name, species="electron", rotation_classes=("o", "v"),
                            internal="active", label="hdiag"):
    """The diagonal orbital-Hessian preconditioner ``h_pq = H_pq,pq`` over the rotation
    blocks of ``species``, as explicit block JSONL IR.

    pdaggerq sums repeated generator labels, so it cannot emit the rank-2 diagonal
    directly. Instead this relabels the column rotation indices of the (unfused, block-
    resolved) same-block Hessian onto the row (electron b->a, j->i): the diagonal of a
    sum is the sum of the per-term diagonals, and each term drops to rank-2.

    Active-space: the exact diagonal of each rotation block (``rotation_classes`` over
    c/o/v/x, internal active). An x-active block's diagonal carries the *diagonal* of
    the x-block integral (g[x,.,x,.]) -- one repeated x, O(N_vir), which neocc builds
    as a J/K diagonal. Per-block targets ``hdiag["<row><col>"]``; default ("o","v")
    stays bare ``hdiag``. Verified against diag(orbital_hessian_ir) to ~1e-14."""
    if species not in _ROT_ROW:
        raise ValueError(f"species must be 'electron' or 'proton', not {species!r}")
    if internal != "active":
        raise ValueError("internal indices are RDM-contracted and must be 'active'")
    single = tuple(rotation_classes) == ("o", "v")
    colrow = _HESS_DIAG_RELABEL[species]
    if species == "proton" and not any(op in model(name).H for op in ("fp", "gep")):
        raise ValueError("proton-species diagonal requires a NEO model")

    def block_hessian(hi, lo):                            # diagonal: row = col = (hi, lo)
        if species == "proton":
            terms, letters = _proton_hessian_terms(name), ["na", "nb", "ni", "nj"]
        else:
            terms, letters = _electron_hessian_terms(name), ["a", "b", "i", "j"]
        ext = {letters[0]: hi, letters[1]: hi, letters[2]: lo, letters[3]: lo}
        return einsums.parse_ir(_block_resolve(terms, "H", letters, ext_classes=ext, drop_inactive_rdm=True))

    out, seen = [], set()
    for hi, lo in _rotation_blocks(rotation_classes, species):
        tgt = label + ("" if single else f'["{hi}{lo}"]')
        for st in block_hessian(hi, lo):
            ops = [{**o, "indices": [colrow.get(l, l) for l in o["indices"]]} for o in st["operands"]]
            t = st["target"]
            dseen, cls = [], []
            for lb, c in zip((colrow.get(l, l) for l in t["indices"]), t["classes"]):
                if lb not in dseen:        # dedupe [a,a,i,i] -> [a,i]
                    dseen.append(lb)
                    cls.append(c)
            out.append(json.dumps({
                "target": {"name": tgt, "indices": dseen, "classes": cls, "is_intermediate": False},
                "is_assignment": tgt not in seen, "coeff": st["coeff"], "operands": ops}))
            seen.add(tgt)
    return out


def _sigma_from_hessian(hess_lines, row_pos, col_pos, trial, label, seen):
    """Turn a Hessian block H[.] into a sigma-vector contribution sigma[row] +=
    coeff * (operands) * trial[col], by contracting the column rotation indices of
    each term against the trial tensor instead of leaving them open. ``row_pos`` /
    ``col_pos`` index into the Hessian target's four indices (row = sigma's open
    indices, col = contracted with the trial)."""
    out = []
    for line in hess_lines:
        st = json.loads(line)
        t = st["target"]
        row = {"name": label, "indices": [t["indices"][p] for p in row_pos],
               "classes": [t["classes"][p] for p in row_pos], "is_intermediate": False}
        col_cls = [t["classes"][p] for p in col_pos]
        kap = {"name": f'{trial}["{"".join(col_cls)}"]', "indices": [t["indices"][p] for p in col_pos],
               "classes": col_cls, "is_intermediate": False}
        out.append(json.dumps({"target": row, "is_assignment": label not in seen,
                               "coeff": st["coeff"], "operands": st["operands"] + [kap]}))
        seen.add(label)
    return out


# ---------------------------------------------------------------------------------
# Orbital gradient/Hessian by ROTATION-DIFFERENTIATING THE FIXED-RDM ENERGY.
#
# The orbital gradient and Hessian ARE the first and second derivatives of the energy
# that energy_from_rdm_ir traces, taken at FIXED RDMs with the integrals rotated by
# exp(kappa). Deriving them that way -- rather than from pq_helper's commutators --
# gives h, g AND gep uniformly from one rule, and is what the finite-difference test
# checks. It replaces (a) pq_helper's DOUBLE commutator, whose two-body piece is wrong
# (the single one is fine), (b) the unsymmetrized <[[H,A],B]> that the Hessian used to
# emit, and (c) the hand-derived _GEP_*_HESS_TERMS.
#
# Guarded by models_test.test_orbital_gradient_finite_difference.
# ---------------------------------------------------------------------------------

_ROT_INTEGRALS = ("h", "f", "g")     # these rotate; the RDMs (D*) are FIXED; d is a delta


def _energy_bare_terms(name):
    """The fixed-RDM energy in bare form -- the quantity the orbital gradient/Hessian
    differentiate. _block_resolve maps the all-proton h/D1 to hp/D1_n and the mixed g/D2
    to gep/D2_ep. The D2 pairing is the CONSUMER's (last two slots swapped vs g), i.e.
    the same [0,1,3,2] pairing energy_from_rdm_ir uses. h/hp are the BARE cores (see the
    contract on energy_from_rdm_ir): every mean field comes from the two-body terms."""
    H = model(name).H
    terms = ["1.0 h(p,q) D1(p,q)",
             "0.5 g(p,q,r,s) D2(p,q,s,r)"]
    if any(op in H for op in ("fp", "gep")):
        terms += ["1.0 h(np,nq) D1(np,nq)",                 # -> hp . D1_n
                  "1.0 g(p,np,q,nq) D2(np,p,q,nq)"]         # -> gep . D2_ep
    if "vp" in H:                                           # >=2 quantum protons: p-p two-body
        # -> vp . D2_n, the proton analogue of the electron 1/2 g.D2 (same pairing).
        # It vanishes identically for ONE quantum proton (a 2-RDM needs two particles), so
        # a single-proton finite-difference test cannot see whether it is present -- which
        # is exactly how it went missing from the OO quantities. Guarded by the two-proton
        # arm of models_test.test_orbital_gradient_finite_difference.
        terms += ["0.5 g(np,nq,nr,ns) D2(np,nq,ns,nr)"]     # -> vp . D2_n
    return terms


def _rot_deriv(terms, vir, occ, species):
    """d/dkappa^{species}_{vir,occ} of each bare-form term, at FIXED RDMs.

    Only the INTEGRALS rotate. For an integral slot carrying an index ``l`` of this
    species, the rotated tensor's derivative contributes::

        + delta(l, occ) * [that slot -> vir]   - delta(l, vir) * [that slot -> occ]

    A general (summed) ``l`` lets the delta collapse: every OTHER occurrence of ``l`` in
    the term becomes the constrained index. An external ``l`` (the row's own a/i) keeps an
    explicit Kronecker ``d(l, .)`` -- which is where the Hessian's delta terms come from.
    Applying this twice gives the second derivative (chain rule through the slots)."""
    out = []
    for term in terms:
        coeff, tensors = _parse_rdm_term(term)
        for ti, (nm, idx) in enumerate(tensors):
            if nm not in _ROT_INTEGRALS:
                continue
            for s, l in enumerate(idx):
                sp, fx = _classify_letter(l)
                if sp != species:
                    continue
                for sgn, put, cons in ((+1.0, vir, occ), (-1.0, occ, vir)):
                    new = [(n, list(ix)) for n, ix in tensors]
                    new[ti][1][s] = put
                    extra = []
                    if fx is None:                       # general -> collapse the delta
                        for k, (n, ix) in enumerate(new):
                            for m in range(len(ix)):
                                if (k, m) != (ti, s) and ix[m] == l:
                                    ix[m] = cons
                    else:                                # external -> explicit Kronecker
                        extra = [("d", [l, cons])]
                    body = " ".join(f"{n}({','.join(ix)})" for n, ix in new + extra)
                    out.append(f"{sgn * coeff} {body}")
    return out


def _swap_labels(terms, pairs):
    """Swap index labels pairwise in every term (used to symmetrize the Hessian)."""
    m = {}
    for x, y in pairs:
        m[x] = y
        m[y] = x
    out = []
    for term in terms:
        coeff, tensors = _parse_rdm_term(term)
        body = " ".join(f"{n}({','.join(m.get(l, l) for l in ix)})" for n, ix in tensors)
        out.append(f"{coeff} {body}")
    return out


def _halve(terms):
    return [f"{0.5 * _parse_rdm_term(t)[0]} " + t.split(None, 1)[1] for t in terms]


def _same_species_hessian(name, vir, occ, vir2, occ2, species):
    """d2E/dkappa2 for one species. The two rotation generators do NOT commute, so the
    second derivative of exp(kappa) is the SYMMETRIZED double derivative
    1/2 (M + M^T) -- emitting M alone (as the old double-commutator route did) is not
    the Hessian and is not even symmetric under (row)<->(col)."""
    G = _rot_deriv(_energy_bare_terms(name), vir, occ, species)
    M = _rot_deriv(G, vir2, occ2, species)
    return _halve(M) + _halve(_swap_labels(M, [(vir, vir2), (occ, occ2)]))


def _electron_hessian_terms(name):
    """Fixed-RDM electron-electron orbital Hessian (rows a,i; cols b,j)."""
    return _same_species_hessian(name, "a", "i", "b", "j", "e")


def _proton_hessian_terms(name):
    """Fixed-RDM proton-proton orbital Hessian (rows na,ni; cols nb,nj)."""
    return _same_species_hessian(name, "na", "ni", "nb", "nj", "p")


def _cross_hessian_terms(name):
    """Fixed-RDM electron-proton CROSS Hessian d2E/dkappa^e_ai dkappa^p_nbNj. The
    electron and proton rotation generators act on different species and COMMUTE, so the
    mixed partial needs no symmetrization. Only gep survives (h/g/hp carry one species)."""
    G = _rot_deriv(_energy_bare_terms(name), "a", "i", "e")
    return _rot_deriv(G, "nb", "nj", "p")


def orbital_sigma_ir(name, species="electron", rotation_classes=("o", "v"), internal="active",
                     label="sigma", trial="kappa", trial_n="kappa_n"):
    """Matrix-free orbital Hessian-vector product (``sigma = H . kappa``) for ``species``
    as explicit block JSONL IR -- the primitive for trust-region OO in a large basis,
    which never materializes H.

    Active-space + large ``N_vir``: the sigma **row** (output) rotation index ranges
    over ``rotation_classes`` (core c, active o/v, external x); the trial ``kappa``
    spans the same rotation blocks. Each block is built by contracting the class-split
    Hessian's **column** against ``kappa["<col>"]``, so the column's external (x) index
    is *folded* (density-driven, O(N_vir)) and only the row's external index stays free
    -- one free inactive-virtual index per term, the J/K shape neocc's engine consumes.
    The ``x-x`` Hessian block is never formed; it enters sigma only as this folded
    contraction. Default ``("o","v")`` reproduces the plain active-active sigma.

    NEO: ``sigma^e = H^ee.kappa^e + H^ep.kappa^p`` and ``sigma^p = H^pp.kappa^p +
    (H^ep)^T.kappa^e`` (default rotation only for now; the cross-block active-space
    split is the follow-up). neocc supplies D1/D2 (active), the integral blocks, and the
    trial each micro-iteration."""
    if species not in _ROT_ROW:
        raise ValueError(f"species must be 'electron' or 'proton', not {species!r}")
    if internal != "active":
        raise ValueError("internal indices are RDM-contracted and must be 'active'")
    is_neo = any(op in model(name).H for op in ("fp", "gep"))
    single = tuple(rotation_classes) == ("o", "v")
    seen, out = set(), []

    def same_species(sp):                                 # H^ss . kappa^s over row x col blocks
        terms = _proton_hessian_terms(name) if sp == "proton" else _electron_hessian_terms(name)
        lets = ["na", "nb", "ni", "nj"] if sp == "proton" else ["a", "b", "i", "j"]
        tr = trial_n if sp == "proton" else trial
        res = []
        for rhi, rlo in _rotation_blocks(rotation_classes, sp):
            rsuf = "" if single else f'["{rhi}{rlo}"]'
            for chi, clo in _rotation_blocks(rotation_classes, sp):
                hess = _block_resolve(terms, "H", lets, drop_inactive_rdm=True,
                                      ext_classes={lets[0]: rhi, lets[1]: chi, lets[2]: rlo, lets[3]: clo})
                res += _sigma_from_hessian(hess, (0, 2), (1, 3), tr, label + rsuf, seen)
        return res

    def cross(row_sp):                                    # H^ep coupling: sigma^e += H^ep.kappa^p
        res = []                                          #                sigma^p += (H^ep)^T.kappa^e
        for ehi, elo in _rotation_blocks(rotation_classes, "electron"):
            for phi, plo in _rotation_blocks(rotation_classes, "proton"):
                cr = _block_resolve(_cross_hessian_terms(name), "H", ["a", "nb", "i", "nj"],
                                    drop_inactive_rdm=True,
                                    ext_classes={"a": ehi, "i": elo, "nb": phi, "nj": plo})
                if row_sp == "electron":                  # row = electron (a,i); contract proton col (nb,nj)
                    rsuf = "" if single else f'["{ehi}{elo}"]'
                    res += _sigma_from_hessian(cr, (0, 2), (1, 3), trial_n, label + rsuf, seen)
                else:                                     # row = proton (nb,nj); contract electron col (a,i)
                    rsuf = "" if single else f'["{phi}{plo}"]'
                    res += _sigma_from_hessian(cr, (1, 3), (0, 2), trial, label + rsuf, seen)
        return res

    if species == "electron":
        out += same_species("electron")
        if is_neo:
            out += cross("electron")
    else:
        if not is_neo:
            raise ValueError("proton-species sigma requires a NEO model")
        out += same_species("proton")
        out += cross("proton")
    return out
