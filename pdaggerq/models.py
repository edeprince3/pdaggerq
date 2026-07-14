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
* single-proton NEO: every ``vp`` model has a ``<name>-1p`` counterpart (e.g.
  ``neo-ccsd-1p``) auto-derived by dropping ``vp`` and the >=2-proton amplitudes -- the
  exact equations for one quantum proton. Bit-for-bit with the full model there (the
  dropped pieces are identically zero) but cheaper, and, having no ``vp``, it takes the
  plain SI-free proton Fock (no dressing). A consumer dispatches on the proton count:
  1 -> ``-1p``, >=2 -> the full ``vp`` model.

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

from ._pdaggerq import pq_helper, pq_graph
from . import einsums
from .spin import get_spin_labels

__all__ = [
    "Model", "MODELS", "PROJECTION", "EXCITATION", "H_ELEC", "H_NEO", "H_NEO_PP",
    "model", "lambda_amps",
    "energy_graph",
    "residual_graph", "residual_ir", "spin_cases", "residual_blocks",
    "lambda_graph", "lambda_ir",
    "gradient_graph", "gradient_ir",
    "hessian_graph", "hessian_ir",
    "rdm_graph", "rdm_ir", "rdm_block_ir",
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
H_NEO    = ("f", "v", "fp", "gep")        # single-proton NEO (no proton-proton term)
H_NEO_PP = ("f", "v", "fp", "gep", "vp")  # + proton-proton fluctuation (multi-proton)


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
    that needs >=2 protons. Both are identically zero for a single proton (``vp``'s two-
    body part annihilates the lone proton; a proton-rank>=2 amplitude is antisymmetric in
    the one occupied proton), so the reduced model is bit-for-bit with the full one for a
    single proton -- but cheaper (no ``vp`` terms, fewer amplitudes) and, without ``vp``,
    it wants the plain SI-free proton Fock rather than the dressed one."""
    H = tuple(h for h in m.H if h != "vp")
    T = [a for a in m.T if _proton_count(a) <= 1]
    return Model(m.name + "-1p", H, T)


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


#: representative line-class sizes for the dimension-aware optimizer cost model
#: (pq_graph option "dims"). pq_graph's default metric counts every summation line
#: equally, so it mis-ranks NEO candidates badly: a nuclear-occupied line (O = number
#: of quantum protons, typically 1!) is scored like an electronic virtual line. With
#: dims set, candidate intermediates/orderings are ranked by numeric flop estimates at
#: these sizes instead. The values are *ratios*, not real basis sizes -- chosen
#: representative of NEO targets (v/o ~ 4, protonic basis ~ o, DF aux ~ 3v); codegen is
#: frozen per model, so decisions are optimal near these ratios. "O" is filled in
#: per-model by :func:`_dims_for` (1 for single-proton models, 2 when the model carries
#: >=2-proton amplitudes). Electron-only models keep pq_graph's scale-safe default
#: metric (no dims) so their codegen is unchanged.
DIMS = {"o": 10.0, "v": 40.0, "V": 10.0, "L": 1.0, "Q": 120.0}


def _dims_for(name):
    """Dimension table for the optimizer cost model: :data:`DIMS` with the nuclear
    occupied size set to the model's proton count, or None (dimension-blind legacy
    metric) for electron-only models."""
    protons = max((_proton_count(a) for a in model(name).T), default=0)
    if protons == 0:
        return None
    return {**DIMS, "O": float(protons)}


def _optimized(pq, label, df, opt_level, dims=None, gep_traces=True):
    return _optimized_multi([(label, pq)], df, opt_level, dims, gep_traces)


def _optimized_multi(labeled_pqs, df, opt_level, dims=None, gep_traces=True):
    # nthreads=1: the pq_graph optimizer's substitution/fusion passes race on the
    # lazy caches (link_vector_, flop_scale_, base_hash_, ...) of Linkage objects that
    # are shared by shared_ptr across candidate intermediates -- forget()/compute_scaling()
    # mutate them mid-flight. At >1 thread this makes the *chosen* substitutions, and thus
    # the emitted code, nondeterministic (all variants are numerically identical; it is a
    # byte-reproducibility bug, not a math one -- ThreadSanitizer flags consolidate.cc /
    # substitute.cc / linkage.cc). Model codegen must be byte-reproducible (neocc freezes
    # it), so pin the optimizer to a single thread here; omp_set_num_threads(nthreads_)
    # then serializes every optimizer loop regardless of OMP_NUM_THREADS. Drop this once
    # the Linkage caches are made thread-safe (guard each getter/forget with Linkage::mtx_).
    # Guarded by models_test.test_opt_level_safe_default.
    options = {"opt_level": opt_level, "density_fitting": df, "nthreads": 1}
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


def _projected_pq(m, left):
    """pq_helper holding ``<left| e^-T H e^T |0>`` for a model, simplified."""
    pq = pq_helper("fermi")
    pq.set_left_operators([left])
    for h in m.H:
        pq.add_st_operator(1.0, [h], list(m.T))
    pq.simplify()
    return pq


def energy_graph(name, df=True, opt_level=None):
    """Optimized pq_graph for the correlation energy ``<0| e^-T H e^T |0>``."""
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    pq = _projected_pq(m, ["1"])
    return _optimized(pq, "energy", df, opt_level, _dims_for(name))


def residual_graph(name, amplitude, df=True, opt_level=None, label="R",
                   spin_case=None, nuclear_spin="high-spin"):
    """Optimized pq_graph for the amplitude residual
    ``<proj(amplitude)| e^-T H e^T |0> = 0``.

    spin_case : None -> spin-orbital (no blocking, the default). Otherwise a spin
                block name from :func:`spin_cases` (e.g. "abab", or NEO "aa_n") --
                the equation is restricted to that block via ``block_by_spin``.
    nuclear_spin : "high-spin" (single nuclear channel) or "full" -- see
                :mod:`pdaggerq.spin`.
    """
    opt_level = _opt_level_for(name, opt_level)
    m = model(name)
    if amplitude not in m.T:
        raise ValueError(f"model {name!r} has no amplitude {amplitude!r}; T={list(m.T)}")
    if amplitude not in PROJECTION:
        raise KeyError(f"no projection defined for amplitude {amplitude!r}")
    pq = _projected_pq(m, [PROJECTION[amplitude]])
    if spin_case is not None:
        cases = get_spin_labels([[PROJECTION[amplitude]]], nuclear_spin)
        if spin_case not in cases:
            raise ValueError(f"unknown spin_case {spin_case!r} for {amplitude!r}; "
                             f"choose from {sorted(cases)}")
        pq.block_by_spin(cases[spin_case])
    return _optimized(pq, label, df, opt_level, _dims_for(name))


def residual_ir(name, amplitude, df=True, opt_level=None, label="R",
                spin_case=None, nuclear_spin="high-spin"):
    """The amplitude residual as ``to_strings("ir")`` JSONL lines."""
    g = residual_graph(name, amplitude, df=df, opt_level=opt_level, label=label,
                       spin_case=spin_case, nuclear_spin=nuclear_spin)
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

    The emitted H is (vir,occ,vir,occ), i.e. o^2 v^2 -- the size of t2 -- so a consumer stores
    it and gets the Hessian DIAGONAL (preconditioner) and the SIGMA product
    ``sigma_ai = sum_bj H_(ai),(bj) kappa_bj`` from it directly, both trivially and with no
    v^4 object anywhere. There is therefore no separate T,Lambda diag/sigma emitter.

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
        # validated in pq_graph/tests/neo_rdm_energy_test.py. The e-p energy is
        # E_ep = +gep.D2_ep (gep carries the charge sign; see energy_from_rdm_ir).
        cP, cE, cE2, cP2 = block[0], block[1], block[2], block[3]
        LE = take("e", cE); LP = take("p", cP); LP2 = take("p", cP2); LE2 = take("e", cE2)
        return f"e2({LP},{LE},{LP2},{LE2})", [(LP, cP), (LE, cE), (LE2, cE2), (LP2, cP2)]

    raise ValueError(f"unknown RDM tensor {tensor!r}; choose D1/D1_n/D2/D2_n/D2_ep")


def _block_ir_from_strings(name, op, consumer, tgt):
    """Emit one RDM block's IR straight from ``pq.strings()``, bypassing ``pq_graph``.

    pq_graph's nuclear-index relabelling collapses a block's *internal* proton indices onto
    its *open* ones -- it cannot keep two same-class proton labels distinct -- silently
    restricting the contraction. That corrupts every block carrying both open and internal
    proton indices (``D1_n``, ``D2_n``, ``D2_ep``); the pure-electron blocks are unaffected,
    which is why it went unnoticed. Emitting from the simplified strings keeps every index
    distinct, so all tensors go through here.

    Kronecker deltas ``d(p,q)`` become ``Id["cc"]`` (the consumer supplies an identity);
    ``P(i,j)`` antisymmetrizers are expanded into signed statements over the open indices.
    """
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

    def cls_of(lab):                                     # occ-check case-insensitive: open
        nuc = len(lab) > 1 and lab[0] == "n"              # proton labels are nI/nA (upper),
        b = lab[1] if nuc else lab[0]                     # internal ones ni/na (lower)
        occ = b.lower() in "ijklmno"
        return ("O" if occ else "V") if nuc else ("o" if occ else "v")

    base_used = {"o": 0, "v": 0, "O": 0, "V": 0}
    base_map = {}
    for lab in out_pq:                                   # open indices get fixed leading letters
        c = cls_of(lab); base_map[lab] = LET[c][base_used[c]]; base_used[c] += 1
    out_idx = [base_map[lab] for lab in out_pq]

    stmts = []
    for term in strings:
        coeff = float(term[0])
        perms, factors = [], []
        for tok in term[1:]:
            if tok.startswith("P("):
                perms.append(tok[2:-1].split(","))
            elif "(" in tok:
                factors.append(tok)
        used = dict(base_used); m = dict(base_map)

        def letter(lab):
            if lab not in m:
                c = cls_of(lab); m[lab] = LET[c][used[c]]; used[c] += 1
            return m[lab]

        operands = []
        for tok in factors:
            nm = tok[:tok.index("(")]; idx = tok[tok.index("(") + 1:-1].split(",")
            letters = [letter(i) for i in idx]; classes = [cls_of(i) for i in idx]
            if nm == "d":                                # Kronecker delta -> identity block
                nm = f'Id["{classes[0]}{classes[1]}"]'
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
                "is_assignment": not stmts, "coeff": sgn * coeff, "operands": operands}))
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


def rdm_energy_reference(name, seed=17, no=2, nv=2, nO=1, nV=2):
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
