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

from ._pdaggerq import pq_helper, pq_graph

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


def residual_graph(name, amplitude, df=True, opt_level=6, label="R"):
    """Optimized pq_graph for the amplitude residual
    ``<proj(amplitude)| e^-T H e^T |0> = 0``."""
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
    return _optimized(pq, label, df, opt_level)


def residual_ir(name, amplitude, df=True, opt_level=6, label="R"):
    """The amplitude residual as ``to_strings("ir")`` JSONL lines."""
    g = residual_graph(name, amplitude, df=df, opt_level=opt_level, label=label)
    return g.to_strings("ir")
