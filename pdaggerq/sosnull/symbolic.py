"""Fermionic normal ordering with symbolic Gram-to-operator index maps."""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass, replace

from itertools import product, permutations

from math import factorial

from string import ascii_letters

from typing import Iterable





@dataclass(frozen=True, order=True)
class PatternGenerator:
    r"""The symbolic word ``A_pattern(I)`` with one index per pattern slot."""

    pattern: str

    def __post_init__(self) -> None:
        if any(symbol not in "xo" for symbol in self.pattern):
            raise ValueError("patterns may contain only 'x' (creator) and 'o' (annihilator)")

    @property
    def degree(self) -> int:
        return len(self.pattern)

    @property
    def particle_change(self) -> int:
        return self.pattern.count("x") - self.pattern.count("o")

    @property
    def parity(self) -> str:
        r"""``"even"`` or ``"odd"``, following the degree.

        ``Delta N = n_x - n_o`` and ``degree = n_x + n_o`` differ by ``2 n_o``, so
        ``Delta N == degree (mod 2)`` always.  A particle-change block therefore
        never mixes parities: every ``Delta N`` block holds patterns of one degree
        parity only, and the Gram matrix is block diagonal with respect to the
        even/odd split as well as with respect to ``Delta N``.

        This is what makes every particle-conserving product ``even``-degree
        (``even+even`` and ``odd+odd`` are both even), which in turn is why only
        ``q_6``, ``q_8``, ... exist and never ``q_5`` or ``q_7``.

        It does **not** imply that the two sectors can be treated independently
        when their normal-ordered branches contribute to the same operator.
        """

        return "even" if self.degree % 2 == 0 else "odd"

    @property
    def name(self) -> str:
        return self.pattern or "1"


@dataclass(frozen=True, order=True)
class SpinPatternGenerator(PatternGenerator):
    r"""An ``x/o`` generator with one ``a/b`` spin label per operator slot."""

    spins: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.spins) != self.degree:
            raise ValueError("spin and operator patterns must have the same length")
        if any(spin not in "ab" for spin in self.spins):
            raise ValueError("spin patterns may contain only 'a' (alpha) and 'b' (beta)")

    @property
    def alpha_particle_change(self) -> int:
        return sum(
            1 if symbol == "x" else -1
            for symbol, spin in zip(self.pattern, self.spins)
            if spin == "a"
        )

    @property
    def beta_particle_change(self) -> int:
        return sum(
            1 if symbol == "x" else -1
            for symbol, spin in zip(self.pattern, self.spins)
            if spin == "b"
        )

    @property
    def twice_sz_change(self) -> int:
        return self.alpha_particle_change - self.beta_particle_change

    @property
    def spin_block_key(self) -> tuple[int, int]:
        r"""Return ``(Delta N, 2 Delta S_z)`` for Gram blocking."""

        return self.particle_change, self.twice_sz_change

    @property
    def name(self) -> str:
        return f"{self.pattern}/{self.spins}"

    def index_labels(self, side: str) -> tuple[str, ...]:
        """Return labels such as ``I0_a`` in this generator's slot order."""

        return tuple(
            f"{side}{slot}_{spin}" for slot, spin in enumerate(self.spins)
        )


@dataclass(frozen=True)
class Rank2SpinBlock:
    """One rank-2 SOS block from Eqs. 36--40, before forming products."""

    name: str
    generators: tuple[SpinPatternGenerator, ...]
    antisymmetric_spatial_pairs: bool = False

    @property
    def spin_block_key(self) -> tuple[int, int]:
        keys = {generator.spin_block_key for generator in self.generators}
        if len(keys) != 1:
            raise ValueError(f"block {self.name!r} mixes symmetry sectors")
        return next(iter(keys))

    @property
    def gram_family_shape(self) -> tuple[int, int]:
        size = len(self.generators)
        return size, size

    def dimension(self, number_of_spatial_orbitals: int) -> int:
        """Return the full Gram dimension for a spatial orbital count."""

        if number_of_spatial_orbitals < 0:
            raise ValueError("number of spatial orbitals must be non-negative")
        if self.antisymmetric_spatial_pairs:
            spatial_dimension = (
                number_of_spatial_orbitals
                * (number_of_spatial_orbitals - 1)
                // 2
            )
        else:
            spatial_dimension = number_of_spatial_orbitals**2
        return len(self.generators) * spatial_dimension


_RANK2_SPIN_BLOCK_SPECS = {
    # Eqs. 30, 36
    "G": (("xo", "aa"), ("xo", "bb"), ("ox", "aa"), ("ox", "bb")),
    # Eqs. 31, 37--38.  Spins are in operator-slot order, so alpha -> beta
    # is a*(beta) a(alpha), namely xo/ba.
    "G_alpha_beta": (("xo", "ba"), ("ox", "ab")),
    "G_beta_alpha": (("xo", "ab"), ("ox", "ba")),
    # Eqs. 32, 34, 39
    "D_alpha_alpha": (("oo", "aa"),),
    "D_beta_beta": (("oo", "bb"),),
    "D_alpha_beta": (("oo", "ba"), ("oo", "ab")),
    # Eqs. 33, 35, 40
    "Q_alpha_alpha": (("xx", "aa"),),
    "Q_beta_beta": (("xx", "bb"),),
    "Q_alpha_beta": (("xx", "ba"), ("xx", "ab")),
}

_ANTISYMMETRIC_SPATIAL_PAIR_BLOCKS = {
    "D_alpha_alpha",
    "D_beta_beta",
    "Q_alpha_alpha",
    "Q_beta_beta",
}


def generate_rank2_spin_block(name: str) -> Rank2SpinBlock:
    r"""Generate one spin- and particle-number-blocked SOS Gram basis.

    The returned generators are the paper's ``O`` monomials in operator-slot
    order.  This function does not form ``O^dagger O`` or generate Wick maps.

    Available names are ``G``, ``G_alpha_beta``, ``G_beta_alpha``,
    ``D_alpha_alpha``, ``D_beta_beta``, ``D_alpha_beta``,
    ``Q_alpha_alpha``, ``Q_beta_beta``, and ``Q_alpha_beta``.
    """

    try:
        specs = _RANK2_SPIN_BLOCK_SPECS[name]
    except KeyError as error:
        available = ", ".join(_RANK2_SPIN_BLOCK_SPECS)
        raise ValueError(
            f"unknown rank-2 spin block {name!r}; choose one of: {available}"
        ) from error

    return Rank2SpinBlock(
        name=name,
        generators=tuple(
            SpinPatternGenerator(pattern, spins) for pattern, spins in specs
        ),
        antisymmetric_spatial_pairs=name in _ANTISYMMETRIC_SPATIAL_PAIR_BLOCKS,
    )


def generate_rank2_spin_blocks() -> tuple[Rank2SpinBlock, ...]:
    """Generate the complete catalog of nine rank-2 spin blocks."""

    return tuple(generate_rank2_spin_block(name) for name in _RANK2_SPIN_BLOCK_SPECS)


def generate_einsum(
    wick_branch: WickBranch,
    canonical_label: Mapping[int, tuple[tuple[str, ...], tuple[str, ...]]],
) -> str:
    """Return the ``C_<degree> += ...`` einsum statement for one Wick branch.

    ``canonical_label`` maps total output degrees (0, 2, 4, ...) to pairs of
    creator and annihilator coefficient label tuples. The entry selected by
    ``wick_branch.degree`` must match the branch's remaining operator degree,
    after contractions.

    Annihilator coefficient labels run opposite their operator-word order.
    Each label must be a distinct single ASCII letter, and each group's
    length must match the corresponding group in ``gram_to_canonical_perm``.

    The input tensor is ``blocks_tensors[delta_n][(sigma, tau)]``, with axes
    in the original I-slot, J-slot order. Contracted slots share a summed
    label. Dummy labels are renamed locally if they conflict with the output.
    The returned statement expects ``einsum``, ``blocks_tensors``, and an
    initialized ``C_<degree>`` in scope. Fully contracted branches update
    ``C_0`` using the degree-zero label pair ``((), ())``.
    """
    # 1. Check the input and validate the canonical labels.
    # Select labels for the operator degree remaining after contractions, then
    # check the creator/annihilator counts, einsum syntax, and Gram block charge.
    if not isinstance(canonical_label, Mapping):
        raise TypeError("canonical_label must be a dictionary keyed by output degree")
    if wick_branch.degree not in canonical_label:
        raise ValueError(
            f"canonical_label has no labels for WickBranch degree {wick_branch.degree}"
        )
    selected_label = canonical_label[wick_branch.degree] # grab the canonical label set to use

    try:
        # the selected_label variable should be a tuple of (I, J) labels where
        # I and J are tuples of labels in the canonical set.
        if isinstance(selected_label, str):
            raise ValueError
        creators, annihilators = selected_label
        canonical_y = (tuple(creators), tuple(annihilators))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "canonical_label must contain creator and annihilator label sequences"
        ) from error

    # join canonical labels into one long list of labels
    # remember that annihilator labels (canonical_y[1]) are
    # reverse the order of the operator word.  This ensures
    # that a gram matrix that is symmetric hermitain has 
    # off diagonal elements corresponding to hermitian conjugate
    # operator pairs.  Also this means that reshaping from
    # the matrix to tensor form (assuming a C-order matrix) can
    # just use .reshape(<tensor_label_shape>, order='C')
    labels = canonical_y[0] + canonical_y[1]
    if len(labels) != wick_branch.degree:
        raise ValueError(
            f"canonical label degree {len(labels)} does not match "
            f"WickBranch degree {wick_branch.degree}"
        )
    if tuple(map(len, canonical_y)) != tuple(
        map(len, wick_branch.gram_to_canonical_perm)
    ):
        raise ValueError(
            "canonical creator and annihilator label counts must match the branch"
        )
    if any(
        not isinstance(label, str) or len(label) != 1 or label not in ascii_letters
        for label in labels
    ):
        raise ValueError("canonical labels must be single ASCII letters for einsum")
    if len(set(labels)) != len(labels):
        raise ValueError("canonical labels must be distinct")

    delta_n = wick_branch.sigma.count("x") - wick_branch.sigma.count("o")
    if delta_n != wick_branch.tau.count("x") - wick_branch.tau.count("o"):
        raise ValueError("sigma and tau must have the same particle change")

    # 2. Substitute the canonical labels into the branch's Gram indices.
    # First rename summed dummy indices that would collide with free output
    # labels. Use a local copy so the original WickBranch remains reusable.
    used_labels = set(labels)
    dummy_indices = []
    for dummy in wick_branch.dummy_indices:
        if len(dummy) != 1 or dummy not in ascii_letters or dummy in used_labels:
            dummy = next((label for label in ascii_letters if label not in used_labels), None)
            if dummy is None:
                raise ValueError("not enough distinct ASCII letters for einsum indices")
        dummy_indices.append(dummy)
        used_labels.add(dummy)
    branch = replace(wick_branch, dummy_indices=tuple(dummy_indices))

    # The inverse map places free labels and repeated contraction labels back
    # into the original I-slot, J-slot order of the Gram coefficient tensor.
    gram_label = branch.p_inv_y2g_sigma_tau(canonical_y)

    # 3. Generate the einsum update from Gram axes to canonical axes.
    # Concatenate each pair of label tuples before joining their letters:
    # e.g. (('I',), ('M',)) becomes 'IM'. Include the branch's sign and weight.
    subscripts = "".join(gram_label[0] + gram_label[1]) + "->" + "".join(labels)
    tensor = f"blocks_tensors[{delta_n}][({wick_branch.sigma!r}, {wick_branch.tau!r})]"
    return (
        f"C_{wick_branch.degree} += {wick_branch.coeff!r} "
        f"* einsum({subscripts!r}, {tensor})"
    )


@dataclass(frozen=True)
class WickBranch:
    gram_degree: int
    degree: int
    coeff: float
    sigma: str
    tau: str
    gram_labels: tuple[tuple[str, str]]
    gram_to_canonical_perm: tuple[tuple[str, str]]
    # canonical_to_gram_perm: tuple[tuple[str, str]]
    dummy_indices: tuple[str, ...]
    contractions: tuple[tuple[str, str], ...]

    def g2y_einsum(
        self,
        canonical_label: Mapping[int, tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> str:
        """Return the canonical update using the degree-indexed label dictionary."""
        return generate_einsum(self, canonical_label)

    def p_g2y_sigma_tau(self, raw_labels):
        # user specifies raw_labels which is their label set for I_J.
        # I_J is a label -> operator map where operator is defined by sigma,tau
        # N(I_J) takes the output of that map an normal orders
        # p_g2y(N(I_J)) -> new label permutation of the normal ordered op
        # so output of funciton is a different label-> opertor map where defintiion is sigma ='xxx...' tau= 'xxx...'
        # note tau is daggered after you read it.
        
        # index_transform tells you where the indices in the sigma,tau map end up
        # index_transform_left = dict(zip(self.gram_labels[0], self.gram_to_canonical_perm[0]))
        # index_transform_right = dict(zip(self.gram_labels[1], self.gram_to_canonical_perm[1]))
        index_transform = dict(zip(self.gram_labels[0] + self.gram_labels[1], 
                                   self.gram_to_canonical_perm[0] + self.gram_to_canonical_perm[1]))

        index_to_raw_label = dict(zip(self.gram_labels[0] + self.gram_labels[1], raw_labels[0] + raw_labels[1]))
        new_idx_transform = [index_to_raw_label[index_transform[rl]] for rl in index_transform.keys()]
        return (tuple(new_idx_transform[:len(self.gram_to_canonical_perm[0])]), tuple(new_idx_transform[len(self.gram_to_canonical_perm[0]):]))

        # return (tuple([index_to_raw_label[index_transform_left[rl]] for rl in index_transform_left.keys()]), 
        #         tuple([index_to_raw_label[index_transform_right[rl]] for rl in index_transform_right.keys()])
        #         )

    def p_inv_y2g_sigma_tau(self, raw_labels):
        # direct permutation
        if self.gram_degree == len(raw_labels[0]) + len(raw_labels[1]):
            # E(y8) = a*(A), a*(B), a*(C), a*(D), a(P), a(Q), a(R), a(S)
            # then we first note that this label is the ouptut of `p_g2y_sigma_tau`
            # so the inverse of this function we want to know which gram matrix
            # indices contribue to this y8 index (coeff, or operator...how ever you want to say it)
            # (('j', 'k', 'l', 'r'), ('t', 'q', 'p', 'm')) = (("A", "B", "C", "D"), ("S", "R", "Q", "P"))
            # in code this is self.gram_to_caonical_perm = raw_labels 
            # or dict(zip(self.gram_to_canonical_perm[0] + self.gram_to_canonical_perm[1],
            #             raw_labels[0] + raw_labels[1] 
            # ))
            index_to_raw_label = dict(zip(self.gram_to_canonical_perm[0] + self.gram_to_canonical_perm[1], 
                                          raw_labels[0] + raw_labels[1]
                                          )
                                    )
            # which is the map we need to keep in mind, 
            # j->A, k->B, l->C, r->D, t->S, q->R, p->Q, m->P
            # but we want to know what the original gram indices map (('j','k','l','m'),('p','q','r','t'))
            # so now just iterate through this order and make the substitution.
            # (('A','B','C','P'), ('Q','R', 'D', 'S'))
            # candidate_inverse_I_J = (('A','B','C','P'), ('Q', 'R', 'D', 'S'))
            candidate_inverse_I_J = (
                tuple([index_to_raw_label[rl] for rl in self.gram_labels[0]]),
                tuple([index_to_raw_label[rl] for rl in self.gram_labels[1]])
            )
            check_ABCDSRQP = self.p_g2y_sigma_tau(candidate_inverse_I_J)
            assert check_ABCDSRQP == raw_labels
            return candidate_inverse_I_J
        # some number of contractions has happened.
        else:
            # E(y6) = a*(A), a*(B), a*(C), a(P), a(Q), a(R)
            # then we first note that this label is the ouptut of `p_g2y_sigma_tau_contraction`
            # so the inverse of this function we want to know which gram matrix
            # indices contribue to this y6 index (coeff, or operator...how ever you want to say it)
            # (('j', 'k', 'l',), ('q', 'p', 'm')) = (("A", "B", "C"), ("R", "Q", "P"))
            # or dict(zip(self.gram_to_canonical_perm[0] + self.gram_to_canonical_perm[1],
            #             raw_labels[0] + raw_labels[1] 
            # ))
            # we will also need to map contracted indices to their contraction variable
            # these come as a list of tuples so we want to take all those indices map them to their 
            # dummy indices
            dummy_index_to_raw_label = {}
            for kdx, (idx, jdx) in enumerate(self.contractions):
                dummy_index_to_raw_label[idx] = self.dummy_indices[kdx]
                dummy_index_to_raw_label[jdx] = self.dummy_indices[kdx]
            index_to_raw_label = dict(zip(self.gram_to_canonical_perm[0] + 
                                          self.gram_to_canonical_perm[1], 
                                          raw_labels[0] + 
                                          raw_labels[1]
                                          )
                                    )
            index_to_raw_label.update(dummy_index_to_raw_label)

            # which is the map we need to keep in mind, 
            # j->A, k->B, l->C, q->R, p->Q, m->P, r->u, t->u
            # but we want to know what the original gram indices map (('j','k','l','m'),('p','q','r','t'))
            # so now just iterate through this order and make the substitution.
            # (('A','B','C','P'), ('Q','R', 'u', 'u'))
            candidate_inverse_I_J = (
                tuple([index_to_raw_label[rl] for rl in self.gram_labels[0]]),
                tuple([index_to_raw_label[rl] for rl in self.gram_labels[1]])
            )
            check_ABCDSRQP = self.p_g2y_sigma_tau(candidate_inverse_I_J)
            assert check_ABCDSRQP == raw_labels
            return candidate_inverse_I_J

 






def generate_pattern_generators(
    degree: int,
    *,
    homogeneous: bool = False,
    minimal: bool = True,
) -> tuple[PatternGenerator, ...]:
    r"""Generate word patterns, including lower degrees unless requested otherwise.

    With ``minimal=True`` (the default) only the CAR-reduced patterns
    ``x^p o^q`` are produced -- the manuscript's generator basis of Eq. (66).
    Every raw word normal orders into those, so for the degree-*at-most*-``d``
    basis the two choices span the same ``V_d`` and generate the same SOS cone;
    the full word set merely reparameterises it redundantly
    (``span{x, oxx, xox, xxo} = span{x, xxo}``).

    Counting
    --------
    The two sizes have different growth, and it is the *raw* set that is
    exponential:

    * **raw** -- a pattern is a free choice of ``x`` or ``o`` in each of ``m``
      slots, so there are ``2^m`` words of degree ``m`` and

      .. math::
         \sum_{m=0}^{d} 2^m = 2^{d+1} - 1

      words of degree at most ``d``.  Exponential in ``d``.

    * **minimal** -- a CAR-reduced pattern is fixed once the *number* of leading
      creators is chosen, because the ``x``'s must all precede the ``o``'s.
      Degree ``m`` therefore contributes exactly the ``m + 1`` patterns
      ``x^p o^(m-p)``, ``p = 0..m``, and

      .. math::
         \sum_{m=0}^{d} (m+1) = \frac{(d+1)(d+2)}{2}

      in total.  Quadratic in ``d``.

    The exponential factor removed is precisely the ordering freedom that CAR
    normal ordering eliminates: the ``C(m,p)`` interleavings of ``p`` creators
    and ``m-p`` annihilators all normal order onto the single canonical word
    ``x^p o^(m-p)`` plus strictly lower-degree terms, and
    ``sum_p C(m,p) = 2^m`` collapses to one representative per ``p``.

    Within a fixed particle-change block the collapse is sharper still.  A raw
    block of degree ``m`` holds ``C(m, (m+eta)/2)`` patterns, whereas the
    minimal block holds *at most one per degree* -- ``p`` and ``q`` are pinned
    by ``p - q = eta`` and ``p + q = m`` -- so its size is just the number of
    admissible degrees,

    .. math::
       |\Sigma_\eta^{\min}| = \left\lfloor \frac{d - |\eta|}{2} \right\rfloor + 1 ,

    e.g. ``d = 3, eta = +1`` gives ``{x, xxo}``, which is Appendix C's
    ``o_(+1) = (O_ijk, a^dag_l)``.

    Equivalence caveat
    ------------------
    The equivalence is specific to the degree-at-most basis.  Under
    ``homogeneous=True`` the mixed words of degree exactly ``d`` normal order
    into *lower*-degree monomials as well, so dropping them shrinks the span:
    ``minimal=False, homogeneous=True`` is the setting that reproduces
    Mazziotti's factorised ``(2,4)`` Table-I families, which are built from
    ``S_sigma`` for all sixteen quartic patterns.
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")
    degrees: Iterable[int] = (degree,) if homogeneous else range(degree + 1)
    if minimal:
        generators = [
            PatternGenerator("x" * creators + "o" * (rank - creators))
            for rank in degrees
            for creators in range(rank + 1)
        ]
    else:
        generators = [
            PatternGenerator("".join(symbols))
            for rank in degrees
            for symbols in product("xo", repeat=rank)
        ]
    return tuple(
        sorted(
            generators,
            key=lambda generator: (
                generator.particle_change,
                generator.degree,
                generator.pattern,
            ),
        )
    )


def _operator_token(symbol: str, label: str) -> str:
    return f"a*({label})" if symbol == "x" else f"a({label})"


def _adjoint_symbol(symbol: str) -> str:
    return "o" if symbol == "x" else "x"

def _indices_to_operators(I_idx, J_idx, sigma, tau):
    A_sigma_I = [_operator_token(sym, label) for (sym, label) in zip(sigma, I_idx)]
    A_tau_J_dag = [_operator_token(_adjoint_symbol(sym), label) for (sym, label) in zip(reversed(tau), reversed(J_idx))]
    ladder_word = A_sigma_I + A_tau_J_dag
    return ladder_word


def sn_x_sn_permutations(creator_indices, annihilator_indices):
    """Yield every S_n x S_n index permutation and its common weight."""
    creators = tuple(creator_indices)
    annihilators = tuple(annihilator_indices)

    if len(creators) != len(annihilators):
        raise ValueError("creator and annihilator ranks must match")

    weight = 1.0 / factorial(len(creators)) ** 2

    for creator_perm, annihilator_perm in product(
        permutations(creators),
        permutations(annihilators),
    ):
        yield creator_perm, annihilator_perm, weight

def _sn_x_sn_index_assignments(I_idx, J_idx, sigma, tau):
    """Yield permuted (I, J) arguments for A_sigma(I) A_tau(J)^dagger."""
    sigma_pattern = getattr(sigma, "pattern", sigma)
    tau_pattern = getattr(tau, "pattern", tau)

    if len(I_idx) != len(sigma_pattern) or len(J_idx) != len(tau_pattern):
        raise ValueError("index and pattern lengths must agree")

    # Slots in the actual product-word order. For the J side, taking the
    # adjoint reverses the word and exchanges x <-> o.
    # Each slot is:
    # (Gram side, original slot index, effective x/o symbol, index label)

    A_sigma_I_slots = [
        ("I", k, sym, label)
        for k, (sym, label) in enumerate(zip(sigma_pattern, I_idx))
    ]

    A_tau_J_dag_slots = [
        ("J", k, _adjoint_symbol(sym), label)
        for k, sym, label in zip(
            reversed(range(len(tau_pattern))),
            reversed(tau_pattern),
            reversed(J_idx),
        )
    ]
    slots = A_sigma_I_slots + A_tau_J_dag_slots

    def slot_label(slot):
        _, _, _, label = slot
        return label

    creator_slots = [slot for slot in slots if slot[2] == "x"]
    annihilator_slots = [slot for slot in slots if slot[2] == "o"]

    creator_labels = tuple(map(slot_label, creator_slots))
    annihilator_labels = tuple(map(slot_label, annihilator_slots))

    for creator_perm, annihilator_perm, weight in sn_x_sn_permutations(
        creator_labels,
        annihilator_labels,
    ):
        permuted_I = list(I_idx)
        permuted_J = list(J_idx)

        for slots_, labels in (
            (creator_slots, creator_perm),
            (annihilator_slots, annihilator_perm),
        ):
            for (side, k, _, _), label in zip(slots_, labels):
                target = permuted_I if side == "I" else permuted_J
                target[k] = label

        yield tuple(permuted_I), tuple(permuted_J), weight

def generate_wickbranch_antisymm(
    sigma: PatternGenerator,
    tau: PatternGenerator,
    *,
    antisymmetrize: bool = True,
) -> list[WickBranch]:
    """Build Wick branches, optionally averaging the Gram-index permutations.

    With ``antisymmetrize=False``, normal order only the original index
    assignment, with unit weight.
    """

    from .pq_backend import PDaggerQSymbolicBackend

    sigma_pattern = getattr(sigma, "pattern", sigma)
    tau_pattern = getattr(tau, "pattern", tau)
    # Both modes require matching particle changes for the Gram block.
    if sigma_pattern.count("x") - sigma_pattern.count("o") != (
        tau_pattern.count("x") - tau_pattern.count("o")
    ):
        raise ValueError("creator and annihilator ranks must match")
    I_idx = tuple(f"I{k}" for k in range(len(sigma_pattern)))
    J_idx = tuple(f"J{k}" for k in range(len(tau_pattern)))
    gram_degree = len(I_idx) + len(J_idx)
    backend = PDaggerQSymbolicBackend()
    branches = []

    # Choose the full permutation orbit or the original labels with unit weight.
    assignments = (
        _sn_x_sn_index_assignments(I_idx, J_idx, sigma_pattern, tau_pattern)
        if antisymmetrize else ((I_idx, J_idx, 1.0),)
    )
    for permuted_I, permuted_J, weight in assignments:
        ladder_word = _indices_to_operators(
            permuted_I,
            permuted_J,
            sigma_pattern,
            tau_pattern,
        )
        if ladder_word:
            terms = backend.normal_order_ladder_word(ladder_word, weight)
        else:
            # Identity is already normal ordered; retain its scalar coefficient.
            terms = ((weight,),)

        for term in terms:
            factors = term[1:]
            creators = tuple(
                factor[3:-1] for factor in factors if factor.startswith("a*(")
            )
            annihilators = tuple(
                factor[2:-1] for factor in factors if factor.startswith("a(")
            )
            contractions = tuple(
                tuple(factor[2:-1].split(","))
                for factor in factors
                if factor.startswith("d(")
            )
            canonical_indices = (creators, tuple(reversed(annihilators)))

            branches.append(
                WickBranch(
                    coeff=float(term[0]),
                    gram_degree=gram_degree,
                    degree=len(creators) + len(annihilators),
                    gram_labels=(permuted_I, permuted_J),
                    gram_to_canonical_perm=canonical_indices,
                    sigma=sigma_pattern,
                    tau=tau_pattern,
                    dummy_indices=tuple("uvwxz"[: len(contractions)]),
                    contractions=contractions,
                )
            )

    return branches

if __name__ == "__main__":
    from pdaggerq.sosnull import pq_backend
    backend = pq_backend.PDaggerQSymbolicBackend()
    examples = (
        # One creator and one annihilator: a single S_1 x S_1 assignment.
        ("x", "x"),
        # The particle-hole G example: four S_2 x S_2 assignments.
        ("xo", "xo"),
        # Unequal generator degrees with the same particle change.
        ("x", "xxo"),
    )

    # for sigma_pattern, tau_pattern in examples:
    #     I_idx = tuple(f"I{k}" for k in range(len(sigma_pattern)))
    #     J_idx = tuple(f"J{k}" for k in range(len(tau_pattern)))
    #     assignments = tuple(
    #         _sn_x_sn_index_assignments(
    #             I_idx,
    #             J_idx,
    #             sigma_pattern,
    #             tau_pattern,
    #         )
    #     )
    #     print(sigma_pattern, tau_pattern) #"".join([_adjoint_symbol(x) for x in list(reversed(tau_pattern))]))
    #     for aa in assignments:
    #         print(f"{aa=}")
    #         A_sigma_I = [_operator_token(sym, label) for (sym, label) in zip(sigma_pattern, aa[0])]
    #         A_tau_J_dag = [_operator_token(_adjoint_symbol(sym), label) for (sym, label) in zip(reversed(tau_pattern), reversed(aa[1]))]
    #         ladder_word = A_sigma_I + A_tau_J_dag
    #         print(f"{ladder_word=}")
    #         terms = backend.normal_order_ladder_word(ladder_word, aa[-1])
    #         for tt in terms:
    #             print("\t", tt)
    #         print()

    #     print("-" * 80)

    from collections import defaultdict
    canonical_ops = {}
    canonical_equations = defaultdict(list)
    create_coninical = 'IJKL'
    annihilate_canonical = 'MNOP'

    for nn in range(0, 5, 2):
        creator_y = [create_coninical[ii] for ii in range(nn//2)]
        annihilator_y = [annihilate_canonical[ii] for ii in range(nn//2)]
        print((tuple(creator_y), tuple(annihilator_y)))
        canonical_ops[nn] = (tuple(creator_y), tuple(annihilator_y))

    sigma = PatternGenerator('xo')
    tau = PatternGenerator('xo')
    wick_branches = generate_wickbranch_antisymm(sigma, tau)

    for degree in [0, 2, 4]:
        yn = canonical_ops[degree]
        print(f"{degree=}")
        for wb in wick_branches:
            if wb.degree == degree:
                print(wb)
                p_inv_yn = wb.p_inv_y2g_sigma_tau(yn)
                print(p_inv_yn)
