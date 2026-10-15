from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Iterable

import pdaggerq


def get_degrees(pq_terms):
    degrees = []
    for term in pq_terms:
        counts = 0
        for tt in term[1:]:
            counts += 1 if tt.startswith(("a(", "a*(")) else 0
        degrees.append(counts)
    return degrees


class PDaggerQSymbolicBackend:
    """Normal order explicit ladder words in pdaggerq's true vacuum."""

    name = "pdaggerq-true-vacuum-symbolic-deltas"

    def __init__(self) -> None:
        self._pq = pdaggerq.pq_helper("true")

    def normal_order_ladder_word(
        self,
        word: Iterable[str],
        coefficient: float = 1.0,
    ) -> tuple[tuple[str, ...], ...]:
        """Normal order ``word`` and lexically canonicalize each result.

        The prefactor is applied after pdaggerq performs the CAR expansion.
        pdaggerq serializes nonintegral input coefficients at limited precision,
        while a unit-coefficient CAR expansion has integral coefficients.  The
        post-expansion scaling therefore preserves the caller's float exactly.
        Terms containing a delta between ``_a`` and ``_b`` labels are removed.
        """

        scale = _finite_decimal(coefficient)
        if scale == 0:
            return ()
        try:
            self._pq.add_operator_product(1.0, list(word))
            self._pq.simplify()
            normal_ordered_terms = tuple(
                (
                    str(_finite_decimal(term[0]) * scale),
                    *term[1:],
                )
                for term in self._pq.strings()
            )
            # return self.canonicalize_normal_ordered_terms(normal_ordered_terms)
            return self.filter_spin_mismatched_delta_terms(normal_ordered_terms)
        finally:
            self._pq.clear()

    @staticmethod
    def filter_spin_mismatched_delta_terms(
        terms: Iterable[Iterable[str]],
    ) -> tuple[tuple[str, ...], ...]:
        """Drop terms containing a Kronecker delta between opposite spins.

        Spin-orbital labels use a final ``_a`` or ``_b`` suffix.  A delta with
        two known, unequal suffixes is identically zero, so its entire product
        term is removed.  Unlabeled and same-spin deltas are left unchanged.
        """

        filtered_terms = []
        for raw_term in terms:
            term = tuple(raw_term)
            spin_mismatch = False

            for factor in term[1:]:
                open_parenthesis = factor.find("(")
                if not factor.startswith("d") or open_parenthesis < 0:
                    continue
                if not factor.endswith(")"):
                    continue

                delta_name = factor[:open_parenthesis]
                if delta_name.startswith("d_"):
                    delta_spins = delta_name[2:]
                    if len(delta_spins) == 2 and delta_spins[0] != delta_spins[1]:
                        spin_mismatch = True
                        break

                labels = tuple(
                    label.strip()
                    for label in factor[open_parenthesis + 1 : -1].split(",")
                )
                if len(labels) != 2:
                    continue

                left_spin = _spin_suffix(labels[0])
                right_spin = _spin_suffix(labels[1])
                if (
                    left_spin is not None
                    and right_spin is not None
                    and left_spin != right_spin
                ):
                    spin_mismatch = True
                    break

            if not spin_mismatch:
                filtered_terms.append(term)

        return tuple(filtered_terms)

    # def canonicalize_normal_ordered_terms(
    #     self,
    #     terms: Iterable[Iterable[str]],
    # ) -> tuple[tuple[str, ...], ...]:
    #     """Put normal-ordered terms into one lexical fermionic convention.

    #     pdaggerq moves creators to the left of annihilators, but it does not
    #     consistently sort operators that are already within those two groups.
    #     This method sorts both groups lexicographically and multiplies the
    #     coefficient by the parity of the required permutations.  Kronecker
    #     deltas are canonicalized separately using ``d(i,j) = d(j,i)`` and do
    #     not affect the fermionic sign.  Identical resulting terms are collected.
    #     """

    #     collected: defaultdict[tuple[str, ...], Decimal] = defaultdict(Decimal)

    #     for raw_term in terms:
    #         term = tuple(raw_term)
    #         if not term:
    #             raise ValueError("normal-ordered terms must include a coefficient")

    #         coefficient = _finite_decimal(term[0])
    #         creators: list[str] = []
    #         annihilators: list[str] = []
    #         deltas: list[tuple[str, str]] = []
    #         seen_annihilator = False

    #         for factor in term[1:]:
    #             if factor.startswith("a*(") and factor.endswith(")"):
    #                 if seen_annihilator:
    #                     raise ValueError(
    #                         f"term is not vacuum-normal ordered: {term!r}"
    #                     )
    #                 label = factor[3:-1].strip()
    #                 if not label:
    #                     raise ValueError(f"invalid creator factor {factor!r}")
    #                 creators.append(label)
    #             elif factor.startswith("a(") and factor.endswith(")"):
    #                 seen_annihilator = True
    #                 label = factor[2:-1].strip()
    #                 if not label:
    #                     raise ValueError(f"invalid annihilator factor {factor!r}")
    #                 annihilators.append(label)
    #             elif factor.startswith("d(") and factor.endswith(")"):
    #                 labels = tuple(
    #                     label.strip() for label in factor[2:-1].split(",")
    #                 )
    #                 if len(labels) != 2 or not all(labels):
    #                     raise ValueError(f"invalid delta factor {factor!r}")
    #                 deltas.append(tuple(sorted(labels)))
    #             else:
    #                 raise ValueError(f"unsupported pdaggerq factor {factor!r}")

    #         sorted_creators, creator_sign = self._signed_lexical_sort(creators)
    #         sorted_annihilators, annihilator_sign = self._signed_lexical_sort(
    #             annihilators
    #         )
    #         if creator_sign == 0 or annihilator_sign == 0:
    #             continue

    #         coefficient *= creator_sign * annihilator_sign
    #         canonical_factors = (
    #             tuple(f"a*({label})" for label in sorted_creators)
    #             + tuple(f"a({label})" for label in sorted_annihilators)
    #             + tuple(f"d({left},{right})" for left, right in sorted(deltas))
    #         )
    #         collected[canonical_factors] += coefficient

    #     return tuple(
    #         (format(coefficient, "+f"), *factors)
    #         for factors, coefficient in sorted(collected.items())
    #         if coefficient != 0
    #     )

    # @staticmethod
    # def _signed_lexical_sort(labels: Iterable[str]) -> tuple[tuple[str, ...], int]:
    #     """Return lexical order and the fermionic sign of that permutation."""

    #     labels = tuple(labels)
    #     if len(set(labels)) != len(labels):
    #         return tuple(sorted(labels)), 0

    #     inversions = sum(
    #         labels[left] > labels[right]
    #         for left in range(len(labels))
    #         for right in range(left + 1, len(labels))
    #     )
    #     sign = -1 if inversions % 2 else 1
    #     return tuple(sorted(labels)), sign


def _finite_decimal(raw: object) -> Decimal:
    """Parse a coefficient without accepting NaN or infinities."""

    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid coefficient {raw!r}") from error
    if not value.is_finite():
        raise ValueError(f"coefficient must be finite, got {raw!r}")
    return value


def _spin_suffix(label: str) -> str | None:
    """Return the final ``a``/``b`` spin component of a spin-orbital label."""

    _, separator, spin = label.rpartition("_")
    return spin if separator and spin in {"a", "b"} else None


def _integer_coefficient(raw: str) -> int:
    value = Decimal(raw)
    integral = value.to_integral_value()
    if value != integral:
        raise ValueError(f"expected an integral CAR coefficient, got {raw!r}")
    return int(integral)
