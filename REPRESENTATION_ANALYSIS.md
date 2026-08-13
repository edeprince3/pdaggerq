# Representation Analysis: Linkages, Terms, and Einsum

This document delineates the internal representations used by the pq_graph
optimizer / code generator, with particular attention to how **contractions**
(produced by substitution) and **additions** (produced by fusion) are encoded,
unpacked, pruned, substituted, and printed.  It highlights the representational
inconsistencies that lead to the corruption observed in
`ccsdt_with_spin_codegen.py` at `max_temps >= 359`.

## 1. Two Parallel Representations of a Contraction

Throughout the codebase a single tensor expression lives in two forms:

### Representation A — Linkage Tree

```
Linkage : public Vertex
  VertexPtr  left_, right_;    // children
  bool       addition_;         // false ⇒ contraction (*), true ⇒ addition (+)
  long       id_;               // ≥ 0 ⇒ temp, -1 ⇒ not a temp
  line_vector lines_;           // external lines (computed in build_connections)
  ConnecMap  connec_map_;       // left/right line pairing
  shape      flop_scale_, mem_scale_;
```

A contraction `A * B` is `Linkage(A, B, addition=false)`.
An addition `A + B` is `Linkage(A, B, addition=true)`.

### Representation B — Term (flat RHS)

```
Term
  VertexPtr      lhs_;          // result vertex
  VertexPtr      eq_;           // equation vertex (usually == lhs)
  vertex_vector  rhs_;          // **flat** list of factor vertices (a product)
  double         coefficient_;
  perm_list      term_perms_;   // symmetry-permutation pairs
  size_t         perm_type_;    // 0 / 1 / 2 / 3 / 6
  bool           is_assignment_;// true ⇒ "= " (initialise), false ⇒ " += "
```

`rhs_` is always interpreted as a **product** of its entries; there is no
per-element `addition_` flag in `vertex_vector`.  Additions live entirely
inside a single Linkage node placed as one entry of `rhs_`.

### Key invariant

`term_linkage()` = `Linkage::link(rhs_)`, building a right-leaning binary tree
of products.  `link_vector()` is its inverse: flatten the tree back into a flat
list — but it only ever reproduces a **product**, never an **addition**.

## 2. How `build_connections` treats additions vs contractions

In `Linkage::build_connections` (linkage.cc:47-239):

- **Contraction** (`addition_ == false`): internal lines (appearing on both
  sides) are contracted away; `lines_` = union of external lines.
- **Addition** (`addition_ == true`): the right operand's external lines are
  ignored — `lines_` is taken **entirely from `left_`**:

```cpp
for (uint_fast8_t i = 0; i < left_half; i++) {
    if (left_ext_idx[i] || addition_)   // addition_ ⇒ always add left lines
        add_line(left_lines[i]);
}
if (!addition_) {                       // skip right lines for additions
    for (uint_fast8_t i = 0; i < right_half; i++)
        if (right_ext_idx[i]) add_line(right_lines[i]);
}
```

Consequence: for an addition `(L + R)`, the **axis order of the result is
dictated solely by `L`'s line order**.  If the addition is later swapped to
`(R + L)`, the `lines_` re-order to `R`'s line order — silently changing the
einsum index string that the printer will emit for every usage.

`Linkage::relabel()` (linkage.cc:241) is a **no-op** for additions:

```cpp
VertexPtr Linkage::relabel() const {
    // TODO: fix relabeling for vertices with additions.
    return clone();
    ...
}
```

so canonical relabeling is unavailable to normalise the order.

## 3. `link_vector` — flattening a tree to a product list

`link_vector(regenerate, fully_expand)` (linkage_ops.cc:130-199) flattens the
**left spine** of the tree, treating every node along the way as a
contraction:

```cpp
if (left_->is_linked() && !left_->empty()) {
    if (!left_->is_expandable() && !fully_expand)
        result.push_back(left_);                   // stop at temp/addition
    else
        result += left_->link_vector(regenerate, fully_expand);
}
if (fully_expand && right_->is_linked() && !right_->empty())
    result += right_->link_vector(regenerate, fully_expand);
else if (!right_->empty() && fabs(right_->value()-1) > 1e-8)
    result.push_back(right_);
```

- If the root is an **addition**, `is_expandable()` is `false` (because
  `expand_addition` defaults to `false`), so the root is returned as a single
  entry and we never recurse — the addition stays intact as one entry.
- If the addition is hidden behind a temp (root is a temp), `is_expandable` is
  also `false` and the temp is returned as-is.
- The danger arises only if an addition's `id_` is later reset to `-1` (so it
  is no longer a temp AND `is_expandable` returns true due to caller passing
  `expand_addition=true`).  In that case `link_vector` would flatten the
  addition's left subtree as a product — silently losing the `+` operator.

## 4. `expand_rhs` — unpacking a linkage into a term

`Term::expand_rhs(const VertexPtr &term_link)` (term.cc:262-299):

```cpp
if (term_link->is_expandable(true))               // expand_scalar=true
    rhs_ = term_link->link_vector();                // ← flattens as product
else if (term_link->is_linked() && !term_link->is_temp() && !term_link->is_addition())
    rhs_ = { as_link(term_link)->left(), as_link(term_link)->right() };
else
    rhs_ = { term_link };                           // keep as single vertex
```

The three branches cover:

1. **Plain contraction** → `link_vector()` flattens left spine into
   `{A, B, C, ...}` — fast path.
2. **Addition** (`!is_temp && is_addition`) → `is_expandable(true)` is `false`,
   branch 2 is also `false` (because of `!is_addition()`), so it falls to
   branch 3: `rhs_ = { addition }`.  The addition survives as a single rhs
   vertex.
3. **Temp** → `is_expandable` and branch 2 are both skipped; `rhs_ = { temp }`.

So **a non-temp addition is preserved as one rhs entry**, while a plain
contraction is unpacked into many.  The unpacked form and the packed form must
remain mutually consistent, but as we will see in §7 neither `reorder()` nor
`prune()` upholds this consistency.

## 5. `term_linkage()` and `reorder()` round-trip

`term_linkage()` caches a `Linkage::link(rhs_)` — a right-leaning product tree
of `rhs_`.  `reorder()` (term.cc:301-326) is the standard re-optimization:

```cpp
compute_scaling(recompute);
best_linkage = term_linkage()->best_permutation();
expand_rhs(best_linkage);
```

`best_permutation()` (linkage_ops.cc:569-620) calls `permutations()`.  For an
**addition** the `permutations()` function (linkage_ops.cc:523-545) has a
dedicated special case:

```cpp
if (is_addition()) {
    const LinkagePtr &left_perm  = as_link(left_)->best_permutation();
    const LinkagePtr &right_perm = as_link(right_)->best_permutation();
    bool same_left_right = *left_perm == *right_perm;
    if (!same_left_right) {
        result.push_back(as_link(left_perm + right_perm));   // L+R
        result.push_back(as_link(right_perm + left_perm));   // R+L
    }
    return result;
}
```

`best_permutation()` then picks the "best" of `{ L+R, R+L }` using a sequence
of tie-breaks (linkage_ops.cc:590-608):

1. flops,
2. memory,
3. prefer right operand that is **not expandable** (i.e. a temp or leaf),
4. fewer `lines()`,
5. lexicographic `name()`.

**Hazard 1 — operand swap:** for `A + ratio*T_b` where `A` is a regular
contraction and `T_b` is a temp, tie-break 3 prefers `ratio*T_b + A` because
`A` (non-temp) on the right is expandable while `T_b` is not.  Swapping is
algebraically a no-op for the scalar sum, **but `lines_` is re-derived from
`left_` only**, so the addition's axis order now tracks `ratio*T_b`'s line
order.  If `T_b` and `A` have different line orders (which they need not —
the fusion guard checks `lines()` equality, not order), Z's axis order changes
silently.

More dangerously, **`reorder()` is called on declaration terms whose rhs_ is a
non-temp addition** (id=-1):
- `term_linkage()` returns the addition itself,
- `best_permutation()` enters the addition branch and may pick the swapped
  permutation,
- `expand_rhs(best_linkage)` falls to branch 3 (`rhs_ = { best_linkage }`),
  so `rhs_` now contains the **swapped** addition, while `lhs` (= merged
  vertex) still has the original (pre-swap) `lines_` order.

When this happens, the printer emits `+= ... einsum('b...->a...', ... )` for
the right operand, producing a **mis-indexed** `+=` line.  In my earlier run
with `expand_rhs` disabled the swap never happened.

## 6. Substitution: replacing a sub-contraction with a temp

`Term::substitute(linkage, graph_perms)` (substitute.cc:796-900):

```cpp
for (const auto &graph_perm : graph_perms) {
    auto matches = graph_perm->find_links(linkage);
    if (matches.empty()) continue;
    new_term_linkage = graph_perm;
    for (auto &m : matches) {
        MutableVertexPtr new_link = m->shallow();
        as_link(new_link)->copy_misc(linkage);          // copy id_, addition_, reused_
        new_term_linkage = as_link(new_term_linkage->replace(m, new_link).first);
    }
    // score ...
}
expand_rhs(best_linkage);
```

- It finds the linkage *by structural equality* (`Linkage::find_links` uses
  `operator==`, which ignores `id_` and `addition_` for sub-tree comparison —
  only `similar_root` checks `addition_`).  So a contraction temp will only
  match contractions, and an addition temp will only match additions.
- After replacement `expand_rhs` is invoked, which for a non-temp result will
  *flatten* the term — potentially destroying an addition that was packed at
  the root.
- `copy_misc` preserves `id_`, `addition_`, and `reused_` from the substitute
  linkage — so the matched sub-tree takes on the temp's identity consistently.

## 7. Pruning: `prune()` in consolidate.cc

`prune(keep_single_use)` (consolidate.cc:44-286) does four conceptually
separate things:

### 7.1 Identify single-use temps

```cpp
for (auto &vertex : term->rhs())
    num_occurrences += as_link(vertex)->count(temp, /*enter_temps=*/false);
```

`count(target, enter_temps=false, enter_additions=true)` (linkage_ops.cc:380)
stops descending into **temps**.  So for a usage term whose rhs_ = `{ Z }`
where `Z = (T_a + ratio*T_b)` (a temp with id ≥ 0), `count(T_a, false)` returns
**0** — T_a living *inside* Z is invisible to the usage tally.  Only the **Z
declaration term** (whose rhs_ *is* the addition with `id=-1`, i.e. not a temp)
contributes to T_a's count:

- `count(T_a, false)` on the addition root → `is_temp()` false, recurses into
  `left_`=T_a, which matches ⇒ count = 1.
- T_a's own declaration term (`T_a = …`) has lhs = T_a; its rhs_ does not
  contain T_a ⇒ that's the declaration, not a usage.

So after fusion: `count(T_a) == 1` (only the Z declaration).  When
`keep_single_use` is `false` (the final prune in `optimize()`), T_a is
scheduled for removal.

### 7.2 Remove the temp via `replace_id`

The `remove_unused` lambda wraps `Linkage::replace_id(temp, -1)`:

```cpp
auto remove_unused = [&sorted_to_remove](VertexPtr vertex){
    if (vertex->is_linked()) {
        for (auto &temp : sorted_to_remove) {
            auto [new_vertex, replaced] = as_link(vertex)->replace_id(temp, -1);
            if (replaced) vertex = new_vertex;
        }
    }
    return make_pair(vertex, made_replacement);
};
```

`replace_id(temp, -1)` (linkage_ops.cc:300-339) walks the tree, changing only
the `id_` field of any vertex that matches `temp` by `same_temp` (same id +
structure).  It does **not** change `addition_` or re-derive `lines_`.  The
shallow copy preserves `addition_` via `copy_misc` inside `set_properties`.
This is the cleanest part of prune and works even inside additions.

### 7.3 The `expand_rhs` / `reorder` block (opt_level ≥ 6)

```cpp
for (auto &term_ptr : all_terms) {
    MutableLinkagePtr term_link = as_link(term.term_linkage()->shallow());
    if (!term_link->is_temp()) continue;   // skip additive declarations etc.
    else term_link->factor();              // no-op (Linkage::factor returns false)

    term.expand_rhs(term_link);
    term.reorder(true);
}
```

For a term where **the term_linkage is a temp** (single-temp rhs, e.g. `R = T`),
- `expand_rhs(T)` ⇒ branch 3, `rhs_ = { T }`.
- `reorder(true)` calls `compute_scaling(true)` which recomputes
  `term_linkage_` from `rhs_`; then `best_permutation()` on a temp returns the
  identity; then `expand_rhs(best_linkage)` ⇒ `rhs_ = { T }` again.

For a declaration term whose `term_linkage()` is a **non-temp addition** (the
fused `Z = (A + ratio*T_b)` case, before the LHS is given an id), `is_temp()`
of the root is *false* (id=-1), so the loop skips this term.

**The loop does NOT itself corrupt additions.**  But its presence in
`prune()` runs `reorder(true)` on every surviving term — and on terms that
have just had a temp inlined by §7.2, which means the addition node inside
their rhs may now be re-examined by `reorder()`'s `best_permutation()` call,
triggering the operand-swap described in §5.

### 7.4 Recursive prune

`prune` recurses (`num_removed = prune(keep_single_use)`) until nothing is
removed.  Each pass rebuilds `all_temp_set` and re-counts.  Once inlined
temps become id=-1, they stop appearing in `all_temp_set` (the `id() == -1`
skip at line 73), so the recursion terminates.

## 8. Fusion: encoding a sum as a single temp

`LinkMerger::merge()` in fusion.cc:598-800.

### 8.1 Detecting a fusible pair

`LinkMerger::populate()` (fusion.cc:250-416) finds pairs `(T_a, T_b)` such
that for every paired usage term the **truncation terms** are structurally
identical.  A trunc term is built by replacing the temp with a *dummy vertex*
(`0.0 * Vertex("dummy")`) and sorting the rhs by name.  Two terms are
fuse-able iff, after this deletion, their rhs are structurally equivalent.

### 8.2 Building the fused linkage

For each paired truncation `i`:

```cpp
merged_vertex = target_infos[i].link->shallow();   // T_a
for (auto &merge_info : merge_infos) {
    MutableLinkagePtr target_vertex = as_link(merge_info[i].link->shallow()); // T_b
    if (fabs(ratio - 1.0) > 1e-10)
         merged_vertex = merged_vertex + ratio * target_vertex;
    else merged_vertex = merged_vertex + target_vertex;
}
as_link(merged_vertex)->factor();                   // no-op
bool last_add_bool = merged_vertex->is_addition();
as_link(merged_vertex)->copy_misc(target_infos[i].link);
as_link(merged_vertex)->is_addition() = last_add_bool;
as_link(merged_vertex)->id() = max_id;              // becomes temp Z
```

`merged_vertex` is an **addition linkage**:
```
Z = (T_a + ratio * T_b)
```
with `T_a`'s `lines_` (because `build_connections` for the addition takes the
left operand's lines — §2).

### 8.3 Rewriting target usage terms

```cpp
// overwrite the target terms with the new terms
size_t idx = 0;
for (auto &link_info : link_tracker_.link_track_map_[target_link])
    *link_info.term = new_terms[idx++];
```

The new term for index `i`:

```cpp
Term new_term = target_infos[i].trunc_term;
for (auto &vertex : new_term.rhs())
    if (vertex->is_linked())
        vertex = as_link(vertex)->replace(dummy, merged_vertex).first;
new_term.request_update();
new_term.reorder();
new_terms[i] = new_term.shallow();
```

- Dummy replaced by the **full addition** `Z` in the rhs.
- `reorder()` is called on the new term.  Inside, `term_linkage()` builds a
  fresh tree; if the rhs's root is an addition (which it is, since Z is an
  addition), then `best_permutation()` may swap the operands — §5.

**Hazard 2:** `reorder()` can swap the addition's operands here.  If the
swap happens, then the `lines_` of the `Z` instance stored in `new_terms[i]`
differ from the original `merged_vertex`'s `lines_` (which were derived from
`T_a`).  Every future pass that looks at `Z`'s `lines()` (printing, further
match checks) will see the swapped order.

### 8.4 Building the Z declaration

```cpp
Term new_def = chosen_decl->shallow();
new_def.eq()  = merged_vertex_init;
new_def.lhs() = merged_vertex_init;

MutableLinkagePtr merged_vertex_copy = as_link(merged_vertex_init->shallow());
merged_vertex_copy->id() = -1;
new_def.expand_rhs(merged_vertex_copy);     // branch 3: rhs_ = { add_id_minus_1 }
```

`merged_vertex_init` is the canonical instance built from slot `i=0` (with
labels relabeled — though relabel is also a no-op, §2).  It is shared across
all `new_terms[i]` via `eq.substitute(new_merged_link, true)` (line 792), so
**all rewritten target terms will reference the same `Z` instance** whose
`lines_` were derived from `target_infos[0]`'s `T_a` instance.

### 8.5 Destroying the merge temp's usage terms

```cpp
for (auto &merge_link : merge_links) {
    merge_link->forget(true);
    for (auto &link_info : link_tracker_.link_track_map_[merge_link])
        if (link_info.term) link_info.term->lhs() = nullptr;     // ← destroys term
}
```

Every term that referenced `T_b` gets its LHS set to `null`.  At lines
776-785 those terms are then physically removed from their equations:

```cpp
for (auto &[name, eq] : pq_graph_.equations()) {
    vector<Term> new_terms;
    for (auto &term : eq.terms())
        if (term.lhs() != nullptr) new_terms.push_back(term);
    eq.terms() = new_terms;
}
```

### 8.6 The "naïve invariant" and why it breaks

The implicit assumption is:

> *"For every usage term of `T_b` (the merge temp), the matching usage term of
> `T_a` (the target temp) shares the identical trunc rhs.  Therefore the
> rewrite of the `T_a`-usage into `… + Z + …` subsumes every `T_b`-usage's
> contribution (scaled by `ratio`), so dropping the `T_b`-usages loses
> nothing."*

This is **only true if every `T_b`-usage has a matched `T_a`-usage in the same
paired list with the same trunc rhs at the same sort position**.  There are
two ways this can fail:

#### Failure mode A: permed siblings

`populate()` sorts each link's info list by `(trunc_term.str, term.str)`.  If
two paired lists sort the same way position-by-position, every slot matches.
But the `LinkTracker::prune()` step (fusion.cc:184-235) imposes another
constraint: perms must be *consistent per link*.  It can *reject* some entries
from one list but not the other, leaving the lists truncated.  If the lists
end up with different lengths after prune, populate rejects the pair (size
check); if the same length but different content, populate's pair-wise
connectivity check fails and the pair is dropped.  So in *theory* we never
reach `merge()` with a mis-paired group.

#### Failure mode B: the **declaration term of Z** provides a new "usage" of
`T_b` that wasn't in the original `link_track_map_[T_b]`

When `merge()` builds `new_def` (§8.4) and later inserts it into the `temp`
equation (line 797), the **new Z declaration term** has rhs_ = `{ add(T_a,
ratio*T_b) }`.  Now:

- `Z` is a usage of `T_a` — and is also the *target's* declaration template.
- The OLD target declaration term (`T_a = …`) is **not** removed by `merge()`.
  `merge()` only walks `link_track_map_[T_b]` to null LHSes; it does **not**
  null `link_declare_map_[T_a]` or `link_declare_map_[T_b]`.  So `T_a` and
  `T_b` retain their original declaration terms (dead, but resident).

This is benign by itself, but it means the next `prune()` call sees **both**
`T_a` and `T_b` appearing exactly once each (inside the Z declaration), with
`keep_single_use=false` ⇒ both are inlined via `replace_id` ⇒ temporary
expressions get pasted into Z.  That inlining is algebraically correct but
produces a *much larger* rhs addition tree — which then feeds back into the
swap hazard of §5 on the *next* `reorder()` pass.

## 9. Printing: how additions are unpacked to `+=` lines

`Term::str()` (graph_printing.cc:486-672) handles three cases in order:

1. **Permutation terms** — expands a symmetry-permuted term into its multiple
   permuted siblings (lines 500-562).  ⚠ **Permutation unfolding happens at
   print time**, not in the stored term.  The storage representation is a
   single term with `perm_type_ > 0`; the printer is what materialises the 2
   / 3 / 6 sibling lines.

2. **Non-temp additions** (lines 567-581) — unpacks an addition into two
   printed lines: `=` for the left operand, `+=` for the right.

   ```cpp
   if (term_link->is_addition() && !term_link->is_temp()) {
       Term left_term = *this, right_term = *this;
       left_term.expand_rhs(term_link->left());
       right_term.expand_rhs(term_link->right());
       right_term.is_assignment_ = false;            // force +=
       return left_term.str() + '\n' + right_term.str();
   }
   ```

   - `right_term` keeps the same `lhs_` so `+=` correctly accumulates to Z.
   - **The recursion is structural**: if the left operand is itself an
     addition, it will be re-printed as another `=` / `+=` pair.  But if an
     operand is a temp (e.g. `T_a` inside the fused Z), the `is_temp()` guard
     at the top of this block fires (`!term_link->is_temp()` ⇒ skip), so the
     temp *leaf* is delegated to the printer.
   - Note: this branch **fires before the prune-inlined addition case**.  If
     the rhs is a *temp* whose definition is an addition (like the pruned Z in
     §7), the printer will *not* expand it here — instead it falls through to
     `format_term`, which prints `Z[...] = einsum(...)` using the LHS labels
     of Z as the output.

3. **General term** (line 671) — delegates to `printer->format_term(*this)`.

### 9.1 EinsumPrinter::format_term (einsum_printer.cc:162-262)

```cpp
einsum_string = t.term_linkage(true)->str();    // recursively built by Vertex::str
if (lhs_string != rhs_string)
    einsum_string = "einsum('" + rhs_string + "->" + lhs_string + "', " + einsum_string + " )";
```

The output indices come **from the term's LHS vertex's lines** (line 197-199);
the input indices come **from each rhs operand's lines** (line 76-81 of
`format_contraction`).  `Linkage::str()` (linkage.h:484) defers to
`str(format_temp=true)`, which recursively walks the tree printing each leaf
with its `lines_`.

**Print-time consistency requirement:** for every temp usage `Z[labels_i]`
the label sequence **must correspond positionally to the same axes** that Z's
declaration assigned.  If `Z`'s `lines_` was swapped between declaration and
usage (§5), the printer will emit a different `labels_i` string at the usage
site — `Z["bcakij"]` at one place and `Z["bcakij"]` at another — even though
they refer to the same stored tensor.  The einsum will then contract the
wrong axes.

## 10. Historical Hypothesis (Superseded)

Combining §5, §7, §8, and §9, here is the precise failure mechanism in
`ccsdt_with_spin_codegen.py` at `max_temps = 359`:

1. `substitute()` runs in batched mode.  When `temp_counts_["temp"]` reaches
   `max_temps - batch_size_ + 1`, one additional substitution creates one
   more candidate intermediate than at `max_temps = 358`.

2. After the substitution round, `merge_intermediates()` is invoked at full
   depth.  One **additional fusion group** (the 42nd vs the 41 from the
   passing run) is now permitted because the extra temp provides a 42nd
   fusible partner.

3. For this 42nd group, `merge()`:
   - Builds `Z = (A + ratio*T_b)` as an addition (§8.2).
   - Calls `new_term.reorder()` on the rewritten target term (§8.3),
     triggering `best_permutation()` on the addition's non-temp form
     (id=-1).
   - **`best_permutation()` swaps the operands** because `A` (non-temp) on
     the right is expandable while `T_b` (temp) is not (tie-break 3 of §5).
   - `Z`'s `lines_` are now taken from `ratio*T_b` rather than from `A`, even
     though `T_a` and `T_b` (as fused links) had *equal* lines per the
     populate guard — the line **order** within the addition's `lines_`
     effectively shifted to `T_b`'s local label sequence.

4. `Z`'s declaration (§8.4) has `lhs = merged_vertex_init`, which was
   constructed from slot `i=0` *before* any potential swap and whose
   `lines_` follow `A`'s label sequence.  Now the declaration `lhs` and the
   swapped `rhs` addition have *different axis orderings*.

5. `prune(false)` runs (§7).  With `keep_single_use=false`, it inlines
   `T_a` and `T_b` into the Z declaration via `replace_id(-1)`.  The
   declaration's rhs becomes raw `einsum(... eri t2 t2 ...) + einsum(... t2
   tmps_...)` — but the swap from step 3 means the *outer* addition of the
   declaration has the swapped axis order; the printer (§9) emits the
   addition's `+=` second line against the declaration's *original* LHS axis
   order, prepending a spurious `einsum('b...->a...', ...)` permutation that
   transposes the operand.

6. The user-observable symptom: the generated code's einsum strings differ
   from the `max_temps=358` reference at exactly the two contributions that
   this 42nd group's `T_b` was supposed to fold in — one permuted sibling of
   the bcjm / bcjl pair goes missing or is transposed (the inner einsum
   generates a `cc` double-contraction on the same axis, corrupting the
   energy by ~3×10⁻⁶).

## 11. Earlier Diagnosis (Superseded)

The root cause is the **inconsistency between the stored Term representation
of a permuted expression (a single term with `perm_type_ > 0`) and the fusion
algorithm's rewriting model (which assumes one stored term = one printed
line)**.  When a merge temp appears in a permuted term, only ONE sibling of
that permuted term is rewritten by the fusion; the other siblings — which the
printer materialises later from the same stored term — are silently destroyed
when the merge temp's usage term is nulled in §8.5.

### Adopted fix: reject permuted terms from fusion

In `LinkTracker::prune()` (fusion.cc:184-235), add an early `remove_link`
check that rejects any linkage whose usage set includes a term with
`perm_type_ != 0`.  This declines fusion opportunities for permuted terms;
the cost is some foregone optimisation of permuted residuals, but
correctness is preserved.

Why this is the right level to address the bug:

- The printer's permutation unfolding (§9, graph_printing.cc:500-562) is a
  fundamental design choice of the codebase — permed siblings are emitted
  from a single stored term.  Changing this would require either (a)
  expanding permed terms at insertion time, with a large memory and
  performance cost, or (b) teaching the fusion algorithm to rewrite every
  permuted sibling individually, which in turn requires the fusion rewrite
  to be symmetric in the line permutations — a non-trivial extension.
- The substitution algorithm (`Term::substitute`, §6) works correctly for
  permuted terms because it operates on the linkage *subtree*, which is
  identical across siblings.  Only the fusion algorithm has this problem,
  because it physically nulls the original usage term rather than modifying
  a subtree.
- Rejecting permuted terms from fusion is a one-line predicate at the entry
  to the fusion-candidate pruning step.  It is cheap, conservative, and
  leaves the rest of the optimisation pipeline (substitution, merge_terms,
  prune) untouched.

### Fix alternative considered and rejected

An earlier version of this analysis proposed restricting
`Linkage::permutations()` to emit only the identity for additions, on the
theory that the `L+R ↔ R+L` swap was the immediate corruption vector (§5).
Empirical testing disproved this: applying that change broke
`PQ_FUSION_MAX_MERGES=41` runs that previously passed, meaning the swap is
load-bearing for some correct fusion outputs.  The fix was reverted.

### What *not* to do

- Do **not** try to "remember" the canonical axis order across swaps.  This
  would require `Linkage::relabel()` to actually relabel additions, which is
  explicitly marked as unimplemented (linkage.cc:241-302) and would be a much
  larger surgery.
- Do **not** attempt to rewrite `link_vector()` to honour `addition_` — that
  would break dozens of call-sites that rely on `link_vector()` returning a
  flat product list.
- Do **not** disable the entire `prune()` opt_level≥6 block at
  consolidate.cc:266-277 — that block is no longer the bug, and disabling
  it impedes the rest of the optimisation without addressing the permuted-
  sibling destruction.

## 12. Earlier Summary Table (Superseded)

| Subsystem               | What it assumes about additions                    | What it actually does                                  | Consistent? |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------------ | ----------- |
| `build_connections`     | addition result lines = left operand's lines      | drops right operand's external lines                   | ✅ (definition) |
| `link_vector`           | tree is a product; additions are opaque leaves    | returns addition root as a single entry               | ✅ |
| `expand_rhs`            | additions are not expandable                       | delegates to `link_vector` only for contractions      | ✅ |
| `permutations` / `best_permutation` | operand swap preserves semantics     | **swaps L+R ↔ R+L**, silently reorders `lines_`       | ❌ |
| `relabel`               | additions can be relabeled                         | no-op for additions                                    | ❌ |
| `substitute`            | addition temps match addition sub-trees            | matches by structure; preserves `addition_`            | ✅ |
| `prune().count`         | temps are opaque, additions transparent            | `enter_temps=false` skips into Z → T_a/T_b invisible  | ✅ (by design) |
| `prune().replace_id`    | id replacement preserves structure                 | only changes `id_`; calls `set_properties`              | ✅ |
| `prune().expand_rhs` block | reorder does not swap addition operands          | invokes `reorder()` which may swap                      | ❌ |
| `LinkMerger::merge`     | rewriting T_a usage captures all T_b contributions| rewrites only paired T_a-usages; nulls all T_b-usages  | ⚠ ties to populate guard |
| `LinkMerger::merge`     | target's old declaration is removed                | only T_b usages are nulled; T_a/T_b declarations stay  | ❌ |
| `LinkMerger::merge`     | `reorder()` does not mutate the merged vertex     | `reorder()` may swap addition operand on the new term | ❌ |
| `Term::str` (printer)   | addition's LHS axis order matches both operands   | emits `einsum('rhs->lhs', ...)` for the right operand  | ❌ (downstream of §5) |

The "Consistent?" column summarises the mismatches that compound to produce
the 359-temps failure.

## 13. Corrected Diagnosis and Fix

The earlier permutation-only diagnosis in sections 10-12 is superseded by the
following verified explanation.  `expand_permutations=True` reproduces the
failure, so stored term permutations are not required for the bug.

### 13.1 Nested fusion representation

Suppose an earlier fusion has produced:

```text
T2 = A - B - 0.5*C
```

Internally, `T2` is a `Linkage` with `addition_ == true` and a valid temp id.
If a later fusion combines it with `T1`, the intended result is:

```text
Z = T1 + r*T2
```

The right-hand side of `Z` is therefore a product node whose right child is
itself an addition temp.  This is a valid linkage tree and must remain valid
when printed or inlined.

The first representation defect was in `EinsumPrinter::format_contraction()`.
It parenthesized only operands for which `is_expandable()` was true.  Addition
nodes intentionally return false from that method, so a scalar/product around
an inlined addition was emitted as:

```python
-1.0 * A + B + 0.5 * C
```

instead of:

```python
-1.0 * (A + B + 0.5 * C)
```

The first expression changes the algebra because the scalar applies only to
the first operand.  The corrected printer now parenthesizes non-temp addition
operands whenever they occur inside a contraction:

```cpp
if (op->is_expandable(false, true))
    s = "(" + s + ")";
```

Temp additions remain named tensor operands; non-temp additions are grouped
before multiplication.

### 13.2 Fusion group ownership

`LinkMerger::merge()` mutates terms in place.  For each target linkage it:

1. Rewrites target usage terms with the fused linkage.
2. Sets the merge-link usage terms' `lhs` to `nullptr`.
3. Removes null terms from the equations.

Consequently, two fusion groups may not share a `Term*`.  The previous
ref/comp pruning loop could accept a group as a `comp_link` and later reject
it as a `ref_link`, leaving the earlier insertion in
`new_link_merge_map`.  That stale group survived into `merge()`, where an
earlier nullification could invalidate a term later read by another group.

The corrected pruning pass builds complete groups and greedily accepts only
groups whose combined target/declaration/usage `Term*` set is disjoint from
all previously accepted groups.  This prevents stale overlapping groups while
still allowing a merge link to be an addition.  Nested fusion is no longer
disabled.

### 13.3 Result

The implementation now supports nested additions and has been verified with:

- `max_temps=339`
- `max_temps=340`
- `max_temps=363`
- `max_temps=364`
- `max_temps=400`
- `max_temps=-1` (unlimited)

Each produces the expected CCSDT energy with `opt_level=6`, batched
substitution, and `expand_permutations=True`.
