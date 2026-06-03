# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Custom clingo propagators for the Spack dependency solver.

These propagators implement constraints incrementally during the solve, pruning
incompatible partial assignments as soon as they are detected rather than waiting
for a full model to be evaluated.

Only the clingo 5.x (CFFI/legacy) API is supported here.  The clingo 6 API has
a different method signature for ``init``, ``propagate``, and ``decide`` and
would require a separate implementation.
"""

import collections
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    # Avoid importing clingo at module level; the bootstrap mechanism in
    # compat.py handles the import at runtime.  These annotations are for
    # type-checkers only.
    import clingo  # novm

# Deferred import: resolved at first use to allow the bootstrap mechanism to
# load clingo before we access its symbols.
_clingo_module = None


def _clingo():
    global _clingo_module
    if _clingo_module is None:
        from .compat import clingo as _clingo_import

        _clingo_module = _clingo_import()
    return _clingo_module


class TargetCompatibilityPropagator:
    """Enforce that every non-build dependency edge has compatible targets.

    For each edge ``attr("depends_on", Parent, Child, Type)`` with
    ``Type != "build"``, the targets chosen for Parent and Child must satisfy
    ``target_compatible(ParentTarget, ChildTarget)``.

    The propagator watches ``attr("depends_on", ...)`` literals and fires
    ``check()`` at every propagation fixpoint (``check_mode = Fixpoint``), so
    violations are detected as soon as both endpoints of an edge have an
    assigned target — the same timing as clingo's native acyclicity propagator.

    ``decide()`` provides domain-heuristic guidance: when the solver is about to
    branch on a target that would create an incompatibility with an already-
    assigned parent, it suggests the parent's target instead.

    This class follows the clingo 5.x (CFFI/legacy) propagator interface::

        init(self, init: clingo.PropagateInit) -> None
        propagate(self, control: clingo.PropagateControl,
                  changes: Sequence[int]) -> None
        check(self, control: clingo.PropagateControl) -> None
        decide(self, thread_id: int, assignment: clingo.Assignment,
               fallback: int) -> int
    """

    def __init__(self) -> None:
        # Set of (parent_target, child_target) pairs that are allowed.
        self._compatible: Set[Tuple[str, str]] = set()

        # (node_symbol, target_string) -> solver literal for
        # attr("node_target", Node, Target).
        self._node_target_lit: Dict[Tuple[object, str], int] = {}

        # Reverse of _node_target_lit: solver literal -> (node_symbol, target_string).
        # Used by decide() for O(1) fallback lookup instead of an O(N) linear scan.
        # Only stores conditional (non-always-true) literals to avoid collisions on
        # slit=1 (multiple always-true atoms share that value; decide() is never
        # called with an always-true fallback anyway).
        self._lit_to_node_target: Dict[int, Tuple[object, str]] = {}

        # node_symbol -> [(target_string, solver_literal), ...]
        self._targets_by_node: Dict[object, List[Tuple[str, int]]] = collections.defaultdict(list)

        # solver literal for a depends_on edge -> (parent_symbol, child_symbol)
        self._edge_of_lit: Dict[int, Tuple[object, object]] = {}

        # node_symbol -> [(edge_solver_lit, child_symbol), ...]
        self._edges_by_parent: Dict[object, List[Tuple[int, object]]] = collections.defaultdict(
            list
        )

        # node_symbol -> [(edge_solver_lit, parent_symbol), ...]
        self._edges_by_child: Dict[object, List[Tuple[int, object]]] = collections.defaultdict(
            list
        )

    # ------------------------------------------------------------------
    # clingo 5.x propagator interface
    # ------------------------------------------------------------------

    def init(self, init: "clingo.PropagateInit") -> None:
        """Build internal tables from the grounded program.

        Called once before each solve step.
        """
        self._compatible = set()
        self._node_target_lit = {}
        self._lit_to_node_target = {}
        self._targets_by_node = collections.defaultdict(list)
        self._edge_of_lit = {}
        self._edges_by_parent = collections.defaultdict(list)
        self._edges_by_child = collections.defaultdict(list)

        # Fire check() at every propagation fixpoint, not just on total
        # assignments.  This matches the timing of clingo's native acyclicity
        # propagator and lets the USC optimiser use our constraint information
        # when proving lower bounds, rather than only at the final model stage.
        init.check_mode = _clingo().PropagatorCheckMode.Fixpoint

        atoms = init.symbolic_atoms
        top = init.assignment

        # 1. Collect compatible (parent_target, child_target) pairs.
        #    Emitted as facts: target_compatible(ParentTarget, ChildTarget).
        #    Target names are string literals in the generated ASP, so use .string.
        for atom in atoms.by_signature("target_compatible", 2):
            args = atom.symbol.arguments
            self._compatible.add((args[0].string, args[1].string))

        # 2. Collect attr("node_target", Node, Target) atoms.
        #    arity is 3: ("node_target", Node, Target).
        #    We only need the literal→(node, target) mapping for nogood construction
        #    and for decide(); we do NOT add watches here to avoid solver-literal
        #    collisions that occur when multiple atoms share slit=1.
        for atom in atoms.by_signature("attr", 3):
            args = atom.symbol.arguments
            if args[0].string != "node_target":
                continue
            node_sym = args[1]
            target_str = args[2].string
            slit = init.solver_literal(atom.literal)
            self._node_target_lit[(node_sym, target_str)] = slit
            self._targets_by_node[node_sym].append((target_str, slit))
            if not top.is_true(slit):
                self._lit_to_node_target[slit] = (node_sym, target_str)

        # 3. Collect attr("depends_on", Parent, Child, Type) atoms for
        #    non-build edges. arity is 4: ("depends_on", Parent, Child, Type).
        #    We watch edge literals so that propagate() fires when an edge is
        #    newly established during search.
        for atom in atoms.by_signature("attr", 4):
            args = atom.symbol.arguments
            if args[0].string != "depends_on":
                continue
            if args[3].string == "build":
                continue
            parent_sym = args[1]
            child_sym = args[2]
            eslit = init.solver_literal(atom.literal)
            self._edge_of_lit[eslit] = (parent_sym, child_sym)
            self._edges_by_parent[parent_sym].append((eslit, child_sym))
            self._edges_by_child[child_sym].append((eslit, parent_sym))
            init.add_watch(eslit)

    def propagate(self, control: "clingo.PropagateControl", changes: List[int]) -> None:
        """Check each newly-true edge literal for target incompatibility."""
        assignment = control.assignment
        for lit in changes:
            if lit not in self._edge_of_lit:
                continue
            parent_sym, child_sym = self._edge_of_lit[lit]
            tp = self._true_target(assignment, parent_sym)
            tc = self._true_target(assignment, child_sym)
            if self._violates(tp, tc):
                assert tp is not None and tc is not None
                parent_tlit = self._node_target_lit[(parent_sym, tp)]
                child_tlit = self._node_target_lit[(child_sym, tc)]
                if not control.add_nogood([lit, parent_tlit, child_tlit]):
                    return

    def check(self, control: "clingo.PropagateControl") -> None:
        """Scan all true edges for target incompatibility.

        With check_mode=Fixpoint this fires after every unit-propagation
        fixpoint, catching violations as soon as both endpoints of an edge
        have an assigned target — even when the edge itself is always-true
        (slit=1) and never appears in propagate() changes.
        """
        assignment = control.assignment
        for eslit, (parent_sym, child_sym) in self._edge_of_lit.items():
            if not assignment.is_true(eslit):
                continue
            tp = self._true_target(assignment, parent_sym)
            tc = self._true_target(assignment, child_sym)
            if not self._violates(tp, tc):
                continue
            assert tp is not None and tc is not None
            parent_tlit = self._node_target_lit[(parent_sym, tp)]
            child_tlit = self._node_target_lit[(child_sym, tc)]
            if not control.add_nogood([eslit, parent_tlit, child_tlit]):
                return

    def decide(self, thread_id: int, assignment: "clingo.Assignment", fallback: int) -> int:
        """Steer the search toward compatible target assignments.

        When the solver is about to branch on a target literal that would
        create an incompatibility with an already-assigned parent's target,
        suggest the parent's target for this node instead.  This replicates
        the domain heuristic that clingo's ``#edge`` acyclicity propagator
        provides implicitly.
        """
        if fallback <= 0:
            return 0
        entry = self._lit_to_node_target.get(fallback)
        if entry is None:
            return 0
        node_sym, target_str = entry
        # Check incoming non-build edges: if any assigned parent is incompatible,
        # suggest the parent's target for this node instead.
        for _eslit, parent_sym in self._edges_by_child.get(node_sym, []):
            if not assignment.is_true(_eslit):
                continue
            tp = self._true_target(assignment, parent_sym)
            if not (tp and self._violates(tp, target_str)):
                continue
            alt_lit = self._node_target_lit.get((node_sym, tp))
            if alt_lit and not assignment.is_true(alt_lit) and not assignment.is_false(alt_lit):
                return alt_lit
        return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _true_target(self, assignment: "clingo.Assignment", node_sym: object) -> Optional[str]:
        """Return the currently-assigned target for *node_sym*, or None."""
        for target_str, slit in self._targets_by_node.get(node_sym, []):
            if assignment.is_true(slit):
                return target_str
        return None

    def _violates(self, tp: Optional[str], tc: Optional[str]) -> bool:
        """Return True when the (parent, child) target pair is incompatible."""
        return tp is not None and tc is not None and (tp, tc) not in self._compatible
