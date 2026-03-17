# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Index for condition provenance and cause-chain reconstruction.

This module replaces the second clingo solve (error_messages.lp) with a
Python-side index that reconstructs the causal chain between conditions.
"""
import enum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import spack.version

from .requirements import RequirementOrigin

CauseType = Tuple[str, str]


class ConditionOrigin(enum.Enum):
    """Origin of a condition, used for provenance in error messages."""

    DEPENDS_ON = "depends_on"
    CONFLICT = "conflict"
    PROVIDES = "provides"
    VARIANT_CONDITION = "variant_cond"
    VARIANT_VALUE = "variant_value"
    REQUIRE_CONFIG = "require_config"
    PREFER_CONFIG = "prefer_config"
    CONFLICT_CONFIG = "conflict_config"
    REQUIRE_DIRECTIVE = "require_dir"
    INPUT_SPEC = "input_spec"
    LITERAL = "literal"
    RUNTIME = "runtime"
    SPLICE = "splice"
    UNKNOWN = "unknown"


#: Map RequirementOrigin -> ConditionOrigin
REQUIREMENT_ORIGIN_MAP: Dict[RequirementOrigin, ConditionOrigin] = {
    RequirementOrigin.REQUIRE_YAML: ConditionOrigin.REQUIRE_CONFIG,
    RequirementOrigin.PREFER_YAML: ConditionOrigin.PREFER_CONFIG,
    RequirementOrigin.CONFLICT_YAML: ConditionOrigin.CONFLICT_CONFIG,
    RequirementOrigin.DIRECTIVE: ConditionOrigin.REQUIRE_DIRECTIVE,
    RequirementOrigin.INPUT_SPECS: ConditionOrigin.INPUT_SPEC,
}


class ConditionRecord(NamedTuple):
    trigger_id: int
    effect_id: Optional[int]
    pkg_name: str
    msg: Optional[str]
    origin: ConditionOrigin


class ConditionIndex:
    """Index of conditions, triggers, and effects for cause-chain reconstruction.

    Built during SpackSolverSetup and queried by ErrorHandler to replace the
    second clingo solve that previously derived condition_cause/4 and error_cause/4.
    """

    def __init__(self):
        # condition_id -> record
        self.conditions: Dict[int, ConditionRecord] = {}
        # trigger_id -> list of attribute signatures (tuples)
        self.trigger_signatures: Dict[int, List[tuple]] = {}
        # effect_id -> list of attribute signatures (tuples)
        self.effect_signatures: Dict[int, List[tuple]] = {}
        # attribute_signature -> list of effect_ids that impose it
        self.sig_to_effects: Dict[tuple, List[int]] = {}
        # effect_id -> condition_id
        self.effect_to_condition: Dict[int, int] = {}
        # trigger_id -> condition_id
        self.trigger_to_condition: Dict[int, int] = {}
        # pkg -> [(trigger_id, constraint_id, msg)]
        self.conflicts: Dict[str, List[Tuple[int, int, str]]] = {}

    def register_condition(
        self,
        condition_id: int,
        trigger_id: int,
        effect_id: Optional[int],
        pkg_name: str,
        msg: Optional[str],
        origin: ConditionOrigin,
    ) -> None:
        self.conditions[condition_id] = ConditionRecord(
            trigger_id, effect_id, pkg_name, msg, origin
        )
        self.trigger_to_condition[trigger_id] = condition_id
        if effect_id is not None:
            self.effect_to_condition[effect_id] = condition_id

    def register_trigger(self, trigger_id: int, signatures: List[tuple]) -> None:
        self.trigger_signatures[trigger_id] = [_normalize_sig(s) for s in signatures]

    def register_effect(self, effect_id: int, signatures: List[tuple]) -> None:
        normalized = [_normalize_sig(s) for s in signatures]
        self.effect_signatures[effect_id] = normalized
        for sig in normalized:
            self.sig_to_effects.setdefault(sig, []).append(effect_id)

    def register_conflict(
        self, pkg_name: str, trigger_id: int, constraint_id: int, msg: str
    ) -> None:
        self.conflicts.setdefault(pkg_name, []).append((trigger_id, constraint_id, msg))

    def compute_condition_causes(
        self, condition_holds: Dict[int, str]
    ) -> Dict[CauseType, List[CauseType]]:
        """Reconstruct the condition_cause/4 relation from the index.

        Args:
            condition_holds: mapping from condition_id -> node_id_str for each
                condition that held in the model.

        Returns:
            Dict mapping (condition_id_str, node_id_str) -> list of
            (cause_condition_id_str, cause_node_id_str) tuples.
        """
        result: Dict[CauseType, List[CauseType]] = {}

        for cond2_id, node2_str in condition_holds.items():
            record2 = self.conditions.get(cond2_id)
            if record2 is None:
                continue

            trigger_sigs = self.trigger_signatures.get(record2.trigger_id, [])
            if not trigger_sigs:
                continue

            # Collect all effect_ids that match any trigger signature
            matching_effect_ids: Set[int] = set()
            for sig in trigger_sigs:
                # Direct match
                for eid in self.sig_to_effects.get(sig, []):
                    matching_effect_ids.add(eid)

                # Special case: trigger requires ("node", Pkg) should also
                # match effects that impose ("dependency_holds", Parent, Pkg, Type)
                if len(sig) == 2 and sig[0] == "node":
                    pkg_name = sig[1]
                    for effect_sig, effect_ids in self.sig_to_effects.items():
                        if (
                            len(effect_sig) == 4
                            and effect_sig[0] == "dependency_holds"
                            and effect_sig[2] == pkg_name
                        ):
                            for eid in effect_ids:
                                matching_effect_ids.add(eid)

            # For each matching effect, find the owning condition
            key2 = (str(cond2_id), node2_str)
            for eid in matching_effect_ids:
                cond1_id = self.effect_to_condition.get(eid)
                if cond1_id is None or cond1_id not in condition_holds:
                    continue
                node1_str = condition_holds[cond1_id]
                key1 = (str(cond1_id), node1_str)
                result.setdefault(key2, []).append(key1)

        return result

    def compute_error_causes(
        self,
        errors: List[Tuple[str, str, str]],
        condition_holds: Dict[int, str],
        actual_versions: Optional[Dict[str, str]] = None,
    ) -> Dict[Tuple[str, str], List[CauseType]]:
        """Reconstruct error_cause/4 from the index.

        Args:
            errors: list of (error_type_str, weight_str, node_str) tuples
            condition_holds: mapping from condition_id -> node_id_str
            actual_versions: optional node_str -> version_str for filtering
                "other side" version constraints

        Returns:
            Dict mapping (error_type_str, node_str) -> list of (cond_id_str, node_str)
        """
        result: Dict[Tuple[str, str], List[CauseType]] = {}

        for error_type_str, _weight_str, node_str in errors:
            error_key = (error_type_str, node_str)

            # Parse the error type functor
            paren_idx = error_type_str.find("(")
            if paren_idx >= 0:
                functor = error_type_str[:paren_idx]
            else:
                functor = error_type_str

            if functor == "version_constraint_unsatisfied":
                self._error_cause_version(
                    error_key, error_type_str, condition_holds, result, actual_versions
                )
            elif functor == "no_valid_provider":
                self._error_cause_no_provider(error_key, node_str, condition_holds, result)
            elif functor == "variant_value_conflict":
                self._error_cause_variant(error_key, error_type_str, condition_holds, result)
            elif functor == "conflict":
                self._error_cause_conflict(error_key, error_type_str, condition_holds, result)

        return result

    def _add_causes_from_effects(
        self,
        error_key: Tuple[str, str],
        effect_ids: List[int],
        condition_holds: Dict[int, str],
        result: Dict[Tuple[str, str], List[CauseType]],
    ) -> None:
        """Add causes for held conditions owning the given effects."""
        for eid in effect_ids:
            cond_id = self.effect_to_condition.get(eid)
            if cond_id is None or cond_id not in condition_holds:
                continue
            node_str = condition_holds[cond_id]
            entry = (str(cond_id), node_str)
            if entry not in result.get(error_key, []):
                result.setdefault(error_key, []).append(entry)

    def _error_cause_version(
        self,
        error_key: Tuple[str, str],
        error_type_str: str,
        condition_holds: Dict[int, str],
        result: Dict[Tuple[str, str], List[CauseType]],
        actual_versions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Causes for version_constraint_unsatisfied errors."""
        # Extract constraint from error_type_str: version_constraint_unsatisfied(Constraint)
        constraint = _extract_single_arg(error_type_str)
        if constraint is None:
            return

        # Extract package from node_str
        node_str = error_key[1]
        pkg = _extract_pkg_from_node_str(node_str)
        if pkg is None:
            return

        # Find effects imposing node_version_satisfies for this package+constraint
        sig_key = ("node_version_satisfies", pkg, constraint)
        effect_ids = self.sig_to_effects.get(sig_key, [])
        self._add_causes_from_effects(error_key, effect_ids, condition_holds, result)

        # Also find effects imposing other version constraints on the same package
        # (the "other side" of the conflict). If we know the actual version, only
        # include constraints that the actual version satisfies — otherwise they
        # are not contributing to the conflict.
        actual_ver_str = actual_versions.get(node_str) if actual_versions else None
        for sig, eids in self.sig_to_effects.items():
            if (
                len(sig) >= 3
                and sig[0] == "node_version_satisfies"
                and sig[1] == pkg
                and sig[2] != constraint
            ):
                if actual_ver_str is not None:
                    try:
                        actual_ver = spack.version.Version(actual_ver_str)
                        other_constraint = spack.version.from_string(sig[2])
                        if not actual_ver.satisfies(other_constraint):
                            continue
                    except Exception:
                        pass  # on parse failure, include the constraint
                self._add_causes_from_effects(error_key, eids, condition_holds, result)

    def _error_cause_no_provider(
        self,
        error_key: Tuple[str, str],
        node_str: str,
        condition_holds: Dict[int, str],
        result: Dict[Tuple[str, str], List[CauseType]],
    ) -> None:
        """Causes for no_valid_provider errors."""
        virtual = _extract_pkg_from_node_str(node_str)
        if virtual is None:
            return

        effect_ids = []
        for sig, eids in self.sig_to_effects.items():
            if len(sig) >= 3 and sig[0] == "dependency_holds" and sig[2] == virtual:
                effect_ids.extend(eids)
        self._add_causes_from_effects(error_key, effect_ids, condition_holds, result)

    def _error_cause_variant(
        self,
        error_key: Tuple[str, str],
        error_type_str: str,
        condition_holds: Dict[int, str],
        result: Dict[Tuple[str, str], List[CauseType]],
    ) -> None:
        """Causes for variant_value_conflict errors."""
        args = _extract_args(error_type_str)
        if len(args) < 3:
            return

        variant, value1, value2 = args[0], args[1], args[2]
        pkg = _extract_pkg_from_node_str(error_key[1])
        if pkg is None:
            return

        for val in (value1, value2):
            effect_ids = self.sig_to_effects.get(("variant_set", pkg, variant, val), [])
            self._add_causes_from_effects(error_key, effect_ids, condition_holds, result)

    def _error_cause_conflict(
        self,
        error_key: Tuple[str, str],
        error_type_str: str,
        condition_holds: Dict[int, str],
        result: Dict[Tuple[str, str], List[CauseType]],
    ) -> None:
        """Causes for conflict() errors."""
        msg = _extract_single_arg(error_type_str)
        if msg is None:
            return

        pkg = _extract_pkg_from_node_str(error_key[1])
        if pkg is None:
            return

        for trigger_id, constraint_id, conflict_msg in self.conflicts.get(pkg, []):
            if conflict_msg != msg:
                continue
            # Add trigger side cause
            if trigger_id in condition_holds:
                node_str = condition_holds[trigger_id]
                result.setdefault(error_key, []).append((str(trigger_id), node_str))
            # Add constraint side cause
            if constraint_id in condition_holds:
                node_str = condition_holds[constraint_id]
                result.setdefault(error_key, []).append((str(constraint_id), node_str))


def _normalize_sig(sig: tuple) -> tuple:
    """Convert all elements of a signature tuple to strings for consistent matching."""
    return tuple(str(x) for x in sig)


def _find_matching_paren(s: str, open_idx: int) -> int:
    """Find the closing ')' that matches the '(' at open_idx, respecting nesting and quotes."""
    depth = 0
    in_quote = False
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == '"':
            in_quote = not in_quote
        elif not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
    return -1


def _extract_single_arg(functor_str: str) -> Optional[str]:
    """Extract a single argument from a functor string like 'name(arg)'."""
    start = functor_str.find("(")
    if start < 0:
        return None
    end = _find_matching_paren(functor_str, start)
    if end < 0 or end <= start:
        return None
    return functor_str[start + 1 : end].strip('"')


def _extract_args(functor_str: str) -> List[str]:
    """Extract arguments from a functor string like 'name(a,b,c)'.

    Handles quoted strings that may contain commas by tracking quote depth.
    """
    start = functor_str.find("(")
    if start < 0:
        return []
    end = _find_matching_paren(functor_str, start)
    if end < 0 or end <= start:
        return []
    inner = functor_str[start + 1 : end]

    args = []
    current: List[str] = []
    depth = 0
    in_quote = False
    for ch in inner:
        if ch == '"' and depth == 0:
            in_quote = not in_quote
        elif ch == "(" and not in_quote:
            depth += 1
            current.append(ch)
        elif ch == ")" and not in_quote:
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0 and not in_quote:
            args.append("".join(current).strip().strip('"'))
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip().strip('"'))
    return args


def _extract_pkg_from_node_str(node_str: str) -> Optional[str]:
    """Extract the package name from a node(ID, Pkg) string representation."""
    # node_str looks like 'node(0,"pkg-name")' or 'node(0,pkg_name)'
    start = node_str.find(",")
    end = node_str.rfind(")")
    if start < 0 or end < 0 or end <= start:
        return None
    return node_str[start + 1 : end].strip().strip('"')
