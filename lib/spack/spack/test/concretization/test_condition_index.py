# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Unit tests for ConditionIndex cause-chain reconstruction."""
import pytest

from spack.solver.condition_index import ConditionIndex, ConditionOrigin


@pytest.fixture()
def index():
    return ConditionIndex()


def test_register_and_lookup(index):
    """Register conditions, triggers, effects; verify all lookups return correct data."""
    index.register_condition(1, 10, 20, "pkg-a", "msg1", ConditionOrigin.DEPENDS_ON)
    index.register_trigger(10, [("node", "pkg-a")])
    index.register_effect(20, [("variant_set", "pkg-a", "v", "true")])

    assert 1 in index.conditions
    assert index.conditions[1].trigger_id == 10
    assert index.conditions[1].effect_id == 20
    assert index.conditions[1].pkg_name == "pkg-a"
    assert index.conditions[1].origin == ConditionOrigin.DEPENDS_ON

    assert index.trigger_signatures[10] == [("node", "pkg-a")]
    assert index.effect_signatures[20] == [("variant_set", "pkg-a", "v", "true")]
    assert index.effect_to_condition[20] == 1
    assert index.trigger_to_condition[10] == 1

    assert ("variant_set", "pkg-a", "v", "true") in index.sig_to_effects
    assert 20 in index.sig_to_effects[("variant_set", "pkg-a", "v", "true")]


def test_cause_chain_simple(index):
    """C1's effect imposes variant_set, C2's trigger requires it. Verify C1 causes C2."""
    # C1: condition 1, trigger 10, effect 20
    index.register_condition(1, 10, 20, "pkg-a", "C1 msg", ConditionOrigin.DEPENDS_ON)
    index.register_trigger(10, [("node", "pkg-a")])
    index.register_effect(20, [("variant_set", "pkg-a", "v", "true")])

    # C2: condition 2, trigger 11, effect 21
    index.register_condition(2, 11, 21, "pkg-a", "C2 msg", ConditionOrigin.DEPENDS_ON)
    index.register_trigger(11, [("variant_set", "pkg-a", "v", "true")])
    index.register_effect(21, [("node_version_satisfies", "pkg-a", "1.0")])

    # Both conditions held
    condition_holds = {1: "0", 2: "0"}
    causes = index.compute_condition_causes(condition_holds)

    # C1 should be a cause of C2
    assert ("2", "0") in causes
    cause_ids = [c[0] for c in causes[("2", "0")]]
    assert "1" in cause_ids


def test_cause_chain_dependency_special_case(index):
    """C1's effect imposes dependency_holds, C2's trigger requires node. Verify match."""
    # C1 imposes dependency_holds(parent, child, run)
    index.register_condition(1, 10, 20, "parent", "C1", ConditionOrigin.DEPENDS_ON)
    index.register_trigger(10, [("node", "parent")])
    index.register_effect(20, [("dependency_holds", "parent", "child", "run")])

    # C2 requires node(child) in its trigger
    index.register_condition(2, 11, 21, "child", "C2", ConditionOrigin.DEPENDS_ON)
    index.register_trigger(11, [("node", "child")])
    index.register_effect(21, [("variant_set", "child", "v", "true")])

    condition_holds = {1: "0", 2: "0"}
    causes = index.compute_condition_causes(condition_holds)

    assert ("2", "0") in causes
    cause_ids = [c[0] for c in causes[("2", "0")]]
    assert "1" in cause_ids


def test_cause_chain_no_false_positives(index):
    """Conditions that held but whose signatures don't match should not produce causes."""
    index.register_condition(1, 10, 20, "pkg-a", "C1", ConditionOrigin.DEPENDS_ON)
    index.register_trigger(10, [("node", "pkg-a")])
    index.register_effect(20, [("variant_set", "pkg-a", "v", "true")])

    index.register_condition(2, 11, 21, "pkg-b", "C2", ConditionOrigin.DEPENDS_ON)
    index.register_trigger(11, [("variant_set", "pkg-b", "w", "false")])
    index.register_effect(21, [("node_version_satisfies", "pkg-b", "2.0")])

    condition_holds = {1: "0", 2: "0"}
    causes = index.compute_condition_causes(condition_holds)

    # C1 and C2 signatures don't overlap -> no cause relationship
    assert ("2", "0") not in causes


def test_error_cause_conflict(index):
    """Register a conflict, simulate error(conflict(Msg), ...) and verify causes."""
    trigger_id = 10
    constraint_id = 11
    msg = "pkg-a: 'x' conflicts with 'y'"

    index.register_condition(trigger_id, 100, None, "pkg-a", "trigger", ConditionOrigin.CONFLICT)
    index.register_condition(
        constraint_id, 101, None, "pkg-a", "constraint", ConditionOrigin.CONFLICT
    )
    index.register_conflict("pkg-a", trigger_id, constraint_id, msg)

    condition_holds = {trigger_id: "0", constraint_id: "0"}
    error_tuples = [(f'conflict("{msg}")', "0", 'node(0,"pkg-a")')]

    causes = index.compute_error_causes(error_tuples, condition_holds)
    error_key = (f'conflict("{msg}")', 'node(0,"pkg-a")')

    assert error_key in causes
    cause_ids = {c[0] for c in causes[error_key]}
    assert str(trigger_id) in cause_ids
    assert str(constraint_id) in cause_ids


def test_error_cause_variant_conflict(index):
    """Simulate variant_value_conflict error, verify causes for both values."""
    # Effect imposing value1
    index.register_condition(1, 10, 20, "pkg", "sets v=a", ConditionOrigin.LITERAL)
    index.register_effect(20, [("variant_set", "pkg", "v", "a")])

    # Effect imposing value2
    index.register_condition(2, 11, 21, "pkg", "sets v=b", ConditionOrigin.DEPENDS_ON)
    index.register_effect(21, [("variant_set", "pkg", "v", "b")])

    condition_holds = {1: "0", 2: "0"}
    error_tuples = [('variant_value_conflict("v","a","b")', "10", 'node(0,"pkg")')]

    causes = index.compute_error_causes(error_tuples, condition_holds)
    error_key = ('variant_value_conflict("v","a","b")', 'node(0,"pkg")')

    assert error_key in causes
    cause_ids = {c[0] for c in causes[error_key]}
    assert "1" in cause_ids
    assert "2" in cause_ids


def test_condition_origin_preserved(index):
    """Register conditions with specific origins, verify they're accessible."""
    index.register_condition(1, 10, 20, "pkg", "msg", ConditionOrigin.LITERAL)
    index.register_condition(2, 11, 21, "pkg", "msg", ConditionOrigin.DEPENDS_ON)
    index.register_condition(3, 12, None, "pkg", "msg", ConditionOrigin.CONFLICT)

    assert index.conditions[1].origin == ConditionOrigin.LITERAL
    assert index.conditions[2].origin == ConditionOrigin.DEPENDS_ON
    assert index.conditions[3].origin == ConditionOrigin.CONFLICT
