# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
from functools import cmp_to_key
from typing import Dict, List, NamedTuple, Iterable

import spack.deptypes as dt
from spack.spec import Spec # Already imported, but good to note
from spack.traverse import by_dag_hash, traverse_nodes


class Splice(NamedTuple):
    #: The spec being spliced into a parent
    splice_spec: "Spec"
    #: The name of the child that splice spec is replacing
    child_name: str
    #: The hash of the child that `splice_spec` is replacing
    child_hash: str


def _resolve_collected_splices(
    specs: List["Spec"], splices: Dict["Spec", List["Splice"]]
) -> Dict["Spec", "Spec"]:
    """After all of the specs have been concretized, apply all immediate splices.
    Returns a dict mapping original specs to their resolved counterparts
    """

    def splice_cmp(s1: "Spec", s2: "Spec") -> int:
        """This function can be used to sort a list of specs such that that any
        spec which will be spliced into a parent comes after the parent it will
        be spliced into. This order ensures that transitive splices will be
        executed in the correct order.
        """

        s1_splices_list: List["Splice"] = splices.get(s1, [])
        s2_splices_list: List["Splice"] = splices.get(s2, [])
        if any(s2.dag_hash() == splice_item.splice_spec.dag_hash() for splice_item in s1_splices_list):
            return -1
        elif any(s1.dag_hash() == splice_item.splice_spec.dag_hash() for splice_item in s2_splices_list):
            return 1
        else:
            return 0

    splice_order_list: List["Spec"] = sorted(specs, key=cmp_to_key(splice_cmp))
    # traverse_nodes can return an Iterable[Spec], ensure it's a list for reversed
    reverse_topo_order_iter: Iterable["Spec"] = (
        x for x in traverse_nodes(splice_order_list, order="topo", key=by_dag_hash) if x in specs
    )
    # Convert iterator to list before reversing if needed, or ensure traverse_nodes returns a Sequence
    # For now, assuming traverse_nodes result can be directly used with reversed if it's a sequence,
    # or converted to list if it's a generic iterable.
    # Python's reversed() works on sequences or objects with __reversed__().
    # If traverse_nodes returns a generator, it needs to be converted to a list first.
    # Let's assume it's a list or sequence for now.
    reverse_topo_order_list: List["Spec"] = list(reverse_topo_order_iter)


    already_resolved_dict: Dict["Spec", "Spec"] = {}
    current_spec: "Spec"
    for current_spec in reversed(reverse_topo_order_list): # Use reversed on the list
        immediate_splices: List["Splice"] = splices.get(current_spec, [])
        if not immediate_splices and not any(
            edge.spec in already_resolved_dict for edge in current_spec.edges_to_dependencies()
        ):
            continue
        new_spliced_spec: "Spec" = current_spec.copy(deps=False)
        new_spliced_spec.clear_caches(ignore=("package_hash",))
        new_spliced_spec.build_spec = current_spec # type: ignore[assignment] # build_spec is Optional[Spec]
        edge: "spack.dependency.Dependency" # Assuming edge is of this type
        for edge in current_spec.edges_to_dependencies():
            depflag_val: "dt.DepFlag" = edge.depflag & ~dt.BUILD
            if any(edge.spec.dag_hash() == splice_item.child_hash for splice_item in immediate_splices):
                # Ensure splice_item is correctly typed or cast if necessary
                splice_item_match: "Splice" = [
                    s_item for s_item in immediate_splices if s_item.child_hash == edge.spec.dag_hash()
                ][0]
                # If the spec being splice in is also spliced
                spliced_in_spec: "Spec" = already_resolved_dict.get(
                    splice_item_match.splice_spec, splice_item_match.splice_spec
                )
                new_spliced_spec.add_dependency_edge(
                    spliced_in_spec, depflag=depflag_val, virtuals=edge.virtuals
                )
            elif edge.spec in already_resolved_dict:
                new_spliced_spec.add_dependency_edge(
                    already_resolved_dict[edge.spec], depflag=depflag_val, virtuals=edge.virtuals
                )
            else:
                new_spliced_spec.add_dependency_edge(
                    edge.spec, depflag=depflag_val, virtuals=edge.virtuals
                )
        already_resolved_dict[current_spec] = new_spliced_spec
    return already_resolved_dict
