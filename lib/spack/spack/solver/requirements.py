# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import enum
from typing import List, NamedTuple, Optional, Sequence, Tuple, Any, Dict

from llnl.util import tty

import spack.config
import spack.error
import spack.package_base
import spack.repo
import spack.spec
from spack.util.spack_yaml import get_mark_from_yaml_data


class RequirementKind(enum.Enum):
    """Purpose / provenance of a requirement"""

    #: Default requirement expressed under the 'all' attribute of packages.yaml
    DEFAULT = enum.auto()
    #: Requirement expressed on a virtual package
    VIRTUAL = enum.auto()
    #: Requirement expressed on a specific package
    PACKAGE = enum.auto()


class RequirementRule(NamedTuple):
    """Data class to collect information on a requirement"""

    pkg_name: str
    policy: str
    requirements: Sequence["spack.spec.Spec"]
    condition: "spack.spec.Spec"
    kind: "RequirementKind"
    message: Optional[str]


class RequirementParser:
    """Parses requirements from package.py files and configuration, and returns rules."""

    def __init__(self, configuration: "spack.config.Configuration") -> None:
        self.config: "spack.config.Configuration" = configuration
        self.runtime_pkgs: Set[str] = spack.repo.PATH.packages_with_tags("runtime")
        self.compiler_pkgs: Set[str] = spack.repo.PATH.packages_with_tags("compiler")

    def rules(
        self, pkg: "spack.package_base.PackageBase"
    ) -> List["RequirementRule"]:
        result: List["RequirementRule"] = []
        result.extend(self.rules_from_package_py(pkg))
        result.extend(self.rules_from_require(pkg))
        result.extend(self.rules_from_prefer(pkg))
        result.extend(self.rules_from_conflict(pkg))
        return result

    def rules_from_package_py(
        self, pkg: "spack.package_base.PackageBase"
    ) -> List["RequirementRule"]:
        rules: List["RequirementRule"] = []
        # Assuming pkg.requirements is Dict[spack.spec.Spec, List[Tuple[Sequence[spack.spec.Spec], str, Optional[str]]]]
        for when_spec, requirement_list in pkg.requirements.items():
            for requirements_seq, policy_str, message_str in requirement_list:
                rules.append(
                    RequirementRule(
                        pkg_name=pkg.name,
                        policy=policy_str,
                        requirements=requirements_seq,
                        kind=RequirementKind.PACKAGE,
                        condition=when_spec,
                        message=message_str,
                    )
                )
        return rules

    def rules_from_virtual(self, virtual_str: str) -> List["RequirementRule"]:
        kind_req, requests_req = self._raw_yaml_data(
            virtual_str, section="require", virtual=True
        )
        result: List["RequirementRule"] = self._rules_from_requirements(
            virtual_str, requests_req, kind=kind_req
        )

        kind_pref, requests_pref = self._raw_yaml_data(
            virtual_str, section="prefer", virtual=True
        )
        result.extend(
            self._rules_from_preferences(virtual_str, preferences=requests_pref, kind=kind_pref)
        )

        kind_conf, requests_conf = self._raw_yaml_data(
            virtual_str, section="conflict", virtual=True
        )
        result.extend(
            self._rules_from_conflicts(virtual_str, conflicts=requests_conf, kind=kind_conf)
        )

        return result

    def rules_from_require(
        self, pkg: "spack.package_base.PackageBase"
    ) -> List["RequirementRule"]:
        kind, requirements_data = self._raw_yaml_data(pkg.name, section="require")
        return self._rules_from_requirements(pkg.name, requirements_data, kind=kind)

    def rules_from_prefer(
        self, pkg: "spack.package_base.PackageBase"
    ) -> List["RequirementRule"]:
        kind, preferences_data = self._raw_yaml_data(pkg.name, section="prefer")
        return self._rules_from_preferences(pkg.name, preferences=preferences_data, kind=kind)

    def _rules_from_preferences(
        self, pkg_name: str, *, preferences: List[Any], kind: "RequirementKind"
    ) -> List["RequirementRule"]:
        result: List["RequirementRule"] = []
        for item in preferences:
            spec, condition, message = self._parse_prefer_conflict_item(item)
            result.append(
                # A strong preference is defined as:
                #
                # require:
                # - any_of: [spec_str, "@:"]
                RequirementRule(
                    pkg_name=pkg_name,
                    policy="any_of",
                    requirements=[spec, spack.spec.Spec("@:")],
                    kind=kind,
                    message=message,
                    condition=condition,
                )
            )
        return result

    def rules_from_conflict(
        self, pkg: "spack.package_base.PackageBase"
    ) -> List["RequirementRule"]:
        kind, conflicts_data = self._raw_yaml_data(pkg.name, section="conflict")
        return self._rules_from_conflicts(pkg.name, conflicts=conflicts_data, kind=kind)

    def _rules_from_conflicts(
        self, pkg_name: str, *, conflicts: List[Any], kind: "RequirementKind"
    ) -> List["RequirementRule"]:
        result: List["RequirementRule"] = []
        for item in conflicts:
            spec, condition, message = self._parse_prefer_conflict_item(item)
            result.append(
                # A conflict is defined as:
                #
                # require:
                # - one_of: [spec_str, "@:"]
                RequirementRule(
                    pkg_name=pkg_name,
                    policy="one_of",
                    requirements=[spec, spack.spec.Spec("@:")],
                    kind=kind,
                    message=message,
                    condition=condition,
                )
            )
        return result

    def _parse_prefer_conflict_item(
        self, item: Union[str, Dict[str, Any]]
    ) -> Tuple["spack.spec.Spec", "spack.spec.Spec", Optional[str]]:
        # The item is either a string or an object with at least a "spec" attribute
        spec: "spack.spec.Spec"
        condition: "spack.spec.Spec"
        message: Optional[str]
        if isinstance(item, str):
            spec = parse_spec_from_yaml_string(item)
            condition = spack.spec.Spec()
            message = None
        else:
            spec = parse_spec_from_yaml_string(item["spec"])
            condition = spack.spec.Spec(item.get("when")) # type: ignore[arg-type]
            message = item.get("message")
        return spec, condition, message

    def _raw_yaml_data(
        self, pkg_name: str, *, section: str, virtual: bool = False
    ) -> Tuple["RequirementKind", List[Any]]:
        config_pkgs: Dict[str, Any] = self.config.get("packages")
        data: List[Any] = config_pkgs.get(pkg_name, {}).get(section, [])
        kind: "RequirementKind" = RequirementKind.PACKAGE

        if virtual:
            return RequirementKind.VIRTUAL, data

        if not data:
            data = config_pkgs.get("all", {}).get(section, [])
            kind = RequirementKind.DEFAULT
        return kind, data

    def _rules_from_requirements(
        self, pkg_name: str, requirements: Union[str, List[Any]], *, kind: "RequirementKind"
    ) -> List["RequirementRule"]:
        """Manipulate requirements from packages.yaml, and return a list of tuples
        with a uniform structure (name, policy, requirements).
        """
        requirements_list: List[Any]
        if isinstance(requirements, str):
            requirements_list = [requirements]
        else:
            requirements_list = requirements

        rules: List["RequirementRule"] = []
        current_requirement: Dict[str, Any]
        for requirement_item in requirements_list:
            # A string is equivalent to a one_of group with a single element
            if isinstance(requirement_item, str):
                current_requirement = {"one_of": [requirement_item]}
            else:
                current_requirement = requirement_item

            policy_str: str
            for current_policy in ("spec", "one_of", "any_of"):
                if current_policy not in current_requirement:
                    continue
                policy_str = current_policy
                break
            else: # Should not happen if input is validated
                continue


            constraints_data: Union[str, List[str]] = current_requirement[policy_str]
            constraints_list_str: List[str]
            # "spec" is for specifying a single spec
            if policy_str == "spec":
                constraints_list_str = [constraints_data] if isinstance(constraints_data, str) else [] # type: ignore[list-item]
                policy_str = "one_of"
            else:
                constraints_list_str = constraints_data if isinstance(constraints_data, list) else []


            # validate specs from YAML first, and fail with line numbers if parsing fails.
            parsed_constraints: List["spack.spec.Spec"] = [
                parse_spec_from_yaml_string(c, named=(kind == RequirementKind.VIRTUAL))
                for c in constraints_list_str
            ]
            when_str_data: Optional[str] = current_requirement.get("when")
            when_spec_obj: "spack.spec.Spec" = parse_spec_from_yaml_string(when_str_data) if when_str_data else spack.spec.Spec()

            final_constraints: List["spack.spec.Spec"] = [
                x for x in parsed_constraints if not self.reject_requirement_constraint(pkg_name, constraint=x, kind=kind)
            ]
            if not final_constraints:
                continue

            rules.append(
                RequirementRule(
                    pkg_name=pkg_name,
                    policy=policy_str,
                    requirements=final_constraints,
                    kind=kind,
                    message=current_requirement.get("message"),
                    condition=when_spec_obj,
                )
            )
        return rules

    def reject_requirement_constraint(
        self, pkg_name: str, *, constraint: "spack.spec.Spec", kind: "RequirementKind"
    ) -> bool:
        """Returns True if a requirement constraint should be rejected"""
        # If it's a specific package requirement, it's never rejected
        if kind != RequirementKind.DEFAULT:
            return False

        # Reject requirements with dependencies for runtimes and compilers
        # These are usually requests on compilers, in the form of %<compiler>
        involves_dependencies = bool(constraint.dependencies())
        if involves_dependencies and (
            pkg_name in self.runtime_pkgs or pkg_name in self.compiler_pkgs
        ):
            tty.debug(f"[{__name__}] Rejecting '{constraint}' for compiler package {pkg_name}")
            return True

        # Requirements under all: are applied only if they are satisfiable considering only
        # package rules, so e.g. variants must exist etc. Otherwise, they are rejected.
        try:
            s = spack.spec.Spec(pkg_name)
            s.constrain(constraint)
            s.validate_or_raise()
        except spack.error.SpackError as e:
            tty.debug(
                f"[{__name__}] Rejecting the default '{constraint}' requirement "
                f"on '{pkg_name}': {str(e)}",
                level=2,
            )
            return True
        return False


    def parse_spec_from_yaml_string(
    string: Optional[str], *, named: bool = False
) -> "spack.spec.Spec":
    """Parse a spec from YAML and add file/line info to errors, if it's available.

    Parse a ``Spec`` from the supplied string, but also intercept any syntax errors and
    add file/line information for debugging using file/line annotations from the string.

    Args:
        string: a string representing a ``Spec`` from config YAML.
        named: if True, the spec must have a name
    """
    if string is None:
        return spack.spec.Spec()

    try:
        result: "spack.spec.Spec" = spack.spec.Spec(string)
    except spack.error.SpecSyntaxError as e:
        mark: Optional[Any] = get_mark_from_yaml_data(string) # Assuming get_mark_from_yaml_data can return None
        if mark:
            msg_err: str = f"{mark.name}:{mark.line + 1}: {str(e)}"
            raise spack.error.SpecSyntaxError(msg_err) from e
        raise e

    if named is True and not result.name:
        msg_named_err: str = f"expected a named spec, but got '{string}' instead"
        mark_named: Optional[Any] = get_mark_from_yaml_data(string)

        # Add a hint in case it's dependencies
        deps_list: List["spack.spec.Spec"] = result.dependencies()
        if len(deps_list) == 1:
            msg_named_err = f"{msg_named_err}. Did you mean '{deps_list[0]}'?"

        if mark_named:
            msg_named_err = f"{mark_named.name}:{mark_named.line + 1}: {msg_named_err}"

        raise spack.error.SpackError(msg_named_err)

    return result
