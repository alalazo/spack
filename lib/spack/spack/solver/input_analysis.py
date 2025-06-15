# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Classes to analyze the input of a solve, and provide information to set up the ASP problem"""
import collections
from typing import Dict, List, NamedTuple, Set, Tuple, Union, Iterable

import _vendoring.archspec.cpu

from llnl.util import lang, tty

import spack.binary_distribution
import spack.config
import spack.deptypes as dt
import spack.platforms
import spack.repo
import spack.spec
import spack.store
from spack.error import SpackError


class PossibleGraph(NamedTuple):
    real_pkgs: Set[str]
    virtuals: Set[str]
    edges: Dict[str, Set[str]]


class PossibleDependencyGraph:
    """Returns information needed to set up an ASP problem"""

    def unreachable(self, *, pkg_name: str, when_spec: "spack.spec.Spec") -> bool:
        """Returns true if the context can determine that the condition cannot ever
        be met on pkg_name.
        """
        raise NotImplementedError

    def candidate_targets(self) -> List["_vendoring.archspec.cpu.Microarchitecture"]:
        """Returns a list of targets that are candidate for concretization"""
        raise NotImplementedError

    def possible_dependencies(
        self,
        *specs: Union["spack.spec.Spec", str],
        allowed_deps: "dt.DepFlag",
        transitive: bool = True,
        strict_depflag: bool = False,
        expand_virtuals: bool = True,
    ) -> "PossibleGraph":
        """Returns the set of possible dependencies, and the set of possible virtuals.

        Runtime packages, which may be injected by compilers, needs to be added to specs if
        the dependency is not explicit in the package.py recipe.

        Args:
            transitive: return transitive dependencies if True, only direct dependencies if False
            allowed_deps: dependency types to consider
            strict_depflag: if True, only the specific dep type is considered, if False any
                deptype that intersects with allowed deptype is considered
            expand_virtuals: expand virtual dependencies into all possible implementations
        """
        raise NotImplementedError


class NoStaticAnalysis(PossibleDependencyGraph):
    """Implementation that tries to minimize the setup time (i.e. defaults to give fast
    answers), rather than trying to reduce the ASP problem size with more complex analysis.
    """

    def __init__(
        self, *, configuration: "spack.config.Configuration", repo: "spack.repo.RepoPath"
    ) -> None:
        self.configuration: "spack.config.Configuration" = configuration
        self.repo: "spack.repo.RepoPath" = repo
        self._platform_condition: "spack.spec.Spec" = spack.spec.Spec(
            f"platform={spack.platforms.host()} target={_vendoring.archspec.cpu.host().family}:"
        )

        try:
            self.libc_pkgs: List[str] = [x.name for x in self.providers_for("libc")]
        except spack.repo.UnknownPackageError:
            self.libc_pkgs = []

    def is_virtual(self, name: str) -> bool:
        return self.repo.is_virtual(name)

    @lang.memoized
    def is_allowed_on_this_platform(self, *, pkg_name: str) -> bool:
        """Returns true if a package is allowed on the current host"""
        pkg_cls: type[spack.package_base.PackageBase] = self.repo.get_pkg_class(pkg_name)
        no_condition: "spack.spec.Spec" = spack.spec.Spec()
        # Assuming pkg_cls.requirements is Dict[spack.spec.Spec, List[Tuple[List[spack.spec.Spec], Any, Any]]]
        for when_spec_req, conditions_req in pkg_cls.requirements.items():
            # Restrict analysis to unconditional requirements
            if when_spec_req != no_condition:
                continue
            for requirements_list, _, _ in conditions_req:
                if not any(x.intersects(self._platform_condition) for x in requirements_list):
                    tty.debug(f"[{__name__}] {pkg_name} is not for this platform")
                    return False
        return True

    def providers_for(self, virtual_str: str) -> List["spack.spec.Spec"]:
        """Returns a list of possible providers for the virtual string in input."""
        return self.repo.providers_for(virtual_str)

    def can_be_installed(self, *, pkg_name: str) -> bool:
        """Returns True if a package can be installed, False otherwise."""
        return True

    def unreachable(self, *, pkg_name: str, when_spec: "spack.spec.Spec") -> bool:
        """Returns true if the context can determine that the condition cannot ever
        be met on pkg_name.
        """
        return False

    def candidate_targets(self) -> List["_vendoring.archspec.cpu.Microarchitecture"]:
        """Returns a list of targets that are candidate for concretization"""
        platform: "spack.platforms.Platform" = spack.platforms.host() # type: ignore[name-defined]
        default_target: "_vendoring.archspec.cpu.Microarchitecture" = _vendoring.archspec.cpu.TARGETS[platform.default] # type: ignore[index]

        # Construct the list of targets which are compatible with the host
        candidate_targets_list: List["_vendoring.archspec.cpu.Microarchitecture"] = [default_target] + default_target.ancestors # type: ignore[operator]
        granularity: str = self.configuration.get("concretizer:targets:granularity")
        host_compatible: bool = self.configuration.get("concretizer:targets:host_compatible")

        # Add targets which are not compatible with the current host
        if not host_compatible:
            additional_targets_in_family: List["_vendoring.archspec.cpu.Microarchitecture"] = sorted( # type: ignore[no-redef]
                [
                    t
                    for t in _vendoring.archspec.cpu.TARGETS.values() # type: ignore[attr-defined]
                    if (t.family.name == default_target.family.name and t not in candidate_targets_list) # type: ignore[attr-defined]
                ],
                key=lambda x: len(x.ancestors), # type: ignore[attr-defined]
                reverse=True,
            )
            candidate_targets_list += additional_targets_in_family

        # Check if we want only generic architecture
        if granularity == "generic":
            candidate_targets_list = [t for t in candidate_targets_list if t.vendor == "generic"] # type: ignore[attr-defined]

        return candidate_targets_list

    def possible_dependencies(
        self,
        *specs: Union["spack.spec.Spec", str],
        allowed_deps: "dt.DepFlag",
        transitive: bool = True,
        strict_depflag: bool = False,
        expand_virtuals: bool = True,
    ) -> "PossibleGraph":
        stack: List[str] = [x for x in self._package_list(specs)]
        virtuals: Set[str] = set()
        edges: Dict[str, Set[str]] = {}

        while stack:
            pkg_name: str = stack.pop()

            if pkg_name in edges:
                continue

            edges[pkg_name] = set()

            # Since libc is not buildable, there is no need to extend the
            # search space with libc dependencies.
            if pkg_name in self.libc_pkgs:
                continue

            pkg_cls_dep: type[spack.package_base.PackageBase] = self.repo.get_pkg_class(pkg_name=pkg_name)
            # Assuming pkg_cls.dependencies_by_name returns Dict[str, Dict[spack.spec.Spec, List[spack.dependency.Dependency]]]
            for dep_name_key, conditions_map in pkg_cls_dep.dependencies_by_name(when=True).items():
                if all(self.unreachable(pkg_name=pkg_name, when_spec=x) for x in conditions_map):
                    tty.debug(
                        f"[{__name__}] Not adding {dep_name_key} as a dep of {pkg_name}, because "
                        f"conditions cannot be met"
                    )
                    continue

                if not self._has_deptypes(
                    conditions_map, allowed_deps=allowed_deps, strict=strict_depflag
                ):
                    continue

                if dep_name_key in virtuals:
                    continue

                current_dep_names: Set[str] = set()
                if self.is_virtual(dep_name_key):
                    virtuals.add(dep_name_key)
                    if expand_virtuals:
                        providers_list: List["spack.spec.Spec"] = self.providers_for(dep_name_key)
                        current_dep_names = {spec.name for spec in providers_list}
                else:
                    current_dep_names = {dep_name_key}

                edges[pkg_name].update(current_dep_names)

                if not transitive:
                    continue

                for current_dep_name_item in current_dep_names:
                    if current_dep_name_item in edges:
                        continue

                    if not self._is_possible(pkg_name=current_dep_name_item):
                        continue

                    stack.append(current_dep_name_item)

        real_packages_set: Set[str] = set(edges)
        if not transitive:
            # We exit early, so add children from the edges information
            for _root_pkg, children_pkgs in edges.items():
                real_packages_set.update(x for x in children_pkgs if self._is_possible(pkg_name=x))

        return PossibleGraph(real_pkgs=real_packages_set, virtuals=virtuals, edges=edges)

    def _package_list(
        self, specs: Iterable[Union["spack.spec.Spec", str]]
    ) -> List[str]: # Changed from Tuple to Iterable
        stack_list: List[str] = []
        current_spec_obj: "spack.spec.Spec"
        for current_spec_item in specs:
            if isinstance(current_spec_item, str):
                current_spec_obj = spack.spec.Spec(current_spec_item)
            else:
                current_spec_obj = current_spec_item

            if self.repo.is_virtual(current_spec_obj.name):
                stack_list.extend([p.name for p in self.providers_for(current_spec_obj.name)])
                continue

            stack_list.append(current_spec_obj.name)
        return sorted(list(set(stack_list))) # Use list for set creation

    def _has_deptypes(
        self,
        dependencies: Dict["spack.spec.Spec", List["spack.dependency.Dependency"]], # Added type for dependencies
        *,
        allowed_deps: "dt.DepFlag",
        strict: bool,
    ) -> bool:
        if strict is True:
            return any(
                dep.depflag == allowed_deps for deplist in dependencies.values() for dep in deplist
            )
        return any(
            dep.depflag & allowed_deps for deplist in dependencies.values() for dep in deplist
        )

    def _is_possible(self, *, pkg_name: str) -> bool: # Added type for pkg_name
        try:
            return self.is_allowed_on_this_platform(pkg_name=pkg_name) and self.can_be_installed(
                pkg_name=pkg_name
            )
        except spack.repo.UnknownPackageError:
            return False


class StaticAnalysis(NoStaticAnalysis):
    """Performs some static analysis of the configuration, store, etc. to provide more precise
    answers on whether some packages can be installed, or used as a provider.

    It increases the setup time, but might decrease the grounding and solve time considerably,
    especially when requirements restrict the possible choices for providers.
    """

    def __init__(
        self,
        *,
        configuration: "spack.config.Configuration",
        repo: "spack.repo.RepoPath",
        store: "spack.store.Store",
        binary_index: "spack.binary_distribution.BinaryCacheIndex",
    ) -> None:
        super().__init__(configuration=configuration, repo=repo)
        self.store: "spack.store.Store" = store
        self.binary_index: "spack.binary_distribution.BinaryCacheIndex" = binary_index

    @lang.memoized
    def providers_for(self, virtual_str: str) -> List["spack.spec.Spec"]: # Overrides-Signature
        candidates: List["spack.spec.Spec"] = super().providers_for(virtual_str)
        result: List["spack.spec.Spec"] = []
        for spec_item in candidates:
            if not self._is_provider_candidate(pkg_name=spec_item.name, virtual=virtual_str):
                continue
            result.append(spec_item)
        return result

    @lang.memoized
    def buildcache_specs(self) -> List["spack.spec.Spec"]:
        self.binary_index.update()
        return self.binary_index.get_all_built_specs()

    @lang.memoized
    def can_be_installed(self, *, pkg_name: str) -> bool: # Overrides-Signature
        if self.configuration.get(f"packages:{pkg_name}:buildable", True) is True:
            return True

        if self.configuration.get(f"packages:{pkg_name}:externals", []) != []:
            return True

        reuse_cfg: Union[bool, str, Dict[str,Union[bool,List[str]]]] = self.configuration.get("concretizer:reuse") # type: ignore[assignment]
        if reuse_cfg is not False and self.store.db.query(pkg_name):
            return True

        if reuse_cfg is not False and any(x.name == pkg_name for x in self.buildcache_specs()):
            return True

        tty.debug(f"[{__name__}] {pkg_name} cannot be installed")
        return False

    @lang.memoized
    def _is_provider_candidate(self, *, pkg_name: str, virtual: str) -> bool:
        if not self.is_allowed_on_this_platform(pkg_name=pkg_name):
            return False

        if not self.can_be_installed(pkg_name=pkg_name):
            return False

        virtual_spec_obj: "spack.spec.Spec" = spack.spec.Spec(virtual)
        # Assuming when_spec can also be a pkg_name string for this check based on usage
        if self.unreachable(pkg_name=virtual_spec_obj.name, when_spec=spack.spec.Spec(pkg_name)):
            tty.debug(f"[{__name__}] {pkg_name} cannot be a provider for {virtual}")
            return False

        return True

    @lang.memoized
    def unreachable(self, *, pkg_name: str, when_spec: "spack.spec.Spec") -> bool: # Overrides-Signature
        """Returns true if the context can determine that the condition cannot ever
        be met on pkg_name.
        """
        candidates_req: Union[str, List[str]] = self.configuration.get(f"packages:{pkg_name}:require", []) # type: ignore[assignment]
        if not candidates_req and pkg_name != "all":
            return self.unreachable(pkg_name="all", when_spec=when_spec)

        if not candidates_req:
            return False

        if isinstance(candidates_req, str):
            candidates_req = [candidates_req]

        union_requirement_spec: "spack.spec.Spec" = spack.spec.Spec()
        for c_str in candidates_req:
            if not isinstance(c_str, str): # Should not happen given previous check
                continue
            try:
                union_requirement_spec.constrain(c_str)
            except SpackError:
                # Less optimized, but shouldn't fail
                pass

        if not union_requirement_spec.intersects(when_spec):
            return True

        return False


def create_graph_analyzer() -> "PossibleDependencyGraph":
    static_analysis_cfg: bool = spack.config.CONFIG.get("concretizer:static_analysis", False)
    if static_analysis_cfg:
        return StaticAnalysis(
            configuration=spack.config.CONFIG, # type: ignore[arg-type]
            repo=spack.repo.PATH, # type: ignore[arg-type]
            store=spack.store.STORE, # type: ignore[arg-type]
            binary_index=spack.binary_distribution.BINARY_INDEX, # type: ignore[arg-type]
        )
    return NoStaticAnalysis(configuration=spack.config.CONFIG, repo=spack.repo.PATH) # type: ignore[arg-type]


class Counter:
    """Computes the possible packages and the maximum number of duplicates
    allowed for each of them.

    Args:
        specs: abstract specs to concretize
        tests: if True, add test dependencies to the list of possible packages
    """

    def __init__(
        self,
        specs: List["spack.spec.Spec"],
        tests: bool,
        possible_graph: "PossibleDependencyGraph",
    ) -> None:
        self.possible_graph: "PossibleDependencyGraph" = possible_graph
        self.specs: List["spack.spec.Spec"] = specs
        self.link_run_types: "dt.DepFlag" = dt.LINK | dt.RUN | dt.TEST
        self.all_types: "dt.DepFlag" = dt.ALL
        if not tests:
            self.link_run_types = dt.LINK | dt.RUN
            self.all_types = dt.LINK | dt.RUN | dt.BUILD

        self._possible_dependencies: Set[str] = set()
        self._possible_virtuals: Set[str] = {
            x.name for x in specs if spack.repo.PATH.is_virtual(x.name) # type: ignore[attr-defined]
        }

    def possible_dependencies(self) -> Set[str]:
        """Returns the list of possible dependencies"""
        self.ensure_cache_values()
        return self._possible_dependencies

    def possible_virtuals(self) -> Set[str]:
        """Returns the list of possible virtuals"""
        self.ensure_cache_values()
        return self._possible_virtuals

    def ensure_cache_values(self) -> None:
        """Ensure the cache values have been computed"""
        if self._possible_dependencies:
            return
        self._compute_cache_values()

    def possible_packages_facts(
        self, gen: "spack.solver.asp.ProblemInstanceBuilder", fn: "spack.solver.asp._AspFunctionBuilder" # type: ignore[name-defined]
    ) -> None:
        """Emit facts associated with the possible packages"""
        raise NotImplementedError("must be implemented by derived classes")

    def _compute_cache_values(self) -> None:
        raise NotImplementedError("must be implemented by derived classes")


class NoDuplicatesCounter(Counter):
    def _compute_cache_values(self) -> None: # Overrides-Signature
        graph_result: "PossibleGraph" = self.possible_graph.possible_dependencies(
            *self.specs, allowed_deps=self.all_types
        )
        self._possible_dependencies, self._possible_virtuals_from_graph, _ = graph_result
        self._possible_virtuals.update(self._possible_virtuals_from_graph)

    def possible_packages_facts(
        self, gen: "spack.solver.asp.ProblemInstanceBuilder", fn: "spack.solver.asp._AspFunctionBuilder" # type: ignore[name-defined]
    ) -> None: # Overrides-Signature
        gen.h2("Maximum number of nodes (packages)")
        for package_name_item in sorted(self.possible_dependencies()):
            gen.fact(fn.max_dupes(package_name_item, 1))
        gen.newline()
        gen.h2("Maximum number of nodes (virtual packages)")
        for package_name_virt in sorted(self.possible_virtuals()):
            gen.fact(fn.max_dupes(package_name_virt, 1))
        gen.newline()
        gen.h2("Possible package in link-run subDAG")
        for name_item in sorted(self.possible_dependencies()):
            gen.fact(fn.possible_in_link_run(name_item))
        gen.newline()


class MinimalDuplicatesCounter(NoDuplicatesCounter):
    def __init__(
        self,
        specs: List["spack.spec.Spec"],
        tests: bool,
        possible_graph: "PossibleDependencyGraph",
    ) -> None:
        super().__init__(specs, tests, possible_graph)
        self._link_run: Set[str] = set()
        self._direct_build: Set[str] = set()
        self._total_build: Set[str] = set()
        self._link_run_virtuals: Set[str] = set()

    def _compute_cache_values(self) -> None: # Overrides-Signature
        link_run_graph: "PossibleGraph" = self.possible_graph.possible_dependencies(
            *self.specs, allowed_deps=self.link_run_types
        )
        self._link_run, link_run_virtuals_graph, _ = link_run_graph
        self._possible_virtuals.update(link_run_virtuals_graph)
        self._link_run_virtuals.update(link_run_virtuals_graph)

        for x_pkg_name in self._link_run:
            direct_build_graph: "PossibleGraph" = self.possible_graph.possible_dependencies(
                x_pkg_name, allowed_deps=dt.BUILD, transitive=False, strict_depflag=True
            )
            direct_build_reals, direct_build_virtuals, _ = direct_build_graph
            self._possible_virtuals.update(direct_build_virtuals)
            self._direct_build.update(direct_build_reals)

        total_build_graph: "PossibleGraph" = self.possible_graph.possible_dependencies(
            *self._direct_build, allowed_deps=self.all_types
        )
        _, total_build_virtuals_graph, _ = total_build_graph
        self._possible_virtuals.update(total_build_virtuals_graph)
        self._possible_dependencies = set(self._link_run) | set(self._total_build)

    def possible_packages_facts(
        self, gen: "spack.solver.asp.ProblemInstanceBuilder", fn: "spack.solver.asp._AspFunctionBuilder" # type: ignore[name-defined]
    ) -> None: # Overrides-Signature
        build_tools_set: Set[str] = set()
        for current_tag_item in ("build-tools", "compiler"):
            build_tools_set.update(spack.repo.PATH.packages_with_tags(current_tag_item))

        gen.h2("Packages with at most a single node")
        for package_name_single in sorted(self.possible_dependencies() - build_tools_set):
            gen.fact(fn.max_dupes(package_name_single, 1))
        gen.newline()

        gen.h2("Packages with multiple possible nodes (build-tools)")
        default_max_dupes: int = spack.config.CONFIG.get("concretizer:duplicates:max_dupes:default", 2)
        for package_name_multi in sorted(self.possible_dependencies() & build_tools_set):
            max_dupes_val: int = spack.config.CONFIG.get(
                f"concretizer:duplicates:max_dupes:{package_name_multi}", default_max_dupes
            )
            gen.fact(fn.max_dupes(package_name_multi, max_dupes_val))
            if max_dupes_val > 1:
                gen.fact(fn.multiple_unification_sets(package_name_multi))
        gen.newline()

        gen.h2("Maximum number of nodes (link-run virtuals)")
        for package_name_link_virt in sorted(self._link_run_virtuals):
            gen.fact(fn.max_dupes(package_name_link_virt, 1))
        gen.newline()

        gen.h2("Maximum number of nodes (other virtuals)")
        for package_name_other_virt in sorted(self.possible_virtuals() - self._link_run_virtuals):
            max_dupes_other_val: int = spack.config.CONFIG.get(
                f"concretizer:duplicates:max_dupes:{package_name_other_virt}", default_max_dupes
            )
            gen.fact(fn.max_dupes(package_name_other_virt, max_dupes_other_val))
        gen.newline()

        gen.h2("Possible package in link-run subDAG")
        for name_link_run in sorted(self._link_run):
            gen.fact(fn.possible_in_link_run(name_link_run))
        gen.newline()


class FullDuplicatesCounter(MinimalDuplicatesCounter):
    def possible_packages_facts(
        self, gen: "spack.solver.asp.ProblemInstanceBuilder", fn: "spack.solver.asp._AspFunctionBuilder" # type: ignore[name-defined]
    ) -> None: # Overrides-Signature
        pkg_counter: "collections.Counter[str]" = collections.Counter(
            list(self._link_run) + list(self._total_build) + list(self._direct_build)
        )
        gen.h2("Maximum number of nodes")
        for pkg_item, count_val in sorted(pkg_counter.items(), key=lambda x: (x[1], x[0])):
            actual_count: int = min(count_val, 2)
            gen.fact(fn.max_dupes(pkg_item, actual_count))
        gen.newline()

        gen.h2("Build unification sets ")
        build_tools_unif_set: Set[str] = set()
        for current_tag_unif in ("build-tools", "compiler"):
            build_tools_unif_set.update(spack.repo.PATH.packages_with_tags(current_tag_unif))

        for name_unif in sorted(self.possible_dependencies() & build_tools_unif_set):
            gen.fact(fn.multiple_unification_sets(name_unif))
        gen.newline()

        gen.h2("Possible package in link-run subDAG")
        for name_link_run_full in sorted(self._link_run):
            gen.fact(fn.possible_in_link_run(name_link_run_full))
        gen.newline()

        virtual_counter: "collections.Counter[str]" = collections.Counter(
            list(self._link_run_virtuals) + list(self._possible_virtuals)
        )
        gen.h2("Maximum number of virtual nodes")
        for pkg_virt_item, count_virt_val in sorted(
            virtual_counter.items(), key=lambda x: (x[1], x[0])
        ):
            gen.fact(fn.max_dupes(pkg_virt_item, count_virt_val))
        gen.newline()


def create_counter(
    specs: List["spack.spec.Spec"], tests: bool, possible_graph: "PossibleDependencyGraph"
) -> "Counter":
    strategy_str: str = spack.config.CONFIG.get("concretizer:duplicates:strategy", "none")
    if strategy_str == "full":
        return FullDuplicatesCounter(specs, tests=tests, possible_graph=possible_graph)
    if strategy_str == "minimal":
        return MinimalDuplicatesCounter(specs, tests=tests, possible_graph=possible_graph)
    return NoDuplicatesCounter(specs, tests=tests, possible_graph=possible_graph)
