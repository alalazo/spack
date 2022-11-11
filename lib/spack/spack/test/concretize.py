# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import posixpath
import sys

import jinja2
import pytest

import archspec.cpu

import llnl.util.lang

import spack.compilers
import spack.concretize
import spack.error
import spack.hash_types as ht
import spack.platforms
import spack.repo
import spack.variant as vt
from spack.concretize import find_spec
from spack.solver.asp import UnsatisfiableSpecError
from spack.spec import Spec
from spack.version import ver


@pytest.fixture(
    params=[
        # Mocking the host detection
        "haswell",
        "broadwell",
        "skylake",
        "icelake",
        # Using preferred targets from packages.yaml
        "icelake-preference",
        "cannonlake-preference",
    ]
)
def current_host(request, monkeypatch):
    # is_preference is not empty if we want to supply the
    # preferred target via packages.yaml
    cpu, _, is_preference = request.param.partition("-")
    target = archspec.cpu.TARGETS[cpu]

    monkeypatch.setattr(spack.platforms.Test, "default", cpu)
    monkeypatch.setattr(spack.platforms.Test, "front_end", cpu)
    if not is_preference:
        monkeypatch.setattr(archspec.cpu, "host", lambda: target)
        yield target
    else:
        with spack.config.override("packages:all", {"target": [cpu]}):
            yield target


@pytest.fixture()
def repo_with_changing_recipe(tmpdir_factory, mutable_mock_repo):
    repo_namespace = "changing"
    repo_dir = tmpdir_factory.mktemp(repo_namespace)

    repo_dir.join("repo.yaml").write(
        """
repo:
  namespace: changing
""",
        ensure=True,
    )

    packages_dir = repo_dir.ensure("packages", dir=True)
    root_pkg_str = """
class Root(Package):
    homepage = "http://www.example.com"
    url      = "http://www.example.com/root-1.0.tar.gz"

    version(1.0, sha256='abcde')
    depends_on('changing')

    conflicts('changing~foo')
"""
    packages_dir.join("root", "package.py").write(root_pkg_str, ensure=True)

    changing_template = """
class Changing(Package):
    homepage = "http://www.example.com"
    url      = "http://www.example.com/changing-1.0.tar.gz"


{% if not delete_version %}
    version(1.0, sha256='abcde')
{% endif %}
    version(0.9, sha256='abcde')

{% if not delete_variant %}
    variant('fee', default=True, description='nope')
{% endif %}
    variant('foo', default=True, description='nope')
{% if add_variant %}
    variant('fum', default=True, description='nope')
    variant('fum2', default=True, description='nope')
{% endif %}
"""

    with spack.repo.use_repositories(str(repo_dir), override=False) as repository:

        class _ChangingPackage(object):
            default_context = [
                ("delete_version", True),
                ("delete_variant", False),
                ("add_variant", False),
            ]

            def __init__(self, repo_directory):
                self.repo_dir = repo_directory
                self.repo = spack.repo.Repo(str(repo_directory))

            def change(self, changes=None):
                changes = changes or {}
                context = dict(self.default_context)
                context.update(changes)
                # Remove the repo object and delete Python modules
                repository.remove(self.repo)
                # TODO: this mocks a change in the recipe that should happen in a
                # TODO: different process space. Leaving this comment as a hint
                # TODO: in case tests using this fixture start failing.
                if sys.modules.get("spack.pkg.changing.changing"):
                    del sys.modules["spack.pkg.changing.changing"]
                    del sys.modules["spack.pkg.changing.root"]
                    del sys.modules["spack.pkg.changing"]

                # Change the recipe
                t = jinja2.Template(changing_template)
                changing_pkg_str = t.render(**context)
                packages_dir.join("changing", "package.py").write(changing_pkg_str, ensure=True)

                # Re-add the repository
                self.repo = spack.repo.Repo(str(self.repo_dir))
                repository.put_first(self.repo)

        _changing_pkg = _ChangingPackage(repo_dir)
        _changing_pkg.change(
            {"delete_version": False, "delete_variant": False, "add_variant": False}
        )
        yield _changing_pkg


@pytest.fixture
def sanity_check_concretize(default_mock_concretization):
    def _func(spec_str):
        abstract_spec = spack.spec.Spec(spec_str)
        concrete_spec = default_mock_concretization(spec_str)
        if abstract_spec.versions.concrete:
            assert abstract_spec.versions == concrete_spec.versions
        if abstract_spec.variants:
            for name in abstract_spec.variants:
                avariant = abstract_spec.variants[name]
                cvariant = concrete_spec.variants[name]
                assert avariant.value == cvariant.value
        if abstract_spec.compiler_flags:
            for flag in abstract_spec.compiler_flags:
                aflag = abstract_spec.compiler_flags[flag]
                cflag = concrete_spec.compiler_flags[flag]
                assert set(aflag) <= set(cflag)
        for name in spack.repo.path.get_pkg_class(abstract_spec.name).variants:
            assert name in concrete_spec.variants
        for flag in concrete_spec.compiler_flags.valid_compiler_flags():
            assert flag in concrete_spec.compiler_flags
        if abstract_spec.compiler and abstract_spec.compiler.concrete:
            assert abstract_spec.compiler == concrete_spec.compiler
        if abstract_spec.architecture and abstract_spec.architecture.concrete:
            assert abstract_spec.architecture == concrete_spec.architecture
        return abstract_spec, concrete_spec

    return _func


@pytest.mark.usefixtures("mock_packages")
class TestConcretize(object):
    @pytest.mark.parametrize(
        "spec_str",
        [
            # no_deps
            "libelf",
            "libelf@0.8.13",
            # dag
            "callpath",
            "mpileaks",
            "libelf",
            "indirect-mpich",
            # withdeps
            "mpileaks ^mpich2",
            # variant
            "mpich+debug",
            "mpich~debug",
            "mpich debug=True",
            "mpich",
            # compiler flags
            'mpich cppflags="-O3"',
            'mpich cppflags=="-O3"',
            # with virtual
            "mpileaks ^mpi",
            "mpileaks ^mpi@:1.1",
            "mpileaks ^mpi@2:",
            "mpileaks ^mpi@2.1",
            "mpileaks ^mpi@2.2",
            "mpileaks ^mpi@2.2",
            "mpileaks ^mpi@:1",
            "mpileaks ^mpi@1.2:2",
            # conflict not triggered
            "conflict",
            "conflict%clang~foo",
            "conflict-parent%gcc",
            # dependent with single valued variant type
            "singlevalue-variant-dependent-type",
            # package with multiple virtual dependencies
            "hypre",
            "hypre ^openblas",
            "hypre ^openblas ^netlib-lapack",
            "hypre ^openblas-with-lapack",
        ],
    )
    def test_concretize(self, sanity_check_concretize, spec_str):
        sanity_check_concretize(spec_str)

    def test_concretize_mention_build_dep(self, sanity_check_concretize):
        _, concrete_spec = sanity_check_concretize("cmake-client ^cmake@3.21.3")

        # Check parent's perspective of child
        to_dependencies = concrete_spec.edges_to_dependencies(name="cmake")
        assert len(to_dependencies) == 1
        assert set(to_dependencies[0].deptypes) == set(["build"])

        # Check child's perspective of parent
        cmake = concrete_spec["cmake"]
        from_dependents = cmake.edges_from_dependents(name="cmake-client")
        assert len(from_dependents) == 1
        assert set(from_dependents[0].deptypes) == set(["build"])

    def test_concretize_preferred_version(self, sanity_check_concretize):
        _, concrete_spec = sanity_check_concretize("python")
        assert concrete_spec.versions == ver("2.7.11")
        _, concrete_spec = sanity_check_concretize("python@3.5.1")
        assert concrete_spec.versions == ver("3.5.1")

    @pytest.mark.parametrize(
        "spec_str,expected_satisfies",
        [
            ("mpileaks ^mpich2@1.1", "mpich2@1.1"),
            ("mpileaks ^mpich2@1.2", "mpich2@1.2"),
            ("mpileaks ^mpich2@:1.5", "mpich2@:1.5"),
            ("mpileaks ^mpich2@:1.3", "mpich2@:1.3"),
            ("mpileaks ^mpich2@:1.2", "mpich2@:1.2"),
            ("mpileaks ^mpich2@:1.1", "mpich2@:1.1"),
            ("mpileaks ^mpich2@1.1:", "mpich2@1.1:"),
            ("mpileaks ^mpich2@1.5:", "mpich2@1.5:"),
            ("mpileaks ^mpich2@1.3.1:1.4", "mpich2@1.3.1:1.4"),
        ],
    )
    def test_concretize_with_restricted_virtual(
        self, sanity_check_concretize, spec_str, expected_satisfies
    ):
        _, concrete_spec = sanity_check_concretize(spec_str)
        assert concrete_spec["mpich2"].satisfies(expected_satisfies)

    @pytest.mark.parametrize(
        "unsatisfiable_constraint,provider_str",
        [
            ("mpich2@:1.0", "mpi@2.1"),
            ("mpich2@:1.1", "mpi@2.2"),
            ("mpich@:1", "mpi@2"),
            ("mpich@:1", "mpi@3"),
            ("mpich2", "mpi@3"),
        ],
    )
    def test_concretize_with_provides_when(
        self, mock_packages, unsatisfiable_constraint, provider_str
    ):
        """Make sure insufficient versions of MPI are not in providers list when
        we ask for some advanced version.
        """
        assert not any(
            s.satisfies(unsatisfiable_constraint)
            for s in mock_packages.providers_for(provider_str)
        )

    def test_provides_handles_multiple_providers_of_same_version(self, mock_packages):
        providers = mock_packages.providers_for("mpi@3.0")
        # Note that providers are repo-specific, so we don't misinterpret
        # providers, but vdeps are not namespace-specific, so we can
        # associate vdeps across repos.
        assert Spec("builtin.mock.multi-provider-mpi@1.10.3") in providers
        assert Spec("builtin.mock.multi-provider-mpi@1.10.2") in providers
        assert Spec("builtin.mock.multi-provider-mpi@1.10.1") in providers
        assert Spec("builtin.mock.multi-provider-mpi@1.10.0") in providers
        assert Spec("builtin.mock.multi-provider-mpi@1.8.8") in providers

    def test_different_compilers_get_different_flags(self, default_mock_concretization):
        client = default_mock_concretization(
            "cmake-client %gcc@11.1.0 platform=test os=fe target=fe"
            + " ^cmake %clang@12.2.0 platform=test os=fe target=fe"
        )
        cmake = client["cmake"]
        assert set(client.compiler_flags["cflags"]) == set(["-O0", "-g"])
        assert set(cmake.compiler_flags["cflags"]) == set(["-O3"])
        assert set(client.compiler_flags["fflags"]) == set(["-O0", "-g"])
        assert not set(cmake.compiler_flags["fflags"])

    def test_concretize_compiler_flag_propagate(self, default_mock_concretization):
        spec = default_mock_concretization("hypre cflags=='-g' ^openblas")
        assert spec.satisfies("^openblas cflags='-g'")

    @pytest.mark.not_on_windows
    @pytest.mark.only_clingo(
        "Optional compiler propagation isn't deprecated for original concretizer"
    )
    def test_concretize_compiler_flag_does_not_propagate(self, default_mock_concretization):
        spec = default_mock_concretization("hypre cflags='-g' ^openblas")
        assert not spec.satisfies("^openblas cflags='-g'")

    @pytest.mark.not_on_windows
    @pytest.mark.only_clingo(
        "Optional compiler propagation isn't deprecated for original concretizer"
    )
    def test_concretize_propagate_compiler_flag_not_passed_to_dependent(
        self, default_mock_concretization
    ):
        spec = default_mock_concretization("hypre cflags=='-g' ^openblas cflags='-O3'")
        assert set(spec.compiler_flags["cflags"]) == set(["-g"])
        assert spec.satisfies("^openblas cflags='-O3'")

    def test_compiler_inherited_upwards(self, default_mock_concretization):
        spec = default_mock_concretization("dt-diamond ^dt-diamond-bottom%clang")
        for dep in spec.traverse():
            assert dep.satisfies("%clang")

    def test_architecture_inheritance(self, default_mock_concretization):
        """test_architecture_inheritance is likely to fail with an
        UnavailableCompilerVersionError if the architecture is concretized
        incorrectly.
        """
        spec = default_mock_concretization("cmake-client %gcc@11.1.0 os=fe ^ cmake")
        assert spec["cmake"].architecture == spec.architecture

    def test_concretize_enable_disable_compiler_existence_check(self, mutable_config):
        with spack.concretize.enable_compiler_existence_check():
            with pytest.raises(spack.concretize.UnavailableCompilerVersionError):
                Spec("dttop %gcc@100.100").concretized()

        with spack.concretize.disable_compiler_existence_check():
            spec = Spec("dttop %gcc@100.100").concretized()
            assert spec.satisfies("%gcc@100.100")
            assert spec["dtlink3"].satisfies("%gcc@100.100")

    def test_mixing_compilers_only_affects_subdag(self, mutable_config):
        spack.config.set("packages:all:compiler", ["clang", "gcc"])
        spec = Spec("dt-diamond%gcc ^dt-diamond-bottom%clang").concretized()
        for dep in spec.traverse():
            assert ("%clang" in dep) == (dep.name == "dt-diamond-bottom")

    @pytest.mark.only_clingo("Fixing the parser broke this test for the original concretizer")
    def test_architecture_deep_inheritance(self, config, mock_targets):
        """Make sure that indirect dependencies receive architecture
        information from the root even when partial architecture information
        is provided by an intermediate dependency.
        """
        spec_str = "mpileaks %gcc@4.5.0 os=CNL target=nocona ^dyninst os=CNL ^callpath os=CNL"
        spec = Spec(spec_str).concretized()
        for s in spec.traverse(root=False):
            assert s.architecture.target == spec.architecture.target

    def test_compiler_flags_from_user_are_grouped(self, default_mock_concretization):
        spec = default_mock_concretization('a%gcc cflags="-O -foo-flag foo-val" platform=test')
        cflags = spec.compiler_flags["cflags"]
        assert any(x == "-foo-flag foo-val" for x in cflags)

    @pytest.mark.only_clingo("Original concretizer cannot resolve this constraint")
    def test_concretize_multi_provider(self, default_mock_concretization):
        concrete_spec = default_mock_concretization("mpileaks ^multi-provider-mpi ^mpi@3.0")
        assert concrete_spec["mpi"].version == ver("1.10.3")

    def test_concretize_two_virtuals_with_dual_provider_and_a_conflict(self):
        """Test a package with multiple virtual dependencies and force a
        provider that provides both, and another conflicting package that
        provides one.
        """
        s = Spec("hypre ^openblas-with-lapack ^netlib-lapack")
        with pytest.raises(spack.error.SpackError):
            s.concretize()

    @pytest.mark.not_on_windows
    @pytest.mark.only_clingo(
        "Optional compiler propagation isn't deprecated for original concretizer"
    )
    def test_concretize_propagate_disabled_variant(self, default_mock_concretization):
        """Test a package variant value was passed from its parent."""
        spec = default_mock_concretization("hypre~~shared ^openblas")
        assert spec.satisfies("^openblas~shared")

    def test_concretize_propagated_variant_is_not_passed_to_dependent(
        self, default_mock_concretization
    ):
        """Test a package variant value was passed from its parent."""
        spec = default_mock_concretization("hypre~~shared ^openblas+shared")
        assert spec.satisfies("^openblas+shared")

    @pytest.mark.not_on_windows
    def test_no_matching_compiler_specs(self, mock_low_high_config):
        # only relevant when not building compilers as needed
        with spack.concretize.enable_compiler_existence_check():
            s = Spec("a %gcc@0.0.0")
            with pytest.raises(spack.concretize.UnavailableCompilerVersionError):
                s.concretize()

    def test_no_compilers_for_arch(self, config):
        s = Spec("a arch=linux-rhel0-x86_64")
        with pytest.raises(spack.error.SpackError):
            s.concretize()

    def test_virtual_is_fully_expanded_for_callpath(self, default_mock_concretization):
        # force dependence on fake "zmpi" by asking for MPI 10.0
        spec = Spec("callpath ^mpi@10.0")
        assert len(spec.dependencies(name="mpi")) == 1
        assert "fake" not in spec

        spec = default_mock_concretization("callpath ^mpi@10.0")
        assert len(spec.dependencies(name="zmpi")) == 1
        assert all(not d.dependencies(name="mpi") for d in spec.traverse())
        assert all(x in spec for x in ("zmpi", "mpi"))

        edges_to_zmpi = spec.edges_to_dependencies(name="zmpi")
        assert len(edges_to_zmpi) == 1
        assert "fake" in edges_to_zmpi[0].spec

    def test_virtual_is_fully_expanded_for_mpileaks(self, default_mock_concretization):
        spec = Spec("mpileaks ^mpi@10.0")
        assert len(spec.dependencies(name="mpi")) == 1
        assert "fake" not in spec

        spec = default_mock_concretization("mpileaks ^mpi@10.0")
        assert len(spec.dependencies(name="zmpi")) == 1
        assert len(spec.dependencies(name="callpath")) == 1

        callpath = spec.dependencies(name="callpath")[0]
        assert len(callpath.dependencies(name="zmpi")) == 1

        zmpi = callpath.dependencies(name="zmpi")[0]
        assert len(zmpi.dependencies(name="fake")) == 1

        assert all(not d.dependencies(name="mpi") for d in spec.traverse())
        assert all(x in spec for x in ("zmpi", "mpi"))

    @pytest.mark.parametrize("compiler_str", ["clang", "gcc", "gcc@10.2.1", "clang@:12.0.0"])
    def test_compiler_inheritance(self, default_mock_concretization, compiler_str):
        spec_str = "mpileaks %{0}".format(compiler_str)
        spec = default_mock_concretization(spec_str)
        assert spec["libdwarf"].compiler.satisfies(compiler_str)
        assert spec["libelf"].compiler.satisfies(compiler_str)

    def test_external_package(self, default_mock_concretization):
        spec = default_mock_concretization("externaltool%gcc")
        assert spec["externaltool"].external_path == posixpath.sep + posixpath.join(
            "path", "to", "external_tool"
        )
        assert "externalprereq" not in spec
        assert spec["externaltool"].compiler.satisfies("gcc")

    def test_external_package_module(self, default_mock_concretization):
        # No tcl modules on darwin/linux machines
        # and Windows does not (currently) allow for bash calls
        # TODO: improved way to check for this.
        platform = spack.platforms.real_host().name
        if platform == "darwin" or platform == "linux" or platform == "windows":
            return

        spec = default_mock_concretization("externalmodule")
        assert spec["externalmodule"].external_modules == ["external-module"]
        assert "externalprereq" not in spec
        assert spec["externalmodule"].compiler.satisfies("gcc")

    def test_nobuild_package(self):
        """Test that a non-buildable package raise an error if no specs
        in packages.yaml are compatible with the request.
        """
        spec = Spec("externaltool%clang")
        with pytest.raises(spack.error.SpecError):
            spec.concretize()

    def test_external_and_virtual(self, default_mock_concretization):
        spec = default_mock_concretization("externaltest")
        assert spec["externaltool"].external_path == posixpath.sep + posixpath.join(
            "path", "to", "external_tool"
        )
        assert spec["stuff"].external_path == posixpath.sep + posixpath.join(
            "path", "to", "external_virtual_gcc"
        )
        assert spec["externaltool"].compiler.satisfies("gcc")
        assert spec["stuff"].compiler.satisfies("gcc")

    def test_find_spec_parents(self):
        """Tests the spec finding logic used by concretization."""
        s = Spec.from_literal({"a +foo": {"b +foo": {"c": None, "d+foo": None}, "e +foo": None}})

        assert "a" == find_spec(s["b"], lambda s: "+foo" in s).name

    def test_find_spec_children(self):
        s = Spec.from_literal({"a": {"b +foo": {"c": None, "d+foo": None}, "e +foo": None}})

        assert "d" == find_spec(s["b"], lambda s: "+foo" in s).name

        s = Spec.from_literal({"a": {"b +foo": {"c+foo": None, "d": None}, "e +foo": None}})

        assert "c" == find_spec(s["b"], lambda s: "+foo" in s).name

    def test_find_spec_sibling(self):

        s = Spec.from_literal({"a": {"b +foo": {"c": None, "d": None}, "e +foo": None}})

        assert "e" == find_spec(s["b"], lambda s: "+foo" in s).name
        assert "b" == find_spec(s["e"], lambda s: "+foo" in s).name

        s = Spec.from_literal({"a": {"b +foo": {"c": None, "d": None}, "e": {"f +foo": None}}})

        assert "f" == find_spec(s["b"], lambda s: "+foo" in s).name

    def test_find_spec_self(self):
        s = Spec.from_literal({"a": {"b +foo": {"c": None, "d": None}, "e": None}})
        assert "b" == find_spec(s["b"], lambda s: "+foo" in s).name

    def test_find_spec_none(self):
        s = Spec.from_literal({"a": {"b": {"c": None, "d": None}, "e": None}})
        assert find_spec(s["b"], lambda s: "+foo" in s) is None

    def test_compiler_child(self, default_mock_concretization):
        s = default_mock_concretization("mpileaks%clang target=x86_64 ^dyninst%gcc")
        assert s["mpileaks"].satisfies("%clang")
        assert s["dyninst"].satisfies("%gcc")

    def test_conflicts_in_spec(self, config, conflict_spec):
        s = Spec(conflict_spec)
        with pytest.raises(spack.error.SpackError):
            s.concretize()

    @pytest.mark.only_clingo("Test is specific to the new concretizer")
    def test_conflicts_show_cores(self, config, conflict_spec, monkeypatch):
        s = Spec(conflict_spec)
        with pytest.raises(spack.error.SpackError) as e:
            s.concretize()

        assert "conflict" in e.value.message

    def test_conflict_in_all_directives_true(self, config):
        s = Spec("when-directives-true")
        with pytest.raises(spack.error.SpackError):
            s.concretize()

    @pytest.mark.parametrize("spec_str", ["conflict@10.0%clang+foo"])
    def test_no_conflict_in_external_specs(self, mutable_config, spec_str):
        # Modify the configuration to have the spec with conflict
        # registered as an external
        ext = Spec(spec_str)
        data = {"externals": [{"spec": spec_str, "prefix": "/fake/path"}]}
        spack.config.set("packages::{0}".format(ext.name), data)
        ext.concretize()  # failure raises exception

    def test_regression_issue_4492(self, default_mock_concretization):
        # Constructing a spec which has no dependencies, but is otherwise
        # concrete is kind of difficult. What we will do is to concretize
        # a spec, and then modify it to have no dependency and reset the
        # cache values.
        s = default_mock_concretization("mpileaks").copy()
        assert s.concrete

        # Remove the dependencies and reset caches
        s.clear_dependencies()
        s._concrete = False
        assert not s.concrete

    @pytest.mark.regression("7239")
    def test_regression_issue_7239(self, default_mock_concretization):
        # Constructing a SpecBuildInterface from another SpecBuildInterface
        # results in an inconsistent MRO

        s = default_mock_concretization("mpileaks")

        assert llnl.util.lang.ObjectWrapper not in type(s).__mro__

        # Spec wrapped in a build interface
        build_interface = s["mpileaks"]
        assert llnl.util.lang.ObjectWrapper in type(build_interface).__mro__

        # Mimics asking the build interface from a build interface
        build_interface = s["mpileaks"]["mpileaks"]
        assert llnl.util.lang.ObjectWrapper in type(build_interface).__mro__

    @pytest.mark.regression("7705")
    def test_regression_issue_7705(self, default_mock_concretization):
        # spec.package.provides(name) doesn't account for conditional
        # constraints in the concretized spec
        s = default_mock_concretization("simple-inheritance~openblas")
        assert not s.package.provides("lapack")

    @pytest.mark.regression("7941")
    def test_regression_issue_7941(self, default_mock_concretization):
        # The string representation of a spec containing
        # an explicit multi-valued variant and a dependency
        # might be parsed differently than the originating spec
        spec_str = "a foobar=bar ^b"
        spec_str_after_roundtrip = str(Spec(spec_str))
        s = default_mock_concretization(spec_str)
        t = default_mock_concretization(spec_str_after_roundtrip)
        assert s.dag_hash() == t.dag_hash()

    @pytest.mark.parametrize(
        "abstract_specs",
        [
            # Establish a baseline - concretize a single spec
            ("mpileaks",),
            # When concretized together with older version of callpath
            # and dyninst it uses those older versions
            ("mpileaks", "callpath@0.9", "dyninst@8.1.1"),
            # Handle recursive syntax within specs
            ("mpileaks", "callpath@0.9 ^dyninst@8.1.1", "dyninst"),
            # Test specs that have overlapping dependencies but are not
            # one a dependency of the other
            ("mpileaks", "direct-mpich"),
        ],
    )
    def test_simultaneous_concretization_of_specs(self, config, abstract_specs):

        abstract_specs = [Spec(x) for x in abstract_specs]
        concrete_specs = spack.concretize.concretize_specs_together(*abstract_specs)

        # Check there's only one configuration of each package in the DAG
        names = set(dep.name for spec in concrete_specs for dep in spec.traverse())
        for name in names:
            name_specs = set(spec[name] for spec in concrete_specs if name in spec)
            assert len(name_specs) == 1

        # Check that there's at least one Spec that satisfies the
        # initial abstract request
        for aspec in abstract_specs:
            assert any(cspec.satisfies(aspec) for cspec in concrete_specs)

        # Make sure the concrete spec are top-level specs with no dependents
        for spec in concrete_specs:
            assert not spec.dependents()

    @pytest.mark.parametrize("spec", ["noversion", "noversion-bundle"])
    def test_noversion_pkg(self, config, spec):
        """Test concretization failures for no-version packages."""
        with pytest.raises(spack.error.SpackError):
            Spec(spec).concretized()

    @pytest.mark.not_on_windows
    # Include targets to prevent regression on 20537
    @pytest.mark.parametrize(
        "spec, best_achievable",
        [
            ("mpileaks%gcc@4.4.7 ^dyninst@10.2.1 target=x86_64:", "core2"),
            ("mpileaks%gcc@4.8 target=x86_64:", "haswell"),
            ("mpileaks%gcc@5.3.0 target=x86_64:", "broadwell"),
            ("mpileaks%apple-clang@5.1.0 target=x86_64:", "x86_64"),
        ],
    )
    @pytest.mark.regression("13361", "20537")
    def test_adjusting_default_target_based_on_compiler(
        self, spec, best_achievable, current_host, mock_targets, mutable_config
    ):
        best_achievable = archspec.cpu.TARGETS[best_achievable]
        expected = best_achievable if best_achievable < current_host else current_host
        with spack.concretize.disable_compiler_existence_check():
            s = Spec(spec).concretized()
            assert str(s.architecture.target) == str(expected)

    @pytest.mark.regression("8735,14730")
    def test_compiler_version_matches_any_entry_in_compilers_yaml(
        self, default_mock_concretization
    ):
        # Ensure that a concrete compiler with different compiler version
        # doesn't match (here it's 10.2 vs. 10.2.1)
        with pytest.raises(spack.concretize.UnavailableCompilerVersionError):
            default_mock_concretization("mpileaks %gcc@10.2")

        # An abstract compiler with a version list could resolve to 4.5.0
        s = default_mock_concretization("mpileaks %gcc@10.2:")
        assert str(s.compiler.version) == "10.2.1"

    def test_concretize_anonymous(self, config):
        with pytest.raises(spack.error.SpackError):
            s = Spec("+variant")
            s.concretize()

    @pytest.mark.parametrize("spec_str", ["mpileaks ^%gcc", "mpileaks ^cflags=-g"])
    def test_concretize_anonymous_dep(self, config, spec_str):
        with pytest.raises(spack.error.SpackError):
            s = Spec(spec_str)
            s.concretize()

    @pytest.mark.parametrize(
        "spec_str,expected_str",
        [
            # Unconstrained versions select default compiler (gcc@4.5.0)
            ("bowtie@1.4.0", "%gcc@10.2.1"),
            # Version with conflicts and no valid gcc select another compiler
            ("bowtie@1.3.0", "%clang@12.0.0"),
            # If a higher gcc is available still prefer that
            ("bowtie@1.2.2 os=redhat6", "%gcc@11.1.0"),
        ],
    )
    @pytest.mark.only_clingo("Original concretizer cannot work around conflicts")
    def test_compiler_conflicts_in_package_py(
        self, default_mock_concretization, spec_str, expected_str
    ):
        s = default_mock_concretization(spec_str)
        assert s.satisfies(expected_str)

    @pytest.mark.parametrize(
        "spec_str,expected,unexpected",
        [
            ("conditional-variant-pkg@1.0", ["two_whens"], ["version_based", "variant_based"]),
            ("conditional-variant-pkg@2.0", ["version_based", "variant_based"], ["two_whens"]),
            (
                "conditional-variant-pkg@2.0~version_based",
                ["version_based"],
                ["variant_based", "two_whens"],
            ),
            (
                "conditional-variant-pkg@2.0+version_based+variant_based",
                ["version_based", "variant_based", "two_whens"],
                [],
            ),
        ],
    )
    def test_conditional_variants(
        self, default_mock_concretization, spec_str, expected, unexpected
    ):
        s = default_mock_concretization(spec_str)

        for var in expected:
            assert s.satisfies("%s=*" % var)
        for var in unexpected:
            assert not s.satisfies("%s=*" % var)

    @pytest.mark.parametrize(
        "bad_spec",
        [
            "@1.0~version_based",
            "@1.0+version_based",
            "@2.0~version_based+variant_based",
            "@2.0+version_based~variant_based+two_whens",
        ],
    )
    def test_conditional_variants_fail(self, config, bad_spec):
        with pytest.raises((spack.error.UnsatisfiableSpecError, vt.InvalidVariantForSpecError)):
            _ = Spec("conditional-variant-pkg" + bad_spec).concretized()

    @pytest.mark.parametrize(
        "spec_str,expected,unexpected",
        [
            ("py-extension3 ^python@3.5.1", [], ["py-extension1"]),
            ("py-extension3 ^python@2.7.11", ["py-extension1"], []),
            ("py-extension3@1.0 ^python@2.7.11", ["patchelf@0.9"], []),
            ("py-extension3@1.1 ^python@2.7.11", ["patchelf@0.9"], []),
            ("py-extension3@1.0 ^python@3.5.1", ["patchelf@0.10"], []),
        ],
    )
    def test_conditional_dependencies(
        self, default_mock_concretization, spec_str, expected, unexpected
    ):
        s = default_mock_concretization(spec_str)

        for dep in expected:
            msg = '"{0}" is not in "{1}" and was expected'
            assert dep in s, msg.format(dep, spec_str)

        for dep in unexpected:
            msg = '"{0}" is in "{1}" but was unexpected'
            assert dep not in s, msg.format(dep, spec_str)

    @pytest.mark.parametrize(
        "spec_str,patched_deps",
        [
            ("patch-several-dependencies", [("libelf", 1), ("fake", 2)]),
            ("patch-several-dependencies@1.0", [("libelf", 1), ("fake", 2), ("libdwarf", 1)]),
            (
                "patch-several-dependencies@1.0 ^libdwarf@20111030",
                [("libelf", 1), ("fake", 2), ("libdwarf", 2)],
            ),
            ("patch-several-dependencies ^libelf@0.8.10", [("libelf", 2), ("fake", 2)]),
            ("patch-several-dependencies +foo", [("libelf", 2), ("fake", 2)]),
        ],
    )
    def test_patching_dependencies(self, default_mock_concretization, spec_str, patched_deps):
        s = default_mock_concretization(spec_str)

        for dep, num_patches in patched_deps:
            assert s[dep].satisfies("patches=*")
            assert len(s[dep].variants["patches"].value) == num_patches

    @pytest.mark.regression("267,303,1781,2310,2632,3628")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    @pytest.mark.parametrize(
        "spec_str, expected",
        [
            # Need to understand that this configuration is possible
            # only if we use the +mpi variant, which is not the default
            ("fftw ^mpich", ["+mpi"]),
            # This spec imposes two orthogonal constraints on a dependency,
            # one of which is conditional. The original concretizer fail since
            # when it applies the first constraint, it sets the unknown variants
            # of the dependency to their default values
            ("quantum-espresso", ["^fftw@1.0+mpi"]),
            # This triggers a conditional dependency on ^fftw@1.0
            ("quantum-espresso", ["^openblas"]),
            # This constructs a constraint for a dependency og the type
            # @x.y:x.z where the lower bound is unconditional, the upper bound
            # is conditional to having a variant set
            ("quantum-espresso", ["^libelf@0.8.12"]),
            ("quantum-espresso~veritas", ["^libelf@0.8.13"]),
        ],
    )
    def test_working_around_conflicting_defaults(
        self, default_mock_concretization, spec_str, expected
    ):
        s = default_mock_concretization(spec_str)
        assert s.concrete
        for constraint in expected:
            assert s.satisfies(constraint)

    @pytest.mark.regression("4635")
    @pytest.mark.parametrize(
        "spec_str,expected",
        [("cmake", ["%clang"]), ("cmake %gcc", ["%gcc"]), ("cmake %clang", ["%clang"])],
    )
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_external_package_and_compiler_preferences(self, mutable_config, spec_str, expected):
        packages_yaml = {
            "all": {
                "compiler": ["clang", "gcc"],
            },
            "cmake": {
                "externals": [{"spec": "cmake@3.4.3", "prefix": "/usr"}],
                "buildable": False,
            },
        }
        spack.config.set("packages", packages_yaml)
        s = Spec(spec_str).concretized()

        assert s.external
        for condition in expected:
            assert s.satisfies(condition)

    @pytest.mark.regression("5651")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_package_with_constraint_not_met_by_external(self, mutable_config):
        """Check that if we have an external package A at version X.Y in
        packages.yaml, but our spec doesn't allow X.Y as a version, then
        a new version of A is built that meets the requirements.
        """
        packages_yaml = {"libelf": {"externals": [{"spec": "libelf@0.8.13", "prefix": "/usr"}]}}
        spack.config.set("packages", packages_yaml)

        # quantum-espresso+veritas requires libelf@:0.8.12
        s = Spec("quantum-espresso+veritas").concretized()
        assert s.satisfies("^libelf@0.8.12")
        assert not s["libelf"].external

    @pytest.mark.regression("9744")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_cumulative_version_ranges_with_different_length(self, default_mock_concretization):
        s = default_mock_concretization("cumulative-vrange-root")
        assert s.satisfies("^cumulative-vrange-bottom@2.2")

    @pytest.mark.regression("9937")
    def test_dependency_conditional_on_another_dependency_state(self, default_mock_concretization):
        root_str = "variant-on-dependency-condition-root"
        dep_str = "variant-on-dependency-condition-a"
        spec_str = "{0} ^{1}".format(root_str, dep_str)

        s = default_mock_concretization(spec_str)
        assert s.satisfies("^variant-on-dependency-condition-b")

        s = default_mock_concretization(spec_str + "+x")
        assert s.satisfies("^variant-on-dependency-condition-b")

        s = default_mock_concretization(spec_str + "~x")
        assert not s.satisfies("^variant-on-dependency-condition-b")

    @pytest.mark.regression("8082")
    @pytest.mark.parametrize(
        "spec_str,expected", [("cmake %gcc", "%gcc"), ("cmake %clang", "%clang")]
    )
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_compiler_constraint_with_external_package(self, mutable_config, spec_str, expected):
        packages_yaml = {
            "cmake": {"externals": [{"spec": "cmake@3.4.3", "prefix": "/usr"}], "buildable": False}
        }
        spack.config.set("packages", packages_yaml)

        s = Spec(spec_str).concretized()
        assert s.external
        assert s.satisfies(expected)

    @pytest.mark.regression("20976")
    @pytest.mark.parametrize(
        "compiler,spec_str,expected,xfailold",
        [
            (
                "gcc",
                "external-common-python %clang",
                "%clang ^external-common-openssl%gcc ^external-common-gdbm%clang",
                False,
            ),
            (
                "clang",
                "external-common-python",
                "%clang ^external-common-openssl%clang ^external-common-gdbm%clang",
                True,
            ),
        ],
    )
    def test_compiler_in_nonbuildable_external_package(
        self, compiler, spec_str, expected, xfailold, mutable_config
    ):
        """Check that the compiler of a non-buildable external package does not
        spread to other dependencies, unless no other commpiler is specified."""
        packages_yaml = {
            "external-common-openssl": {
                "externals": [
                    {"spec": "external-common-openssl@1.1.1i%" + compiler, "prefix": "/usr"}
                ],
                "buildable": False,
            }
        }
        spack.config.set("packages", packages_yaml)

        s = Spec(spec_str).concretized()
        if xfailold and spack.config.get("config:concretizer") == "original":
            pytest.xfail("This only works on the ASP-based concretizer")
        assert s.satisfies(expected)
        assert "external-common-perl" not in [d.name for d in s.dependencies()]

    @pytest.mark.only_clingo("This test is specific to the ASP-based concretizer")
    def test_external_packages_have_consistent_hash(self, default_mock_concretization):
        s = Spec("externaltool")
        s._old_concretize()
        t = default_mock_concretization("externaltool")
        assert s.dag_hash() == t.dag_hash()

    def test_external_that_would_require_a_virtual_dependency(self, default_mock_concretization):
        s = default_mock_concretization("requires-virtual")
        assert s.external
        assert "stuff" not in s

    def test_transitive_conditional_virtual_dependency(self, default_mock_concretization):
        s = default_mock_concretization("transitive-conditional-virtual-dependency")

        # The default for conditional-virtual-dependency is to have
        # +stuff~mpi, so check that these defaults are respected
        assert "+stuff" in s["conditional-virtual-dependency"]
        assert "~mpi" in s["conditional-virtual-dependency"]

        # 'stuff' is provided by an external package, so check it's present
        assert "externalvirtual" in s

    @pytest.mark.regression("20040")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_conditional_provides_or_depends_on(self, default_mock_concretization):
        # Check that we can concretize correctly a spec that can either
        # provide a virtual or depend on it based on the value of a variant
        s = default_mock_concretization("conditional-provider +disable-v1")
        assert "v1-provider" in s
        assert s["v1"].name == "v1-provider"
        assert s["v2"].name == "conditional-provider"

    @pytest.mark.regression("20079")
    @pytest.mark.parametrize(
        "spec_str,tests_arg,with_dep,without_dep",
        [
            # Check that True is treated correctly and attaches test deps
            # to all nodes in the DAG
            ("a", True, ["a"], []),
            ("a foobar=bar", True, ["a", "b"], []),
            # Check that a list of names activates the dependency only for
            # packages in that list
            ("a foobar=bar", ("a",), ["a"], ["b"]),
            ("a foobar=bar", ("b",), ["b"], ["a"]),
            # Check that False disregard test dependencies
            ("a foobar=bar", False, [], ["a", "b"]),
        ],
    )
    def test_activating_test_dependencies(
        self, default_mock_concretization, spec_str, tests_arg, with_dep, without_dep
    ):
        s = default_mock_concretization(spec_str, tests=tests_arg)

        for pkg_name in with_dep:
            msg = "Cannot find test dependency in package '{0}'"
            node = s[pkg_name]
            assert node.dependencies(deptype="test"), msg.format(pkg_name)

        for pkg_name in without_dep:
            msg = "Test dependency in package '{0}' is unexpected"
            node = s[pkg_name]
            assert not node.dependencies(deptype="test"), msg.format(pkg_name)

    @pytest.mark.regression("20019")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_compiler_match_is_preferred_to_newer_version(self, default_mock_concretization):
        # This spec depends on openblas. Openblas has a conflict
        # that doesn't allow newer versions with gcc@4.4.0. Check
        # that an old version of openblas is selected, rather than
        # a different compiler for just that node.
        spec_str = "simple-inheritance+openblas %gcc@10.1.0 os=redhat6"
        s = default_mock_concretization(spec_str)

        assert "openblas@0.2.15" in s
        assert s["openblas"].satisfies("%gcc@10.1.0")

    @pytest.mark.regression("19981")
    def test_target_ranges_in_conflicts(self, config):
        with pytest.raises(spack.error.SpackError):
            Spec("impossible-concretization").concretized()

    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_target_compatibility(self, config):
        with pytest.raises(spack.error.SpackError):
            Spec("libdwarf target=x86_64 ^libelf target=x86_64_v2").concretized()

    @pytest.mark.regression("20040")
    def test_variant_not_default(self, default_mock_concretization):
        s = default_mock_concretization("ecp-viz-sdk")

        # Check default variant value for the package
        assert "+dep" in s["conditional-constrained-dependencies"]

        # Check that non-default variant values are forced on the dependency
        d = s["dep-with-variants"]
        assert "+foo+bar+baz" in d

    @pytest.mark.regression("20055")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_custom_compiler_version(self, default_mock_concretization):
        s = default_mock_concretization("a %gcc@10foo os=redhat6")
        assert "%gcc@10foo" in s

    def test_all_patches_applied(self, default_mock_concretization):
        uuidpatch = (
            "a60a42b73e03f207433c5579de207c6ed61d58e4d12dd3b5142eb525728d89ea"
            if not sys.platform == "win32"
            else "d0df7988457ec999c148a4a2af25ce831bfaad13954ba18a4446374cb0aef55e"
        )
        localpatch = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        spec = default_mock_concretization("conditionally-patch-dependency+jasper")
        assert (uuidpatch, localpatch) == spec["libelf"].variants["patches"].value

    def test_dont_select_version_that_brings_more_variants_in(self, default_mock_concretization):
        s = default_mock_concretization("dep-with-variants-if-develop-root")
        assert s["dep-with-variants-if-develop"].satisfies("@1.0")

    @pytest.mark.regression("20244,20736")
    @pytest.mark.parametrize(
        "spec_str,is_external,expected",
        [
            # These are all externals, and 0_8 is a version not in package.py
            ("externaltool@1.0", True, "@1.0"),
            ("externaltool@0.9", True, "@0.9"),
            ("externaltool@0_8", True, "@0_8"),
            # This external package is buildable, has a custom version
            # in packages.yaml that is greater than the ones in package.py
            # and specifies a variant
            ("external-buildable-with-variant +baz", True, "@1.1.special +baz"),
            ("external-buildable-with-variant ~baz", False, "@1.0 ~baz"),
            ("external-buildable-with-variant@1.0: ~baz", False, "@1.0 ~baz"),
            # This uses an external version that meets the condition for
            # having an additional dependency, but the dependency shouldn't
            # appear in the answer set
            ("external-buildable-with-variant@0.9 +baz", True, "@0.9"),
            # This package has an external version declared that would be
            # the least preferred if Spack had to build it
            ("old-external", True, "@1.0.0"),
        ],
    )
    def test_external_package_versions(
        self, default_mock_concretization, spec_str, is_external, expected
    ):
        s = default_mock_concretization(spec_str)
        assert s.external == is_external
        assert s.satisfies(expected)

    @pytest.mark.parametrize("dev_first", [True, False])
    @pytest.mark.parametrize(
        "spec", ["dev-build-test-install", "dev-build-test-dependent ^dev-build-test-install"]
    )
    @pytest.mark.parametrize("mock_db", [True, False])
    def test_reuse_does_not_overwrite_dev_specs(
        self, dev_first, spec, mock_db, tmpdir, monkeypatch, mutable_config
    ):
        """Test that reuse does not mix dev specs with non-dev specs.

        Tests for either order (dev specs are not reused for non-dev, and
        non-dev specs are not reused for dev specs)
        Tests for a spec in which the root is developed and a spec in
        which a dep is developed.
        Tests for both reuse from database and reuse from buildcache"""
        # dev and non-dev specs that are otherwise identical
        spec = Spec(spec)
        dev_spec = spec.copy()
        dev_constraint = "dev_path=%s" % tmpdir.strpath
        dev_spec["dev-build-test-install"].constrain(dev_constraint)

        # run the test in both orders
        first_spec = dev_spec if dev_first else spec
        second_spec = spec if dev_first else dev_spec

        # concretize and setup spack to reuse in the appropriate manner
        first_spec.concretize()

        def mock_fn(*args, **kwargs):
            return [first_spec]

        if mock_db:
            monkeypatch.setattr(spack.store.db, "query", mock_fn)
        else:
            monkeypatch.setattr(spack.binary_distribution, "update_cache_and_get_specs", mock_fn)

        # concretize and ensure we did not reuse
        with spack.config.override("concretizer:reuse", True):
            second_spec.concretize()
        assert first_spec.dag_hash() != second_spec.dag_hash()

    @pytest.mark.skipif(
        sys.version_info[:2] == (2, 7), reason="Fixture fails intermittently with Python 2.7"
    )
    @pytest.mark.regression("20292")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    @pytest.mark.parametrize(
        "context",
        [
            {"add_variant": True, "delete_variant": False},
            {"add_variant": False, "delete_variant": True},
            {"add_variant": True, "delete_variant": True},
        ],
    )
    def test_reuse_installed_packages_when_package_def_changes(
        self, context, mutable_database, repo_with_changing_recipe, mutable_config
    ):
        # Install a spec
        root = Spec("root").concretized()
        dependency = root["changing"].copy()
        root.package.do_install(fake=True, explicit=True)

        # Modify package.py
        repo_with_changing_recipe.change(context)

        # Try to concretize with the spec installed previously
        new_root_with_reuse = Spec("root ^/{0}".format(dependency.dag_hash())).concretized()

        new_root_without_reuse = Spec("root").concretized()

        # validate that the graphs are the same with reuse, but not without
        assert ht.build_hash(root) == ht.build_hash(new_root_with_reuse)
        assert ht.build_hash(root) != ht.build_hash(new_root_without_reuse)

        # DAG hash should be the same with reuse since only the dependency changed
        assert root.dag_hash() == new_root_with_reuse.dag_hash()

        # Structure and package hash will be different without reuse
        assert root.dag_hash() != new_root_without_reuse.dag_hash()

    @pytest.mark.regression("20784")
    def test_concretization_of_test_dependencies(self, default_mock_concretization):
        # With clingo we emit dependency_conditions regardless of the type
        # of the dependency. We need to ensure that there's at least one
        # dependency type declared to infer that the dependency holds.
        s = default_mock_concretization("test-dep-with-imposed-conditions")
        assert "c" not in s

    @pytest.mark.only_clingo("Known failure of the original concretizer")
    @pytest.mark.parametrize(
        "spec_str", ["wrong-variant-in-conflicts", "wrong-variant-in-depends-on"]
    )
    def test_error_message_for_inconsistent_variants(self, default_mock_concretization, spec_str):
        with pytest.raises(RuntimeError, match="not found in package"):
            default_mock_concretization(spec_str)

    @pytest.mark.regression("22533")
    @pytest.mark.parametrize(
        "spec_str,variant_name,expected_values",
        [
            # Test the default value 'auto'
            ("mvapich2", "file_systems", ("auto",)),
            # Test setting a single value from the disjoint set
            ("mvapich2 file_systems=lustre", "file_systems", ("lustre",)),
            # Test setting multiple values from the disjoint set
            ("mvapich2 file_systems=lustre,gpfs", "file_systems", ("lustre", "gpfs")),
        ],
    )
    def test_mv_variants_disjoint_sets_from_spec(
        self, default_mock_concretization, spec_str, variant_name, expected_values
    ):
        s = default_mock_concretization(spec_str)
        assert set(expected_values) == set(s.variants[variant_name].value)

    @pytest.mark.regression("22533")
    def test_mv_variants_disjoint_sets_from_packages_yaml(self, mutable_config):
        external_mvapich2 = {
            "mvapich2": {
                "buildable": False,
                "externals": [{"spec": "mvapich2@2.3.1 file_systems=nfs,ufs", "prefix": "/usr"}],
            }
        }
        spack.config.set("packages", external_mvapich2)

        s = Spec("mvapich2").concretized()
        assert set(s.variants["file_systems"].value) == set(["ufs", "nfs"])

    @pytest.mark.regression("22596")
    def test_external_with_non_default_variant_as_dependency(self, default_mock_concretization):
        # This package depends on another that is registered as an external
        # with 'buildable: true' and a variant with a non-default value set
        s = default_mock_concretization("trigger-external-non-default-variant")
        assert "~foo" in s["external-non-default-variant"]
        assert "~bar" in s["external-non-default-variant"]
        assert s["external-non-default-variant"].external

    @pytest.mark.regression("22871")
    @pytest.mark.parametrize(
        "spec_str,expected_os",
        [
            ("mpileaks", "os=debian6"),
            # To trigger the bug in 22871 we need to have the same compiler
            # spec available on both operating systems
            ("mpileaks%gcc@10.2.1 platform=test os=debian6", "os=debian6"),
            ("mpileaks%gcc@10.2.1 platform=test os=redhat6", "os=redhat6"),
        ],
    )
    def test_os_selection_when_multiple_choices_are_possible(
        self, default_mock_concretization, spec_str, expected_os
    ):
        s = default_mock_concretization(spec_str)
        for node in s.traverse():
            assert node.satisfies(expected_os)

    @pytest.mark.regression("22718")
    @pytest.mark.parametrize(
        "spec_str,expected_compiler",
        [("mpileaks", "%gcc@10.2.1"), ("mpileaks ^mpich%clang@12.0.0", "%clang@12.0.0")],
    )
    def test_compiler_is_unique(self, default_mock_concretization, spec_str, expected_compiler):
        s = default_mock_concretization(spec_str)
        for node in s.traverse():
            assert node.satisfies(expected_compiler)

    @pytest.mark.parametrize(
        "spec_str,expected_dict",
        [
            # Check the defaults from the package (libs=shared)
            ("multivalue-variant", {"libs=shared": True, "libs=static": False}),
            # Check that libs=static doesn't extend the default
            ("multivalue-variant libs=static", {"libs=shared": False, "libs=static": True}),
        ],
    )
    def test_multivalued_variants_from_cli(
        self, default_mock_concretization, spec_str, expected_dict
    ):
        s = default_mock_concretization(spec_str)
        for constraint, value in expected_dict.items():
            assert s.satisfies(constraint) == value

    @pytest.mark.regression("22351")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    @pytest.mark.parametrize(
        "spec_str,expected",
        [
            # Version 1.1.0 is deprecated and should not be selected, unless we
            # explicitly asked for that
            ("deprecated-versions", ["deprecated-versions@1.0.0"]),
            ("deprecated-versions@1.1.0", ["deprecated-versions@1.1.0"]),
        ],
    )
    def test_deprecated_versions_not_selected(
        self, default_mock_concretization, spec_str, expected
    ):
        s = default_mock_concretization(spec_str)
        for abstract_spec in expected:
            assert abstract_spec in s

    @pytest.mark.regression("24196")
    def test_version_badness_more_important_than_default_mv_variants(
        self, default_mock_concretization
    ):
        # If a dependency had an old version that for some reason pulls in
        # a transitive dependency with a multi-valued variant, that old
        # version was preferred because of the order of our optimization
        # criteria.
        s = default_mock_concretization("root")
        assert s["gmt"].satisfies("@2.0")

    @pytest.mark.regression("24205")
    def test_provider_must_meet_requirements(self, default_mock_concretization):
        # A package can be a provider of a virtual only if the underlying
        # requirements are met.
        with pytest.raises((RuntimeError, spack.error.UnsatisfiableSpecError)):
            default_mock_concretization("unsat-virtual-dependency")

    @pytest.mark.regression("23951")
    def test_newer_dependency_adds_a_transitive_virtual(self, default_mock_concretization):
        # Ensure that a package doesn't concretize any of its transitive
        # dependencies to an old version because newer versions pull in
        # a new virtual dependency. The possible concretizations here are:
        #
        # root@1.0 <- middle@1.0 <- leaf@2.0 <- blas
        # root@1.0 <- middle@1.0 <- leaf@1.0
        #
        # and "blas" is pulled in only by newer versions of "leaf"
        s = default_mock_concretization("root-adds-virtual")
        assert s["leaf-adds-virtual"].satisfies("@2.0")
        assert "blas" in s

    @pytest.mark.regression("26718")
    def test_versions_in_virtual_dependencies(self, default_mock_concretization):
        # Ensure that a package that needs a given version of a virtual
        # package doesn't end up using a later implementation
        s = default_mock_concretization("hpcviewer@2019.02")
        assert s["java"].satisfies("virtual-with-versions@1.8.0")

    @pytest.mark.regression("26866")
    def test_non_default_provider_of_multiple_virtuals(self, default_mock_concretization):
        s = default_mock_concretization("many-virtual-consumer ^low-priority-provider")
        assert s["mpi"].name == "low-priority-provider"
        assert s["lapack"].name == "low-priority-provider"

        for virtual_pkg in ("mpi", "lapack"):
            for pkg in spack.repo.path.providers_for(virtual_pkg):
                if pkg.name == "low-priority-provider":
                    continue
                assert pkg not in s

    @pytest.mark.regression("27237")
    @pytest.mark.only_clingo("Original concretizer cannot reuse specs")
    @pytest.mark.parametrize(
        "spec_str,expect_installed",
        [("mpich", True), ("mpich+debug", False), ("mpich~debug", True)],
    )
    def test_concrete_specs_are_not_modified_on_reuse(
        self, mutable_database, spec_str, expect_installed, config
    ):
        # Test the internal consistency of solve + DAG reconstruction
        # when reused specs are added to the mix. This prevents things
        # like additional constraints being added to concrete specs in
        # the answer set produced by clingo.
        with spack.config.override("concretizer:reuse", True):
            s = spack.spec.Spec(spec_str).concretized()
        assert s.installed is expect_installed
        assert s.satisfies(spec_str, strict=True)

    @pytest.mark.regression("26721,19736")
    @pytest.mark.only_clingo("Original concretizer cannot use sticky variants")
    def test_sticky_variant_in_package(self, default_mock_concretization):
        # Here we test that a sticky variant cannot be changed from its default value
        # by the ASP solver if not set explicitly. The package used in the test needs
        # to have +allow-gcc set to be concretized with %gcc and clingo is not allowed
        # to change the default ~allow-gcc
        with pytest.raises(spack.error.SpackError):
            spack.spec.Spec("sticky-variant %gcc").concretized()

        s = default_mock_concretization("sticky-variant+allow-gcc %gcc")
        assert s.satisfies("%gcc") and s.satisfies("+allow-gcc")

        s = default_mock_concretization("sticky-variant %clang")
        assert s.satisfies("%clang") and s.satisfies("~allow-gcc")

    @pytest.mark.only_clingo(
        "Original concretizer doesn't resolve concrete versions to known ones"
    )
    def test_do_not_invent_new_concrete_versions_unless_necessary(
        self, default_mock_concretization
    ):
        # ensure we select a known satisfying version rather than creating
        # a new '2.7' version.
        assert ver("2.7.11") == default_mock_concretization("python@2.7").version

        # Here there is no known satisfying version - use the one on the spec.
        assert ver("2.7.21") == default_mock_concretization("python@2.7.21").version

    @pytest.mark.parametrize(
        "spec_str",
        [
            "conditional-values-in-variant@1.62.0 cxxstd=17",
            "conditional-values-in-variant@1.62.0 cxxstd=2a",
            "conditional-values-in-variant@1.72.0 cxxstd=2a",
            # Ensure disjoint set of values work too
            "conditional-values-in-variant@1.72.0 staging=flexpath",
        ],
    )
    @pytest.mark.only_clingo("Original concretizer doesn't resolve conditional values in variants")
    def test_conditional_values_in_variants(self, default_mock_concretization, spec_str):
        with pytest.raises((RuntimeError, spack.error.UnsatisfiableSpecError)):
            default_mock_concretization(spec_str)

    @pytest.mark.only_clingo("Original concretizer doesn't resolve conditional values in variants")
    def test_conditional_values_in_conditional_variant(self, default_mock_concretization):
        """Test that conditional variants play well with conditional possible values"""
        s = default_mock_concretization("conditional-values-in-variant@1.50.0")
        assert "cxxstd" not in s.variants

        s = default_mock_concretization("conditional-values-in-variant@1.60.0")
        assert "cxxstd" in s.variants

    @pytest.mark.only_clingo("Original concretizer cannot account for target granularity")
    def test_target_granularity(self, mutable_config):
        # The test architecture uses core2 as the default target. Check that when
        # we configure Spack for "generic" granularity we concretize for x86_64
        default_target = spack.platforms.test.Test.default
        generic_target = archspec.cpu.TARGETS[default_target].generic.name
        s = Spec("python")
        assert s.concretized().satisfies("target=%s" % default_target)
        with spack.config.override("concretizer:targets", {"granularity": "generic"}):
            assert s.concretized().satisfies("target=%s" % generic_target)

    @pytest.mark.only_clingo("Original concretizer cannot account for host compatibility")
    def test_host_compatible_concretization(self, mutable_config):
        # Check that after setting "host_compatible" to false we cannot concretize.
        # Here we use "k10" to set a target non-compatible with the current host
        # to avoid a lot of boilerplate when mocking the test platform. The issue
        # is that the defaults for the test platform are very old, so there's no
        # compiler supporting e.g. icelake etc.
        s = Spec("python target=k10")
        assert s.concretized()
        with spack.config.override("concretizer:targets", {"host_compatible": True}):
            with pytest.raises(spack.error.SpackError):
                s.concretized()

    @pytest.mark.only_clingo("Original concretizer cannot account for host compatibility")
    def test_add_microarchitectures_on_explicit_request(self, mutable_config):
        # Check that if we consider only "generic" targets, we can still solve for
        # specific microarchitectures on explicit requests
        with spack.config.override("concretizer:targets", {"granularity": "generic"}):
            s = Spec("python target=k10").concretized()
        assert s.satisfies("target=k10")

    @pytest.mark.skipif(
        sys.version_info[:2] == (2, 7), reason="Fixture fails intermittently with Python 2.7"
    )
    @pytest.mark.regression("29201")
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_delete_version_and_reuse(
        self, mutable_database, mutable_config, repo_with_changing_recipe
    ):
        """Test that we can reuse installed specs with versions not
        declared in package.py
        """
        root = Spec("root").concretized()
        root.package.do_install(fake=True, explicit=True)
        repo_with_changing_recipe.change({"delete_version": True})

        with spack.config.override("concretizer:reuse", True):
            new_root = Spec("root").concretized()

        assert root.dag_hash() == new_root.dag_hash()

    @pytest.mark.regression("29201")
    @pytest.mark.skipif(
        sys.version_info[:2] == (2, 7), reason="Fixture fails intermittently with Python 2.7"
    )
    @pytest.mark.only_clingo("Known failure of the original concretizer")
    def test_installed_version_is_selected_only_for_reuse(
        self, mutable_database, repo_with_changing_recipe, mutable_config
    ):
        """Test that a version coming from an installed spec is a possible
        version only for reuse
        """
        # Install a dependency that cannot be reused with "root"
        # because of a conflict a variant, then delete its version
        dependency = Spec("changing@1.0~foo").concretized()
        dependency.package.do_install(fake=True, explicit=True)
        repo_with_changing_recipe.change({"delete_version": True})

        with spack.config.override("concretizer:reuse", True):
            new_root = Spec("root").concretized()

        assert not new_root["changing"].satisfies("@1.0")

    @pytest.mark.regression("28259")
    def test_reuse_with_unknown_namespace_dont_raise(self, mutable_config, mock_custom_repository):
        with spack.repo.use_repositories(mock_custom_repository, override=False):
            s = Spec("c").concretized()
            assert s.namespace != "builtin.mock"
            s.package.do_install(fake=True, explicit=True)

        with spack.config.override("concretizer:reuse", True):
            s = Spec("c").concretized()
        assert s.namespace == "builtin.mock"

    @pytest.mark.regression("28259")
    def test_reuse_with_unknown_package_dont_raise(self, mutable_config, tmpdir, monkeypatch):
        builder = spack.repo.MockRepositoryBuilder(tmpdir, namespace="myrepo")
        builder.add_package("c")
        with spack.repo.use_repositories(builder.root, override=False):
            s = Spec("c").concretized()
            assert s.namespace == "myrepo"
            s.package.do_install(fake=True, explicit=True)

        del sys.modules["spack.pkg.myrepo.c"]
        del sys.modules["spack.pkg.myrepo"]
        builder.remove("c")
        with spack.repo.use_repositories(builder.root, override=False) as repos:
            # TODO (INJECT CONFIGURATION): unclear why the cache needs to be invalidated explicitly
            repos.repos[0]._pkg_checker.invalidate()
            with spack.config.override("concretizer:reuse", True):
                s = Spec("c").concretized()
            assert s.namespace == "builtin.mock"

    @pytest.mark.parametrize(
        "specs,expected",
        [
            (["libelf", "libelf@0.8.10"], 1),
            (["libdwarf%gcc", "libelf%clang"], 2),
            (["libdwarf%gcc", "libdwarf%clang"], 4),
            (["libdwarf^libelf@0.8.12", "libdwarf^libelf@0.8.13"], 4),
            (["hdf5", "zmpi"], 3),
            (["hdf5", "mpich"], 2),
            (["hdf5^zmpi", "mpich"], 4),
            (["mpi", "zmpi"], 2),
            (["mpi", "mpich"], 1),
        ],
    )
    @pytest.mark.only_clingo("Original concretizer cannot concretize in rounds")
    def test_best_effort_coconcretize(self, config, specs, expected):
        import spack.solver.asp

        specs = [spack.spec.Spec(s) for s in specs]
        solver = spack.solver.asp.Solver()
        solver.reuse = False
        concrete_specs = set()
        for result in solver.solve_in_rounds(specs):
            for s in result.specs:
                concrete_specs.update(s.traverse())

        assert len(concrete_specs) == expected

    @pytest.mark.parametrize(
        "specs,expected_spec,occurances",
        [
            # The algorithm is greedy, and it might decide to solve the "best"
            # spec early in which case reuse is suboptimal. In this case the most
            # recent version of libdwarf is selected and concretized to libelf@0.8.13
            (
                [
                    "libdwarf@20111030^libelf@0.8.10",
                    "libdwarf@20130207^libelf@0.8.12",
                    "libdwarf@20130729",
                ],
                "libelf@0.8.12",
                1,
            ),
            # Check we reuse the best libelf in the environment
            (
                [
                    "libdwarf@20130729^libelf@0.8.10",
                    "libdwarf@20130207^libelf@0.8.12",
                    "libdwarf@20111030",
                ],
                "libelf@0.8.12",
                2,
            ),
            (["libdwarf@20130729", "libdwarf@20130207", "libdwarf@20111030"], "libelf@0.8.13", 3),
            # We need to solve in 2 rounds and we expect mpich to be preferred to zmpi
            (["hdf5+mpi", "zmpi", "mpich"], "mpich", 2),
        ],
    )
    @pytest.mark.only_clingo("Original concretizer cannot concretize in rounds")
    def test_best_effort_coconcretize_preferences(self, config, specs, expected_spec, occurances):
        """Test package preferences during coconcretization."""
        import spack.solver.asp

        specs = [spack.spec.Spec(s) for s in specs]
        solver = spack.solver.asp.Solver()
        solver.reuse = False
        concrete_specs = {}
        for result in solver.solve_in_rounds(specs):
            concrete_specs.update(result.specs_by_input)

        counter = 0
        for spec in concrete_specs.values():
            if expected_spec in spec:
                counter += 1
        assert counter == occurances, concrete_specs

    @pytest.mark.only_clingo("Original concretizer cannot reuse")
    def test_coconcretize_reuse_and_virtuals(self, default_mock_concretization):
        import spack.solver.asp

        reusable_specs = []
        for s in ["mpileaks ^mpich", "zmpi"]:
            reusable_specs.extend(default_mock_concretization(s).traverse(root=True))

        root_specs = [spack.spec.Spec("mpileaks"), spack.spec.Spec("zmpi")]

        with spack.config.override("concretizer:reuse", True):
            solver = spack.solver.asp.Solver()
            setup = spack.solver.asp.SpackSolverSetup()
            result, _, _ = solver.driver.solve(setup, root_specs, reuse=reusable_specs)

        for spec in result.specs:
            assert "zmpi" in spec

    @pytest.mark.regression("30864")
    @pytest.mark.only_clingo("Original concretizer cannot reuse")
    def test_misleading_error_message_on_version(
        self, default_mock_concretization, mutable_database
    ):
        # For this bug to be triggered we need a reusable dependency
        # that is not optimal in terms of optimization scores.
        # We pick an old version of "b"
        import spack.solver.asp

        reusable_specs = [default_mock_concretization("non-existing-conditional-dep@1.0")]
        root_spec = spack.spec.Spec("non-existing-conditional-dep@2.0")

        with spack.config.override("concretizer:reuse", True):
            solver = spack.solver.asp.Solver()
            setup = spack.solver.asp.SpackSolverSetup()
            with pytest.raises(
                spack.solver.asp.UnsatisfiableSpecError,
                match="'dep-with-variants' satisfies '@999'",
            ):
                solver.driver.solve(setup, [root_spec], reuse=reusable_specs)

    @pytest.mark.regression("31148")
    @pytest.mark.only_clingo("Original concretizer cannot reuse")
    def test_version_weight_and_provenance(self, default_mock_concretization):
        """Test package preferences during coconcretization."""
        import spack.solver.asp

        reusable_specs = [default_mock_concretization(spec_str) for spec_str in ("b@0.9", "b@1.0")]
        root_spec = spack.spec.Spec("a foobar=bar")

        with spack.config.override("concretizer:reuse", True):
            solver = spack.solver.asp.Solver()
            setup = spack.solver.asp.SpackSolverSetup()
            result, _, _ = solver.driver.solve(setup, [root_spec], reuse=reusable_specs)
            # The result here should have a single spec to build ('a')
            # and it should be using b@1.0 with a version badness of 2
            # The provenance is:
            # version_declared("b","1.0",0,"package_py").
            # version_declared("b","0.9",1,"package_py").
            # version_declared("b","1.0",2,"installed").
            # version_declared("b","0.9",3,"installed").
            #
            # Depending on the target, it may also use gnuconfig
            result_spec = result.specs[0]
            num_specs = len(list(result_spec.traverse()))

            criteria = [
                (num_specs - 1, None, "number of packages to build (vs. reuse)"),
                (2, 0, "version badness"),
            ]

            for criterion in criteria:
                assert criterion in result.criteria
            assert result_spec.satisfies("^b@1.0")

    @pytest.mark.regression("31169")
    @pytest.mark.only_clingo("Original concretizer cannot reuse")
    def test_not_reusing_incompatible_os_or_compiler(self, default_mock_concretization):
        import spack.solver.asp

        root_spec_str = "b"
        root_spec = spack.spec.Spec(root_spec_str)
        s = default_mock_concretization(root_spec_str)
        wrong_compiler, wrong_os = s.copy(), s.copy()
        wrong_compiler.compiler = spack.spec.CompilerSpec("gcc@12.1.0")
        wrong_os.architecture = spack.spec.ArchSpec("test-ubuntu2204-x86_64")
        reusable_specs = [wrong_compiler, wrong_os]
        with spack.config.override("concretizer:reuse", True):
            solver = spack.solver.asp.Solver()
            setup = spack.solver.asp.SpackSolverSetup()
            result, _, _ = solver.driver.solve(setup, [root_spec], reuse=reusable_specs)
        concrete_spec = result.specs[0]
        assert concrete_spec.satisfies("%{}".format(s.compiler))
        assert concrete_spec.satisfies("os={}".format(s.architecture.os))

    def test_git_hash_assigned_version_is_preferred(self, default_mock_concretization):
        hash = "a" * 40
        c = default_mock_concretization("develop-branch-version@%s=develop" % hash)
        assert hash in str(c)

    @pytest.mark.not_on_windows
    @pytest.mark.only_clingo("Original concretizer cannot account for git hashes")
    @pytest.mark.parametrize("git_ref", ("a" * 40, "0.2.15", "main"))
    def test_git_ref_version_is_equivalent_to_specified_version(
        self, default_mock_concretization, git_ref
    ):
        spec_str = "develop-branch-version@git.%s=develop" % git_ref
        s = Spec(spec_str)
        c = default_mock_concretization(spec_str)
        assert git_ref in str(c)
        assert s.satisfies("@develop")
        assert s.satisfies("@0.1:")

    @pytest.mark.not_on_windows
    @pytest.mark.only_clingo("Original concretizer cannot account for git hashes")
    @pytest.mark.parametrize("git_ref", ("a" * 40, "0.2.15", "fbranch"))
    def test_git_ref_version_errors_if_unknown_version(self, config, git_ref):
        # main is not defined in the package.py for this file
        s = Spec("develop-branch-version@git.%s=main" % git_ref)
        with pytest.raises(
            UnsatisfiableSpecError,
            match="The reference version 'main' for package 'develop-branch-version'",
        ):
            s.concretized()

    @pytest.mark.regression("31484")
    @pytest.mark.skipif(
        sys.version_info[:2] == (2, 7), reason="Fixture fails intermittently with Python 2.7"
    )
    @pytest.mark.only_clingo("Use case not supported by the original concretizer")
    def test_installed_externals_are_reused(
        self, mutable_config, mutable_database, repo_with_changing_recipe
    ):
        """Test that external specs that are in the DB can be reused."""
        external_conf = {
            "changing": {
                "buildable": False,
                "externals": [{"spec": "changing@1.0", "prefix": "/usr"}],
            }
        }
        spack.config.set("packages", external_conf)

        # Install the external spec
        external1 = Spec("changing@1.0").concretized()
        external1.package.do_install(fake=True, explicit=True)
        assert external1.external

        # Modify the package.py file
        repo_with_changing_recipe.change({"delete_variant": True})

        # Try to concretize the external without reuse and confirm the hash changed
        with spack.config.override("concretizer:reuse", False):
            external2 = Spec("changing@1.0").concretized()
        assert external2.dag_hash() != external1.dag_hash()

        # ... while with reuse we have the same hash
        with spack.config.override("concretizer:reuse", True):
            external3 = Spec("changing@1.0").concretized()
        assert external3.dag_hash() == external1.dag_hash()

    @pytest.mark.regression("31484")
    @pytest.mark.only_clingo("Use case not supported by the original concretizer")
    def test_user_can_select_externals_with_require(self, mutable_config, mutable_database):
        """Test that users have means to select an external even in presence of reusable specs."""
        external_conf = {
            "mpi": {"buildable": False},
            "multi-provider-mpi": {
                "externals": [{"spec": "multi-provider-mpi@2.0.0", "prefix": "/usr"}]
            },
        }
        spack.config.set("packages", external_conf)

        # mpich and others are installed, so check that
        # fresh use the external, reuse does not
        with spack.config.override("concretizer:reuse", False):
            mpi_spec = Spec("mpi").concretized()
            assert mpi_spec.name == "multi-provider-mpi"

        with spack.config.override("concretizer:reuse", True):
            mpi_spec = Spec("mpi").concretized()
            assert mpi_spec.name != "multi-provider-mpi"

        external_conf["mpi"]["require"] = "multi-provider-mpi"
        spack.config.set("packages", external_conf)

        with spack.config.override("concretizer:reuse", True):
            mpi_spec = Spec("mpi").concretized()
            assert mpi_spec.name == "multi-provider-mpi"

    @pytest.mark.regression("31484")
    @pytest.mark.only_clingo("Use case not supported by the original concretizer")
    def test_installed_specs_disregard_conflicts(
        self, mutable_config, mutable_database, monkeypatch
    ):
        """Test that installed specs do not trigger conflicts. This covers for the rare case
        where a conflict is added on a package after a spec matching the conflict was installed.
        """
        # Add a conflict to "mpich" that match an already installed "mpich~debug"
        pkg_cls = spack.repo.path.get_pkg_class("mpich")
        monkeypatch.setitem(pkg_cls.conflicts, "~debug", [(spack.spec.Spec(), None)])

        # If we concretize with --fresh the conflict is taken into account
        with spack.config.override("concretizer:reuse", False):
            s = Spec("mpich").concretized()
            assert s.satisfies("+debug")

        # If we concretize with --reuse it is not, since "mpich~debug" was already installed
        with spack.config.override("concretizer:reuse", True):
            s = Spec("mpich").concretized()
            assert s.satisfies("~debug")

    @pytest.mark.regression("32471")
    @pytest.mark.only_clingo("Use case not supported by the original concretizer")
    def test_require_targets_are_allowed(self, mutable_config, mutable_database):
        """Test that users can set target constraints under the require attribute."""
        # Configuration to be added to packages.yaml
        external_conf = {"all": {"require": "target=%s" % spack.platforms.test.Test.front_end}}
        spack.config.set("packages", external_conf)

        with spack.config.override("concretizer:reuse", False):
            spec = Spec("mpich").concretized()

        for s in spec.traverse():
            assert s.satisfies("target=%s" % spack.platforms.test.Test.front_end)

    def test_external_python_extensions_have_dependency(self, mutable_config):
        """Test that python extensions have access to a python dependency"""
        external_conf = {
            "py-extension1": {
                "buildable": False,
                "externals": [{"spec": "py-extension1@2.0", "prefix": "/fake"}],
            }
        }
        spack.config.set("packages", external_conf)

        spec = Spec("py-extension2").concretized()

        assert "python" in spec["py-extension1"]
        assert spec["python"] == spec["py-extension1"]["python"]
