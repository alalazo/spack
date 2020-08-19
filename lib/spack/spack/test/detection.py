# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import contextlib
import glob
import os
import os.path

import pytest

import spack.config
import spack.detection
import spack.repo
import spack.spec
import spack.util.path
import spack.util.spack_yaml


@pytest.fixture
def executables_found(monkeypatch):
    def _factory(result):
        def _mock_search(path_hints=None):
            return result

        monkeypatch.setattr(spack.detection.path, 'executables_in_path', _mock_search)
    return _factory


def test_find_external_single_package(mock_executable, executables_found):
    pkgs_to_check = [spack.repo.get('cmake')]
    executables_found({
        mock_executable("cmake", output='echo "cmake version 1.foo"'): 'cmake'
    })

    pkg_to_entries = spack.detection.by_executable(pkgs_to_check)

    pkg, entries = next(iter(pkg_to_entries.items()))
    single_entry = next(iter(entries))

    assert single_entry.spec == spack.spec.Spec('cmake@1.foo')


def test_find_external_two_instances_same_package(mock_executable, executables_found):
    pkgs_to_check = [spack.repo.get('cmake')]

    # Each of these cmake instances is created in a different prefix
    cmake_path1 = mock_executable(
        "cmake", output='echo "cmake version 1.foo"', subdir=('base1', 'bin')
    )
    cmake_path2 = mock_executable(
        "cmake", output='echo "cmake version 3.17.2"', subdir=('base2', 'bin')
    )
    executables_found({
        cmake_path1: 'cmake',
        cmake_path2: 'cmake'
    })

    pkg_to_entries = spack.detection.by_executable(pkgs_to_check)

    pkg, entries = next(iter(pkg_to_entries.items()))
    spec_to_path = dict((e.spec, e.prefix) for e in entries)
    assert spec_to_path[spack.spec.Spec('cmake@1.foo')] == (
        spack.detection.executable_prefix(os.path.dirname(cmake_path1)))
    assert spec_to_path[spack.spec.Spec('cmake@3.17.2')] == (
        spack.detection.executable_prefix(os.path.dirname(cmake_path2)))


def test_find_external_update_config(mutable_config):
    entries = [
        spack.detection.DetectedPackage(
            spack.spec.Spec.from_detection('cmake@1.foo'), '/x/y1/'
        ),
        spack.detection.DetectedPackage(
            spack.spec.Spec.from_detection('cmake@3.17.2'), '/x/y2/'
        ),
    ]
    pkg_to_entries = {'cmake': entries}

    scope = spack.config.default_modify_scope('packages')
    spack.detection.update_configuration(pkg_to_entries, scope=scope, buildable=True)

    pkgs_cfg = spack.config.get('packages')
    cmake_cfg = pkgs_cfg['cmake']
    cmake_externals = cmake_cfg['externals']

    assert {'spec': 'cmake@1.foo', 'prefix': '/x/y1/'} in cmake_externals
    assert {'spec': 'cmake@3.17.2', 'prefix': '/x/y2/'} in cmake_externals


def test_get_executables(working_env, mock_executable):
    cmake_path1 = mock_executable("cmake", output="echo cmake version 1.foo")
    os.environ['PATH'] = ':'.join([os.path.dirname(cmake_path1)])
    path_to_exe = spack.detection.executables_in_path()
    assert path_to_exe[cmake_path1] == 'cmake'


def test_find_external_merge(mutable_config, mutable_mock_repo):
    """Check that 'spack find external' doesn't overwrite an existing spec
    entry in packages.yaml.
    """
    pkgs_cfg_init = {
        'find-externals1': {
            'externals': [{
                'spec': 'find-externals1@1.1',
                'prefix': '/preexisting-prefix/'
            }],
            'buildable': False
        }
    }

    mutable_config.update_config('packages', pkgs_cfg_init)
    entries = [
        spack.detection.DetectedPackage(
            spack.spec.Spec.from_detection('find-externals1@1.1'), '/x/y1/'
        ),
        spack.detection.DetectedPackage(
            spack.spec.Spec.from_detection('find-externals1@1.2'), '/x/y2/'
        )
    ]
    pkg_to_entries = {'find-externals1': entries}
    scope = spack.config.default_modify_scope('packages')
    spack.detection.update_configuration(pkg_to_entries, scope=scope, buildable=True)

    pkgs_cfg = spack.config.get('packages')
    pkg_cfg = pkgs_cfg['find-externals1']
    pkg_externals = pkg_cfg['externals']

    assert {'spec': 'find-externals1@1.1',
            'prefix': '/preexisting-prefix/'} in pkg_externals
    assert {'spec': 'find-externals1@1.2',
            'prefix': '/x/y2/'} in pkg_externals


def candidate_packages():
    """Return the list of packages with a corresponding
    detection_test.yaml file.
    """
    # Directories where we have repositories
    repo_dirs = [spack.util.path.canonicalize_path(x)
                 for x in spack.config.get('repos')]

    # Compute which files need to be tested
    to_be_tested = []
    for repo_dir in repo_dirs:
        pattern = os.path.join(
            repo_dir, 'packages', '**', 'detection_test.yaml'
        )
        pkgs_with_tests = [os.path.basename(os.path.dirname(x))
                           for x in glob.glob(pattern)]
        to_be_tested.extend(pkgs_with_tests)

    return to_be_tested


@pytest.mark.detection
@pytest.mark.parametrize('package_name', candidate_packages())
def test_package_detection(mock_executable, package_name):
    def detection_tests_for(pkg):
        pkg_dir = os.path.dirname(
            spack.repo.path.filename_for_package_name(pkg)
        )
        detection_data = os.path.join(pkg_dir, 'detection_test.yaml')
        with open(detection_data) as f:
            return spack.util.spack_yaml.load(f)

    @contextlib.contextmanager
    def setup_test_layout(layout):
        hints, to_be_removed = set(), []
        for binary in layout:
            exe = mock_executable(
                binary['name'], binary['output'], subdir=binary['subdir']
            )
            to_be_removed.append(exe)
            hints.add(os.path.dirname(str(exe)))

        yield list(hints)

        for exe in to_be_removed:
            os.unlink(exe)

    # Retrieve detection test data for this package and cycle over each
    # of the scenarios that are encoded
    detection_tests = detection_tests_for(package_name)
    if 'paths' not in detection_tests:
        msg = 'Package "{0}" has no detection tests based on PATH'
        pytest.skip(msg.format(package_name))

    for test in detection_tests['paths']:
        # Setup the mock layout for detection. The context manager will
        # remove mock files when it's finished.
        with setup_test_layout(test['layout']) as abs_path_to_exe:
            entries = spack.detection.by_executable(
                [spack.repo.get(package_name)], abs_path_to_exe
            )
            specs = set(x.spec for x in entries[package_name])
            results = test['results']
            # If no result was expected, check that nothing was detected
            if not results:
                msg = 'No spec was expected [detected={0}]'
                assert not specs, msg.format(sorted(specs))
                continue

            # If we expected results check that all of the expected
            # specs were detected.
            for result in results:
                spec, msg = result['spec'], 'Not able to detect "{0}"'
                assert spack.spec.Spec(spec) in specs, msg.format(str(spec))
