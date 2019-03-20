# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
from spack.spec_list import SpecList
from spack.spec import Spec


class TestSpecList(object):
    default_input = ['mpileaks', '$mpis',
             {'matrix': [['zlib'],['$gccs', '%intel@18']]}, 'libelf']

    default_reference = {'gccs': SpecList('gccs', ['%gcc@4.9.3', '%gcc@7.3.0']),
                 'mpis': SpecList('mpis', ['mvapich2@2.2', 'openmpi@3.1.0']),
                 }

    default_expansion = ['mpileaks', 'mvapich2@2.2', 'openmpi@3.1.0',
                          {'matrix': [
                              ['zlib'],
                              ['%gcc@4.9.3', '%gcc@7.3.0', '%intel@18'],
                              ]},
                          'libelf']

    default_constraints = [[Spec('mpileaks')],
                            [Spec('mvapich2@2.2')],
                            [Spec('openmpi@3.1.0')],
                            [Spec('zlib'), Spec('%gcc@4.9.3')],
                            [Spec('zlib'), Spec('%gcc@7.3.0')],
                            [Spec('zlib'), Spec('%intel@18')],
                            [Spec('libelf')]]

    default_specs = [Spec('mpileaks'), Spec('mvapich2@2.2'),
                            Spec('openmpi@3.1.0'), Spec('zlib%gcc@4.9.3'),
                            Spec('zlib%gcc@7.3.0'), Spec('zlib%intel@18'),
                            Spec('libelf')]

    def test_spec_list_expansions(self):
        speclist = SpecList('specs', self.default_input, self.default_reference)

        assert speclist.specs_as_yaml_list == self.default_expansion
        assert speclist.specs_as_constraints == self.default_constraints
        assert speclist.specs == self.default_specs

    def test_spec_list_constraint_ordering(self):
        specs = [{'matrix': [
            ['^mvapich2'],
            ['%gcc@4.9.3'],
            ['zlib', 'libelf'],
            ['~shared'],
            ['cflags=-O3', 'cflags="-g -O0"'],
            ['^foo']
        ]}]

        speclist = SpecList('specs', specs)

        expected_specs = [
            Spec('zlib cflags=-O3 ~shared %gcc@4.9.3 ^foo ^mvapich2'),
            Spec('zlib cflags="-g -O0" ~shared %gcc@4.9.3 ^foo ^mvapich2'),
            Spec('libelf cflags=-O3 ~shared %gcc@4.9.3 ^foo ^mvapich2'),
            Spec('libelf cflags="-g -O0" ~shared %gcc@4.9.3 ^foo ^mvapich2'),
        ]
        assert speclist.specs == expected_specs

    def test_spec_list_add(self):
        speclist = SpecList('specs', self.default_input, self.default_reference)

        assert speclist.specs_as_yaml_list == self.default_expansion
        assert speclist.specs_as_constraints == self.default_constraints
        assert speclist.specs == self.default_specs

        speclist.add('libdwarf')

        assert speclist.specs_as_yaml_list == self.default_expansion + ['libdwarf']
        assert speclist.specs_as_constraints == self.default_constraints + [[Spec('libdwarf')]]
        assert speclist.specs == self.default_specs + [Spec('libdwarf')]

    def test_spec_list_remove(self):
        speclist = SpecList('specs', self.default_input, self.default_reference)

        print speclist._list
        print self.default_input
        assert speclist.specs_as_yaml_list == self.default_expansion
        assert speclist.specs_as_constraints == self.default_constraints
        assert speclist.specs == self.default_specs

        speclist.remove('libelf')

        assert speclist.specs_as_yaml_list + ['libelf'] == self.default_expansion
        assert speclist.specs_as_constraints + [[Spec('libelf')]] == self.default_constraints
        assert speclist.specs + [Spec('libelf')] == self.default_specs

    def test_spec_list_update_reference(self):
        speclist = SpecList('specs', self.default_input, self.default_reference)

        assert speclist.specs_as_yaml_list == self.default_expansion
        assert speclist.specs_as_constraints == self.default_constraints
        assert speclist.specs == self.default_specs

        orig_reference = self.default_reference
        self.default_reference['mpis'].add('mpich@3.3')
        speclist.update_reference(self.default_reference)

        expansion = self.default_expansion[:]
        expansion.insert(3, 'mpich@3.3')
        constraints = self.default_constraints[:]
        constraints.insert(3,[Spec('mpich@3.3')])
        specs = self.default_specs[:]
        specs.insert(3, Spec('mpich@3.3'))
        assert speclist.specs_as_yaml_list == expansion
        assert speclist.specs_as_constraints == constraints
        assert speclist.specs == specs

        self.default_reference = orig_reference
