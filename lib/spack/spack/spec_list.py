# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import itertools
import string

from spack.spec import Spec, InvalidDependencyError
from spack.variant import UnknownVariantError
from spack.error import SpackError


def spec_ordering_key(s):
    if s.startswith('^'):
        return 5
    elif s.startswith('/'):
        return 4
    elif s.startswith('%'):
        return 3
    elif any(s.startswith(c) for c in '~-+@') or '=' in s:
        return 2
    else:
        return 1

class SpecList(object):

    def __init__(self, name='specs', yaml_list=[], reference={}):
        self.name = name
        self._reference = reference  # TODO: Do we need to defensively copy here?

        self._list = yaml_list[:]

        # Expansions can be expensive to compute and difficult to keep updated
        # We cache results and invalidate when self._list changes
        self._expanded_list = None
        self._constraints = None
        self._specs = None
        self._concrete_specs = None


    @property
    def specs_as_yaml_list(self):
        if self._expanded_list is None:
            self._expanded_list = self._expand_references(self._list)
        return self._expanded_list

    @property
    def specs_as_constraints(self):
        if self._constraints is None:
            constraints = []
            for item in self.specs_as_yaml_list:
                if isinstance(item, dict):  # matrix of specs
                    excludes = item.get('exclude', [])
                    for combo in itertools.product(*(item['matrix'])):
                        # Test against the excludes using a single spec
                        ordered_combo = sorted(combo, key=spec_ordering_key)
                        test_spec = Spec(' '.join(ordered_combo))
                        if any(test_spec.satisfies(x) for x in excludes):
                            continue

                        # Add as list of constraints
                        constraints.append([Spec(x) for x in ordered_combo])
                else:  # individual spec
                    constraints.append([Spec(item)])
            self._constraints = constraints

        return self._constraints

    @property
    def specs(self):
        if self._specs is None:
            specs = []
            # This could be slightly faster done directly from yaml_list,
            # but this way is easier to maintain.
            for constraint_list in self.specs_as_constraints:
                spec = constraint_list[0]
                for const in constraint_list[1:]:
                    spec.constrain(const)
                specs.append(spec)
            self._specs = specs

        return self._specs

    @property
    def concrete_specs(self):
        if self._concrete_specs is None:
            concrete_specs = []
            for constraint_list in self.specs_as_constraints:
                # Get the named spec even if out of order
                root_spec = [s for s in constraint_list if s.name]
                if len(root_spec) != 1:
                    m = 'Spec %s is not a valid concretization target' % s.name
                    m += 'all specs must have a single name constraint for '
                    m += 'concretization.'
                    raise InvalidSpecConstraintError(m)
                constraint_list.remove(root_spec[0])

                invalid_constraints = []
                while True:
                    # Attach all anonymous constraints to one named spec
                    s = root_spec[0]
                    for c in constraint_list:
                        if c not in invalid_constraints:
                            s.constrain(c)
                    try:
                        concrete_specs.append(s.concretized())
                        break
                    except InvalidDependencyError as e:
                        dep_index = e.message.index('depend on ') + len('depend on ')
                        invalid_msg = e.message[dep_index:]
                        invalid_deps_string = ['^' + d.strip(',')
                                               for d in invalid_depstring.split()
                                               if d != 'or']
                        invalid_deps = [c for c in constraints
                                        if any(c.satisfies(invd)
                                               for invd in invalid_deps_string)]
                        if len(invalid_deps) != len(invalid_deps_strings):
                            raise e
                        invalid_constraints.extend(invalid_deps)
                    except UnknownVariantError as e:
                        invalid_variants = re.findall(r"'(\w+)'", e.message)
                        invalid_deps = [c for c in constraints
                                        if any(name in c.variants
                                               for name in invalid_variants)]
                        if len(invalid_dpes) != len(invalid_variants):
                            raise e
                        invalid_constraints.extend(invalid_deps)
            self._concrete_specs = concrete_specs

        return self._concrete_specs

    def add(self, spec):
        self._list.append(str(spec))

        # expanded list can be updated without invalidation
        self._expanded_list.append(str(spec))

        # Invalidate cache variables when we change the list
        self._constraints = None
        self._specs = None
        self._concrete_specs = None

    def remove(self, spec):
        # Get spec to remove from list
        remove = [s for s in self._list
                  if ((isinstance(s, basestring) and not s.startswith('$')) or
                      isinstance(s, Spec)) and Spec(s) == Spec(spec)]
        assert len(remove) == 1
        self._list.remove(remove[0])

        # invalidate cache variables when we change the list
        self._expanded_list = None
        self._constraints = None
        self._specs = None
        self._concrete_specs = None

    def update_reference(self, reference):
        self._reference = reference
        self._expanded_list = None
        self._constraints = None
        self._specs = None
        self._concrete_specs = None

    def _expand_references(self, yaml):
        if isinstance(yaml, list):
            for idx, item in enumerate(yaml):
                if isinstance(item, basestring) and item.startswith('$'):
                    name = item[1:]
                    if name in self._reference:
                        ret = [self._expand_references(i) for i in yaml[:idx]]
                        ret += self._reference[name].specs_as_yaml_list
                        ret += [self._expand_references(i)
                                for i in yaml[idx + 1:]]
                        return ret
                    else:
                        msg = 'SpecList %s refers to named list %s ' % (self.name, name)
                        msg += 'which does not appear in its reference dict'
                        raise InvalidReferenceError(msg)
            # No references in this
            return [self._expand_references(item) for item in yaml]
        elif isinstance(yaml, dict):
            # There can't be expansions in dicts
            return dict((name, self._expand_references(val))
                        for (name, val) in yaml.items())
        else:
            # Strings are just returned
            return yaml

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, key):
        return self.specs[key]


class UndefinedReferenceError(SpackError):
    """Error class for undefined references in Spack stacks."""


class InvalidSpecConstraintError(SpackError):
    """Error class for invalid spec constraints at concretize time."""
