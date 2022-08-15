# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Infrastructure used by tests for mocking packages and repos."""
import os.path
import shutil

from llnl.util import filesystem

import spack.provider_index
import spack.util.naming

__all__ = ["MockRepositoryBuilder"]


class MockRepositoryBuilder(object):
    """Build a mock repository in a directory"""

    def __init__(self, root_directory, namespace=None):
        self.namespace = namespace or "mock"
        template = spack.tengine.make_environment().get_template("mock-repository/repo.yaml")
        text = template.render({"namespace": self.namespace})
        # Write the "repo.yaml" file in the appropriate directory
        filename = os.path.join(str(root_directory), self.namespace, "repo.yaml")
        self.root = os.path.dirname(filename)
        filesystem.mkdirp(self.root)
        with open(filename, "w") as f:
            f.write(text)

    def add_package(self, name, dependencies=None):
        """Create a mock package in the repository, using a Jinja2 template.

        Args:
            name (str): name of the new package
            dependencies (list): list of ("dep_spec", "dep_type", "condition") tuples.
                Both "dep_type" and "condition" can default to ``None`` in which case
                ``spack.dependency.default_deptype`` and ``spack.spec.Spec()`` are used.
        """
        dependencies = dependencies or []
        context = {"cls_name": spack.util.naming.mod_to_class(name), "dependencies": dependencies}
        template = spack.tengine.make_environment().get_template("mock-repository/package.pyt")
        text = template.render(context)
        package_py = self.recipe_filename(name)
        filesystem.mkdirp(os.path.dirname(package_py))
        with open(package_py, "w") as f:
            f.write(text)

    def remove(self, name):
        package_py = self.recipe_filename(name)
        shutil.rmtree(os.path.dirname(package_py))

    def recipe_filename(self, name):
        return os.path.join(self.root, "packages", name, "package.py")
