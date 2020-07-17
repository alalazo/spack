# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


class Libgfortran(BundlePackage):
    """Libgfortran is the GNU Fortran runtime library"""

    version('5')
    version('4')

    provides('fortran-runtime')
