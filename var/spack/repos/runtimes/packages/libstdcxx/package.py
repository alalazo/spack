# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


class Libstdcxx(BundlePackage):
    """Libstdcxx is the GNU C++ runtime library"""

    version('6')

    provides('cxx-runtime')
