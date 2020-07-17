# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


class Libcxx(BundlePackage):
    """Libc++ is the LLVM C++ runtime library"""

    version('1')

    provides('cxx-runtime')
    depends_on('libcxx-abi')
