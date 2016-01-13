#!/usr/bin/env python
##############################################################################
# Copyright (c) 2013, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# This file is part of Spack.
# Written by Todd Gamblin, tgamblin@llnl.gov, All rights reserved.
# LLNL-CODE-647188
#
# For details, see https://github.com/llnl/spack
# Please also see the LICENSE file for our notice and the LGPL.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
##############################################################################

from distutils.core import setup

setup(name='spack',
      version='0.8.15',
      description='A flexible package manager designed to support multiple versions, configurations, platforms, and compilers',
      author='Todd Gamblin',
      author_email='tgamblin@llnl.gov',
      url='http://llnl.github.io/spack',
      package_dir={'': 'lib/spack'},
      packages=['spack',
                'spack.util',
                'spack.hooks',
                'spack.compilers',
                'spack.cmd',
                'llnl.util',
                'llnl.util.tty',
                'external',
                'external.yaml'],
      package_data={'spack': ['../env/*']},
      scripts=['bin/spack'])
