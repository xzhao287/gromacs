#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2021- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

if(GMX_PYSCF)

    # CMake flags for PySCF linking
    find_package(Python COMPONENTS Development NumPy)

    # Check if libpython is found
    if (Python_Development.Embed_FOUND AND Python_NumPy_FOUND)
        # libpython found: add libraries and paths to the respecting GROMACS variables
        message(STATUS "Found libpython in ${Python_LIBRARY_DIRS}")
        message(STATUS "Found python include dir: ${Python_INCLUDE_DIRS}")
        message(STATUS "Found numpy include dir: ${Python_NumPy_INCLUDE_DIRS}")
        include_directories(SYSTEM "${Python_INCLUDE_DIRS}")
        include_directories(SYSTEM "${Python_NumPy_INCLUDE_DIRS}")
        link_directories(${Python_LIBRARY_DIRS})
        list(APPEND GMX_COMMON_LIBRARIES ${Python_LIBRARIES})
        add_definitions(-DGMX_PYSCF=1)
    else()
        message(FATAL_ERROR "To build GROMACS with PySCF Interface, libpython and numpy should be installed")
    endif()

endif()
