/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2010- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \brief
 * Implements the gmx wrapper binary.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/commandline/cmdlineinit.h"
#include "gromacs/commandline/cmdlinemodulemanager.h"
#include "gromacs/selection/selhelp.h"
#include "gromacs/trajectoryanalysis/modules.h"
#include "gromacs/utility/exceptions.h"

#include "legacymodules.h"

#if GMX_PYSCF
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL GROMACS_ARRAY_API
#define NUMPY_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#endif

int main(int argc, char* argv[])
{
    gmx::CommandLineProgramContext& context = gmx::initForCommandLine(&argc, &argv);
    try
    {
#if GMX_PYSCF
        wchar_t* program = Py_DecodeLocale(argv[0], NULL);
        if (program == NULL)
        {
            fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
            exit(1);
        }

        Py_SetProgramName(program);
        Py_Initialize();
        fprintf(stderr, "Initialized Python Interpreter.\n");
        if (_import_array() < 0) {
            PyErr_Print();
            Py_FinalizeEx();
            fprintf(stderr, "Failed to import numpy.\n");
            PyMem_RawFree(program);
            exit(1);
        }
        fprintf(stderr, "Imported NumpyArray.\n");
        assert(PyArray_API);
#endif

        gmx::CommandLineModuleManager manager("gmx", &context);
        registerTrajectoryAnalysisModules(&manager);
        registerLegacyModules(&manager);
        manager.addHelpTopic(gmx::createSelectionHelpTopic());
        int rc = manager.run(argc, argv);
        gmx::finalizeForCommandLine();

#if GMX_PYSCF
        PyErr_Print();
        fprintf(stderr, "Printed Python Errors, Finalizing Py Interpreter.\n");
        if (Py_FinalizeEx() < 0)
        {
            PyMem_RawFree(program);
            exit(120);
        }
        PyMem_RawFree(program);
#endif
        return rc;
    }
    catch (const std::exception& ex)
    {
        gmx::printFatalErrorMessage(stderr, ex);
        return gmx::processExceptionAtExitForCommandLine(ex);
    }
}
