/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2021- The GROMACS Authors
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
/*! \internal \file
 * \brief
 * Implements force provider for QMMM
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/stringutil.h"

#include "qmmmforceprovider.h"
#include "qmmminputgenerator.h"

// debug, delete later TODO
#include <cstdio>

#include <fstream>
#include <iostream>
#include <filesystem>

namespace gmx
{

class PythonObjectManager
{
public:
    PyObject* obj = nullptr;

    PythonObjectManager() : obj(nullptr) {}
    explicit PythonObjectManager(PyObject* o) : obj(o) {}
    ~PythonObjectManager() { Py_XDECREF(obj); }

    PyObject* get() { return obj; }
    PyObject** ptr() { return &obj; }
    void reset(PyObject* o) {
        Py_XDECREF(obj);
        obj = o;
    }
    PyObject* release() {
        PyObject* tmp = obj;
        obj = nullptr;
        return tmp;
    }
};

QMMMForceProvider::QMMMForceProvider(const QMMMParameters& parameters,
                                     const LocalAtomSet&   localQMAtomSet,
                                     const LocalAtomSet&   localMMAtomSet,
                                     PbcType               pbcType,
                                     const MDLogger&       logger) :
    parameters_(parameters),
    qmAtoms_(localQMAtomSet),
    mmAtoms_(localMMAtomSet),
    pbcType_(pbcType),
    logger_(logger),
    box_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } },
    pModule_(nullptr)
{
}

QMMMForceProvider::~QMMMForceProvider()
{
    Py_XDECREF(pModule_);
}

bool QMMMForceProvider::isQMAtom(Index globalAtomIndex)
{
    return std::find(qmAtoms_.globalIndex().begin(), qmAtoms_.globalIndex().end(), globalAtomIndex)
           != qmAtoms_.globalIndex().end();
}

void QMMMForceProvider::appendLog(const std::string& msg)
{
    GMX_LOG(logger_.info).asParagraph().appendText(msg);
}

void QMMMForceProvider::calculateForces(const ForceProviderInput& fInput, ForceProviderOutput* fOutput)
{
    // Set pyscf driver name and path
    const std::string pyscfDriverName = parameters_.qmFileNameBase_;
    const std::string pyscfDriverPath = parameters_.qmFilePath_;
    fprintf(stderr, "PySCF drivername is %s\n", pyscfDriverName.c_str());
    fprintf(stderr, "PySCF driver path is %s\n", pyscfDriverPath.c_str());
    //  to enable pyscfdriver from any directory, need to add driverpath to parameters_,
    //  and then add the path to sys.path with C/Python API
    // Load module if not already loaded
    if (!pModule_)
    {
        fprintf(stderr, "Importing pyscfdriver...\n");

        // Test if pyscfdriver exists
        std::string pyscfDriverFullName = pyscfDriverPath + pyscfDriverName + ".py";
        std::ifstream pyscfDriverFile(pyscfDriverFullName);
        if (!pyscfDriverFile.good()) {
            PyErr_Print();
            fprintf(stderr, "PySCF driver file %s%s%s does not exist\n",
            pyscfDriverPath.c_str(), pyscfDriverName.c_str(), ".py");
            Py_Finalize();
        }

        // Import the `sys` module
        PythonObjectManager sysModule(PyImport_ImportModule("sys"));
        if (!sysModule.get()) {
            PyErr_Print();
            std::cerr << "Failed to import sys module." << std::endl;
            Py_Finalize();
        }

        // Get the `sys.path` attribute
        PythonObjectManager sysPath(PyObject_GetAttrString(sysModule.get(), "path"));
        if (!sysPath.get() || !PyList_Check(sysPath.get())) {
            PyErr_Print();
            std::cerr << "Failed to get sys.path." << std::endl;
            Py_Finalize();
        }

        // Add  pyscfDriverPath path to `sys.path`
        PythonObjectManager pyscfPath(PyUnicode_FromString(pyscfDriverPath.c_str()));
        if (PyList_Append(sysPath.get(), pyscfPath.get()) != 0) {
            PyErr_Print();
            std::cerr << "Failed to add DriverPath to SysPath for loading driver module." << std::endl;
            Py_Finalize();
        }

        pModule_ = PyImport_ImportModule(pyscfDriverName.c_str());
        if (!pModule_)
        {
            PyErr_Print();
            Py_FinalizeEx();
            GMX_THROW(gmx::InternalError("Failed to import pyscfdriver module!"));
        }
        fprintf(stderr, "pyscfdriver load successful!\n");
    }

    // Total number of atoms in the system
    size_t numAtomsQM = qmAtoms_.numAtomsGlobal();
    size_t numAtomsMM = mmAtoms_.numAtomsGlobal();
    size_t numAtoms   = numAtomsQM + numAtomsMM;
    // Save box

    copy_mat(fInput.box_, box_);
    // Initialize PBC
    t_pbc pbc;
    set_pbc(&pbc, pbcType_, box_);
    /*
     * 1) We need to gather fInput.x_ in case of MPI / DD setup
     */

    // x - coordinates (gathered across nodes in case of DD)
    std::vector<RVec> x(numAtoms, RVec({ 0.0, 0.0, 0.0 }));
    // Put all atoms into the central box (they might be shifted out of it because of the
    // translation) put_atoms_in_box(pbcType_, fInput.box_, ArrayRef<RVec>(x));

    // Fill cordinates of local QM atoms and add translation
    PythonObjectManager funcCalc;
    const std::string qm_basis   = "631G";
    PythonObjectManager pyQMBasis(PyUnicode_FromString(qm_basis.c_str()));
    PythonObjectManager pyQMMult(PyLong_FromLong(parameters_.qmMultiplicity_));
    PythonObjectManager pyQMCharge(PyLong_FromLong(parameters_.qmCharge_));

    PythonObjectManager pyQMKinds(PyList_New(numAtomsQM));
    PythonObjectManager pyQMCoords(PyList_New(numAtomsQM));
    PythonObjectManager pyQMLocalIndex(PyList_New(numAtomsQM));
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        int       qmLocalIndex  = qmAtoms_.localIndex()[i];
        int       qmGlobalIndex = qmAtoms_.globalIndex()[i];
        PyObject* pyqmLoclNdx   = PyLong_FromLong(qmLocalIndex);
        PyList_SetItem(pyQMLocalIndex.get(), i, pyqmLoclNdx); // Steals reference, no need to decref pyqmLoclNdx
        x[qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]] = fInput.x_[qmLocalIndex];

        PyObject* pySymbol =
                PyUnicode_FromString(periodic_system[parameters_.atomNumbers_[qmGlobalIndex]].c_str());
        PyList_SetItem(pyQMKinds.get(), i, pySymbol); // Steals reference, no need to decref pySymbol

        PyObject* pyCoords_row = PyList_New(3);

        double coordTempX = 10 * (fInput.x_[qmLocalIndex][XX]);
        double coordTempY = 10 * (fInput.x_[qmLocalIndex][YY]);
        double coordTempZ = 10 * (fInput.x_[qmLocalIndex][ZZ]);

        PyObject* pyCoordsX = PyFloat_FromDouble(coordTempX);
        PyObject* pyCoordsY = PyFloat_FromDouble(coordTempY);
        PyObject* pyCoordsZ = PyFloat_FromDouble(coordTempZ);

        PyList_SetItem(pyCoords_row, XX, pyCoordsX);
        PyList_SetItem(pyCoords_row, YY, pyCoordsY);
        PyList_SetItem(pyCoords_row, ZZ, pyCoordsZ);

        PyList_SetItem(pyQMCoords.get(), i, pyCoords_row);
    }

    // Fill cordinates of local MM atoms and add translation
    PythonObjectManager pyMMKinds;
    PythonObjectManager pyMMCharges;
    PythonObjectManager pyMMCoords;
    PythonObjectManager pyMMLocalIndex;
    PythonObjectManager pyLinks;

    PythonObjectManager pyStepNumber(PyLong_FromLongLong(fInput.step_));
    fprintf(stderr, "c++ output step number %" PRId64 " \n", fInput.step_);
    if (fInput.step_ == 0)
    {
        initialInfoGenerator(fInput);
    }
    PythonObjectManager calcResult;

    // determine whether to do a pure QM calculation or a QM/MM calculation
    if (numAtomsMM > 0)
    {
        fprintf(stderr, "Loading qmmmCalc...\n");
        funcCalc.reset(PyObject_GetAttrString(pModule_, "qmmmCalc"));
        if (!funcCalc.get())
        {
            PyErr_Print();
            Py_FinalizeEx();
            GMX_THROW(gmx::InternalError("qmmmCalc load failed!\n"));
        }
        fprintf(stderr, "qmmmCalc load successful!\n");

        pyMMKinds.reset(PyList_New(numAtomsMM));
        pyMMCharges.reset(PyList_New(numAtomsMM));
        pyMMCoords.reset(PyList_New(numAtomsMM));
        pyMMLocalIndex.reset(PyList_New(numAtomsMM));

        for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
        {
            int       mmLocalIndex  = mmAtoms_.localIndex()[i];
            int       mmGlobalIndex = mmAtoms_.globalIndex()[i];
            PyObject* pymmLoclNdx   = PyLong_FromLong(mmLocalIndex);
            PyList_SetItem(pyMMLocalIndex.get(), i, pymmLoclNdx);
            x[mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]] = fInput.x_[mmLocalIndex];

            PyObject* pySymbol = PyUnicode_FromString(
                    periodic_system[parameters_.atomNumbers_[mmGlobalIndex]].c_str());
            PyList_SetItem(pyMMKinds.get(), i, pySymbol);

            PyObject* pyCharge = PyFloat_FromDouble(fInput.chargeA_[mmLocalIndex]);
            PyList_SetItem(pyMMCharges.get(), i, pyCharge);

            PyObject* pyCoords_row = PyList_New(3);
            double coordTempX = 10 * (fInput.x_[mmLocalIndex][XX]);
            double coordTempY = 10 * (fInput.x_[mmLocalIndex][YY]);
            double coordTempZ = 10 * (fInput.x_[mmLocalIndex][ZZ]);

            PyObject* pyCoordsX = PyFloat_FromDouble(coordTempX);
            PyObject* pyCoordsY = PyFloat_FromDouble(coordTempY);
            PyObject* pyCoordsZ = PyFloat_FromDouble(coordTempZ);

            PyList_SetItem(pyCoords_row, XX, pyCoordsX);
            PyList_SetItem(pyCoords_row, YY, pyCoordsY);
            PyList_SetItem(pyCoords_row, ZZ, pyCoordsZ);

            PyList_SetItem(pyMMCoords.get(), i, pyCoords_row);
        }

        size_t    numLinks = parameters_.link_.size();
        pyLinks.reset(PyList_New(numLinks));
        for (size_t i = 0; i < numLinks; i++)
        {
            PyObject* pyLinkQM = nullptr;
            PyObject* pyLinkMM = nullptr;

            for (size_t j = 0; j < numAtomsQM; j++)
            {
                if (qmAtoms_.globalIndex()[j] == parameters_.link_[i].qm)
                {
                    pyLinkQM = PyLong_FromLong(qmAtoms_.localIndex()[j]);
                }
            }

            for (size_t j = 0; j < numAtomsMM; j++)
            {
                if (mmAtoms_.globalIndex()[j] == parameters_.link_[i].mm)
                {
                    pyLinkMM = PyLong_FromLong(mmAtoms_.localIndex()[j]);
                }
            }

            PyObject* pyLinkPair = PyList_New(2);

            PyList_SetItem(pyLinkPair, 0, pyLinkQM);
            PyList_SetItem(pyLinkPair, 1, pyLinkMM);
            PyList_SetItem(pyLinks.get(), i, pyLinkPair);
        }
        calcResult.reset(PyObject_CallFunctionObjArgs(funcCalc.get(),
                                                      pyStepNumber.get(),
                                                      pyQMBasis.get(),
                                                      pyQMMult.get(),
                                                      pyQMCharge.get(),
                                                      pyQMLocalIndex.get(),
                                                      pyQMKinds.get(),
                                                      pyQMCoords.get(),
                                                      pyMMLocalIndex.get(),
                                                      pyMMKinds.get(),
                                                      pyMMCharges.get(),
                                                      pyMMCoords.get(),
                                                      pyLinks.get(),
                                                      NULL));
    }
    else if (numAtomsMM == 0)
    {
        fprintf(stderr, "numAtomsMM = 0, Loading qmCalc...\n");
        funcCalc.reset(PyObject_GetAttrString(pModule_, "qmCalc"));
        if (!funcCalc.get())
        {
            PyErr_Print();
            Py_FinalizeEx();
            GMX_THROW(gmx::InternalError("qmCalc load failed!\n"));
        }
        fprintf(stderr, "qmCalc load successful!\n");

        calcResult.reset(PyObject_CallFunctionObjArgs(
                funcCalc.get(), pyStepNumber.get(), pyQMBasis.get(), pyQMMult.get(),
                pyQMCharge.get(), pyQMKinds.get(), pyQMCoords.get(), NULL));
    }
    else
    {
        GMX_THROW(gmx::InternalError("Unexpected MM atom number."));
    }

    if (!calcResult.get())
    {
        PyErr_Print();
        // this function must be called to print Traceback messges and python output up to that exception -
        // - with import_array() called in gmx.cpp and numpy imported in pyscfdriver.py.
        Py_FinalizeEx();
        fprintf(stderr, "Py_FinalizeEx() run finished\n");
        GMX_THROW(gmx::InternalError("GMX_THORW: calcResult is nullptr!!!"));
    }

    // If we are in MPI / DD conditions then gather coordinates over nodes
    if (havePPDomainDecomposition(&fInput.cr_))
    {
        gmx_sum(3 * numAtoms, x.data()->as_vec(), &fInput.cr_);
    }

    // TODO: fix for MPI version?

    PyObject* pQMMMEnergy = PyTuple_GetItem(calcResult.get(), 0);
    PyObject* pQMForce    = PyTuple_GetItem(calcResult.get(), 1);
    PyObject* pMMForce    = PyTuple_GetItem(calcResult.get(), 2);

    // fprintf(stdout, "Python print test QMForce\n");

    double qmmmEnergy(0);
    if (pQMMMEnergy)
    {
        qmmmEnergy = PyFloat_AsDouble(pQMMMEnergy);
        fprintf(stderr, "GROMACS received energy %f \n", qmmmEnergy);
    }
    else
    {
        PyErr_Print();
        Py_FinalizeEx();
        GMX_THROW(gmx::InternalError("pointer to pyscf returned energy is nullptr\n"));
    }

    if (!pQMForce)
    {
        PyErr_Print();
        Py_FinalizeEx();
        GMX_THROW(gmx::InternalError(
                  "parsing calcResult error, pyobject pQMForce is nullptr\n"));
    }

    if ((!pMMForce) && (numAtomsMM > 0))
    {
        PyErr_Print();
        Py_FinalizeEx();
        GMX_THROW(gmx::InternalError(
                  "parsing calcResult error, pyobject pMMForce is nullptr\n"));
    }

    double         qmForce[numAtomsQM * 3 + 1] = {};
    PyArrayObject* npyQMForce                  = reinterpret_cast<PyArrayObject*>(pQMForce);
    GMX_ASSERT(numAtomsQM == static_cast<size_t>(PyArray_DIM(npyQMForce, 0)), "Check QM atom number");
    GMX_ASSERT(3 == static_cast<int>(PyArray_DIM(npyQMForce, 1)), "Check if 3 columns for QM force");

    double* npyQMForce_cast = static_cast<double*>(PyArray_DATA(npyQMForce));
    for (size_t i = 0; i < numAtomsQM; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            qmForce[i * 3 + j] = npyQMForce_cast[i * 3 + j];
        }
    }
    double mmForce[numAtomsMM * 3 + 1] = {};
    if (numAtomsMM > 0)
    {
        PyArrayObject* npyMMForce = reinterpret_cast<PyArrayObject*>(pMMForce);
        GMX_ASSERT(numAtomsMM == static_cast<size_t>(PyArray_DIM(npyMMForce, 0)),
                   "Check MM atom number");
        GMX_ASSERT(3 == static_cast<int>(PyArray_DIM(npyMMForce, 1)),
                   "Check if 3 columns for MM force");

        double* npyMMForce_cast = static_cast<double*>(PyArray_DATA(npyMMForce));
        for (size_t i = 0; i < numAtomsMM; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                mmForce[i * 3 + j] = npyMMForce_cast[i * 3 + j];
            }
        }
    }

    /*
     * 2) Cast data to double format of libpython
     *    update coordinates and box in PySCF and perform QM calculation
     */
    // x_d - coordinates casted to linear dobule vector for PySCF with parameters_.qmTrans_ added
    std::vector<double> x_d(3 * numAtoms, 0.0);
    for (size_t i = 0; i < numAtoms; i++)
    {
        x_d[3 * i]     = static_cast<double>((x[i][XX]) / c_bohr2Nm);
        x_d[3 * i + 1] = static_cast<double>((x[i][YY]) / c_bohr2Nm);
        x_d[3 * i + 2] = static_cast<double>((x[i][ZZ]) / c_bohr2Nm);
    }

    // box_d - box_ casted to linear dobule vector for PySCF
    std::vector<double> box_d(9);
    for (size_t i = 0; i < DIM; i++)
    {
        box_d[3 * i]     = static_cast<double>(box_[0][i] / c_bohr2Nm);
        box_d[3 * i + 1] = static_cast<double>(box_[1][i] / c_bohr2Nm);
        box_d[3 * i + 2] = static_cast<double>(box_[2][i] / c_bohr2Nm);
    }

    // TODO: maybe need to handle MPI?
    // Update coordinates and box in PySCF
    // Check if we have external MPI library
    if (GMX_LIB_MPI)
    {
        // We have an external MPI library
#if GMX_LIB_MPI
#endif
    }
    else
    {
        // If we have thread-MPI or no-MPI
    }


    /*
     * 3) Get output data
     * We need to fill only local part into fOutput
     */

    // Only main process should add QM + QMMM energy
    if (MAIN(&fInput.cr_))
    {
        double qmEner(0.0);
        qmEner = qmmmEnergy;
        // cp2k_get_potential_energy(force_env_, &qmEner);
        fOutput->enerd_.term[F_EQM] += qmEner * c_hartree2Kj * c_avogadro;
    }

    // Get Forces they are in Hartree/Bohr and will be converted to kJ/mol/nm
    std::vector<double> pyscfForce(3 * numAtoms, 0.0);

    // Fill forces on QM atoms first
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]] =
                qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 0];
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1] =
                qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 1];
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2] =
                qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 2];

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][XX] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][YY] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][ZZ] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2])
                * c_hartreeBohr2Md;
    }

    // Fill forces on MM atoms then
    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]] =
                mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 0];
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1] =
                mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 1];
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2] =
                mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 2];

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][XX] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][YY] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][ZZ] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2])
                * c_hartreeBohr2Md;
    }

}


void QMMMForceProvider::initialInfoGenerator(const ForceProviderInput& fInput)
{

    std::string       QMMM_initInfo = "";
    std::ofstream     recordFile;
    const std::string pyscfRecordName = parameters_.qmFileNameBase_ + "_init.txt";
    std::ifstream pyscfInitRecordFile(pyscfRecordName);
    if (pyscfInitRecordFile.good()) {
        fprintf(stderr, "PySCF initial information exist, will not print again.\n");
    } else {
        recordFile.open(pyscfRecordName.c_str(), std::ios::app);

        double coordx = 0.0, coordy = 0.0, coordz = 0.0;
        double charge = 0.0;
        QMMM_initInfo += formatString("step = ");
        QMMM_initInfo += formatString("%" PRId64, fInput.step_);
        QMMM_initInfo += formatString(", time = %f\n", fInput.t_);
        // QMMM_initInfo += formatString(
        // "          %10s %10s %10s %9s %6s %6s %6s %6s\n", "x", "y", "z", "charge", "i", "local", "global", "collec");
        QMMM_initInfo += formatString(
                "        region     x          y          z       charge      i  local global "
                "collec\n");

        for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
        {

            QMMM_initInfo += formatString(
                    "%4d  %2s QM  ",
                    qmAtoms_.localIndex()[i],
                    periodic_system[parameters_.atomNumbers_[qmAtoms_.globalIndex()[i]]].c_str());

            coordx = (fInput.x_[qmAtoms_.localIndex()[i]][XX]) * 10;
            coordy = (fInput.x_[qmAtoms_.localIndex()[i]][YY]) * 10;
            coordz = (fInput.x_[qmAtoms_.localIndex()[i]][ZZ]) * 10;

            charge = fInput.chargeA_[qmAtoms_.localIndex()[i]];

            QMMM_initInfo += formatString("%10.4lf %10.4lf %10.4lf %7.3lf", coordx, coordy, coordz, charge);
            QMMM_initInfo += formatString("%8d %6d %6d %6d\n",
                                        static_cast<int>(i),
                                        qmAtoms_.localIndex()[i],
                                        qmAtoms_.globalIndex()[i],
                                        qmAtoms_.collectiveIndex()[i]);
        }

        for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
        {

            QMMM_initInfo += formatString(
                    "%4d  %2s MM  ",
                    mmAtoms_.localIndex()[i],
                    periodic_system[parameters_.atomNumbers_[mmAtoms_.globalIndex()[i]]].c_str());

            coordx = (fInput.x_[mmAtoms_.localIndex()[i]][XX]) * 10;
            coordy = (fInput.x_[mmAtoms_.localIndex()[i]][YY]) * 10;
            coordz = (fInput.x_[mmAtoms_.localIndex()[i]][ZZ]) * 10;

            charge = fInput.chargeA_[mmAtoms_.localIndex()[i]];
            QMMM_initInfo += formatString("%10.4lf %10.4lf %10.4lf %7.3lf", coordx, coordy, coordz, charge);
            QMMM_initInfo += formatString("%8d %6d %6d %6d\n",
                                        static_cast<int>(i),
                                        mmAtoms_.localIndex()[i],
                                        mmAtoms_.globalIndex()[i],
                                        mmAtoms_.collectiveIndex()[i]);
        }

        QMMM_initInfo += formatString("\n");
        recordFile << QMMM_initInfo;
        recordFile.close();
    }

    return;
}

} // namespace gmx
