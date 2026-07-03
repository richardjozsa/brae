/*---------------------------------------------------------------------------*\
    amgxSolver, an OpenFOAM lduMatrix::solver that offloads the linear solve
    to NVIDIA AMGX (GPU). Self-contained: converts OF's LDU matrix to CSR and
    calls the AMGX C API directly (no PETSc, no PCAMGX). Selected in fvSolution
    with `solver amgx;`. Load via `libs ("libamgxFoam.so");` in controlDict.
\*---------------------------------------------------------------------------*/
#ifndef amgxSolver_H
#define amgxSolver_H

#include "lduMatrix.H"

namespace Foam
{

class amgxSolver
:
    public lduMatrix::solver
{
    // Private Member Functions
        amgxSolver(const amgxSolver&) = delete;
        void operator=(const amgxSolver&) = delete;

public:

    //- Runtime type information
    TypeName("amgx");

    //- Construct from matrix components and solver controls
    amgxSolver
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Destructor
    virtual ~amgxSolver() = default;

    // Member Functions

        //- Solve the matrix with this solver (solveScalar precision)
        virtual solverPerformance scalarSolve
        (
            solveScalarField& psi,
            const solveScalarField& source,
            const direction cmpt = 0
        ) const;

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const;
};

} // End namespace Foam

#endif
