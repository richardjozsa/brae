// dumpWallDist, write OF v2412's default wall-distance field y_ = wallDist::New(mesh).y()
// (method meshWave), the field kOmegaSST F1/F2/F3 consume. Reference oracle for cf::cellWallDist.
#include "fvCFD.H"
#include "wallDist.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // OF default patchDistMethod is meshWave (wallDist.H: "method meshWave").
    const volScalarField& y = wallDist::New(mesh).y();

    volScalarField yOut
    (
        IOobject
        (
            "y",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        y
    );
    yOut.write();

    Info<< "dumpWallDist: wrote y (wallDist meshWave)"
        << " nCells=" << mesh.nCells()
        << " min=" << gMin(y.primitiveField())
        << " max=" << gMax(y.primitiveField()) << endl;

    return 0;
}
