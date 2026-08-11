// DIAGNOSTIC (not a gate): assemble the same scalar equation OpenFOAM's dumpScalarMatrix builds, and
// compare the matrix cell by cell.
//
//     fvm::div(phi, psi) - fvm::laplacian(gamma, psi)      phi = fvc::flux(U), gamma uniform
//
// with the case's own fvSchemes, so `bounded Gauss linearUpwind grad(omega)` is honoured on both sides.
// Deliberately not the full kOmegaSST omega equation: F1/beta/gamma are protected in OF, so rebuilding
// them in a utility would test my copy of the model rather than OF's. The machinery under investigation --
// the bounded prefix, the upwind matrix, the deferred correction, the laplacian -- is generic and is
// exactly what this assembles.
//
// Reads OF's Ddiag/Dsource from the same case and reports the per-cell difference.
//
//   diag_compare <caseDir> <timeDir> [gamma]

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

namespace {

// Build a GeometricField from a file, honouring the plain BC types this comparison uses.
GeometricField<scalar> loadScalar(const std::string& path, const std::vector<FvPatch>& fvp)
{
    const FieldData<scalar> fd = readField<scalar>(path);
    GeometricField<scalar> f;
    f.internal = fd.internalField;
    for (const FvPatch& q : fvp)
    {
        std::string type = "zeroGradient";
        scalar val = 0;
        bool fixed = false;
        for (const auto& b : fd.boundary)
            if (b.name == q.name) { type = b.type; if (b.valueUniform) val = b.uniformValue; }
        if (q.type == "empty")            f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (type == "fixedValue")  { fixed = true;
                                          f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, true, val, std::vector<scalar>{})); }
        else                              f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        (void)fixed;
    }
    f.evaluateBoundary();
    return f;
}

std::vector<scalar> readInternal(const std::string& path)
{
    return readField<scalar>(path).internalField;
}

}   // namespace

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: diag_compare <caseDir> <timeDir> [gamma]\n"); return 2; }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar gammaVal = argc > 3 ? std::atof(argv[3]) : 1e-3;

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // psi (the converged omega) and U, exactly the files OF read.
    GeometricField<scalar> psi = loadScalar(caseDir + "/" + t + "/omega", fvp);
    const FieldData<vector> ud = readField<vector>(caseDir + "/" + t + "/U");
    // U's REAL boundary conditions. An earlier version forced zeroGradient on every patch, which changed
    // phi = fvc::flux(U) at the boundary faces and put 100% of the apparent diagonal difference in
    // boundary cells -- an artifact of the harness, not of the assembly.
    GeometricField<vector> U;
    U.internal = ud.internalField;
    for (const FvPatch& q : fvp)
    {
        std::string type = "zeroGradient";
        vector val{0, 0, 0};
        for (const auto& b : ud.boundary)
            if (b.name == q.name) { type = b.type; if (b.valueUniform) val = b.uniformValue; }
        if (q.type == "empty")
            U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (type == "fixedValue" || type == "noSlip")
            U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                q, true, type == "noSlip" ? vector{0,0,0} : val, std::vector<vector>{}));
        else
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceBuffer<scalar> dpsi(psi.internal), dphi(phi.internal);
    std::vector<std::vector<scalar>> pb;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) pb.push_back(psi.boundary[pi]->value());
    DeviceBuffer<scalar> dpsib(flattenBoundary(pb));

    // --- assemble exactly as deviceSolveScalarTransport does ---
    DeviceBuffer<scalar> aD, aU, aL, lD, lU, lL, luCorr, lapCorr;
    DeviceBuffer<scalar> gx, gy, gz;
    deviceGaussGrad(dm, dpsi, dpsib, gx, gy, gz);

    deviceDivUpwindCoeffs(dm, dphi, aD, aU, aL);
    deviceLinearUpwindCorr(dm, dphi, gx, gy, gz, luCorr);

    DeviceBuffer<scalar> gammaCell(std::vector<scalar>(static_cast<std::size_t>(nC), gammaVal)), Df;
    deviceInterpolate(dm, gammaCell, Df);
    deviceLaplacianCoeffs(dm, Df, lD, lU, lL, /*nonOrth*/ true);
    deviceAxpy(-1.0, lD, aD);
    deviceLaplacianCorr(dm, Df, gx, gy, gz, lapCorr);

    // bounded: -Sp(div(phi), psi)
    DeviceBuffer<scalar> divPhi, bt;
    deviceDiv(dm, dphi, DeviceBuffer<scalar>(flattenBoundary(phi.boundary)), divPhi);
    deviceHadamard(bt, divPhi, dm.V);
    deviceAxpy(-1.0, bt, aD);

    DeviceBuffer<scalar> src(static_cast<std::size_t>(nC));
    cudaCheck(cudaMemsetAsync(src.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "src zero");
    deviceAxpy(-1.0, luCorr, src);
    if (lapCorr.size()) deviceAxpy(-1.0, lapCorr, src);

    // --- compare against OF ---
    const std::vector<scalar> ofDiag = readInternal(caseDir + "/" + t + "/Ddiag");
    const std::vector<scalar> ofSrc  = readInternal(caseDir + "/" + t + "/Dsource");
    const std::vector<scalar> bDiag = aD.host(), bSrc = src.host();

    auto rel = [](const std::vector<scalar>& a, const std::vector<scalar>& b)
    {
        const std::size_t n = std::min(a.size(), b.size());
        double num = 0, den = 0;
        for (std::size_t i = 0; i < n; ++i) { num += (a[i]-b[i])*(a[i]-b[i]); den += a[i]*a[i]; }
        return den > 0 ? std::sqrt(num/den) : std::sqrt(num);
    };
    auto rng = [](const std::vector<scalar>& v)
    {
        scalar lo = v.empty() ? 0 : v[0], hi = lo;
        for (scalar x : v) { lo = std::fmin(lo, x); hi = std::fmax(hi, x); }
        return std::pair<scalar,scalar>(lo, hi);
    };

    std::printf("cells: OF %zu  brae %zu\n\n", ofDiag.size(), bDiag.size());
    auto od = rng(ofDiag), bd = rng(bDiag);
    std::printf("  diag   OF [%.6e, %.6e]\n", (double)od.first, (double)od.second);
    std::printf("  diag brae [%.6e, %.6e]\n", (double)bd.first, (double)bd.second);
    std::printf("  diag   L2rel = %.6e\n", rel(ofDiag, bDiag));
    std::printf("  source L2rel = %.6e\n", rel(ofSrc, bSrc));

    // worst cell, to point at a mechanism rather than a number
    std::size_t w = 0; double wd = -1;
    for (std::size_t i = 0; i < std::min(ofDiag.size(), bDiag.size()); ++i)
    {
        const double d = std::fabs((double)ofDiag[i] - (double)bDiag[i]);
        if (d > wd) { wd = d; w = i; }
    }
    std::printf("  worst diag cell %zu: OF %.8e  brae %.8e  ratio %.6f\n",
                w, (double)ofDiag[w], (double)bDiag[w], (double)bDiag[w]/(double)ofDiag[w]);

    // Is the difference a BIAS (a missing contribution) or scatter (arithmetic)? And is it at the boundary?
    std::size_t smaller = 0, larger = 0, nb = 0;
    double meanRatio = 0;
    std::vector<char> isBnd(static_cast<std::size_t>(nC), 0);
    for (const FvPatch& q : fvp)
        for (label i = 0; i < q.size; ++i)
            if (q.type != "empty") isBnd[static_cast<std::size_t>(q.faceCells[i])] = 1;
    double sBnd = 0, sInt = 0, dBnd = 0, dInt = 0;
    for (std::size_t i = 0; i < bDiag.size(); ++i)
    {
        const double r = (double)bDiag[i] / (double)ofDiag[i];
        meanRatio += r;
        if (r < 1.0) ++smaller; else ++larger;
        const double d2 = ((double)ofDiag[i]-(double)bDiag[i])*((double)ofDiag[i]-(double)bDiag[i]);
        if (isBnd[i]) { ++nb; dBnd += d2; sBnd += (double)ofDiag[i]*(double)ofDiag[i]; }
        else          {        dInt += d2; sInt += (double)ofDiag[i]*(double)ofDiag[i]; }
    }
    meanRatio /= (double)bDiag.size();
    std::printf("\n  brae/OF mean ratio = %.6f   (smaller in %zu cells, larger in %zu)\n",
                meanRatio, smaller, larger);
    std::printf("  error energy: boundary cells (%zu) %.1f%%   interior %.1f%%\n",
                nb, 100.0*dBnd/(dBnd+dInt), 100.0*dInt/(dBnd+dInt));
    return 0;
}
