// The Phase 4 invariant, on a REAL assembled matrix: WHEN is the pressure operator self-adjoint?
//
// test_interface_invariants.cu asserts that it always is, and it passes. It builds the interface arrays
// BY HAND, so what it actually proves is that the assembly applies a coupling symmetrically when handed
// a symmetric one. True, worth having, and not the claim that matters -- because on
// pimpleFoam/RAS/oscillatingInletPeriodicAMI2D the real matrix is NOT self-adjoint and the synthetic
// fixture could not see it. Jacobi-PCG then ran to its 50-iteration cap with the residual GROWING
// (1.00 -> 1.32) where OpenFOAM's GAMG converged in 12.
//
// So this file builds the interface the way the solver does: a real mesh, buildAMIInterfaces (polygon
// overlap, real weights), buildDeviceAMI, and the real deviceAmiAssembleLaplacian. Nothing is
// hand-written except the mesh.
//
// WHAT THE FOUR LEGS ESTABLISH. The interface contributes ifCoeff_i*w_ij to A[c_i][c_j], and the reverse
// entry comes from the OTHER direction as ifCoeff_j*w_ji. With w_ij = overlap/|Sf_i| and
// ifCoeff ~ gamma*|Sf|*deltaCoeffs, the two agree exactly when deltaCoeffs_i == deltaCoeffs_j. And
// deltaCoeffs is 1/(nf & delta) with delta = patchD - Sum_j w_ij*nbrD_j -- a WEIGHTED AVERAGE over
// partners. So:
//
//     conforming (one partner per face)   -> the average is a single term, both sides agree   -> SYMMETRIC
//     non-conforming (several partners)   -> each side averages a different set               -> NOT
//
// Legs 1-2 pin the first row, legs 3-4 the second. Leg 3 is the one that settled it: an ORDINARY
// cyclicAMI, no periodicity and no tiling, is asymmetric at 1.3e-01 -- worse than the periodic patch
// that prompted the investigation. The tiling was never the cause.
//
// THIS IS NOT A DEFECT TO BE NORMALISED AWAY. OpenFOAM computes cyclicAMI deltaCoeffs per side by the
// same formula and inherits the same asymmetry; that is why its AMI tutorials ship GAMG or PBiCGStab
// for p and not one of them ships PCG. The defect was brae solving the interface pressure system with
// Jacobi-PCG regardless. CG on a non-symmetric operator does not lose efficiency, it fails to converge.
// The fix is in the solver constructor (device_simple_foam.cu): a non-conforming AMI selects BiCGStab,
// exactly as the transonic pressure matrix already did for the same reason. Measured effect on
// oscillatingInletPeriodicAMI2D: 2.23e-01 -> 7.73e-02.
//
// These legs therefore assert the PROPERTY, not a bug -- which is why legs 3 and 4 assert asymmetry and
// must keep doing so. If a future change made them symmetric, the solver selection would silently
// become over-conservative rather than wrong, and this file is where that would be noticed.
#include "box_mesh.cuh"
#include "device_ami.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include "device_interface.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "interface/ami_interface.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// Retype inlet/outlet into a cyclicAMI pair, optionally GRADING the mesh along x.
//
// Grading is the right lever for this, and shearing is not. Shearing one side slides faces off the
// patch edge, so coverage drops and the interface is refused before the invariant can be measured --
// a correct refusal that simply prevents the test. Grading keeps the pair perfectly CONFORMING (both
// sides keep the same y-z discretisation, full coverage) while making the cell adjacent to `inlet`
// thin and the one adjacent to `outlet` thick. That is exactly the asymmetry mechanism under
// suspicion: w_ij is 1:1, but deltaCoeffs_i and deltaCoeffs_j differ, so ifCoeff_i*w_ij and
// ifCoeff_j*w_ji need not agree.
PrimitiveMesh amiMesh(const PrimitiveMesh& src, scalar grade)
{
    std::vector<vector> pts = src.points();
    if (grade != scalar(1))
    {
        scalar xmax = 0;
        for (const vector& v : pts) xmax = std::max(xmax, v.x);
        if (xmax > 0)
            for (vector& v : pts) v.x = xmax*std::pow(v.x/xmax, grade);
    }
    std::vector<PatchInfo> pp = src.patches();
    for (PatchInfo& q : pp)
    {
        if (q.name == "inlet")  { q.type = "cyclicAMI"; q.neighbourPatch = "outlet"; q.transform = "unknown"; }
        if (q.name == "outlet") { q.type = "cyclicAMI"; q.neighbourPatch = "inlet";  q.transform = "unknown"; }
    }
    PrimitiveMesh out;
    out.assign(std::move(pts), src.faceVerts(), src.faceOffsets(), src.owner(), src.neighbour(),
               std::move(pp), src.nCells());
    return out;
}

scalar adjointDefect(const DeviceLduView& A, DeviceBuffer<scalar>& x, DeviceBuffer<scalar>& y)
{
    DeviceBuffer<scalar> Ay, Ax;
    Ay.copyFrom(std::vector<scalar>(x.size(), scalar(0)));
    Ax.copyFrom(std::vector<scalar>(x.size(), scalar(0)));
    deviceAmul(A, y, Ay);
    deviceAmul(A, x, Ax);
    const scalar a = deviceDot(x, Ay), b = deviceDot(y, Ax);
    return std::fabs(a - b)/std::max(std::max(std::fabs(a), std::fabs(b)), scalar(1e-300));
}

// Assemble a pressure-like laplacian with the REAL interface coefficients and measure self-adjointness.
scalar realDefect(const PrimitiveMesh& m, const char* tag)
{
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells();
    const int nIf = (int)m.nInternalFaces();

    const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);
    DeviceAMI ami = buildDeviceAMI(amis);

    // uniform diffusivity, so any asymmetry is the interface's and not the coefficient field's
    DeviceBuffer<scalar> gF, gC, D, U, L;
    gF.copyFrom(std::vector<scalar>((std::size_t)std::max(nIf, 0), scalar(1)));
    gC.copyFrom(std::vector<scalar>((std::size_t)nC, scalar(1)));
    deviceLaplacianCoeffs(dm, gF, D, U, L, false);
    // ...and the interface's own diagonal + ifCoeff, exactly as the pressure equation builds them
    interfaceAssembleLaplacian(ami, gC, D, /*addToDiag*/true);

    std::vector<scalar> hx(nC), hy(nC);
    for (label c = 0; c < nC; ++c) { hx[c] = std::sin(0.7*c) + 1.3; hy[c] = std::cos(0.4*c) - 0.8; }
    DeviceBuffer<scalar> x, y;
    x.copyFrom(hx); y.copyFrom(hy);

    const DeviceLduView A = deviceLduViewAmi(dm, D, U, L, ami.n, ami.ownCell.data(), ami.off.data(),
                                             ami.nbrCell.data(), ami.weight.data(), ami.ifCoeff.data());
    const scalar d = adjointDefect(A, x, y);
    std::printf("        (%s: %d interface faces, %zu stencil entries)\n",
                tag, ami.n, (std::size_t)ami.weight.size());
    return d;
}
} // namespace

int main()
{
    std::printf("== interface invariants on a REAL assembled matrix ==\n");

    const PrimitiveMesh base = boxtest::boxMesh(6, 5, 4);
    PrimitiveMesh ncMesh, perMesh;   // kept for Leg 5, which re-reads them through the predicate

    // ---- Leg 1: a uniform, conforming AMI pair -----------------------------------------------------
    // Both sides identical, so each face maps onto exactly one partner and the two directions' delta
    // coefficients coincide. If even this were asymmetric, the assembly would be at fault rather than
    // the geometry it is handed.
    {
        const scalar d = realDefect(amiMesh(base, 1.0), "uniform");
        std::printf("        adjoint defect %.4e\n", (double)d);
        check(d < 1e-12, "uniform cyclicAMI: the assembled operator is self-adjoint");
    }

    // ---- Leg 2: the two sides' deltaCoeffs differ ---------------------------------------------------
    // Still 1:1 and fully covered, but the cell next to one side is thin and the cell next to the other
    // is thick. Any dependence of the coupling on the OWN side's delta -- rather than on the distance
    // between the two cells, which is shared -- shows up here and nowhere else. This is the property the
    // synthetic fixture could not express, because it supplied the coefficients rather than deriving
    // them from a mesh.
    {
        const scalar d = realDefect(amiMesh(base, 1.8), "graded (deltaCoeffs differ across the pair)");
        std::printf("        adjoint defect %.4e\n", (double)d);
        check(d < 1e-12, "graded cyclicAMI: still self-adjoint when the two sides' deltas differ");
    }

    // ---- Leg 3: an ORDINARY cyclicAMI that is NON-CONFORMING ----------------------------------------
    // The control that decides what the periodic defect below actually is.
    //
    // Grading y by an amount that RAMPS WITH x leaves the y-range untouched at both ends -- so both
    // patches still span 0..ymax and coverage stays complete -- while giving the two sides different y
    // spacings. Every source face then overlaps two target faces instead of one. That is
    // non-conformity WITHOUT periodicity, WITHOUT tiling, and without multiple images.
    //
    // It matters because deltaCoeffs is `1/(nf & delta)` with `delta = patchD - Sum w . nbrD`: the
    // moment a face has more than one partner, its delta is a WEIGHTED AVERAGE over partners, and the
    // reverse direction averages a different set. If this leg is asymmetric too, then asymmetry is a
    // property of every non-conforming AMI -- which is also true of OpenFOAM, whose cyclicAMI
    // deltaCoeffs are computed per side by the same formula -- and the periodic patch is not special.
    {
        std::vector<vector> pts = base.points();
        scalar xmax = 0, ymax = 0;
        for (const vector& v : pts) { xmax = std::max(xmax, v.x); ymax = std::max(ymax, v.y); }
        for (vector& v : pts)
            if (ymax > 0 && xmax > 0)
                v.y = ymax*std::pow(v.y/ymax, scalar(1) + scalar(0.25)*(v.x/xmax));
        std::vector<PatchInfo> pp = base.patches();
        for (PatchInfo& q : pp)
        {
            if (q.name == "inlet")  { q.type = "cyclicAMI"; q.neighbourPatch = "outlet"; q.transform = "unknown"; }
            if (q.name == "outlet") { q.type = "cyclicAMI"; q.neighbourPatch = "inlet";  q.transform = "unknown"; }
        }
        ncMesh.assign(std::move(pts), base.faceVerts(), base.faceOffsets(), base.owner(), base.neighbour(),
                  std::move(pp), base.nCells());
        const scalar d = realDefect(ncMesh, "non-conforming cyclicAMI (no periodicity)");
        std::printf("        adjoint defect %.4e\n", (double)d);
        check(d > 1e-9, "a non-conforming cyclicAMI is NOT self-adjoint -- the property that selects BiCGStab");
    }

    // ---- Leg 4: a cyclicPeriodicAMI, whose stencil carries MULTIPLE IMAGES -------------------------
    // The case that started this. A source face couples to target faces at several periodic images, so
    // the same neighbour cell can appear more than once in one row -- and the first hypothesis was that
    // the two directions' adaptive tiling searches stopped at different image counts, making the
    // stencils non-transposes. They do not: the images are added in +/-k pairs and the counts match.
    //
    // Leg 3 is what makes this leg readable. The periodic patch is asymmetric for the ordinary reason
    // -- several partners per face -- and by LESS than the plain non-conforming pair above, so
    // periodicity is not even an aggravating factor.
    {
        std::vector<vector> pts = base.points();
        for (const PatchInfo& q : base.patches())          // shear one side so the tiling is needed
            if (q.name == "outlet")
                for (label f = q.start; f < q.start + q.size; ++f)
                    for (label k = base.faceOffsets()[f]; k < base.faceOffsets()[f+1]; ++k)
                        pts[base.faceVerts()[k]].y += scalar(0.5);
        std::vector<PatchInfo> pp = base.patches();
        for (PatchInfo& q : pp)
        {
            if (q.name == "inlet")  { q.type = "cyclicPeriodicAMI"; q.neighbourPatch = "outlet";
                                      q.periodicPatch = "wallYmin"; q.maxIter = 36; q.matchTolerance = 1e-4; }
            if (q.name == "outlet") { q.type = "cyclicPeriodicAMI"; q.neighbourPatch = "inlet";
                                      q.periodicPatch = "wallYmin"; q.maxIter = 36; q.matchTolerance = 1e-4; }
            if (q.name == "wallYmin") { q.type = "cyclic"; q.neighbourPatch = "wallYmax"; q.transform = "unknown"; }
            if (q.name == "wallYmax") { q.type = "cyclic"; q.neighbourPatch = "wallYmin"; q.transform = "unknown"; }
        }
        perMesh.assign(std::move(pts), base.faceVerts(), base.faceOffsets(), base.owner(), base.neighbour(),
                  std::move(pp), base.nCells());
        const scalar d = realDefect(perMesh, "cyclicPeriodicAMI (tiled)");
        std::printf("        adjoint defect %.4e\n", (double)d);
        // KNOWN DEFECT, asserted in the direction it currently fails so the suite stays honest AND
        // green: the periodic AMI's two directions are not a transpose pair, so the pressure operator
        // is not self-adjoint and Jacobi-PCG cannot solve it. Measured ~1.1e-03 here against 4e-16 and
        // 4e-15 for the two ordinary cyclicAMI pairs above -- twelve orders of magnitude, so this is a
        // structural property and not round-off.
        //
        // It is the confirmed cause of pimpleFoam/RAS/oscillatingInletPeriodicAMI2D: with the matrix
        // inputs exact to 1e-9, PCG still ran to its 50-iteration cap with the residual GROWING
        // (1.00 -> 1.32) where OpenFOAM's GAMG converged in 12, and BiCGStab -- which tolerates
        // asymmetry -- turned that into a decrease.
        //
        // WHEN IT IS FIXED this leg will fail. That is the intent: flip it to `d < 1e-12` then, and the
        // two cyclicAMI legs above already show what the fixed number looks like.
        check(d > 1e-9, "a tiled cyclicPeriodicAMI is likewise not self-adjoint");
        check(d < 1e-1, "...and no worse than the ordinary non-conforming pair above");
    }

    // ---- Leg 5: THE PREDICATE THE SOLVER ACTUALLY BRANCHES ON --------------------------------------
    // Legs 1-4 establish which interfaces are symmetric. The fix turns that into a solver choice, and
    // it does so with one cheap test in the constructor: `nbrCell.size() > ownCell.size()`, i.e. some
    // face has more than one partner. This leg pins that the predicate agrees with the measurement --
    // false exactly where the defect is ~1e-16 and true exactly where it is not -- because the two
    // could drift apart silently, leaving a non-symmetric operator back on PCG.
    {
        auto nonConforming = [](const PrimitiveMesh& m)
        {
            FvGeometry g; g.build(m);
            const std::vector<FvPatch> fvp = buildPatches(m, g);
            bool nc = false;
            for (const AMIInterface& a : buildAMIInterfaces(m, g, fvp))
                if (a.nbrCell.size() > a.ownCell.size()) nc = true;
            return nc;
        };
        check(!nonConforming(amiMesh(base, 1.0)), "uniform pair: the predicate is false, so it stays on PCG");
        check(!nonConforming(amiMesh(base, 1.8)), "graded pair: still conforming, still PCG");
        check(nonConforming(ncMesh),   "non-conforming pair: the predicate fires, selecting BiCGStab");
        check(nonConforming(perMesh),  "tiled periodic pair: the predicate fires too");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
