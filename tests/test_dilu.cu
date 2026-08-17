// DILU on the device must be OpenFOAM's DILU, not something with the same name.
//
// A preconditioner that is merely "similar" is not a smaller error, it is a different solver: the
// iterate after a loose relTol depends on WHICH modes the preconditioner left un-damped, which is the
// entire reason brae's Jacobi substitution broke LES/vortexShed. So the bar here is not "close" -- it is
// that the level-scheduled parallel sweep reproduces the sequential recurrence exactly.
//
// THE REFERENCE IS A TRANSCRIPTION, not a re-derivation. `ofDilu` below is OF's DILUPreconditioner.C
// written out line for line -- the same loops, the same order, the same divisions -- so a disagreement
// is brae's, not a difference of formulation. The comparison is exact equality (==), and it holds
// because the schedule preserves both the dependency order AND the per-cell summation order: each cell
// gathers its faces in increasing face index, which is the order losort visits them.
//
// Leg 3 is the one that would catch the tempting shortcut. Multi-colour or Chow-Patel ILU parallelise
// beautifully and produce a valid preconditioner that is NOT this one; they would pass a "does it
// converge" test and fail here, which is the point.
#include "box_mesh.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include "device_dilu.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
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

// OpenFOAM DILUPreconditioner.C, transcribed. lPtr = lowerAddr = owner, uPtr = upperAddr = neighbour.
void ofDilu(const std::vector<label>& owner, const std::vector<label>& nei,
            const std::vector<scalar>& diag, const std::vector<scalar>& upper,
            const std::vector<scalar>& lower, const std::vector<scalar>& rA,
            std::vector<scalar>& rD, std::vector<scalar>& wA)
{
    const std::size_t nCells = diag.size(), nFaces = nei.size();   // nei is internal-only; owner spans all faces
    rD = diag;
    for (std::size_t f = 0; f < nFaces; ++f)
        rD[(std::size_t)nei[f]] -= upper[f]*lower[f]/rD[(std::size_t)owner[f]];
    for (std::size_t c = 0; c < nCells; ++c) rD[c] = scalar(1)/rD[c];

    // losort: faces ordered by upper (neighbour) cell, ties by face index -- OF's losortAddr.
    std::vector<label> losort;
    {
        std::vector<std::vector<label>> byU(nCells);
        for (std::size_t f = 0; f < nFaces; ++f) byU[(std::size_t)nei[f]].push_back((label)f);
        for (const auto& v : byU) for (const label f : v) losort.push_back(f);
    }
    wA.assign(nCells, 0);
    for (std::size_t c = 0; c < nCells; ++c) wA[c] = rD[c]*rA[c];
    for (std::size_t f = 0; f < nFaces; ++f)
    {
        const label sf = losort[f];
        wA[(std::size_t)nei[sf]] -= rD[(std::size_t)nei[sf]]*lower[sf]*wA[(std::size_t)owner[sf]];
    }
    for (std::size_t i = nFaces; i-- > 0; )
        wA[(std::size_t)owner[i]] -= rD[(std::size_t)owner[i]]*upper[i]*wA[(std::size_t)nei[i]];
}
} // namespace

int main()
{
    std::printf("== DILU: the device sweep reproduces OpenFOAM's sequential recurrence ==\n");

    // A 3-D box, so the DAG has real depth and cells have several lower- and upper-neighbours. An
    // ASYMMETRIC matrix (upper != lower) is the case DILU exists for; a symmetric one would hide a
    // confusion between the two arrays.
    const PrimitiveMesh m = boxtest::boxMesh(9, 7, 5);
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells();
    const int nIf = (int)m.nInternalFaces();

    std::vector<scalar> hDiag((std::size_t)nC), hUp((std::size_t)nIf), hLo((std::size_t)nIf), hR((std::size_t)nC);
    for (label c = 0; c < nC; ++c) { hDiag[(std::size_t)c] = 12.0 + 0.5*std::sin(0.7*c); hR[(std::size_t)c] = std::cos(0.3*c) + 1.1; }
    for (int f = 0; f < nIf; ++f)
    {
        hUp[(std::size_t)f] = -(1.0 + 0.25*std::sin(0.9*f));      // asymmetric: upper != lower
        hLo[(std::size_t)f] = -(1.0 + 0.25*std::cos(0.4*f));
    }
    DeviceBuffer<scalar> D, U, L, r;
    D.copyFrom(hDiag); U.copyFrom(hUp); L.copyFrom(hLo); r.copyFrom(hR);
    const DeviceLduView A = deviceLduView(dm, D, U, L);

    DeviceDilu d = buildDeviceDilu(m.owner(), m.neighbour(), nC);
    diluUpdate(A, d);
    DeviceBuffer<scalar> w;
    diluApply(A, d, r, w);
    std::vector<scalar> gotRD, gotW;
    d.rD.copyTo(gotRD); w.copyTo(gotW);

    std::vector<scalar> refRD, refW;
    ofDilu(m.owner(), m.neighbour(), hDiag, hUp, hLo, hR, refRD, refW);

    // ---- Leg 1: the preconditioned diagonal ---------------------------------------------------------
    {
        scalar worst = 0; bool exact = true;
        for (label c = 0; c < nC; ++c)
        {
            worst = std::fmax(worst, std::fabs(gotRD[(std::size_t)c] - refRD[(std::size_t)c]));
            if (gotRD[(std::size_t)c] != refRD[(std::size_t)c]) exact = false;
        }
        check(exact, "rD is BIT-IDENTICAL to OF's calcReciprocalD");
        std::printf("        (worst |diff| %.3e over %d cells, %d levels)\n", (double)worst, (int)nC, d.levels());
    }

    // ---- Leg 2: the forward+backward sweep ----------------------------------------------------------
    {
        scalar worst = 0; bool exact = true;
        for (label c = 0; c < nC; ++c)
        {
            worst = std::fmax(worst, std::fabs(gotW[(std::size_t)c] - refW[(std::size_t)c]));
            if (gotW[(std::size_t)c] != refW[(std::size_t)c]) exact = false;
        }
        check(exact, "M^-1 r is BIT-IDENTICAL to OF's precondition()");
        std::printf("        (worst |diff| %.3e)\n", (double)worst);
    }

    // ---- Leg 3: it is not Jacobi (discrimination) ---------------------------------------------------
    // Without this, a diluApply that quietly fell back to w = r/diag would pass Legs 1-2 on any mesh
    // where the off-diagonals happened to be small. DILU differs from Jacobi by construction; assert it.
    {
        DeviceBuffer<scalar> j;
        deviceJacobi(j, r, A.diag);
        std::vector<scalar> hj; j.copyTo(hj);
        scalar rel = 0;
        for (label c = 0; c < nC; ++c)
            rel = std::fmax(rel, std::fabs(hj[(std::size_t)c] - gotW[(std::size_t)c])/std::fmax(std::fabs(gotW[(std::size_t)c]), scalar(1e-30)));
        check(rel > 0.05, "the result differs materially from Jacobi -- this is a real ILU, not a fallback");
        std::printf("        (max relative difference from Jacobi: %.3f)\n", (double)rel);
    }

    // ---- Leg 4: the schedule covers every cell exactly once ------------------------------------------
    // A dropped or duplicated cell would leave part of the field un-preconditioned, which Legs 1-2 would
    // catch only if the omission happened to change a value. Assert the partition directly.
    {
        std::vector<label> fh, bh;
        d.fwdCells.copyTo(fh); d.bwdCells.copyTo(bh);
        std::vector<int> seenF((std::size_t)nC, 0), seenB((std::size_t)nC, 0);
        for (const label c : fh) ++seenF[(std::size_t)c];
        for (const label c : bh) ++seenB[(std::size_t)c];
        bool ok = fh.size() == (std::size_t)nC && bh.size() == (std::size_t)nC;
        for (label c = 0; c < nC; ++c) if (seenF[(std::size_t)c] != 1 || seenB[(std::size_t)c] != 1) ok = false;
        check(ok, "both level schedules are a partition of the cells: every cell exactly once");
    }

    // ---- Leg 5: a second update() on a changed matrix is not stale ----------------------------------
    // rD is recomputed every solve because the momentum diagonal moves every outer corrector. A cached
    // rD would still pass everything above on the first call and be wrong from the second onwards.
    {
        for (label c = 0; c < nC; ++c) hDiag[(std::size_t)c] *= 1.7;
        D.copyFrom(hDiag);
        diluUpdate(A, d);
        diluApply(A, d, r, w);
        std::vector<scalar> gotW2; w.copyTo(gotW2);
        std::vector<scalar> refRD2, refW2;
        ofDilu(m.owner(), m.neighbour(), hDiag, hUp, hLo, hR, refRD2, refW2);
        bool exact = true;
        for (label c = 0; c < nC; ++c) if (gotW2[(std::size_t)c] != refW2[(std::size_t)c]) exact = false;
        check(exact, "update() re-reads the matrix: still exact after the diagonal changes");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
