// A cyclicACMI's non-overlap patch is a WALL only where it is closed.
//
// THE TRAP. cyclicACMI carries a coincident wall -- its nonOverlapPatch -- whose area is (1-mask)*A. On
// the COVERED part of the interface that is a patch of type `wall` with essentially zero area: a wall
// face with no wall behind it. brae counted every one of them, and paid for it twice:
//   - the near-wall epsilon (Cmu^0.75 k^1.5 / (kappa y)) was imposed on cells that are not near a wall
//   - deviceSolveScalarTransport zeroes the AMI interface off-diagonal for wall cells, because their
//     value is fixed -- so epsilon lost its interface coupling entirely
//
// WHAT OF DOES. cyclicACMIFvPatchField::manipulateMatrix does not touch the matrix itself; it re-directs
// to the non-overlap patch field with weights (1 - mask), and epsilonWallFunction acts only where that
// weight exceeds tolerance_ = 1e-5, blending rather than switching in between
// (epsilonWallFunction.C:586). The weight is the patch's areaFraction, which brae has as scaled/raw
// |Sf| -- and rawMagSf falls back to magSf, so it is 1 on every ordinary wall.
//
// Measured on pimpleFoam/RAS/oscillatingInletACMI2D, one step from OpenFOAM's own t=0.01: epsilon on the
// channel's interface column was 8.20x OpenFOAM's and the duct's covered band 4.09x, while the fully
// blocked cells were already right to 3.9e-06. That contrast is the whole diagnosis -- it was never the
// wall function, it was which faces count as wall.
//
// THE FIXTURE. acmi_mesh's two blocks with the blockage patches emitted (withBlockage), so the mask runs
// 0 / 0.5 / 1 / 1 and the blockage carries the complement. Leg 2 is the load-bearing one: the mask-0.5
// face must be neither counted nor skipped but WEIGHTED, and a fixture without one would pass for an
// implementation that just thresholded.
#include "acmi_mesh.cuh"
#include "acmi_area_scaling.cuh"
#include "device_kepsilon.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cyclic_field.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
}

int main()
{
    // Geometry + the ACMI area split, through the ordered entry point so the blockage areas are scaled.
    PrimitiveMesh m = acmitest::twoBlockACMI(acmitest::ACMI_DY, /*withBlockage*/true,
                                            "ACMI1_blockage", /*frontBackEmpty*/true);
    FvGeometry g;
    std::vector<FvPatch> fvp;
    std::vector<AMIInterface> amis;
    buildGeometryPatchesAndAMI(m, g, fvp, amis);

    // A velocity field is only needed for the wall-face bookkeeping; its values do not enter this test.
    GeometricField<vector> U =
        buildCyclicField<vector>(std::vector<vector>(m.nCells()), fvp, {}, /*wallNoSlip*/true);
    U.evaluateBoundary();

    const DeviceWallData w = buildDeviceWallData(m, g, fvp, U);
    const std::vector<scalar> wallW = w.wallW.host();
    const std::vector<label>  isW   = w.isWallCell.host();

    // Per wall face, check its cell against the weight it deserves. The rule is
    //     wallW[c] = max over the cell's wall faces of that face's areaFraction
    // which is what makes a cell touching BOTH a closed wall and a half-open blockage fully a wall cell --
    // the same answer OF reaches by overriding from `walls` first and then blending 0.5 onto a value that
    // is already the wall value.
    std::size_t nOpenOnly = 0, nBlendOnly = 0, nBlocked = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type != "wall") continue;
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label c = fvp[pi].faceCells[i];
            scalar want = 0;   // the rule, recomputed here from the geometry alone
            for (std::size_t pj = 0; pj < fvp.size(); ++pj)
            {
                if (fvp[pj].type != "wall") continue;
                for (label jj = 0; jj < fvp[pj].size; ++jj)
                {
                    if (fvp[pj].faceCells[jj] != c) continue;
                    const label f = fvp[pj].start + jj;
                    const scalar raw = g.rawMagSf(f);
                    const scalar frac = raw > scalar(0) ? g.magSf()[f]/raw : scalar(1);
                    if (frac > want) want = frac;
                }
            }
            if (std::fabs(wallW[c] - want) > scalar(1e-12))
            {
                std::printf("  FAIL cell %d: wallW %.6f, expected %.6f (max areaFraction over its wall faces)\n",
                            (int)c, (double)wallW[c], (double)want);
                ++failures;
            }
            const label wantIsW = want > scalar(1e-5) ? 1 : 0;
            if (isW[c] != wantIsW)
            {
                std::printf("  FAIL cell %d: isWallCell %d, expected %d -- an ACMI blockage face of area\n"
                            "       fraction %.3e is a wall face with no wall behind it\n",
                            (int)c, (int)isW[c], (int)wantIsW, (double)want);
                ++failures;
            }
            if (want <= scalar(1e-5)) ++nOpenOnly;
            else if (want < scalar(1) - scalar(1e-5)) ++nBlendOnly;
            else ++nBlocked;
        }
    }
    std::printf("  wall faces by their cell's weight: %zu fully open, %zu partial, %zu closed\n",
                nOpenOnly, nBlendOnly, nBlocked);

    // ---- vacuity guards: all three kinds must be present, or the legs above prove nothing ----
    if (!nOpenOnly)
    {
        std::printf("  FAIL vacuous: no cell whose ONLY wall face is an open blockage, so the case this\n"
                    "       test exists for never occurs -- without frontBackEmpty every cell touches the\n"
                    "       front/back walls and is a wall cell regardless\n");
        ++failures;
    }
    if (!nBlendOnly)
    {
        std::printf("  FAIL vacuous: no cell at a PARTIAL weight, so a plain 0/1 threshold would pass this\n"
                    "       test just as well as the weighted blend OF actually does\n");
        ++failures;
    }
    if (!nBlocked)
    {
        std::printf("  FAIL vacuous: no fully blocked face, so nothing here checks that a real wall is\n"
                    "       still treated as one\n");
        ++failures;
    }

    // ---- an ordinary wall on a mesh with no ACMI must be unaffected (weight 1 everywhere) ----
    {
        PrimitiveMesh m2 = acmitest::twoBlockACMI(acmitest::ACMI_DY, false, "ACMI1_blockage",
                                                 /*frontBackEmpty*/true);   // no blockage, no scaling
        FvGeometry g2;
        g2.build(m2);
        const std::vector<FvPatch> f2 = buildPatches(m2, g2);
        GeometricField<vector> U2 =
            buildCyclicField<vector>(std::vector<vector>(m2.nCells()), f2, {}, /*wallNoSlip*/true);
        U2.evaluateBoundary();
        const DeviceWallData w2 = buildDeviceWallData(m2, g2, f2, U2);
        const std::vector<scalar> ww2 = w2.wallW.host();
        const std::vector<label>  iw2 = w2.isWallCell.host();
        std::size_t nWallCells = 0;
        for (std::size_t c = 0; c < iw2.size(); ++c)
            if (iw2[c])
            {
                ++nWallCells;
                if (std::fabs(ww2[c] - scalar(1)) > scalar(1e-12))
                {
                    std::printf("  FAIL unscaled mesh: wall cell %zu has weight %.6f, expected exactly 1 --\n"
                                "       rawMagSf must fall back to magSf where nothing was scaled\n",
                                c, (double)ww2[c]);
                    ++failures;
                    break;
                }
            }
        std::printf("  no-ACMI control: %zu wall cells, all at weight 1\n", nWallCells);
        if (!nWallCells)
        { std::printf("  FAIL vacuous: the control mesh has no wall cells at all\n"); ++failures; }
    }

    std::printf("acmi_wall_gate: %d failures\n", failures);
    return failures ? 1 : 0;
}
