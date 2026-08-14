// cyclicACMI mask: the overlap fraction that splits each face between the coupled patch and its wall.
//
// WHAT THE MASK IS -- OF cyclicACMIPolyPatch.C:348:
//
//     srcMask_ = clamp(AMI.srcWeightsSum(), zero_one{});
//
// i.e. the AMI coverage, clamped. It then decides how the face area is shared with the coincident
// nonOverlapPatch wall (cyclicACMIPolyPatch.C:226,252), so a face that has slid off its neighbour stops
// being coupled and becomes solid instead of leaking.
//
// WHY THIS TEST EXISTS, and what it is really guarding. brae infers a translational period for a
// cyclic interface from the difference of the two patch centroids. That is right for a PERIODIC
// cyclicAMI -- the centroid difference IS the period. It is catastrophic for ACMI, whose patches are
// CO-LOCATED and whose centroid difference is the physical slide of the moving zone. Inferring a period
// there re-aligns the patches, makes the mask identically 1 for all time, gives the blockage wall zero
// area, and lets the sliding channel never close -- a case that runs clean and is entirely wrong.
//
// Measured on pimpleFoam/RAS/oscillatingInletACMI2D at t = 0.5, where the inlet channel has slid 0.5:
//
//     inferred separation (0 -0.5 0) ->  0 uncovered,  0 blended, 40 covered
//     separation = 0                 -> 19 uncovered,  1 blended, 20 covered
//     OF's own ACMI report           -> 19, 1, 20
//
// and brae now reproduces OF exactly at t = 0.1/0.25/0.3/0.4/0.5/0.6.
//
// THE FIXTURE is that situation in miniature, with an EXACT answer instead of a reference log. Two
// blocks meet at x = 1; the second is offset in y by 0.375, so the interface only partly overlaps:
//
//     src faces (y):  [0,.25]  [.25,.5]  [.5,.75]  [.75,1]
//     tgt faces (y):           [.375,.625]  [.625,.875]  [.875,1.125]  [1.125,1.375]
//     overlap:          none     .125/.25     full        full
//     mask:              0         0.5          1           1
//
// Every value is a hand-computable fraction, not a tolerance: one uncovered face, one genuinely BLENDED
// face, two covered. The blended face is the point -- a fixture whose faces were all 0 or 1 would pass
// for an implementation that rounded the mask, and would not notice a partial overlap being mishandled.
//
// The offset is also what makes the leg-3 guard meaningful: the two patch centroids differ by exactly
// 0.375 in y, so if the ACMI transform were ever inferred again the mask would collapse to all-ones and
// leg 2 would go red. Leg 3 asserts that trap is armed.
//
// SCOPE: this covers the mask only. The nonOverlapPatch wall carries no faces here because nothing
// consumes them yet -- area scaling is the next step, and a fixture with duplicated faces would have
// wrong cell volumes until that scaling exists (which is precisely why OF scales them:
// "to avoid double-accounting of face areas", cyclicACMIPolyPatch.C:194).
#include "acmi_mesh.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ami_interface.cuh"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void checkMask(const AMIInterface& A, const std::string& nm, const std::vector<scalar>& want)
{
    if (A.mask.size() != want.size())
    {
        std::printf("  FAIL %s: mask has %zu entries, expected %zu\n", nm.c_str(), A.mask.size(), want.size());
        ++failures;
        return;
    }
    std::printf("  %-13s mask =", nm.c_str());
    for (const scalar w : A.mask) std::printf(" %.6f", w);
    std::printf("   (expected");
    for (const scalar w : want) std::printf(" %.3f", w);
    std::printf(")\n");
    for (std::size_t i = 0; i < want.size(); ++i)
        if (std::fabs(A.mask[i] - want[i]) > 1e-12)
        {
            std::printf("  FAIL %s face %zu: mask %.12f, expected %.12f\n", nm.c_str(), i, A.mask[i], want[i]);
            ++failures;
        }
}

}   // namespace

int main()
{
    PrimitiveMesh m = acmitest::twoBlockACMI(acmitest::ACMI_DY);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);

    // -------------------------------------------------------------------------------------------
    // 1. cyclicACMI is BUILT (it used to be skipped outright), flagged, and carries no transform.
    if (amis.size() != 2)
    {
        std::printf("  FAIL built %zu AMI interfaces from 2 cyclicACMI patches\n", amis.size());
        std::printf("test_acmi_mask: %d failures\n", ++failures);
        return 1;
    }
    for (const AMIInterface& A : amis)
    {
        const std::string nm = fvp[A.patch].name;
        if (!A.acmi)
        { std::printf("  FAIL %s: built but not flagged as ACMI\n", nm.c_str()); ++failures; }
        if (mag(A.separation) > 1e-14)
        {
            std::printf("  FAIL %s: ACMI carries separation (%.6f %.6f %.6f). The patches are CO-LOCATED;\n"
                        "       any centroid offset is the zone's physical slide, not a period. Subtracting\n"
                        "       it re-aligns them and pins the mask at 1 for all time.\n",
                        nm.c_str(), A.separation.x, A.separation.y, A.separation.z);
            ++failures;
        }
    }

    // -------------------------------------------------------------------------------------------
    // 2. THE MASK, against hand-computed fractions. The 0.5 entries are the load-bearing ones: a
    // partial overlap that is neither fully coupled nor fully blocked.
    for (const AMIInterface& A : amis)
    {
        const std::string nm = fvp[A.patch].name;
        if (nm == "ACMI1_couple") checkMask(A, nm, {0.0, 0.5, 1.0, 1.0});
        if (nm == "ACMI2_couple") checkMask(A, nm, {1.0, 1.0, 0.5, 0.0});
    }

    // -------------------------------------------------------------------------------------------
    // 3. THE TRAP IS ARMED. The whole test only means something because the two patch centroids
    // genuinely differ -- that difference is what the old code would have subtracted. If a refactor
    // ever made the fixture centroid-aligned, leg 2 would pass under the broken behaviour too.
    {
        const FvPatch& S = fvp[amis[0].patch];
        const FvPatch& T = fvp[amis[0].nbrPatch];
        vector cS{0,0,0}, cT{0,0,0};
        for (label i = 0; i < S.size; ++i) cS += g.Cf()[S.start+i];
        for (label j = 0; j < T.size; ++j) cT += g.Cf()[T.start+j];
        const vector inferred = cT/(scalar)T.size - cS/(scalar)S.size;
        std::printf("  trap: centroid offset the old code would infer = (%.4f %.4f %.4f)\n",
                    inferred.x, inferred.y, inferred.z);
        if (mag(inferred) < 1e-6)
        {
            std::printf("  FAIL vacuous: the two patches are centroid-aligned, so inferring a period would\n"
                        "       be harmless here and leg 2 cannot detect the defect it exists for\n");
            ++failures;
        }
    }

    // -------------------------------------------------------------------------------------------
    // 4. PARTIAL COVERAGE IS NOT REFUSED. cyclicAMI refuses a mean coverage below 0.9 (a leaking
    // interface). Here the mean is 0.625 BY CONSTRUCTION and must be accepted: that is what ACMI is.
    // Reaching this line at all means no exception was thrown; assert the numbers really are low
    // enough that the AMI check would have fired.
    {
        scalar sum = 0;
        std::size_t n = 0;
        for (const scalar w : amis[0].mask) { sum += w; ++n; }
        const scalar mean = sum/scalar(n);
        std::printf("  refusal exemption: mean coverage %.4f accepted (cyclicAMI would refuse below 0.9)\n", mean);
        if (mean >= scalar(0.9))
        {
            std::printf("  FAIL vacuous: mean coverage %.4f is above the cyclicAMI refusal threshold, so this\n"
                        "       fixture never exercises the ACMI exemption\n", mean);
            ++failures;
        }
    }

    // -------------------------------------------------------------------------------------------
    // 5. THE COVERAGE IS APPLIED ONCE, NOT TWICE. For cyclicACMI the overlap fraction ends up in the
    // face AREA (leg 2's mask, consumed by test_acmi_area_scaling). OF therefore takes it back OUT of
    // the interpolation weights, as the last act of cyclicACMIPolyPatch::scalePatchFaceAreas:
    //
    //     "Re-normalise the weights since the effect of overlap is already accounted for in the area"
    //         for (scalar& w : wghts) { w /= sum; }   sum = 1.0;
    //
    // brae kept the weights on AMIInterpolation's non-conformal branch (summing to the coverage) AND
    // scaled the area, so a partially covered face transmitted mask^2 of its flux. On
    // pimpleFoam/RAS/oscillatingInletACMI2D that was 2 faces of 136, and it still cost 0.9% of the
    // interface mass flow, a pressure ramp of 7 down the duct and ~10% velocity error at the interface.
    //
    // THE BLENDED FACE IS THE WHOLE TEST. A fully covered face already sums to 1 and an uncovered one
    // has no stencil, so on a fixture without a genuine 0 < mask < 1 face BOTH conventions agree and
    // this leg is vacuous -- which is why it asserts the blend exists before asserting anything else.
    // The pairing with leg 2 is the point: the mask must STILL read 0.5 (the coverage survives, in the
    // area) while the weights read 1 (it is not there a second time).
    {
        std::size_t blended = 0;
        for (const AMIInterface& A : amis)
        {
            const std::string nm = fvp[A.patch].name;
            for (std::size_t i = 0; i + 1 < A.srcOffset.size(); ++i)
            {
                const label b = A.srcOffset[i], e = A.srcOffset[i+1];
                if (e <= b)   // uncovered: no stencil, nothing to normalise, weightsSum stays 0
                {
                    if (A.weightsSum[i] != scalar(0))
                    {
                        std::printf("  FAIL %s face %zu: empty stencil but weightsSum %.12f, expected 0 --\n"
                                    "       an uncovered face must not be handed a 1 it has nothing to read with\n",
                                    nm.c_str(), i, A.weightsSum[i]);
                        ++failures;
                    }
                    continue;
                }
                if (A.mask[i] > scalar(0) && A.mask[i] < scalar(1)) ++blended;
                scalar s = 0;
                for (label k = b; k < e; ++k) s += A.weight[k];
                std::printf("  %-13s face %zu: mask %.4f  sum(weights) %.12f  weightsSum %.12f\n",
                            nm.c_str(), i, A.mask[i], s, A.weightsSum[i]);
                if (std::fabs(s - scalar(1)) > 1e-12)
                {
                    std::printf("  FAIL %s face %zu: weights sum to %.12f, expected 1. The coverage %.4f is\n"
                                "       already in the face area; keeping it here too transmits mask^2.\n",
                                nm.c_str(), i, s, A.mask[i]);
                    ++failures;
                }
                if (std::fabs(A.weightsSum[i] - scalar(1)) > 1e-12)
                {
                    std::printf("  FAIL %s face %zu: weightsSum %.12f, expected 1 (OF sets sum = 1.0)\n",
                                nm.c_str(), i, A.weightsSum[i]);
                    ++failures;
                }
            }
        }
        if (!blended)
        {
            std::printf("  FAIL vacuous: no face has 0 < mask < 1, so both the right and the wrong\n"
                        "       normalisation give sum(weights) = 1 and this leg proves nothing\n");
            ++failures;
        }
    }

    std::printf("test_acmi_mask: %d failures\n", failures);
    return failures ? 1 : 0;
}
