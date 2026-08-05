#pragma once
// cf GPU offload, on-device boundary-condition evaluation. Encodes each boundary face's BC category
// (0=extrapolated, 1=fixedValue, 2=calculated) + its refValue / deltaCoeffs / |Sf| / faceCell, then
// computes on device: the boundary value() (from the internal field) and the matrix boundary coefficients
// (laplacian internalCoeffs/boundaryCoeffs from a diffusivity, div internalCoeffs/boundaryCoeffs from a
// face flux). This removes the host round-trip: the SIMPLE step's boundary coeffs are built on the GPU.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "fv_geometry.cuh"
#include "geometric_field.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace brae {

struct DeviceBoundary
{
    int n = 0;                              // total boundary faces (patch order = DeviceMesh bndCell order)
    DeviceBuffer<label>  bcType;            // 0 extrapolated, 1 fixedValue, 2 calculated (3=inletOutlet -> resolved
                                            // to 0|1 per face each step by deviceUpdateInletOutlet; 5=mixed/Robin;
                                            // 8=COUPLED (processor): zero matrix coeffs, value injected by the halo)
    DeviceBuffer<label>  ioMask;            // 1 if the face is inletOutlet (bcType recomputed from the flux sign)
    DeviceBuffer<label>  oioMask;           // 1 if the face is outletInlet (freestreamPressure): opposite flux switch
    DeviceBuffer<label>  mixedMask;         // 1 if the face is mixed/Robin (freestreamVelocity/Pressure): vf recomputed
    DeviceBuffer<label>  piovMask;          // 1 if the face is pressureInletOutletVelocity (bcType+refValue recomputed)
    DeviceBuffer<label>  symMask;            // 1 if the face is slip/symmetry (per-comp vf=|n_k|, ref recomputed each step)
    DeviceBuffer<label>  tpMask;             // 1 if the face is totalPressure (fixedValue-p, refValue recomputed each step)
    DeviceBuffer<scalar> valueFraction;     // per-face vf for mixed faces (deviceUpdateMixedFreestream); blends 0->1
    DeviceBuffer<scalar> refValue, p0, deltaCoeffs, magSf;   // p0 = totalPressure reference (constant; refValue = p0 - 0.5*neg(phi)|U|^2)
    DeviceBuffer<label>  faceCell;
};

inline DeviceBoundary buildDeviceBoundary(
    const GeometricField<scalar>& f,
    const std::vector<FvPatch>& fvp,
    const FvGeometry& g)
{
    std::vector<label> ty, fc, io, oio, mx, pv, sm, tp;
    std::vector<scalar> ref, dc, ms, vf, p0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;                     // cyclic = internal-like (handled by appended faces)
        // A processor patch is COUPLED: it stays in the boundary gather (the explicit operators need its bval
        // slot, filled with the halo-interpolated face value by DeviceHalo::scatterBoundaryValues), but must
        // contribute NO matrix coefficients -- its coupling is the interface off-diagonal. bcCategory() is NOT
        // overridden on ProcessorFvPatchField, so it would otherwise report 0 (= zeroGradient) whose
        // valueInternalCoeffs is 1, DOUBLE-COUNTING the interface diagonal.
        const int cat = (fvp[pi].type == "processor") ? 8 : f.boundary[pi]->bcCategory();
        const std::vector<scalar>& val = f.boundary[pi]->value();   // inletOutlet/outletInlet/mixed: value() = refValue; totalPressure: p0
        const std::vector<scalar>* vfp = f.boundary[pi]->valueFractionPtr();   // mixed (cat 5): per-face vf seed
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            // Categories whose VALUE is resolved per-step but whose TYPE is a plain fixedValue: inletOutlet,
            // outletInlet, totalPressure and flowRateInletVelocity(mass). They must map to 1 here -- pushing
            // the category through as a device bcType leaves an unknown type that no evaluator handles.
            ty.push_back((cat == 3 || cat == 4 || cat == 7 || cat == 9) ? 1 : cat);   // 5 stays mixed
            io.push_back(cat == 3 ? 1 : 0);
            oio.push_back(cat == 4 ? 1 : 0);
            mx.push_back(cat == 5 ? 1 : 0);
            pv.push_back(0);
            sm.push_back(0);
            tp.push_back(cat == 7 ? 1 : 0);
            p0.push_back(cat == 7 ? val[i] : 0.0);   // totalPressure: mask + the reference p0
            vf.push_back((cat == 5 && vfp) ? (*vfp)[i] : 0.0);
            ref.push_back(val[i]);
            dc.push_back(fvp[pi].deltaCoeffs[i]);
            ms.push_back(g.magSf()[fvp[pi].start + i]);
            fc.push_back(fvp[pi].faceCells[i]);
        }
    }
    DeviceBoundary db;
    db.n = static_cast<int>(ty.size());
    db.bcType.copyFrom(ty);
    db.ioMask.copyFrom(io);
    db.oioMask.copyFrom(oio);
    db.mixedMask.copyFrom(mx);
    db.piovMask.copyFrom(pv);
    db.symMask.copyFrom(sm);
    db.tpMask.copyFrom(tp);
    db.p0.copyFrom(p0);
    db.valueFraction.copyFrom(vf);
    db.refValue.copyFrom(ref);
    db.deltaCoeffs.copyFrom(dc);
    db.magSf.copyFrom(ms);
    db.faceCell.copyFrom(fc);
    return db;
}

// inletOutlet: recompute the effective per-face bcType from the patch flux sign (OF valueFraction = neg(phi)):
// inflow phiBnd<0 -> fixedValue(1, =inletValue), outflow phiBnd>=0 -> zeroGradient(0). Only ioMask=1 faces change.
// Call at the start of each SIMPLE step (before assembly), with the previous step's boundary flux (matches OF).
void deviceUpdateInletOutlet(DeviceBoundary& db, const DeviceBuffer<scalar>& phiBnd);
void deviceBCValue(const DeviceBoundary& db, const DeviceBuffer<scalar>& internal, DeviceBuffer<scalar>& value);
void deviceBCLaplacianCoeffs(const DeviceBoundary& db, const DeviceBuffer<scalar>& gammaCell,
                             DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC);
// variant taking gamma per BOUNDARY FACE (for nuEff with the true wall nut, not the cell value).
void deviceBCLaplacianCoeffsFace(const DeviceBoundary& db, const DeviceBuffer<scalar>& gammaFace,
                                 DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC);
void deviceBCDivCoeffs(const DeviceBoundary& db, const DeviceBuffer<scalar>& phiB,
                       DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC);
// pEqn.flux() at boundary faces = internalCoeffs*p[faceCell] - boundaryCoeffs.
void deviceMatrixFluxBoundary(const DeviceBoundary& db, const DeviceBuffer<scalar>& iC, const DeviceBuffer<scalar>& bC,
                              const DeviceBuffer<scalar>& p, DeviceBuffer<scalar>& fluxB);

// A vector field's BC is three scalar boundaries (one per component): the laplacian/div internalCoeffs are
// isotropic (component-independent), only the refValue-dependent boundaryCoeffs / value differ by component.
// So the per-component momentum boundary coeffs reuse the scalar kernels on comp[k].
struct DeviceVectorBoundary
{
    DeviceBoundary comp[3];
    int n = 0;
    DeviceBuffer<scalar> nx, ny, nz;   // nx/ny/nz = unit face normal (pressureInletOutletVelocity)
};

// inletOutlet on a vector field: same patch flux for all 3 components -> update each component's bcType.
inline void deviceUpdateInletOutlet(DeviceVectorBoundary& db, const DeviceBuffer<scalar>& phiBnd)
{
    for (int k = 0; k < 3; ++k)
        deviceUpdateInletOutlet(db.comp[k], phiBnd);
}

// mixed (Robin) freestreamVelocity/Pressure updateCoeffs: recompute per-face vf from the flow angle. The normal
// flux U.n = phi_b/|Sf| is exact (lagged like OF + the inletOutlet switch); |U| is the local adjacent-cell speed.
// Writes vf to the 3 U components (velocity sign 0.5-0.5c) and to p (pressure sign 0.5+0.5c); non-mixed faces
// untouched. Call at the start of each step (before assembly), after the io switch.
void deviceUpdateMixedFreestream(DeviceVectorBoundary& dbU, DeviceBoundary& dbP, const DeviceBuffer<scalar>& phiBnd,
                                 const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                                 const DeviceBuffer<scalar>* rhoBnd = nullptr);   // compressible: rho at boundary faces
// constrainHbyA at mixed velocity faces: OF resets phiHbyA_b = U_b.Sf at fixesValue patches (mixed fixesValue=true).
// cf's HbyA boundary value uses HbyA_cell in the (1-vf) part; at mixed faces replace hb[k] with the boundary value
// of U itself (U_b = (1-vf) U_cell + vf U_freestream) so the boundary flux uses U_b, not HbyA_b. Non-mixed faces
// untouched (pure fixedValue already = U_b; zeroGradient is not fixesValue).
void deviceConstrainMixedHbyA(const DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Ux,
                              const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                              DeviceBuffer<scalar>& hbx, DeviceBuffer<scalar>& hby, DeviceBuffer<scalar>& hbz);
// constrainHbyA at slip/symmetry faces: HbyA_b = HbyA_c - n(n.HbyA_c) so the wall flux is exactly 0 (no penetration).
void deviceConstrainSymmetryHbyA(const DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Hx,
                                 const DeviceBuffer<scalar>& Hy, const DeviceBuffer<scalar>& Hz,
                                 DeviceBuffer<scalar>& hbx, DeviceBuffer<scalar>& hby, DeviceBuffer<scalar>& hbz);

// pressureInletOutletVelocity updateCoeffs (directionMixed, valueFraction = neg(phi)*(I - n n)): per piov face by the
// lagged boundary flux sign, set each component's bcType + refValue. Outflow (phi>=0) -> zeroGradient; inflow
// (phi<0) -> fixedValue with refValue = n*(n.U_cell) (tangential 0). Call each step before assembly (after the io
// switch), with the previous step's boundary flux + cell velocity (matches OF lagging).
void deviceUpdatePressureInletOutletVelocity(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& phiBnd,
                                             const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                                             const DeviceBuffer<scalar>& Uz);

// slip/symmetry updateCoeffs (OF basicSymmetry, GENERAL non-axis-aligned): reuses the mixed (cat 5) kernels with a
// PER-COMPONENT valueFraction vf_k = |n_k| and ref_k = U_c[k] - sign(n_k)*(n.U_c). Then valueIC_k = 1-|n_k|,
// gradIC_k = -dc*|n_k|, value = U_c - n(n.U_c), identical to OF's snGradTransformDiag=(|nx|,|ny|,|nz|) coeffs. Call
// each step before assembly with the previous step's cell velocity (the cross-component n.U is lagged, as in a
// segregated solve). Reduces bit-exactly to fixedValue-0(normal)+zeroGradient(tangential) for an axis-aligned plane.
void deviceUpdateSymmetry(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                          const DeviceBuffer<scalar>& Uz);

// totalPressure updateCoeffs (OF totalPressureFvPatchScalarField, incompressible rho=none/psi=none): per face
// refValue = p0 - 0.5*neg(phi_b)*magSqr(U_b)  (inflow phi<0 -> p0 - dynamic head; outflow -> p0). bcType stays
// fixedValue(1). Call each step AFTER the U-boundary updates, with the boundary U (deviceBCValue of current U) + flux.
// flowRateInletVelocity (massFlowRate): set U_b = avgU*n on the masked patch faces. See the .cu.
void deviceUpdateFlowRateInlet(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& maskMagSf, scalar avgU,
                               const DeviceBuffer<scalar>& nx, const DeviceBuffer<scalar>& ny,
                               const DeviceBuffer<scalar>& nz);

void deviceUpdateTotalPressure(DeviceBoundary& db, const DeviceBuffer<scalar>& phiB, const DeviceBuffer<scalar>& Uxb,
                               const DeviceBuffer<scalar>& Uyb, const DeviceBuffer<scalar>& Uzb,
                               const DeviceBuffer<scalar>* rhoBnd = nullptr);   // compressible: rho at the face
                                                                                // (p in Pa -> 0.5*rho*|U|^2)

inline DeviceVectorBoundary buildDeviceVectorBoundary(
    const GeometricField<vector>& f,
    const std::vector<FvPatch>& fvp,
    const FvGeometry& g)
{
    std::vector<label> ty[3], fc, io, oio, mx, pv, sm;
    std::vector<scalar> dc, ms, ref[3], vf[3], nrm[3];
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;                     // cyclic = internal-like (handled by appended faces)
        // A processor patch is COUPLED: it stays in the boundary gather (the explicit operators need its bval
        // slot, filled with the halo-interpolated face value by DeviceHalo::scatterBoundaryValues), but must
        // contribute NO matrix coefficients -- its coupling is the interface off-diagonal. bcCategory() is NOT
        // overridden on ProcessorFvPatchField, so it would otherwise report 0 (= zeroGradient) whose
        // valueInternalCoeffs is 1, DOUBLE-COUNTING the interface diagonal.
        const int cat = (fvp[pi].type == "processor") ? 8 : f.boundary[pi]->bcCategory();
        const bool sym = f.boundary[pi]->isSymmetry();
        const std::vector<vector>& val = f.boundary[pi]->value();   // inletOutlet/mixed: value() = freestreamValue (= refValue)
        const std::vector<scalar>* vfp = f.boundary[pi]->valueFractionPtr();   // mixed (cat 5): per-face vf seed
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            fc.push_back(fvp[pi].faceCells[i]);
            dc.push_back(fvp[pi].deltaCoeffs[i]);
            ms.push_back(g.magSf()[fvp[pi].start + i]);
            io.push_back(cat == 3 ? 1 : 0);
            oio.push_back(cat == 4 ? 1 : 0);   // inletOutlet / outletInlet (same flux for all 3 comps)
            mx.push_back(cat == 5 ? 1 : 0);
            pv.push_back(cat == 6 ? 1 : 0);
            sm.push_back(sym ? 1 : 0);   // mixed / piov / symmetry masks
            const scalar seedVf = (cat == 5 && vfp) ? (*vfp)[i] : 0.0;
            {
                const vector Sf = g.Sf()[fvp[pi].start + i];
                const scalar mg = g.magSf()[fvp[pi].start + i];   // unit face normal
                const scalar inv = mg > 0 ? 1.0 / mg : 0.0;
                nrm[0].push_back(Sf.x * inv);
                nrm[1].push_back(Sf.y * inv);
                nrm[2].push_back(Sf.z * inv);
            }
            const scalar rv[3] = { val[i].x, val[i].y, val[i].z };
            if (sym)   // mixed kernels; vf_k=|n_k|, ref recomputed per step (init = host value v - n(n.v))
            {
                for (int k = 0; k < 3; ++k)
                {
                    ty[k].push_back(5);
                    vf[k].push_back(std::fabs(nrm[k].back()));
                    ref[k].push_back(rv[k]);
                }
            }
            else
            {
                for (int k = 0; k < 3; ++k)
                {
                    // io/oio/flowRateInlet init fixedValue (the device rewrites refValue per step); piov
                    // zeroGradient init. Anything whose VALUE is resolved per-step but whose TYPE is a
                    // plain fixedValue MUST map to 1 -- pushing the category through leaves an unknown
                    // device bcType that no evaluator handles, and the patch then behaves arbitrarily.
                    ty[k].push_back((cat == 3 || cat == 4 || cat == 9) ? 1 : (cat == 6 ? 0 : cat));
                    vf[k].push_back(seedVf);
                    ref[k].push_back(rv[k]);
                }
            }
        }
    }
    DeviceVectorBoundary db;
    db.n = static_cast<int>(fc.size());
    db.nx.copyFrom(nrm[0]);
    db.ny.copyFrom(nrm[1]);
    db.nz.copyFrom(nrm[2]);
    for (int k = 0; k < 3; ++k)
    {
        db.comp[k].n = db.n;
        db.comp[k].bcType.copyFrom(ty[k]);
        db.comp[k].ioMask.copyFrom(io);
        db.comp[k].oioMask.copyFrom(oio);
        db.comp[k].mixedMask.copyFrom(mx);
        db.comp[k].piovMask.copyFrom(pv);
        db.comp[k].symMask.copyFrom(sm);
        db.comp[k].valueFraction.copyFrom(vf[k]);
        db.comp[k].refValue.copyFrom(ref[k]);
        db.comp[k].deltaCoeffs.copyFrom(dc);
        db.comp[k].magSf.copyFrom(ms);
        db.comp[k].faceCell.copyFrom(fc);
    }
    return db;
}

} // namespace brae
