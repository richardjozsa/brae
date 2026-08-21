#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's cyclicAMI coupling as the finite-volume layer uses it.
//
// provenance:
//   openfoam:
//     files:   src/finiteVolume/fields/fvPatchFields/constraint/cyclicAMI/cyclicAMIFvPatchField.C
//              src/meshTools/AMIInterpolation/AMIInterpolation/AMIInterpolation.C
//     symbols: cyclicAMIFvPatchField::updateInterfaceMatrix / patchNeighbourField,
//              AMIInterpolation::interpolateToSource
//   brae:
//     geometry:  src/OpenFOAM/meshes/interface/ami_interface.cuh  (AMIInterface: weights, addressing,
//                deltaCoeffs, corrVec, forwardT -- already validated by ami_weights/ami_geometry)
//     reference: src/finiteVolume/fvMatrices/cyclicAMI/cyclicAMI_cpp.cu
//     cuda:      src/cuda/device_ami.cu
//     tests:     tests/ami_cpp_vs_device.cu
//
// WHY THIS EXISTS. brae's device AMI is one fused path, and every attempt to explain pipeCyclic's
// disagreement with OpenFOAM has run into the same wall: a single number for the whole interface cannot
// say WHICH stage is wrong. Every other defect in this port fell out quickly once a _cpp reference
// existed to compare against stage by stage -- the pitzDaily inlet diffusivity, the Spalding wall seed,
// turbineSiting's profile origin, the LM lambda loop. The AMI had no such reference, so its residual
// (97% of pipeCyclic's momentum residual sits on interface cells, with the interior exactly zero) has
// stayed unexplained through four passes of reading the device code.
//
// THE STAGES ARE THE DEVICE'S STAGES, deliberately: each function below is the host twin of one device
// entry point, takes the same inputs and returns the same outputs, so a disagreement names a kernel.
//
//   interpolate      <-> deviceAmiInterpolate      out[i]   = sum_k w[k]*psi[nbr[k]]            (no transform)
//   interpolateVec   <-> deviceAmiInterpolateVec   out[i]   = sum_k w[k]*(forwardT & U[nbr[k]])
//   faceValue        <-> deviceAmiFaceValue        out[i]   = w_i*psi[own] + (1-w_i)*interp[i]
//   assembleMomentum <-> deviceAmiAssembleMomentum ifCoeff  = -lap + min(phi,0);  diag += lap + max(phi,0)
//                                                  lap      = (w*nuEff[own] + (1-w)*interp(nuEff))*dc*magSf
//   assembleLaplacian<-> deviceAmiAssembleLaplacian ifCoeff = +c;                  diag -= c
//                                                  (the OPPOSITE sign to momentum: the pressure equation
//                                                   carries +laplacian(rAUf,p), momentum -laplacian(nuEff,U))
//   amul             <-> deviceAmiAmul             Apsi[own] += ifCoeff[i]*interp(psi)[i]
//   addH             <-> deviceAmiAddH             H[own]    -= ifCoeff[i]*UN[i]/V[own]
//   flux             <-> deviceAmiFlux             phi[i]    = faceValue(HbyA) . Sf[i]
//   correctFlux      <-> deviceAmiCorrectFlux      phi[i]   -= ifCoeff[i]*(interp(p)[i] - p[own])
//   addDiv           <-> deviceAmiAddDiv           div[own] += phi[i]/V[own]
//
// THE TRANSFORM IS APPLIED BEFORE THE WEIGHTED SUM, not after -- forwardT is constant per interface so
// the two agree, but the device does it that way and a reference that reorders them stops being a
// reference for the thing it is checking.
//
// ONE DIRECTION PER ENTRY. An interface entry couples own <- nbr and nothing else; the reverse coupling
// is a separate entry on the paired patch. That is why amul ADDS to Apsi[own] only, and why the AMG
// agglomeration gives each appended edge upper = ifCoeff with lower = zero.
#include "cf_types.cuh"
#include "ami_interface.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"

#include <vector>

namespace brae {
namespace cpu {
namespace cyclicAMI {

// Per-interface working state: the assembled off-diagonal and face flux, mirroring DeviceAMI's
// ifCoeff/phi so a comparison is field-for-field.
struct State
{
    std::vector<scalar> ifCoeff;   // per src face
    std::vector<scalar> phi;       // per src face
};

std::vector<scalar> interpolate(const AMIInterface& a, const std::vector<scalar>& psi);

std::vector<vector> interpolateVec(const AMIInterface& a, const std::vector<vector>& U);

std::vector<scalar> faceValue(const AMIInterface& a, const std::vector<scalar>& cell);

// OF's momentum interface: the laplacian face coefficient plus the upwind convective split.
void assembleMomentum(const AMIInterface&        a,
                      const std::vector<scalar>& nuEffCell,
                      const std::vector<scalar>& phi,        // the interface flux, per src face
                      State&                     st,
                      std::vector<scalar>&       diag);

// The pressure (laplacian) interface: no convection.
void assembleLaplacian(const AMIInterface&        a,
                       const std::vector<scalar>& gammaCell,
                       State&                     st,
                       std::vector<scalar>&       diag,
                       bool                       addToDiag = true);

// The matrix action: OF cyclicAMIFvPatchField::updateInterfaceMatrix.
void amul(const AMIInterface&        a,
          const State&               st,
          const std::vector<scalar>& psi,
          std::vector<scalar>&       Apsi);

// UEqn.H(): the interface's explicit contribution, per component of an already-interpolated neighbour.
void addH(const AMIInterface&        a,
          const State&               st,
          const std::vector<scalar>& UN,      // interpolateVec's component for this comp
          const std::vector<scalar>& V,
          std::vector<scalar>&       H);

// phiHbyA on the interface faces.
void flux(const AMIInterface&        a,
          const std::vector<vector>& HbyA,
          State&                     st);

// phi -= pEqn.flux() across the interface.
void correctFlux(const AMIInterface&        a,
                 const State&               st,
                 const std::vector<scalar>& p,
                 State&                     phiOut);

// continuity: the interface's share of div(phi).
void addDiv(const AMIInterface&        a,
            const State&               st,
            const std::vector<scalar>& V,
            std::vector<scalar>&       div);

} // namespace cyclicAMI
} // namespace cpu
} // namespace brae
