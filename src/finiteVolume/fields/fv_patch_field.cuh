#pragma once
// brae::fvPatchField<T>, boundary-condition base + the core concrete types, mirroring
// OpenFOAM fvPatchField. evaluate() sets the boundary face VALUES from the internal field;
// snGrad() is the surface-normal gradient. Matrix-coupling coefficients (valueInternalCoeffs
// etc.) come with Phase 3 (assembly). Unsupported types throw, no silent fallback.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "cf_pstream.cuh"
#include <memory>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <cmath>

namespace brae {

template <typename T>
class fvPatchField
{
public:
    explicit fvPatchField(const FvPatch& p) : patch_(p), value_(p.size) {}
    virtual ~fvPatchField() = default;

    // Coupled (processor) patches post their exchange here; others no-op. The owning field
    // calls initEvaluate on all patches, then Pstream::waitAll(), then evaluate on all.
    virtual void initEvaluate(const std::vector<T>& /*internal*/) {}
    virtual void evaluate(const std::vector<T>& internal) = 0;
    virtual bool fixesValue() const = 0;

    // Laplacian/gradient matrix coupling: snGrad = gradientInternalCoeffs * phi_P
    // + gradientBoundaryCoeffs. Default (zeroGradient/empty/calculated) contributes nothing.
    virtual std::vector<T> gradientInternalCoeffs() const { return std::vector<T>(patch_.size, T{}); }
    virtual std::vector<T> gradientBoundaryCoeffs() const { return std::vector<T>(patch_.size, T{}); }

    // BC category for the on-device evaluator: 0 = extrapolated (zeroGradient/empty: value tracks the
    // internal cell, valueIC=1, gradIC=0); 1 = fixedValue (fixedValue/noSlip: value=ref, valueIC=0,
    // gradIC=-deltaCoeffs); 2 = calculated (value=ref but extrapolated coeffs).
    virtual int bcCategory() const { return 0; }

    // symmetryPlane/symmetry needs PER-COMPONENT device categories for vectors (the wall-normal component
    // is reflected = fixedValue 0, the tangential components are zeroGradient). The device builder keys on
    // this flag + the face normal; scalars are plain zeroGradient so it is a no-op for them.
    virtual bool isSymmetry() const { return false; }

    // Convection/value matrix coupling: faceValue = valueInternalCoeffs * phi_P
    // + valueBoundaryCoeffs. Default (zeroGradient/empty/calculated) = (1, 0).
    virtual std::vector<T> valueInternalCoeffs() const { return std::vector<T>(patch_.size, tUniform<T>(1)); }
    virtual std::vector<T> valueBoundaryCoeffs() const { return std::vector<T>(patch_.size, T{}); }

    // Mixed (Robin) BC support for the device builder (bcCategory 5): the per-face valueFraction (null = not mixed)
    // and the freestream sign (true = velocity vf=0.5-0.5 U.n/|U|, false = pressure vf=0.5+0.5 U.n/|U|).
    virtual const std::vector<scalar>* valueFractionPtr() const { return nullptr; }
    virtual bool mixedVelocitySign() const { return true; }

    const std::vector<T>& value() const { return value_; }
    void setValue(const std::vector<T>& v) { value_ = v; }   // e.g. nutkWallFunction writing nut at walls
    // Coupled-patch neighbour (halo) values for the matrix; non-coupled patches return value().
    virtual const std::vector<T>& patchNeighbourField() const { return value_; }
    const FvPatch&        patch() const { return patch_; }

    std::vector<T> patchInternalField(const std::vector<T>& internal) const
    {
        std::vector<T> pif(patch_.size);
        for (label i = 0; i < patch_.size; ++i)
            pif[i] = internal[patch_.faceCells[i]];
        return pif;
    }

protected:
    const FvPatch& patch_;
    std::vector<T> value_;
};

// fixedValue: value is prescribed (uniform or per-face).
template <typename T>
class FixedValuePatchField : public fvPatchField<T>
{
public:
    FixedValuePatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals)
        : fvPatchField<T>(p), uniform_(uniform), uniformValue_(uval), values_(std::move(vals)) {}
    void evaluate(const std::vector<T>&) override
    {
        for (label i = 0; i < this->patch_.size; ++i)
            this->value_[i] = uniform_ ? uniformValue_ : values_[i];
    }
    bool fixesValue() const override { return true; }
    int  bcCategory() const override { return 1; }

    std::vector<T> gradientInternalCoeffs() const override        // -deltaCoeffs
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = tUniform<T>(-this->patch_.deltaCoeffs[i]);
        return r;
    }
    std::vector<T> gradientBoundaryCoeffs() const override        // deltaCoeffs * refValue
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = (uniform_ ? uniformValue_ : values_[i]) * this->patch_.deltaCoeffs[i];
        return r;
    }
    std::vector<T> valueInternalCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); } // 0
    std::vector<T> valueBoundaryCoeffs() const override            // the fixed value
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = uniform_ ? uniformValue_ : values_[i];
        return r;
    }
private:
    bool           uniform_;
    T              uniformValue_;
    std::vector<T> values_;
};

// totalPressure (incompressible, rho=none/psi=none): a fixedValue p whose value is recomputed each step from the patch
// velocity and the boundary flux:  p_b = p0 - 0.5*neg(phi_b)*magSqr(U_b)   (OF totalPressureFvPatchScalarField).
// Inflow (phi<0): p = p0 - 0.5|U|^2 (static = total - dynamic head); outflow (phi>=0): p = p0. The host build uses p0
// as the fixedValue base (value() = p0); the DEVICE recomputes the per-face refValue each step (deviceUpdateTotalPressure).
// bcCategory()=7 marks the face for the device builder. Pressure (scalar) only.
template <typename T>
class TotalPressurePatchField : public FixedValuePatchField<T>
{
public:
    TotalPressurePatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals)
        : FixedValuePatchField<T>(p, uniform, uval, std::move(vals)) {}
    int bcCategory() const override { return 7; }                   // device: totalPressure (per-step refValue)
};

// surfaceNormalFixedValue / uniformNormalFixedValue (velocity U only): U_b = refValue * face_normal
// (OF surfaceNormalFixedValueFvPatchVectorField: refValue_*patch().nf()). refValue is a SCALAR (signed normal-velocity
// magnitude; <0 = inflow). A plain fixedValue once built, the normals are geometric, refValue is constant for steady;
// any time-`ramp`/Function1 is ignored (the converged value, ramp->1 at endTime). Vector field only. bcCategory()=1.
class SurfaceNormalFixedValuePatchField : public FixedValuePatchField<vector>
{
public:
    SurfaceNormalFixedValuePatchField(
        const FvPatch& p,
        bool uniform,
        scalar uval,
        std::vector<scalar> vals)
        : FixedValuePatchField<vector>(p, false, vector{}, build(p, uniform, uval, vals)) {}
private:
    static std::vector<vector> build(
        const FvPatch& p,
        bool uniform,
        scalar uval,
        const std::vector<scalar>& vals)
    {
        std::vector<vector> v(p.size);
        for (label i = 0; i < p.size; ++i)
            v[i] = (uniform ? uval : (i < (label)vals.size() ? vals[i] : scalar(0))) * p.nf[i];
        return v;
    }
};

// timeVaryingMappedFixedValue: a fixedValue whose per-face value is the external boundaryData profile MAPPED onto the
// faces, NEAREST data point to each face centre (exact when the data is finer than the mesh, as in pitzDailyExptInlet's
// 70-point inlet profile). Time interpolation / offset / setAverage not applied (steady; offset 0, setAverage off here).
// bcCategory()=1. Works for scalar (k/epsilon) and vector (U).
template <typename T>
class TimeVaryingMappedPatchField : public FixedValuePatchField<T>
{
public:
    TimeVaryingMappedPatchField(
        const FvPatch& p,
        const std::vector<vector>& pts,
        const std::vector<T>& vals)
        : FixedValuePatchField<T>(p, false, T{}, mapNearest(p, pts, vals)) {}
private:
    static std::vector<T> mapNearest(
        const FvPatch& p,
        const std::vector<vector>& pts,
        const std::vector<T>& vals)
    {
        std::vector<T> v(p.size);
        for (label i = 0; i < p.size; ++i)
        {
            const vector& c = p.Cf[i];
            scalar best = 1e300;
            label bj = 0;
            for (std::size_t j = 0; j < pts.size(); ++j)
            {
                const scalar dx = c.x - pts[j].x, dy = c.y - pts[j].y, dz = c.z - pts[j].z;
                const scalar d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < best)
                {
                    best = d2;
                    bj = (label)j;
                }
            }
            v[i] = vals.empty() ? T{} : vals[bj];
        }
        return v;
    }
};

// zeroGradient: boundary value == adjacent internal cell value.
template <typename T>
class ZeroGradientPatchField : public fvPatchField<T>
{
public:
    explicit ZeroGradientPatchField(const FvPatch& p) : fvPatchField<T>(p) {}
    void evaluate(const std::vector<T>& internal) override
    {
        this->value_ = this->patchInternalField(internal);
    }
    bool fixesValue() const override { return false; }
};

// noSlip: fixed zero (velocity wall).
template <typename T>
class NoSlipPatchField : public fvPatchField<T>
{
public:
    explicit NoSlipPatchField(const FvPatch& p) : fvPatchField<T>(p) {}
    void evaluate(const std::vector<T>&) override
    {
        for (label i = 0; i < this->patch_.size; ++i)
            this->value_[i] = T{};
    }
    bool fixesValue() const override { return true; }
    int  bcCategory() const override { return 1; }

    std::vector<T> gradientInternalCoeffs() const override        // -deltaCoeffs (refValue 0)
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = tUniform<T>(-this->patch_.deltaCoeffs[i]);
        return r;
    }
    std::vector<T> valueInternalCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); } // 0
};

// empty: 2D front/back. Value tracks the internal field (the out-of-plane component is zero,
// so it contributes nothing to in-plane fluxes); the discretisation ignores empty patches.
template <typename T>
class EmptyPatchField : public fvPatchField<T>
{
public:
    explicit EmptyPatchField(const FvPatch& p) : fvPatchField<T>(p) {}
    void evaluate(const std::vector<T>& internal) override
    {
        this->value_ = this->patchInternalField(internal);
    }
    bool fixesValue() const override { return false; }
};

// symmetryPlane / symmetry: a slip plane. For a SCALAR (p,k,epsilon,nut) the normal gradient is zero, so the
// boundary value == the adjacent internal cell value (identical to zeroGradient). For a VECTOR (U) the
// wall-normal component is reflected and the tangential components are mirrored: the boundary value is the
// tangential projection v - n(n.v) (OF basicSymmetry value = (v + (I-2nn)&v)/2 = v - n(n.v)), and the
// snGrad is -n(n.v)*deltaCoeffs. The matrix coupling (per-component fixedValue-0 on the normal axis,
// zeroGradient on the tangential axes) is applied on the device by buildDeviceVectorBoundary, which keys on
// isSymmetry() + the face normal. The vector specialisation below is axis-agnostic for value()/output.
template <typename T>
class SymmetryPlanePatchField : public fvPatchField<T>
{
public:
    explicit SymmetryPlanePatchField(const FvPatch& p) : fvPatchField<T>(p) {}
    void evaluate(const std::vector<T>& internal) override
    {
        this->value_ = this->patchInternalField(internal);     // scalar: zeroGradient
    }
    bool fixesValue() const override { return false; }
    bool isSymmetry() const override { return true; }
};

// vector: remove the wall-normal component (value = v - n (n.v)).
template <> inline void SymmetryPlanePatchField<vector>::evaluate(const std::vector<vector>& internal)
{
    for (label i = 0; i < this->patch_.size; ++i)
    {
        const vector v = internal[this->patch_.faceCells[i]];
        const vector n = this->patch_.nf[i];
        this->value_[i] = v - n * dot(n, v);
    }
}

// Shared storage + read-and-hold value() for the OF calculated / inletOutlet / outletInlet /
// pressureInletOutletVelocity / mixed family, they differ ONLY in which device category (bcCategory) claims the
// face; value_[i] is the same uniform-or-per-face read value (= refValue). Each OF-named subclass below stays a
// distinct type so OpenFOAM developers still read them 1:1 against OF's fvPatchField hierarchy.
template <typename T>
class ExtrapolatedValuePatchField : public fvPatchField<T>
{
public:
    ExtrapolatedValuePatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals)
        : fvPatchField<T>(p), uniform_(uniform), uniformValue_(uval), values_(std::move(vals)) { evaluate({}); }
    void evaluate(const std::vector<T>&) override                  // value() = the read value (OF calculated-style)
    {
        for (label i = 0; i < this->patch_.size; ++i)
            this->value_[i] = refValue(i);
    }
    bool fixesValue() const override { return false; }
protected:
    T refValue(label i) const { return uniform_ ? uniformValue_ : (values_.empty() ? T{} : values_[i]); }
    bool           uniform_;
    T              uniformValue_;
    std::vector<T> values_;
};

// calculated: value is read from file and kept (used by e.g. nut interior/empty patches).
template <typename T>
class CalculatedPatchField : public ExtrapolatedValuePatchField<T>
{
public:
    using ExtrapolatedValuePatchField<T>::ExtrapolatedValuePatchField;   // read-and-hold value(); OF calculatedFvPatchField
    int bcCategory() const override { return 2; }
};

// inletOutlet: flux-conditional mix (OF mixed, valueFraction = neg(phi), refValue = inletValue, refGrad = 0).
// Per face: inflow (phi<0) -> fixedValue = inletValue; outflow (phi>=0) -> zeroGradient. The host evaluate()
// does not see the flux; the DEVICE recomputes the per-face fixedValue|zeroGradient choice every iteration from
// the patch flux (deviceUpdateInletOutlet). value()/refValue = inletValue (also the rest-start boundary value);
// bcCategory()=3 marks the face inletOutlet for the device builder. (Device-resident solver path; the host CPU
// assembly path treats category 3 as its zeroGradient base default, see scope in PORTING_INLETOUTLET_BC.md.)
template <typename T>
class InletOutletPatchField : public ExtrapolatedValuePatchField<T>   // value()/refValue = inletValue
{
public:
    using ExtrapolatedValuePatchField<T>::ExtrapolatedValuePatchField;
    int bcCategory() const override { return 3; }                  // inletOutlet (device: per-face fixedValue|zeroGradient)
};

// outletInlet (freestreamPressure base): the flux-OPPOSITE of inletOutlet, outflow phi>=0 -> fixedValue(outletValue),
// inflow phi<0 -> zeroGradient. Device category 4 (oioMask). value() = outletValue (= refValue).
template <typename T>
class OutletInletPatchField : public ExtrapolatedValuePatchField<T>   // value()/refValue = outletValue (= freestreamValue)
{
public:
    using ExtrapolatedValuePatchField<T>::ExtrapolatedValuePatchField;
    int bcCategory() const override { return 4; }                  // outletInlet (device: per-face fixedValue|zeroGradient, opposite switch)
};

// pressureInletOutletVelocity (OF directionMixedFvPatchVectorField): an outlet that allows backflow. Per face by the
// flux sign (valueFraction = neg(phi)*(I - n n); value = vf&refValue + (I-vf)&pif): outflow (phi>=0) -> zeroGradient
// (full extrapolation); inflow (phi<0) -> the TANGENTIAL velocity is fixed to refValue (default 0) while the NORMAL
// component is zeroGradient, so the pressure sets the inflow speed (value = n*(n.U_cell)). The DEVICE recomputes the
// per-face inflow value each step (deviceUpdatePressureInletOutletVelocity); bcCategory()=6 marks it. Vector-only;
// a non-zero `tangentialVelocity` field is NOT supported (default refValue 0 covers the simpleFoam tutorials).
template <typename T>
class PressureInletOutletVelocityPatchField : public ExtrapolatedValuePatchField<T>   // value() = written seed
{
public:
    using ExtrapolatedValuePatchField<T>::ExtrapolatedValuePatchField;
    int bcCategory() const override { return 6; }                  // device: pressureInletOutletVelocity (outlet, adjustable flux)
};

// mixed (Robin) BC, OF mixedFvPatchField (refGrad = 0). value = (1-vf)*internal + vf*refValue; the per-face
// valueFraction vf blends linearly between zeroGradient (vf=0) and fixedValue (vf=1) in BOTH the value and the
// matrix coeffs (gradientInternalCoeffs = -vf*dc, valueInternalCoeffs = 1-vf, etc.). freestreamVelocity /
// freestreamPressure are mixed with vf = 0.5 -/+ 0.5*(U.n)/|U| (continuous flow-angle blend, NOT the binary
// inletOutlet switch). OF mixedFvPatchField::fixesValue() == true -> references the pressure (needReference false).
// The DEVICE recomputes vf each step from the boundary flux (deviceUpdateMixedFreestream); the host evaluate()
// keeps the seed vf (=0.5) -> host CPU path is approximate for mixed (the device-resident solver is the target),
// matching the InletOutlet host-path scope.
template <typename T>
class MixedPatchField : public ExtrapolatedValuePatchField<T>     // value() = refValue (freestreamValue); base evaluate() sets it
{
public:
    MixedPatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals,
        bool velocitySign)
        : ExtrapolatedValuePatchField<T>(p, uniform, uval, std::move(vals)),
          velocitySign_(velocitySign), vf_(p.size, 0.5) {}
    bool fixesValue() const override { return true; }               // OF mixedFvPatchField::fixesValue() == true
    int  bcCategory() const override { return 5; }                  // device: mixed (per-face valueFraction blend)
    const std::vector<scalar>* valueFractionPtr() const override { return &vf_; }
    bool mixedVelocitySign() const override { return velocitySign_; }
    // OF mixed coeffs with refGrad = 0 (host correctness; the device blends the same way in its kernels):
    std::vector<T> gradientInternalCoeffs() const override        // -vf*deltaCoeffs
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = tUniform<T>(-vf_[i] * this->patch_.deltaCoeffs[i]);
        return r;
    }
    std::vector<T> gradientBoundaryCoeffs() const override        // vf*deltaCoeffs*refValue
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = this->refValue(i) * (vf_[i] * this->patch_.deltaCoeffs[i]);
        return r;
    }
    std::vector<T> valueInternalCoeffs() const override           // 1-vf
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = tUniform<T>(1.0 - vf_[i]);
        return r;
    }
    std::vector<T> valueBoundaryCoeffs() const override           // vf*refValue
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = this->refValue(i) * vf_[i];
        return r;
    }
private:
    bool                velocitySign_;  // true: vf=0.5-0.5 U.n/|U| (velocity); false: 0.5+0.5 ... (pressure)
    std::vector<scalar> vf_;            // per-face valueFraction (seed 0.5; device recomputes from the flow angle)
};

// processor: rank<->rank coupled patch. The exchange receives the NEIGHBOUR cells' values (the
// halo, = patchNeighbourField, used by the matrix). If interpolation weights are set, value() is the
// COUPLED FACE value w*ownCell + (1-w)*neighbour (OpenFOAM coupledFvPatchField::evaluate), so the
// serial fvc operators (grad/flux/interpolate) treat a processor face like an internal face. Without
// weights, value() == the halo (the original behaviour). Mirrors OpenFOAM processor/coupled patch.
template <typename T>
class ProcessorFvPatchField : public fvPatchField<T>
{
public:
    ProcessorFvPatchField(
        const FvPatch& p,
        int neighbProcNo,
        int tag = 0)
        : fvPatchField<T>(p), neighbProcNo_(neighbProcNo), tag_(tag) {}

    void setWeights(std::vector<scalar> w) { weights_ = std::move(w); }

    // A processor patch contributes NOTHING to the matrix as a boundary, its coupling is added
    // as an interface (off-diagonal) separately. So all matrix coeffs are zero (overriding the
    // base valueInternalCoeffs default of 1), letting the serial fvm operators run on the local
    // mesh and produce the correct REAL-boundary coeffs while ignoring processor faces.
    std::vector<T> valueInternalCoeffs()    const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> valueBoundaryCoeffs()    const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> gradientInternalCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> gradientBoundaryCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); }

    void initEvaluate(const std::vector<T>& internal) override
    {
        sendBuf_ = this->patchInternalField(internal);             // own interface cells
        halo_.resize(sendBuf_.size());                             // receive neighbour cells into halo_
        const int ns = static_cast<int>(sendBuf_.size() * (sizeof(T) / sizeof(scalar)));
        Pstream::irecv(reinterpret_cast<scalar*>(halo_.data()), ns, neighbProcNo_, tag_);
        Pstream::isend(reinterpret_cast<const scalar*>(sendBuf_.data()), ns, neighbProcNo_, tag_);
    }
    void evaluate(const std::vector<T>& internal) override
    {
        if (weights_.empty())   // value() = halo
        {
            this->value_ = halo_;
            return;
        }
        const std::vector<T> pif = this->patchInternalField(internal);
        this->value_.resize(halo_.size());
        for (std::size_t i = 0; i < halo_.size(); ++i)
            this->value_[i] = weights_[i] * pif[i] + (1.0 - weights_[i]) * halo_[i];  // coupled face value
    }
    bool fixesValue() const override { return false; }

    const std::vector<T>& patchNeighbourField() const override { return halo_; }

private:
    int                 neighbProcNo_;
    int                 tag_;
    std::vector<T>      sendBuf_;
    std::vector<T>      halo_;
    std::vector<scalar> weights_;
};

// cyclic: a periodic coupled patch. Like the processor patch but the neighbour cells are LOCAL (the
// owner cells of the matched neighbour faces, gathered from `internal`, no MPI). value() is the coupled
// face value w*ownCell + (1-w)*neighbour (with the transform applied to the neighbour for rotational
// cyclic), so the serial fvc operators treat a cyclic face like an internal face. The matrix coupling
// is added as an interface (off-diagonal), so all boundary matrix coeffs are zero. Mirrors OpenFOAM
// cyclicFvPatchField. forwardT = identity for translational periodicity.
template <typename T>
class CyclicFvPatchField : public fvPatchField<T>
{
public:
    CyclicFvPatchField(
        const FvPatch& p,
        std::vector<label> nbrFaceCells,
        std::vector<scalar> weights,
        tensor forwardT = tUniform<tensor>(0))
        : fvPatchField<T>(p), nbrFaceCells_(std::move(nbrFaceCells)), weights_(std::move(weights)),
          forwardT_(forwardT), hasTransform_(false) {}

    // Mark a rotational transform (forwardT applied to vector/tensor neighbour values).
    void setTransform(const tensor& fT)
    {
        forwardT_ = fT;
        hasTransform_ = true;
    }

    std::vector<T> valueInternalCoeffs()    const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> valueBoundaryCoeffs()    const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> gradientInternalCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> gradientBoundaryCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); }

    void evaluate(const std::vector<T>& internal) override
    {
        const std::vector<T> pif = this->patchInternalField(internal);     // own cells
        halo_.resize(nbrFaceCells_.size());
        for (std::size_t i = 0; i < nbrFaceCells_.size(); ++i)
            halo_[i] = transformValue(internal[nbrFaceCells_[i]]);
        this->value_.resize(halo_.size());
        for (std::size_t i = 0; i < halo_.size(); ++i)
            this->value_[i] = weights_[i] * pif[i] + (1.0 - weights_[i]) * halo_[i];   // coupled face value
    }
    bool fixesValue() const override { return false; }

    const std::vector<T>& patchNeighbourField() const override { return halo_; }

private:
    // Apply the cyclic transform to a neighbour value (identity unless rotational; scalars never rotate).
    T transformValue(const T& v) const { return v; }

    std::vector<label>  nbrFaceCells_;
    std::vector<scalar> weights_;
    std::vector<T>      halo_;       // transformed neighbour cell values (= patchNeighbourField)
    tensor              forwardT_;
    bool                hasTransform_;
};

// Rotational transform specialisation for vectors (forwardT & v). Scalars/tensors handled inline above
// (scalar: no rotation; tensor: forwardT & v & forwardT^T, added with C4).
template <> inline vector CyclicFvPatchField<vector>::transformValue(const vector& v) const
{
    return hasTransform_ ? dot(v, transpose(forwardT_)) : v;   // transform(forwardT, v) = forwardT & v = v & forwardT^T
}

// inletOutlet / totalPressure take their operating value from the inletValue slot, falling back to the plain `value`
// when inletValue is omitted. One accessor for that choice (returns a view; the referenced vector outlives the call).
template <typename T>
struct InletOrValue
{
    bool uniform;
    T uniformValue;
    const std::vector<T>& values;
};
template <typename T>
inline InletOrValue<T> inletOrValue(const PatchFieldData<T>& d)
{
    return d.hasInletValue ? InletOrValue<T>{d.inletUniform, d.inletUniformValue, d.inletValues}
                           : InletOrValue<T>{d.valueUniform, d.uniformValue,      d.values};
}

template <typename T>
std::unique_ptr<fvPatchField<T>> makePatchField(const FvPatch& p, const PatchFieldData<T>& d)
{
    if (d.type == "fixedValue" || d.type == "uniformFixedValue" || d.type == "codedFixedValue")
        // uniformFixedValue: steady constant = fixedValue. codedFixedValue: seed with `value`; the NVRTC device kernel
        // (device_coded_bc) OVERWRITES this patch's refValue each step from the compiled `code` snippet (the driver
        // parses the code + wires the solver's coded-BC apply). Building it as fixedValue makes buildField/buildDeviceBoundary succeed.
        return std::make_unique<FixedValuePatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    if (d.type == "codedMixed")   // Robin: seeded as mixed (vf=0.5); the NVRTC device kernel overwrites refValue + valueFraction each step
        return std::make_unique<MixedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values, false);
    if (d.type == "noSlip")          return std::make_unique<NoSlipPatchField<T>>(p);
    if (d.type == "zeroGradient")    return std::make_unique<ZeroGradientPatchField<T>>(p);
    // fixedFluxPressure = fixedGradient p whose gradient constrainPressure sets to (phiHbyA - Sf&U)/(magSf*rAU).
    // At a fixed-velocity patch constrainHbyA gives phiHbyA_b = Sf&U, so that gradient is exactly 0 == zeroGradient.
    // That is its only standard usage (no-flux walls), so we map it to zeroGradient (exact there). See
    // PORTING_PRESSURE_REFERENCE.md; a fixedFluxPressure on a non-fixed-velocity patch would need the gradient path.
    if (d.type == "fixedFluxPressure") return std::make_unique<ZeroGradientPatchField<T>>(p);
    if (d.type == "totalPressure")   // p0 read into the inletValue slot (foam_field_reader); fall back to value
    {
        const auto v = inletOrValue(d);
        return std::make_unique<TotalPressurePatchField<T>>(p, v.uniform, v.uniformValue, v.values);
    }
    if (d.type == "surfaceNormalFixedValue" || d.type == "uniformNormalFixedValue")   // U_b = refValue(scalar) * n
    {
        if constexpr (std::is_same_v<T, vector>)
            return std::make_unique<SurfaceNormalFixedValuePatchField>(p, d.normalRefUniform, d.normalRefUniformValue, d.normalRefValues);
        else throw std::runtime_error("brae: surfaceNormalFixedValue/uniformNormalFixedValue is a velocity (vector) BC");
    }
    if (d.type == "timeVaryingMappedFixedValue")   // boundaryData profile mapped (nearest) onto the faces -> fixedValue
    {
        if (d.hasMapData) return std::make_unique<TimeVaryingMappedPatchField<T>>(p, d.mapPoints, d.mapValues);
        // No mapped data: OF would FATAL here (the `value` entry is only the initial field, not a substitute for
        // boundaryData). Throw rather than silently degrade to fixedValue/zeroGradient -- masks a misconfigured case.
        throw std::runtime_error("brae: timeVaryingMappedFixedValue on patch '" + p.name +
            "' has no boundaryData (constant/boundaryData/" + p.name + "); OF requires it -- not falling back silently");
    }
    if (d.type == "inletOutlet")   // refValue = inletValue (fall back to value if inletValue omitted)
    {
        const auto v = inletOrValue(d);
        return std::make_unique<InletOutletPatchField<T>>(p, v.uniform, v.uniformValue, v.values);
    }
    // turbulent-inlet BCs (inletOutlet-derived): inflow value computed from U/k by the solver (applyTurbulentInlets);
    // here we build an inletOutlet placeholder from the written `value` so buildField succeeds.
    if (d.type == "turbulentIntensityKineticEnergyInlet" || d.type == "turbulentMixingLengthDissipationRateInlet"
        || d.type == "turbulentMixingLengthFrequencyInlet")
        return std::make_unique<InletOutletPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // base `freestream` (e.g. k/omega) derives from inletOutlet in OF -> BINARY flux switch (kept).
    if (d.type == "freestream")
        return std::make_unique<InletOutletPatchField<T>>(p, d.inletUniform, d.inletUniformValue, d.inletValues);
    // freestreamVelocity / freestreamPressure derive from mixedFvPatchField in OF -> CONTINUOUS Robin blend
    // vf = 0.5 -/+ 0.5*(U.n)/|U| (flow-angle, not the binary switch). The device recomputes vf each step.
    if (d.type == "freestreamVelocity")   // mixed, velocity sign (0.5 - 0.5 U.n/|U|)
        return std::make_unique<MixedPatchField<T>>(p, d.inletUniform, d.inletUniformValue, d.inletValues, true);
    if (d.type == "freestreamPressure")   // mixed, pressure sign (0.5 + 0.5 U.n/|U|)
        return std::make_unique<MixedPatchField<T>>(p, d.inletUniform, d.inletUniformValue, d.inletValues, false);
    // pressureInletOutletVelocity (directionMixed): device recomputes the inflow normal-projection each step (cat 6).
    if (d.type == "pressureInletOutletVelocity")
        return std::make_unique<PressureInletOutletVelocityPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    if (d.type == "kqRWallFunction")      return std::make_unique<ZeroGradientPatchField<T>>(p); // zeroGradient wrapper
    if (d.type == "epsilonWallFunction")  return std::make_unique<ZeroGradientPatchField<T>>(p); // boundary face = cell value; near-wall constraint applied separately
    if (d.type == "omegaWallFunction")    return std::make_unique<ZeroGradientPatchField<T>>(p); // kOmegaSST: same as epsilon, wall value set by deviceWallOmegaG0 + setValues
    if (d.type == "nutkWallFunction")     return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // alphatWallFunction: alphat_w = rho_w*nut_w/Prt, i.e. the SAME expression deviceAlphat applies in the
    // cells. Nothing is prescribed at the patch, so it is `calculated` exactly like the nut wall functions
    // -- the model writes the value. Without this row a real OF compressible case is refused at load,
    // because every rhoSimpleFoam tutorial with turbulence ships a 0/alphat.
    // OF v2412 registers this as "compressible::alphatWallFunction"; the unqualified spelling is NOT a
    // valid OF type (OF errors out on it). Both are accepted here so a hand-written case is not rejected
    // for a spelling OF would have caught anyway, but the qualified form is the one real cases use.
    if (d.type == "compressible::alphatWallFunction"
     || d.type == "alphatWallFunction")   return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // alphatJayatillekeWallFunction is NOT the same condition: it adds a thermal-sublayer resistance
    // (the P-function) and gives a different wall heat flux. Accepting it as the simple form would run
    // and converge with the wrong wall heat transfer, so it is refused by name instead.
    if (d.type == "compressible::alphatJayatillekeWallFunction"
     || d.type == "alphatJayatillekeWallFunction")
    {
        throw std::runtime_error(
            "brae: alphatJayatillekeWallFunction is not implemented (patch " + p.name
            + "). It applies a thermal-sublayer P-function that brae does not have, so treating it as "
              "compressible::alphatWallFunction would give the wrong wall heat flux. Use that instead, or wait "
              "for the Jayatilleke port.");
    }
    if (d.type == "atmNutkWallFunction")  return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values); // atmospheric rough-wall nut (z0); device wall nut from deviceBoundaryNut with atmZ0>0
    if (d.type == "nutUSpaldingWallFunction") return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values); // SA wall nut (value set by deviceBoundaryNutSpalding)
    // nutLowReWallFunction: OF calcNut()=0 (resolved viscous sublayer). cf's nutkWallFunction already yields nut=0 for
    // yPlus<yPlusLam, so on the low-Re mesh this BC is used on it is the same; nutUBlendedWallFunction is the same
    // log-law wall-nut family as nutk. Both -> Calculated (device wall nut from deviceBoundaryNut). Validated vs OF.
    if (d.type == "nutLowReWallFunction" || d.type == "nutUBlendedWallFunction")
        return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // cyclic: the device-resident solver couples it via appended internal faces (DeviceMesh), so the host patch
    // field is a no-op placeholder here (its value is unused; the FvPatch type "cyclic" drives the device skip).
    if (d.type == "cyclic" || d.type == "cyclicAMI")          return std::make_unique<ZeroGradientPatchField<T>>(p);
    if (d.type == "empty")           return std::make_unique<EmptyPatchField<T>>(p);
    if (d.type == "symmetryPlane" || d.type == "symmetry" || d.type == "slip")
        return std::make_unique<SymmetryPlanePatchField<T>>(p);  // slip = OF basicSymmetry
    if (d.type == "calculated")      return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // atmBoundaryLayerInlet{Velocity,K,Epsilon,Omega} (OpenFOAM atmBoundaryLayer.C, v2412): Richards-Hoxey log-law
    // profile evaluated per boundary face from the face-centre height z = Cf.zDir - d, a steady fixedValue. Params come
    // from include/ABLConditions (foam_field_reader parses them). u* = kappa|Uref|/ln((Zref+z0)/z0):
    //   U(z)=(u*/kappa)ln((z+z0)/z0)flowDir ; k=u*^2/sqrt(Cmu) ; eps=u*^3/(kappa(z+z0)) ; omega=u*/(sqrt(Cmu)kappa(z+z0)).
    // OF defaults C1=0,C2=1 -> k is height-constant. z clamped >=0 at/below the ground plane (OF max(z-d,0)).
    if (d.type == "atmBoundaryLayerInletVelocity")
    {
        if constexpr (std::is_same_v<T, vector>)
        {
            const scalar kap = d.ablKappa, z0 = d.ablZ0;
            const scalar Ustar = kap * std::fabs(d.ablUref) / std::log((d.ablZref + z0) / z0);
            vector fh = d.ablFlowDir;
            const scalar fm = std::sqrt(fh.x*fh.x + fh.y*fh.y + fh.z*fh.z);
            if (fm > 0)
            {
                fh.x /= fm;
                fh.y /= fm;
                fh.z /= fm;
            }
            const vector zh = d.ablZDir;
            std::vector<vector> vals(p.size);
            for (label i = 0; i < p.size; ++i)
            {
                const vector& c = p.Cf[i];
                scalar zr = c.x*zh.x + c.y*zh.y + c.z*zh.z - d.ablD;
                if (zr < 0) zr = 0;
                const scalar Um = (Ustar / kap) * std::log((zr + z0) / z0);
                vals[i] = vector{fh.x*Um, fh.y*Um, fh.z*Um};
            }
            return std::make_unique<FixedValuePatchField<vector>>(p, false, vector{}, vals);
        }
        else throw std::runtime_error("brae: atmBoundaryLayerInletVelocity is a velocity (vector) BC");
    }
    if (d.type == "atmBoundaryLayerInletK" || d.type == "atmBoundaryLayerInletEpsilon" || d.type == "atmBoundaryLayerInletOmega")
    {
        if constexpr (std::is_same_v<T, scalar>)
        {
            const scalar kap = d.ablKappa, z0 = d.ablZ0, Cmu = d.ablCmu;
            const scalar Ustar = kap * std::fabs(d.ablUref) / std::log((d.ablZref + z0) / z0);
            const vector zh = d.ablZDir;
            const bool isK = (d.type == "atmBoundaryLayerInletK"), isEps = (d.type == "atmBoundaryLayerInletEpsilon");
            std::vector<scalar> vals(p.size);
            for (label i = 0; i < p.size; ++i)
            {
                const vector& c = p.Cf[i];
                scalar zr = c.x*zh.x + c.y*zh.y + c.z*zh.z - d.ablD;
                if (zr < 0) zr = 0;
                vals[i] = isK   ? Ustar*Ustar / std::sqrt(Cmu)
                        : isEps ? Ustar*Ustar*Ustar / (kap * (zr + z0))
                                : Ustar / (std::sqrt(Cmu) * kap * (zr + z0));
            }
            return std::make_unique<FixedValuePatchField<scalar>>(p, false, scalar{}, vals);
        }
        else throw std::runtime_error("brae: atmBoundaryLayerInlet{K,Epsilon,Omega} is a scalar BC");
    }
    // codedFixedValue is handled above (seeded as fixedValue; the NVRTC device kernel overwrites its refValue per step).
    throw std::runtime_error("brae: unsupported BC type '" + d.type + "' on patch " + p.name);
}

} // namespace brae
