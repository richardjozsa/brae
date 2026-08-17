#pragma once
// brae::VelocityComponentLaplacianMotion -- OF velocityComponentLaplacianFvMotionSolver
// (fvMotionSolver/fvMotionSolvers/velocity/componentLaplacian).
//
// A mesh-motion solver that moves the points along ONE coordinate direction, with the motion diffused
// through the mesh by a Laplace equation:
//
//     solve   laplacian(diffusivity, cellMotionU) == 0        (a SCALAR equation: the chosen component)
//     then    pointMotionU = interpolate(cellMotionU)         (volPointInterpolation)
//     then    points[cmpt] += deltaT * pointMotionU           (INCREMENTAL, from the CURRENT points)
//
// This is how pimpleFoam/laminar/movingCone drives a piston: `movingWall` carries pointMotionUx = 1 m/s,
// `fixedWall` and `left` carry 0, and the Laplace solve spreads the 1 -> 0 transition over the mesh so
// the cells stretch instead of tangling. The `directional (1 200 0)` diffusivity makes the mesh 200x
// stiffer across the channel than along it, which is what keeps the motion confined to the x direction.
//
// INCREMENTAL, NOT ABSOLUTE. solidBodyMotion transforms points0 by an absolute function of t precisely
// so round-off cannot accumulate; this solver cannot do that -- the motion is the SOLUTION of an
// equation on the current mesh, so OF adds deltaT*U to the CURRENT points (curPoints()). Using points0
// here would undo every previous step.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "pcg.cuh"
#include "vol_point_interpolation.cuh"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

struct VelocityComponentLaplacianMotion
{
    bool   active = false;
    int    cmpt   = 0;                  // 0=x 1=y 2=z, from `component x|y|z`
    vector diffusivity{1, 1, 1};        // `directional (dx dy dz)`; `uniform` -> (1,1,1)
    std::string fieldName;              // pointMotionUx / ...Uy / ...Uz
};

// constant/dynamicMeshDict for this solver. Returns active=false when the dict names another solver,
// so the caller can fall through to its existing solidBody path.
inline VelocityComponentLaplacianMotion readVelocityComponentLaplacian(const FoamDict& d)
{
    VelocityComponentLaplacianMotion mm;
    std::string solver = d.wordOr("motionSolver", "");
    if (solver.empty()) solver = d.wordOr("solver", "");   // OF getCompat: `solver` is the older key
    if (solver != "velocityComponentLaplacian") return mm;

    const std::string c = d.wordOr("component", "");
    if      (c == "x") mm.cmpt = 0;
    else if (c == "y") mm.cmpt = 1;
    else if (c == "z") mm.cmpt = 2;
    else throw std::runtime_error(
        "brae: velocityComponentLaplacian needs `component x|y|z` (got '" + c + "'). It moves the mesh "
        "along ONE axis; without the axis there is nothing to solve for.");
    mm.fieldName = std::string("pointMotionU") + c;

    // OF motionDiffusivity::New: the first word selects the model. `uniform` is 1 everywhere;
    // `directional (dx dy dz)` makes the face diffusivity n & cmptMultiply(D, n) -- a mesh that resists
    // deformation differently along each axis (directionalDiffusivity.C:correct()).
    const std::string dw = d.wordOr("diffusivity", "uniform");
    if (dw == "directional")
    {
        const std::vector<scalar> v = d.scalarListOr("diffusivity", {});
        if (v.size() < 3)
            throw std::runtime_error("brae: `diffusivity directional` needs three components (dx dy dz).");
        mm.diffusivity = vector{v[v.size()-3], v[v.size()-2], v[v.size()-1]};
    }
    else if (dw != "uniform")
        throw std::runtime_error(
            "brae: mesh-motion diffusivity '" + dw + "' is not implemented (brae has `uniform` and "
            "`directional`). The diffusivity decides HOW the prescribed motion is spread through the "
            "mesh, so substituting another one deforms the mesh differently.");
    mm.active = true;
    return mm;
}

// The per-face diffusivity: OF directionalDiffusivity::correct() is
//     faceDiffusivity = n & cmptMultiply(D, n) = Dx nx^2 + Dy ny^2 + Dz nz^2
// with n = Sf/|Sf|. `uniform` is the same expression with D = (1,1,1), which gives exactly 1.
inline SurfaceScalarField motionFaceDiffusivity(const PrimitiveMesh& m, const FvGeometry& g,
                                                const std::vector<FvPatch>& fvp, const vector& D)
{
    SurfaceScalarField gf;
    const label nIf = m.nInternalFaces();
    gf.internal.resize(nIf);
    auto faceD = [&](label f)
    {
        const scalar a = g.magSf()[f];
        if (!(a > scalar(0))) return scalar(1);
        const vector n{g.Sf()[f].x/a, g.Sf()[f].y/a, g.Sf()[f].z/a};
        return D.x*n.x*n.x + D.y*n.y*n.y + D.z*n.z*n.z;
    };
    for (label f = 0; f < nIf; ++f) gf.internal[f] = faceD(f);
    for (const FvPatch& p : fvp)
    {
        std::vector<scalar> pb(p.size);
        for (label i = 0; i < p.size; ++i) pb[i] = faceD(p.start + i);
        gf.boundary.push_back(std::move(pb));
    }
    return gf;
}

// The cell field the Laplace equation is solved for, with the boundary conditions OF derives from the
// POINT field's (fvMotionSolverCore::cellMotionBoundaryTypes): a fixed point motion becomes a fixed cell
// motion, a slip/symmetry point constraint becomes a zero-gradient cell BC (for a SCALAR, `slip` is
// zeroGradient -- the transform of a scalar is the identity), and empty stays empty.
inline GeometricField<scalar> buildCellMotionField(const FieldData<scalar>& pointFd,
                                                   const std::vector<FvPatch>& fvp,
                                                   label nCells)
{
    GeometricField<scalar> f;
    f.internal.assign(nCells, pointFd.internalUniform ? pointFd.internalUniformValue : scalar(0));
    for (const FvPatch& p : fvp)
    {
        const PatchFieldData<scalar>* d = nullptr;
        for (const PatchFieldData<scalar>& b : pointFd.boundary)
            if (b.name == p.name) { d = &b; break; }
        const std::string type = d ? d->type : std::string("");
        if (p.type == "empty" || type == "empty")
        {
            f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        }
        else if (type == "fixedValue" || type == "uniformFixedValue" || type == "calculated"
              || type == "cellMotion")
        {
            const scalar v = d->valueUniform || d->values.empty()
                           ? d->uniformValue
                           : d->values.front();
            std::vector<scalar> vals;
            if (!d->valueUniform && static_cast<label>(d->values.size()) == p.size) vals = d->values;
            f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(p, vals.empty(), v, vals));
        }
        // `wedge` on the point-motion SCALAR is zeroGradient, exactly as OF specialises
        // wedgeFvPatchField<scalar> (a scalar has no direction to rotate). The axisymmetry is carried by
        // the mesh and by the VECTOR fields, not by the motion component itself.
        else if (type == "slip" || type == "symmetryPlane" || type == "symmetry"
              || type == "wedge" || type == "zeroGradient" || type.empty())
        {
            f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        }
        else
            throw std::runtime_error(
                "brae: point-motion boundary type '" + type + "' on patch '" + p.name + "' is not one "
                "brae maps to a cell-motion boundary condition (fixedValue/uniformFixedValue, "
                "slip/symmetry/zeroGradient, empty). The motion boundary IS the prescribed motion, so "
                "guessing one would move the mesh differently from the case.");
    }
    f.evaluateBoundary();
    return f;
}


// The POINT field's own boundary conditions, as a list of (mesh point, prescribed value).
//
// THE INTERPOLATION IS NOT THE LAST WORD. OF's motion solver interpolates cellMotionU onto the points
// and then calls the POINT field's own boundary conditions, and a valuePointPatchField::evaluate()
// simply SCATTERS its prescribed values into the point field (valuePointPatchField.C:203-211),
// overriding whatever the interpolation produced. Without that step a point where the moving wall meets
// another patch is blended with that patch's faces and comes out slow: measured against OpenFOAM's own
// pointMotionUx on movingCone, brae had 0.9438 where OF has exactly 1 -- a piston face that is 6% slow
// at its rim, on a run that otherwise looks healthy.
//
// Patch order decides a shared point, because that is what scattering in patch order does.
inline std::vector<std::pair<label, scalar>> pointMotionConstraints(
    const PrimitiveMesh& m, const std::vector<FvPatch>& fvp, const FieldData<scalar>& pointFd)
{
    std::vector<std::pair<label, scalar>> out;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const PatchFieldData<scalar>* d = nullptr;
        for (const PatchFieldData<scalar>& b : pointFd.boundary)
            if (b.name == fvp[pi].name) { d = &b; break; }
        if (!d) continue;
        // Only the value-holding point BCs constrain. slip/symmetry/wedge/empty are CONSTRAINT patches
        // and for a scalar they leave the value alone (pointConstraints has nothing to project).
        if (!(d->type == "fixedValue" || d->type == "uniformFixedValue" || d->type == "calculated"
           || d->type == "cellMotion"))
            continue;
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label f = fvp[pi].start + i;
            const scalar v = (d->valueUniform || static_cast<label>(d->values.size()) != fvp[pi].size)
                           ? d->uniformValue : d->values[i];
            const label nv = m.faceSize(f);
            for (label k = 0; k < nv; ++k) out.emplace_back(m.faceVert(f, k), v);
        }
    }
    return out;
}

// One motion step. Returns the NEW point positions; the caller owns movePoints and everything that
// follows from it (geometry rebuild, mesh flux, boundary velocities).
//
// `cellMotion` is carried across steps by the caller: OF keeps cellMotionU_ as a field and its previous
// solution is the next solve's initial guess, which is what makes the solve cheap after the first step.
inline std::vector<vector> velocityComponentLaplacianPoints(
    const VelocityComponentLaplacianMotion& mm,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    GeometricField<scalar>& cellMotion,       // in/out: the solved component field
    const VolPointInterpolation& vpi,
    // The point field's own prescribed values (pointMotionConstraints), applied AFTER the interpolation
    // exactly as OF's pointPatchField::evaluate does.
    const std::vector<std::pair<label, scalar>>& constraints,
    scalar deltaT)
{
    const SurfaceScalarField gf = motionFaceDiffusivity(m, g, fvp, mm.diffusivity);
    FvScalarMatrix M = fvm::laplacian(gf, cellMotion, m, g, fvp);
    // OF solves `laplacian(...) == 0`: no source at all. The boundary values enter through the matrix's
    // boundary coefficients, which fvm::laplacian has already folded in.
    pcg(M, cellMotion.internal, m, fvp, /*tolerance=*/1e-9, /*relTol=*/0.0, /*maxIter=*/1000);
    cellMotion.evaluateBoundary();

    // cell -> point. The boundary VALUES are what a patch point reads, so they have to be the evaluated
    // ones (a fixedValue 1 wall gives exactly 1), not the adjacent cell values.
    std::vector<scalar> bnd;
    bnd.reserve(static_cast<std::size_t>(m.nFaces() - m.nInternalFaces()));
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& v = cellMotion.boundary[pi]->value();
        for (label i = 0; i < fvp[pi].size; ++i) bnd.push_back(v[i]);
    }
    std::vector<scalar> pointMotion = vpi.interpolate(cellMotion.internal, bnd);
    for (const auto& c : constraints) pointMotion[c.first] = c.second;   // the point BCs have the last word

    std::vector<vector> pts = m.points();
    for (std::size_t p = 0; p < pts.size(); ++p)
    {
        scalar* c = (mm.cmpt == 0) ? &pts[p].x : (mm.cmpt == 1) ? &pts[p].y : &pts[p].z;
        *c += deltaT * pointMotion[p];
    }
    return pts;
}

} // namespace brae
