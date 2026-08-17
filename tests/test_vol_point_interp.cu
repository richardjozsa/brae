// volPointInterpolation (cell -> point) and the motion solver it exists for.
//
// OF volPointInterpolation.C:169-256 has two cases and the SPLIT is the whole of it:
//   interior point : value = sum_c w_c cellValue_c / sum w,  w_c = 1/|p - C_c|
//   patch point    : value = sum_f w_f faceValue_f / sum w,  w_f = 1/|p - Cf_f|   -- cells unused
//
// LEG 2 IS THE ONE THAT DECIDES WHETHER A PISTON MOVES AT THE RIGHT SPEED. movingCone prescribes
// `pointMotionUx uniformFixedValue 1` on its moving wall, so the points ON that wall must come out at
// exactly 1. Interpolate them from the surrounding CELLS instead and they get the Laplace solution just
// inside the wall -- always less than 1 -- so the piston lags, on a mesh that still looks well-formed
// and a run that still converges. The test sets the cell field to 0 and the boundary field to 1 so the
// two sources cannot be confused: any leakage of cell data into a patch point shows up immediately.
//
// Leg 4 then runs the whole velocityComponentLaplacian step against an ANALYTIC answer: with a linear
// Laplace solution the point motion is exactly 1 - x, so the moved mesh is known in closed form.
#include "vol_point_interpolation.cuh"
#include "velocity_component_laplacian.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_dict.cuh"
#include "box_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

FoamDict dictFromString(const std::string& body)
{
    const std::string path = "test_vol_point_interp.tmpdict";
    { std::ofstream f(path); f << body; }
    FoamDict d = readDict(path);
    std::remove(path.c_str());
    return d;
}

std::vector<scalar> flatBoundary(const std::vector<FvPatch>& fvp, scalar v)
{
    std::size_t n = 0;
    for (const FvPatch& p : fvp) n += static_cast<std::size_t>(p.size);
    return std::vector<scalar>(n, v);
}
} // namespace

int main()
{
    std::printf("== volPointInterpolation + velocityComponentLaplacian ==\n");

    PrimitiveMesh m = boxtest::boxMesh(4, 4, 4);
    FvGeometry g; g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    VolPointInterpolation vpi;
    vpi.build(m, g, fvp);

    // ---- Leg 1: a LINEAR field is reproduced exactly at interior points -----------------------------
    // An interior point of a uniform hex mesh is surrounded by eight cells placed symmetrically about
    // it, so the inverse-distance weights are all equal and the weighted mean of a linear field is the
    // field's value AT the point. That makes this an equality, not an error bound.
    {
        const std::vector<vector>& C = g.C();
        std::vector<scalar> cell(nC);
        for (label c = 0; c < nC; ++c) cell[c] = 3.0*C[c].x - 2.0*C[c].y + 0.5*C[c].z + 1.25;
        // boundary values from the same linear function, at the face centres
        std::vector<scalar> bnd;
        for (const FvPatch& p : fvp)
            for (label i = 0; i < p.size; ++i)
            {
                const vector& x = g.Cf()[p.start + i];
                bnd.push_back(3.0*x.x - 2.0*x.y + 0.5*x.z + 1.25);
            }
        const std::vector<scalar> pv = vpi.interpolate(cell, bnd);
        scalar worst = 0;
        int interior = 0;
        for (label p = 0; p < m.nPoints(); ++p)
        {
            const vector& x = m.points()[p];
            const scalar want = 3.0*x.x - 2.0*x.y + 0.5*x.z + 1.25;
            if (!vpi.isPatchPoint(p)) { ++interior; worst = std::max(worst, std::fabs(pv[p] - want)); }
        }
        check(interior > 0, "vacuity guard: the mesh has interior points");
        check(worst < 1e-12, "interior points reproduce a linear field exactly");
        std::printf("        (%d interior points, worst error %.3g)\n", interior, (double)worst);
    }

    // ---- Leg 2: a patch point reads the BOUNDARY value, never the cells ----------------------------
    {
        const std::vector<scalar> cell(nC, 0.0);
        const std::vector<scalar> bnd = flatBoundary(fvp, 1.0);
        const std::vector<scalar> pv = vpi.interpolate(cell, bnd);
        scalar worstPatch = 0, worstInterior = 0;
        int nPatch = 0;
        for (label p = 0; p < m.nPoints(); ++p)
        {
            if (vpi.isPatchPoint(p)) { ++nPatch; worstPatch = std::max(worstPatch, std::fabs(pv[p] - 1.0)); }
            else                     worstInterior = std::max(worstInterior, std::fabs(pv[p]));
        }
        check(nPatch > 0, "vacuity guard: the mesh has patch points");
        check(worstPatch < 1e-14, "a patch point takes the boundary value exactly (1, with cells at 0)");
        check(worstInterior < 1e-14, "...and an interior point takes the cell value (0), not the boundary's");
    }

    // ---- Leg 3: an `empty` patch does NOT make its points into patch points -------------------------
    // A 2D case's front/back are empty. If they counted, every point of a 2D mesh would be a patch point
    // and the motion would be pinned to the boundary data everywhere.
    {
        std::vector<FvPatch> fvpE = fvp;
        int nEmpty = 0;
        for (FvPatch& p : fvpE)
            if (p.name.find("Zmin") != std::string::npos || p.name.find("Zmax") != std::string::npos)
            { p.type = "empty"; ++nEmpty; }
        check(nEmpty == 2, "vacuity guard: two patches were made empty");
        VolPointInterpolation vE;
        vE.build(m, g, fvpE);
        // a point on the z=0 plane but away from the x/y walls must now be INTERIOR
        int checked = 0, wrong = 0;
        for (label p = 0; p < m.nPoints(); ++p)
        {
            const vector& x = m.points()[p];
            const bool onZface = (std::fabs(x.z) < 1e-12);
            const bool onXY = (std::fabs(x.x) < 1e-12 || std::fabs(x.x - 4.0) < 1e-12
                            || std::fabs(x.y) < 1e-12 || std::fabs(x.y - 4.0) < 1e-12);
            if (onZface && !onXY) { ++checked; if (vE.isPatchPoint(p)) ++wrong; }
        }
        check(checked > 0 && wrong == 0, "points on an empty patch are treated as interior");
    }

    // ---- Leg 4: the whole motion step ---------------------------------------------------------------
    // Moving wall at x = 0 with motion 1, fixed wall at x = 4 with motion 0, zero-gradient elsewhere.
    // On this uniform mesh the CELL solution of the Laplace equation is exactly linear, so it can be
    // asserted in closed form; the POINT motion cannot be, and deliberately is not. A point on the edge
    // where the inlet meets a side wall draws on faces at two different x (the inlet face at 1 and the
    // side-wall faces at 0.875) and lands between them -- which is exactly what OpenFOAM's rule does
    // too. What IS pinned is the piston claim itself: a point in the INTERIOR of the moving face, whose
    // patch faces all carry the prescribed value, must move at exactly the prescribed speed.
    {
        VelocityComponentLaplacianMotion mm;
        mm.active = true;
        mm.cmpt = 0;
        mm.diffusivity = vector{1, 1, 1};

        GeometricField<scalar> cellMotion;
        cellMotion.internal.assign(nC, 0.0);
        for (const FvPatch& p : fvp)
        {
            const bool xMin = (p.name == "inlet");    // boxMesh names its x faces inlet/outlet
            const bool xMax = (p.name == "outlet");
            if (xMin || xMax)
                cellMotion.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(
                    p, true, xMin ? scalar(1) : scalar(0), std::vector<scalar>{}));
            else
                cellMotion.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        }
        cellMotion.evaluateBoundary();

        const scalar dt = 0.25;
        const std::vector<vector> before = m.points();
        const std::vector<vector> after =
            velocityComponentLaplacianPoints(mm, m, g, fvp, cellMotion, vpi, {}, dt);

        // (a) the Laplace solve: exactly linear across the box
        scalar worstCell = 0;
        for (label c = 0; c < nC; ++c)
            worstCell = std::max(worstCell, std::fabs(cellMotion.internal[c] - (1.0 - g.C()[c].x/4.0)));
        check(worstCell < 1e-8, "the motion equation solves to the analytic linear profile 1 - x/4");

        // (b) the piston itself: points strictly inside the moving face advance at the prescribed speed
        int nFaceInterior = 0;
        scalar worstPiston = 0;
        for (std::size_t p = 0; p < after.size(); ++p)
        {
            const vector& x = before[p];
            const bool onInlet = std::fabs(x.x) < 1e-12;
            const bool onRim   = std::fabs(x.y) < 1e-12 || std::fabs(x.y - 4.0) < 1e-12
                              || std::fabs(x.z) < 1e-12 || std::fabs(x.z - 4.0) < 1e-12;
            if (onInlet && !onRim)
            {
                ++nFaceInterior;
                worstPiston = std::max(worstPiston, std::fabs((after[p].x - x.x) - dt*1.0));
            }
        }
        check(nFaceInterior > 0, "vacuity guard: the moving face has points off its rim");
        check(worstPiston < 1e-12, "a point inside the moving face advances at exactly the prescribed speed");

        // (c) ...and one inside the fixed face does not move at all
        int nFixed = 0;
        scalar worstFixed = 0;
        for (std::size_t p = 0; p < after.size(); ++p)
        {
            const vector& x = before[p];
            const bool onOutlet = std::fabs(x.x - 4.0) < 1e-12;
            const bool onRim    = std::fabs(x.y) < 1e-12 || std::fabs(x.y - 4.0) < 1e-12
                               || std::fabs(x.z) < 1e-12 || std::fabs(x.z - 4.0) < 1e-12;
            if (onOutlet && !onRim) { ++nFixed; worstFixed = std::max(worstFixed, std::fabs(after[p].x - x.x)); }
        }
        check(nFixed > 0 && worstFixed < 1e-12, "a point inside the fixed face does not move");

        // (d) the update is deltaT * the interpolated motion, and nothing else
        std::vector<scalar> bnd;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& v = cellMotion.boundary[pi]->value();
            for (label i = 0; i < fvp[pi].size; ++i) bnd.push_back(v[i]);
        }
        const std::vector<scalar> pm = vpi.interpolate(cellMotion.internal, bnd);
        scalar worstAll = 0, worstYZ = 0;
        for (std::size_t p = 0; p < after.size(); ++p)
        {
            worstAll = std::max(worstAll, std::fabs((after[p].x - before[p].x) - dt*pm[p]));
            worstYZ  = std::max(worstYZ,  std::fabs(after[p].y - before[p].y)
                                        + std::fabs(after[p].z - before[p].z));
        }
        check(worstAll < 1e-14, "every point moves by deltaT * its interpolated motion");
        check(worstYZ == 0.0, "...and nothing moves in the other two components");
    }

    // ---- Leg 5: what the dictionary reader accepts and refuses --------------------------------------
    {
        const VelocityComponentLaplacianMotion a = readVelocityComponentLaplacian(dictFromString(
            "dynamicFvMesh dynamicMotionSolverFvMesh;\nmotionSolver velocityComponentLaplacian;\n"
            "component x;\ndiffusivity directional (1 200 0);\n"));
        check(a.active && a.cmpt == 0, "parser: component x selects the x motion");
        check(std::fabs(a.diffusivity.y - 200.0) < 1e-14, "parser: the directional diffusivity is read");
        check(a.fieldName == "pointMotionUx", "parser: the point field name follows the component");
    }
    {
        const VelocityComponentLaplacianMotion a = readVelocityComponentLaplacian(dictFromString(
            "dynamicFvMesh dynamicMotionSolverFvMesh;\nmotionSolver solidBody;\n"));
        check(!a.active, "negative control: a solidBody dict leaves this solver inactive");
    }
    {
        bool threw = false;
        try
        {
            readVelocityComponentLaplacian(dictFromString(
                "motionSolver velocityComponentLaplacian;\ncomponent x;\ndiffusivity inverseDistance 1(wall);\n"));
        }
        catch (const std::exception&) { threw = true; }
        check(threw, "refusal: an unimplemented diffusivity is rejected, not run as uniform");
    }
    {
        bool threw = false;
        try
        {
            readVelocityComponentLaplacian(dictFromString("motionSolver velocityComponentLaplacian;\n"));
        }
        catch (const std::exception&) { threw = true; }
        check(threw, "refusal: no `component` entry is rejected");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
