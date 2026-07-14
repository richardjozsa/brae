#pragma once
// brae::FvGeometry, cell/face geometry computed EXACTLY per OpenFOAM v2412
// (primitiveMeshTools::makeFaceCentresAndAreas / makeCellCentresAndVols and
// basicFvGeometryScheme weights/deltaCoeffs/nonOrth*), in fp64.
//
// Face quantities (Cf, Sf, magSf) are sized nFaces. The cell-centre estimate + pyramid
// decomposition gives C, V. Interpolation quantities (weights, deltaCoeffs, nonOrth*) are
// defined on internal faces [0, nInternalFaces).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

class FvGeometry
{
public:
    void build(const PrimitiveMesh& m);

    const std::vector<vector>& Cf()    const { return Cf_; }
    const std::vector<vector>& Sf()    const { return Sf_; }
    const std::vector<scalar>& magSf() const { return magSf_; }
    const std::vector<vector>& C()     const { return C_; }
    const std::vector<scalar>& V()     const { return V_; }
    const std::vector<scalar>& weights()            const { return weights_; }
    const std::vector<scalar>& deltaCoeffs()        const { return deltaCoeffs_; }
    const std::vector<scalar>& nonOrthDeltaCoeffs() const { return nonOrthDeltaCoeffs_; }
    const std::vector<vector>& nonOrthCorrectionVectors() const { return nonOrthCorr_; }

private:
    void makeFaceCentresAndAreas(const PrimitiveMesh& m);
    void makeCellCentresAndVols(const PrimitiveMesh& m);
    void makeInterpolation(const PrimitiveMesh& m);

    std::vector<vector> Cf_, Sf_, C_, nonOrthCorr_;
    std::vector<scalar> magSf_, V_, weights_, deltaCoeffs_, nonOrthDeltaCoeffs_;
};

} // namespace brae
