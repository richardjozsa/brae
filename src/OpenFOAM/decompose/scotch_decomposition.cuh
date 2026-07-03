#pragma once
// SCOTCH cell-graph partitioning. Mirrors OpenFOAM scotchDecomp: partition the cell DUAL
// graph (cells = vertices, internal faces = edges) into nParts. Unweighted (matches a
// default decomposePar with no cell/face weights). Returns cellToPart[nCells].
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

std::vector<label> scotchDecompose(const PrimitiveMesh& m, label nParts);

} // namespace brae
