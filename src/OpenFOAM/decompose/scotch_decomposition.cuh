#pragma once
// SCOTCH cell-graph partitioning. Mirrors OpenFOAM scotchDecomp: partition the cell DUAL
// graph (cells = vertices, internal faces = edges) into nParts. Unweighted (matches a
// default decomposePar with no cell/face weights). Returns cellToPart[nCells].
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <vector>
#include <string>

namespace brae {

std::vector<label> scotchDecompose(const PrimitiveMesh& m, label nParts);

// Cached scotchDecompose: reload cacheDir/.brae_decomp_np<N> if newer than cacheDir/owner (mesh unchanged),
// else compute and persist. Static per (mesh, nParts); saves the ~1s decompose on warm launches.
std::vector<label> scotchDecomposeCached(const PrimitiveMesh& m, label nParts, const std::string& cacheDir);

} // namespace brae
