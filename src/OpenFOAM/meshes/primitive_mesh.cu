#include "primitive_mesh.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace brae {

namespace {
constexpr unsigned BRAE_MESH_CACHE_MAGIC = 0x43464D32;   // "CFM2"
template<class T>
void wvec(std::FILE* f, const std::vector<T>& v)
{
    std::size_t n = v.size();
    std::fwrite(&n, sizeof(n), 1, f);
    if (n) std::fwrite(v.data(), sizeof(T), n, f);
}
template<class T>
bool rvec(std::FILE* f, std::vector<T>& v)
{
    std::size_t n;
    if (std::fread(&n, sizeof(n), 1, f) != 1) return false;
    v.resize(n);
    return n == 0 || std::fread(v.data(), sizeof(T), n, f) == n;
}
void wstr(std::FILE* f, const std::string& s)
{
    std::size_t n = s.size();
    std::fwrite(&n, sizeof(n), 1, f);
    if (n) std::fwrite(s.data(), 1, n, f);
}
bool rstr(std::FILE* f, std::string& s)
{
    std::size_t n;
    if (std::fread(&n, sizeof(n), 1, f) != 1) return false;
    s.resize(n);
    return n == 0 || std::fread(&s[0], 1, n, f) == n;
}
} // namespace

void PrimitiveMesh::writeBinary(const std::string& path) const
{
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    unsigned magic = BRAE_MESH_CACHE_MAGIC;
    std::fwrite(&magic, sizeof(magic), 1, f);
    std::fwrite(&nCells_, sizeof(nCells_), 1, f);
    wvec(f, points_);
    wvec(f, faceVerts_);
    wvec(f, faceOffsets_);
    wvec(f, owner_);
    wvec(f, neighbour_);
    std::size_t np = patches_.size();
    std::fwrite(&np, sizeof(np), 1, f);
    for (const auto& p : patches_)
    {
        wstr(f, p.name);
        wstr(f, p.type);
        std::fwrite(&p.start, sizeof(p.start), 1, f);
        std::fwrite(&p.size, sizeof(p.size), 1, f);
        wstr(f, p.neighbourPatch);
        std::size_t ng = p.inGroups.size();
        std::fwrite(&ng, sizeof(ng), 1, f);
        for (const auto& g : p.inGroups) wstr(f, g);
        wstr(f, p.transform);
        std::fwrite(&p.rotationAxis, sizeof(p.rotationAxis), 1, f);
        std::fwrite(&p.rotationCentre, sizeof(p.rotationCentre), 1, f);
    }
    std::fclose(f);
}

bool PrimitiveMesh::loadBinary(const std::string& path)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned magic = 0;
    bool ok = std::fread(&magic, sizeof(magic), 1, f) == 1 && magic == BRAE_MESH_CACHE_MAGIC;
    ok = ok && std::fread(&nCells_, sizeof(nCells_), 1, f) == 1;
    ok = ok && rvec(f, points_) && rvec(f, faceVerts_) && rvec(f, faceOffsets_) && rvec(f, owner_) && rvec(f, neighbour_);
    std::size_t np = 0;
    ok = ok && std::fread(&np, sizeof(np), 1, f) == 1;
    if (ok)
    {
        patches_.resize(np);
        for (auto& p : patches_)
        {
            ok = ok && rstr(f, p.name) && rstr(f, p.type)
                 && std::fread(&p.start, sizeof(p.start), 1, f) == 1 && std::fread(&p.size, sizeof(p.size), 1, f) == 1
                 && rstr(f, p.neighbourPatch);
            std::size_t ng = 0;
            ok = ok && std::fread(&ng, sizeof(ng), 1, f) == 1;
            if (ok)
            {
                p.inGroups.resize(ng);
                for (auto& g : p.inGroups) ok = ok && rstr(f, g);
            }
            ok = ok && rstr(f, p.transform)
                 && std::fread(&p.rotationAxis, sizeof(p.rotationAxis), 1, f) == 1
                 && std::fread(&p.rotationCentre, sizeof(p.rotationCentre), 1, f) == 1;
            if (!ok) break;
        }
    }
    std::fclose(f);
    return ok;
}

void PrimitiveMesh::readPoints(const std::string& dir)
{
    const std::string path = dir + "/points";
    if (foamFormat(path) == "binary")
    {
        points_ = readBinaryPoints(path);
        return;
    }

    TokenStream ts(path);
    const label n = ts.nextLabel();
    ts.expect("(");
    points_.resize(n);
    for (label i = 0; i < n; ++i)
    {
        ts.expect("(");
        points_[i].x = ts.nextScalar();
        points_[i].y = ts.nextScalar();
        points_[i].z = ts.nextScalar();
        ts.expect(")");
    }
    ts.expect(")");
}

void PrimitiveMesh::readFaces(const std::string& dir)
{
    const std::string path = dir + "/faces";
    if (foamFormat(path) == "binary")
    {
        readBinaryCompactFaces(path, faceOffsets_, faceVerts_);
        return;
    }

    TokenStream ts(path);
    const label n = ts.nextLabel();
    ts.expect("(");
    faceOffsets_.resize(n + 1);
    faceOffsets_[0] = 0;
    faceVerts_.clear();
    for (label f = 0; f < n; ++f)
    {
        const label k = ts.nextLabel();   // points in this face
        ts.expect("(");
        for (label j = 0; j < k; ++j) faceVerts_.push_back(ts.nextLabel());
        ts.expect(")");
        faceOffsets_[f + 1] = faceOffsets_[f] + k;
    }
    ts.expect(")");
}

static std::vector<label> readLabelColumn(const std::string& path)
{
    TokenStream ts(path);
    const label n = ts.nextLabel();
    ts.expect("(");
    std::vector<label> v(n);
    for (label i = 0; i < n; ++i) v[i] = ts.nextLabel();
    ts.expect(")");
    return v;
}

static std::vector<label> readLabelFile(const std::string& path)
{
    return (foamFormat(path) == "binary") ? readBinaryLabelList(path) : readLabelColumn(path);
}

void PrimitiveMesh::readOwner(const std::string& dir)     { owner_     = readLabelFile(dir + "/owner"); }
void PrimitiveMesh::readNeighbour(const std::string& dir) { neighbour_ = readLabelFile(dir + "/neighbour"); }

void PrimitiveMesh::readBoundary(const std::string& dir)
{
    TokenStream ts(dir + "/boundary");
    const label np = ts.nextLabel();
    ts.expect("(");
    patches_.clear();
    patches_.reserve(np);
    for (label p = 0; p < np; ++p)
    {
        PatchInfo pi;
        pi.name = ts.next();
        ts.expect("{");
        while (ts.peek() != "}")
        {
            const std::string key = ts.next();
            if      (key == "type")           { pi.type  = ts.next();      ts.expect(";"); }
            else if (key == "nFaces")         { pi.size  = ts.nextLabel(); ts.expect(";"); }
            else if (key == "startFace")      { pi.start = ts.nextLabel(); ts.expect(";"); }
            else if (key == "neighbourPatch") { pi.neighbourPatch = ts.next(); ts.expect(";"); }
            else if (key == "transform")      { pi.transform = ts.next();      ts.expect(";"); }
            else if (key == "rotationAxis")   { ts.expect("("); pi.rotationAxis   = {ts.nextScalar(), ts.nextScalar(), ts.nextScalar()}; ts.expect(")"); ts.expect(";"); }
            else if (key == "rotationCentre") { ts.expect("("); pi.rotationCentre = {ts.nextScalar(), ts.nextScalar(), ts.nextScalar()}; ts.expect(")"); ts.expect(";"); }
            else if (key == "inGroups")   // wordList: "N(g1 g2 ...)" or "(g1 ...)"
            {
                if (ts.peek() != "(") ts.next();                          // optional leading count token
                ts.expect("(");
                while (ts.peek() != ")") pi.inGroups.push_back(ts.next());
                ts.expect(")");
                ts.expect(";");
            }
            else { while (ts.peek() != ";") ts.next(); ts.expect(";"); }  // skip remaining unsupported keys
        }
        ts.expect("}");
        patches_.push_back(std::move(pi));
    }
    ts.expect(")");
}

void PrimitiveMesh::read(const std::string& polyMeshDir)
{
    // BRAE_MESH_CACHE: skip the (slow) ASCII parse on a warm run by reloading a binary blob, provided it is newer than
    // the polyMesh/owner file (so an edited/regenerated mesh auto-invalidates the cache). Same idea as OF reusing
    // decomposePar's processor* dirs, one cold parse, then fast warm starts.
    const std::string cachePath = polyMeshDir + "/.brae_meshcache";
    {   // AUTO warm-load if a valid cache is present (newer than owner), no env needed, so a `-partition` run makes
        // the subsequent solve warm automatically. Stale/foreign caches are rejected (mtime + magic) -> cold parse.
        std::error_code ec;
        namespace fs = std::filesystem;
        const std::string ownerPath = polyMeshDir + "/owner";
        if (fs::exists(cachePath, ec) && fs::exists(ownerPath, ec)
            && fs::last_write_time(cachePath, ec) >= fs::last_write_time(ownerPath, ec)
            && loadBinary(cachePath))
            return;                                          // warm: reloaded from cache
    }
    const bool writeCache = std::getenv("BRAE_MESH_CACHE") != nullptr;   // write only when asked (-partition / env)
    readPoints(polyMeshDir);
    readFaces(polyMeshDir);
    readOwner(polyMeshDir);
    readNeighbour(polyMeshDir);
    readBoundary(polyMeshDir);

    // nCells = max cell index referenced by owner/neighbour, + 1.
    label maxCell = -1;
    for (const label c : owner_)     maxCell = std::max(maxCell, c);
    for (const label c : neighbour_) maxCell = std::max(maxCell, c);
    nCells_ = maxCell + 1;

    if (writeCache) writeBinary(cachePath);                  // cold: write the cache for next time
}

} // namespace brae
