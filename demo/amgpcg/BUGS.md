# AMG/PCG bug audit (single-GPU path)

Static analysis of `src/cuda/device_amg.cu`, `device_pcg.cu`, `device_spmv.cu`, `device_blas.*`,
`device_ldu.cuh`, done 2026-07-23. Findings are ordered applied-first, then deferred.

**All applied fixes are now GPU-VERIFIED** (2026-07-23, GB10) via `demo/amgpcg/verify.sh`:
- regression: `ctest -R "gpu_amg|gamg|pcg"` → 14/14 pass (F1/F3/F5/D2 all live on the default path)
- Chebyshev: `BRAE_CHEBYSHEV=1` at 1M → **79 iters, final 9.0e-9** (was 5000 stalled at 1.6e-3)
- D2: device graph PCG vs host loop on a real simpleFoam case → **same 13 iters**, forces agree to ~0.01%
- F4: `BRAE_AMG_SA` at 1M → 36 iters (25 with SOC), unchanged, so `safeDiag` is the intended no-op

Build note that shaped the audit: `--default-stream per-thread` is set globally
(`CMakeLists.txt:22`), so a bare `<<<>>>` launch is `cudaStreamPerThread` — there is no
legacy-vs-per-thread stream-mixing hazard anywhere in these files.

---

## Applied (GPU-verified 2026-07-23; each is a strict no-op for every currently-working config)

### F1 — Chebyshev spectrum estimate frozen across an evolving operator
`device_amg.cu` `ensureSpectrum`. `spectrumReady` was set once and never invalidated;
`amgGalerkin` rebuilds the coarse operators every SIMPLE step but left the flag set. Across
steps the pressure matrix is re-weighted non-uniformly (face fluxes / Ap evolve with velocity),
so `lambda_max(D^-1 A)` drifts and the frozen Chebyshev interval stops covering the top modes →
the smoother stops damping them. Compounds the seed/iteration bug (below): F1 froze an already
under-estimated step-0 value for the whole run. **Fix:** `A.spectrumReady = false;` at the end
of `amgGalerkin`. Zero cost unless `BRAE_CHEBYSHEV` is on (`ensureSpectrum` early-returns).

### (also) Chebyshev power-iteration seed + interval — the actual 1M divergence
`estimateLambdaMax` seeded from the constant vector (the Laplacian near-null-space), so power
iteration reached `lambda_max` only through round-off and badly under-estimated it in 12 steps;
an under-covered Chebyshev interval amplifies the top modes → divergence, worsening with depth
(stalled at 1.6e-3 on the 1M case). **Fix:** period-7 sawtooth seed (`fillSeedK`), 25 iterations
not 12, and `CHEB_UPPER_SAFETY = 1.2` (was 1.1). Removed the now-dead `fillK`.
**GPU-verified 2026-07-23: converges in 79 iters, final 9.0e-9 (was 5000 stalled).** 79 not ~30
because Chebyshev on UNSMOOTHED aggregation still tracks hierarchy depth; SA is the depth fix, not
the smoother. If it ever stalls again, the bulletproof lever is a Gershgorin upper bound
(`lambda_max(D^-1 A) <= 2` for a Laplacian) instead of the power estimate.

### F3 — dead duplicate branch in the coarsest-solve dispatch
`device_amg.cu` `vcycleAt`. Two identical `else if (n <= SB_MAX) deviceCoarseJacobiSingleBlock(...)`
lines; the second was unreachable. Behaviour was unaffected (both called the same kernel), but it
was a maintenance trap implying an intended larger-coarsest path that no longer exists.
**Fix:** removed the duplicate. Provably behaviour-preserving.

### F4 — unguarded coarse-diagonal division → NaN under smoothed aggregation
`device_amg.cu`, the coarse Jacobi/PCG kernels (`coarseJacobiFusedKernel`,
`coarseJacobiSingleBlockKernel`, `coarsePCGKernel`). Default injection Galerkin always yields a
positive coarse diagonal, but the general RAP under `BRAE_AMG_SA` can produce a near-zero or
negative one (the `BRAE_AMG_DEBUG` block reports exactly this). Dividing by it gives Inf/NaN that
propagates through prolongation into the outer residual and runs the solve to `maxIter` on
garbage. **This matters because SA is the recommended config for large meshes** (see README's
mesh-independence table). **Fix:** new `safeDiag()` helper flooring |d| at 1e-300, applied at the
four coarse-solve divisions. Strict no-op for any diagonal of magnitude > 1e-300, so it cannot
perturb a well-formed operator; it only replaces a divide-by-(near)-zero. Note it prevents the
Inf but cannot fix an indefinite coarse operator — if SA produces negative diagonals the real
fix is in the aggregation/RAP, and `BRAE_AMG_DEBUG` is how you detect it.

### D2 — PCG graph cache omitted tol/relTol/maxIter from its key (the known bug (b), full scope)
`device_amg.cu` `deviceAMGPCGGraph` (local) and `deviceParallelAMGPCGGraph` (distributed), both
using `PCGGraphCache`. The cache was keyed on `psi.data()` alone, but `pcgSetCondK` bakes `tol`,
`relTol` and `maxIter-1` into the captured conditional node. Reusing the cache (same `psi`) with
any control changed silently replayed the stale bound/tolerance — the reason `trace_brae` had to
set `BRAE_PCG_DEVICE=0`. **Fix (applied):** added `keyTol/keyRelTol/keyMaxIter` (sentinel
-1) to `PCGGraphCache`; both capture guards now compare all four and both key-sets store all
four. Zero extra re-capture in the SIMPLE loop (the controls are constant per field there), so
this is a strict no-op for the working case and only re-captures when a control actually changes.
The cache is already per-solver-lifetime (owned by `AMGData`), so this does not touch the
process-global-cache hazard in D1. **GPU-verified 2026-07-23: device vs host both converge in 13 iters, forces agree ~0.01%.**

### F5 — `cudaFuncSetAttribute` on the hot path / inside a captured region
`device_amg.cu` `deviceCoarseJacobiFused`. The >48KB shared-memory opt-in was issued on every
call; that function is reachable from the stream-captured `vcycleAt`, and a non-stream runtime
call during capture is at best ignored and at worst refuses the capture. **Fix:** wrapped in
`std::call_once` (the attribute is a per-function property, so once is enough). Only reachable
with an unusually high `BRAE_AMG_TARGET` or a coarsest level above `SB_MAX`; the default
`TARGET=64` always takes the single-block path, so this did not fire by default.

---

## Deferred (real, but touch the default hot path / multi-GPU / need GPU validation)

(D2 moved to Applied above; D1/D3/D4/D5 keep their original audit labels, so the gap is intentional.)

### D1 — GS device-graph cache is process-global and pointer-keyed (adjacent to the known maxIter bug)
`device_amg.cu`, the `BRAE_GS_DEVICE` path: `static auto& cache = *new std::map<const void*,
GSGraphCache>()`, keyed on `psi.data()` for the whole process. This is the exact pattern the
`PCGGraphCache` header comment (`device_amg.cuh:26-28`) rejects as unsafe: after a field is freed
and its buffer address recycled to a new field (multi-region / transient remesh), the stale entry
replays a graph baked against the wrong/freed buffers → illegal access. It also bakes
`tol/relTol/maxIter` into the captured node, so changed stop criteria on a key hit silently
replay the old bound. Same class in the distributed graphs and the momentum BiCGStab graph.
**Fix (not applied):** make the GS graph cache per-solver-lifetime (own it in the field/solver,
as `AMGData` owns `PCGGraphCache`), and fold `tol/relTol/maxIter` into the key or refresh them
via device scalars. Does not fire for a fixed set of persistent fields on one mesh — the common
steady case — which is why it is deferred rather than applied blind.

### D3 — `BRAE_PARALLEL_GRAPH>=2` silently drops the Nicolaides coarse space (multi-GPU)
`device_pcg.cu` / `deviceParallelAMGPCGGraph`: the direct `deviceParallelAMGPCG` adds a
subdomain-constant coarse correction `R0^T A0^-1 R0` to the block-Jacobi V-cycle; the whole-loop
graph path omits it. So enabling the graph performance flag silently switches to a strictly
weaker one-level preconditioner whose iteration count grows with `nProcs` and can diverge at
scale. **Fix (not applied — multi-GPU, out of the single-GPU scope you asked about):** add the
coarse correction into the captured body, or refuse the graph path when the coarse space is
active and fall back to the direct path.

### D4 — V-cycle graph cache omits `corrScaling` from its key (latent, family of (b))
`AMGGraphCache` keyed on `A.diag` only, while `corrScaling` is read inside the captured
`vcycleAt`. Does not fire with current callers (`corrScaling` is a per-run constant).
**Fix (not applied):** include `corrScaling` and the smoother mode in the key.

### D5 — device PCG alpha/beta divisions unguarded (defense-in-depth only)
`deviceAMGPCGGraph` `alpha = wArA/pAp`, `beta = wArA/wArAold`. Safe in the intended regime (SPD
operator + SPD preconditioner keep these strictly positive until the residual test fires first),
and it mirrors the CPU oracle — but a near-breakdown in the FP32 path would NaN and only stop at
`maxIter`. **Fix (optional):** the same 1e-300 guard used in the coarse PCG.

---

## Audited clean (checked, no defect)

Stream mixing (per-thread default stream, all one stream); no illegal host sync/reduction inside
any captured region (`*Into` device-scalar variants + `cudaMemsetAsync` only; `ensureSpectrum`
runs pre-capture); graph-referenced buffer lifetimes (pool never frees mid-run, `resize` a no-op
at constant size, so no pointer moves under a live graph — except the process-global GS cache,
D1); `fp32Alloc` correctly starts false per fresh `AMGData` and `amgCastFP32` re-casts values
every solve; WHILE-graph re-capture destroys old exec/graph and does not double-free the
conditional body graph; prefix sums / `losort` / coarse addressing; `faceRestrict`/`faceFlip`
`-1-coarseCell` encoding; agglomeration completeness (no empty or self-referential aggregates);
RAP triple-recipe bounds; the cluster ping-pong in `coarseJacobiFusedKernel`; `blockDot`
barriers and warp masks; `coarsePCGKernel` shared-memory sync and sizing; scatter atomics into
freshly zeroed coarse buffers; `scaleFactorK`/`flexBetaK` denominator guards; FP32 cast
directions and SA/GS/Chebyshev gating; convergence tests terminate under `tol=relTol=0` and under
NaN via the `maxIter` guard.
