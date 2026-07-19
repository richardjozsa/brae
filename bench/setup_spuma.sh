#!/usr/bin/env bash
# ============================================================================
#  Build the SPUMA benchmark back-end (the `spuma` column).
#
#  SPUMA (CINECA / EU-exaFOAM, arXiv:2512.22215, GPLv3) is a full GPU fork of OpenFOAM-v2412: it runs the WHOLE
#  SIMPLE loop on the GPU via three generic wrapper kernels on UNIFIED (managed) memory. It is the closest
#  full-residency peer to brae, so it is the fair "other device-resident port" to compare against. Usage after
#  build is stock OpenFOAM plus two flags:  simpleFoam -pool fixedSizeMemoryPool -poolSize <GB>
#
#  Prereqs:
#    - NVIDIA HPC SDK (nvc++, >= 22.3) on PATH or at NVHPC_DIR   -- SPUMA's GPU backend REQUIRES nvc++
#    - CUDA toolkit (>= 11.6)
#    - a system OpenMPI (SPUMA's etc/bashrc defaults to SYSTEMOPENMPI)
#
#  THE TRAPS (all handled below; full story in results/OF_GPU_SETUP_GB10.md and ../../brae_dev_docs/SPUMA_COMPARISON.md):
#   1. GPU backend is OPT-IN at build time: you MUST export have_cuda=true, else you silently get a CPU-only binary.
#   2. GPU arch: NVARCH defaults to 80 (A100). Set it to YOUR arch (GB10 = 121), or the build runs on the wrong GPU.
#   3. AArch64 GOT overflow: on ARM64, every `-cuda` TU embeds a CUDA fatbin stub; the ~640 stubs overflow the -fPIC
#      GOT when BFD ld links libOpenFOAM.so ("R_AARCH64_LD64_GOTPAGE_LO15 ... __NV_CUDA_LOC"). Fix: link the SHARED
#      libs with ld.lld (via a ld->ld.lld PATH shim), keep BFD ld for executables (lld rejects --add-needed).
#      x86_64 does not hit this; the patch is applied only when WM_ARCH is ARM64.
#   4. The default `dummyMemoryPool` is a perf-killer on GPUs; always run with `-pool fixedSizeMemoryPool`.
#
#  Usage:  ./setup_spuma.sh
#  Env:  SPUMA_SRC (clone target, default $HOME/spuma)   NVARCH (default: autodetect, e.g. 121)
#        NVHPC_DIR (HPC SDK compilers/bin, default: autodetect)   JOBS (default 16)
# ============================================================================
set +u   # OpenFOAM etc/bashrc is not nounset-safe; sourcing it under set -u fails
SPUMA_SRC="${SPUMA_SRC:-$HOME/spuma}"
NVARCH="${NVARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}"
JOBS="${JOBS:-16}"
# find nvc++ (NVIDIA HPC SDK)
NVHPC_DIR="${NVHPC_DIR:-$(dirname "$(command -v nvc++ 2>/dev/null || ls /opt/nvidia/hpc_sdk/*/*/compilers/bin/nvc++ 2>/dev/null | head -1)")}"
[ -x "$NVHPC_DIR/nvc++" ] || { echo "ERROR: nvc++ not found. Set NVHPC_DIR to the HPC SDK compilers/bin."; exit 2; }
export PATH="$NVHPC_DIR:$PATH"
echo "== SPUMA build: nvc++=$("$NVHPC_DIR/nvc++" --version | awk 'NF{print;exit}')  NVARCH=$NVARCH =="

echo "== 1. clone SPUMA =="
[ -d "$SPUMA_SRC/.git" ] || git clone --depth 1 https://gitlab-hpc.cineca.it/exafoam/spuma.git "$SPUMA_SRC" || exit 1
cd "$SPUMA_SRC" || exit 2

echo "== 2. environment: Nvidia compiler + GPU backend on (traps 1 & 2) =="
export WM_COMPILER=Nvidia FOAM_SIGFPE=false      # trap 1 (compiler) + SPUMA requirement
export have_cuda=true NVARCH="$NVARCH"            # trap 1 (backend on) + trap 2 (arch)
# shellcheck disable=SC1091
source etc/bashrc
export have_cuda=true NVARCH="$NVARCH" FOAM_SIGFPE=false   # re-assert (wmake reads these at build time)

echo "== 3. AArch64 GOT-overflow fix: link .so with ld.lld (trap 3) =="
if [ "${WM_ARCH:-}" = "linuxARM64" ]; then
  LLD="$(command -v ld.lld || echo "$NVHPC_DIR/../share/llvm/bin/ld.lld")"
  [ -x "$LLD" ] || { echo "ERROR: ld.lld not found (apt install lld). Needed for the ARM64 .so link."; exit 3; }
  mkdir -p "$SPUMA_SRC/ldshim"; ln -sf "$LLD" "$SPUMA_SRC/ldshim/ld"
  RULE=wmake/rules/General/Nvidia/link-c++
  if ! grep -q 'ldshim' "$RULE"; then
    # prepend the lld shim to PATH for the shared-lib link only; executables keep BFD ld.
    sed -i "s|^LINKLIBSO   = \$(CC)|LINKLIBSO   = PATH=\"$SPUMA_SRC/ldshim:\$\$PATH\" \$(CC)|" "$RULE"
    echo "  patched $RULE (LINKLIBSO -> ld.lld)"
  else
    echo "  $RULE already patched"
  fi
else
  echo "  WM_ARCH=${WM_ARCH:-?} is not ARM64; no GOT-overflow patch needed (x86_64 links cleanly)."
fi

echo "== 4. build (nvc++ is slow; expect a while) =="
./Allwmake -j "$JOBS" 2>&1 | grep -viE '#3560|unsupported in nvc|detected during|instantiation of|Remark #3391' | tail -8

BIN="$SPUMA_SRC/platforms/${WM_OPTIONS}/bin/simpleFoam"
if [ -x "$BIN" ] && ldd "$BIN" 2>/dev/null | grep -q libcudart; then
  echo "== DONE: $BIN (GPU-linked: libcudart present) =="
  echo "   Run a case with:   $BIN -pool fixedSizeMemoryPool -poolSize <GB>     # trap 4 (never the default dummy pool)"
  echo "   Source the env first:  export have_cuda=true NVARCH=$NVARCH; source $SPUMA_SRC/etc/bashrc"
else
  echo "== SPUMA build FAILED or produced a non-GPU binary. Check the log above. =="; exit 1
fi
