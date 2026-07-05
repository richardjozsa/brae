#!/bin/sh
# brae installer  ,  clones, builds, and installs the brae CFD engine.
#
#   curl -fsSL https://brae.sh/install.sh | sh
#
# Optional environment overrides:
#   BRAE_DIR        where to clone the source     (default: $HOME/brae)
#   BRAE_REF        branch or tag to build        (default: main)
#   BRAE_CUDA_ARCH  GPU compute capability, e.g. 90, 121, or "native"
#                   (default: auto-detected from nvidia-smi)
#   PREFIX          install prefix for the binary (default: $HOME/.local)
set -eu

REPO="${BRAE_REPO:-https://github.com/simd-ai/brae.git}"
REF="${BRAE_REF:-main}"
DIR="${BRAE_DIR:-$HOME/brae}"
PREFIX="${PREFIX:-$HOME/.local}"
BINDIR="$PREFIX/bin"

# ---- logging ----
if [ -t 1 ]; then
    B=$(printf '\033[1m'); G=$(printf '\033[32m'); Y=$(printf '\033[33m'); R=$(printf '\033[31m'); N=$(printf '\033[0m')
else
    B=''; G=''; Y=''; R=''; N=''
fi
say()  { printf '%s==>%s %s\n' "$G$B" "$N" "$*"; }
warn() { printf '%s!! %s%s\n'  "$Y$B" "$*" "$N" >&2; }
die()  { printf '%serror:%s %s\n' "$R$B" "$N" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

say "installing brae"

# ---- CUDA on PATH (it often lives in /usr/local/cuda/bin) ----
if ! have nvcc && [ -x /usr/local/cuda/bin/nvcc ]; then
    PATH="/usr/local/cuda/bin:$PATH"; export PATH
fi

# ---- required tools ----
missing=''
for t in git cmake nvcc; do have "$t" || missing="$missing $t"; done
have c++ || have g++ || missing="$missing c++"
have make || have ninja || missing="$missing make"
[ -z "$missing" ] || die "missing required tools:$missing
  On Ubuntu/Debian:  sudo apt-get install -y git cmake build-essential
  CUDA toolkit (nvcc), 12.4+ (13.x recommended): https://developer.nvidia.com/cuda-downloads"

# ---- cmake >= 3.24 ----
cmake_ver=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1)
if [ "$(printf '3.24\n%s\n' "$cmake_ver" | sort -V 2>/dev/null | head -n1)" != "3.24" ]; then
    die "cmake $cmake_ver found, brae needs >= 3.24"
fi

# ---- clone or update ----
if [ -d "$DIR/.git" ]; then
    say "updating existing checkout in $DIR"
    git -C "$DIR" fetch --depth 1 origin "$REF"
    git -C "$DIR" reset --hard FETCH_HEAD
else
    say "cloning $REPO ($REF) into $DIR"
    git clone --depth 1 --branch "$REF" "$REPO" "$DIR"
fi

# ---- detect target GPU arch ----
arch="${BRAE_CUDA_ARCH:-}"
if [ -z "$arch" ] && have nvidia-smi; then
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
fi
[ -n "$arch" ] || arch="native"
say "target GPU arch: $arch"

# ---- configure + build ----
cd "$DIR"
say "configuring"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" || die \
"cmake configure failed, usually a missing library.
  On Ubuntu/Debian:  sudo apt-get install -y libopenmpi-dev openmpi-bin libscotch-dev zlib1g-dev"

jobs=$(nproc 2>/dev/null || echo 4)
say "building brae with $jobs jobs (this can take a few minutes)"
cmake --build build -j "$jobs" --target brae || die "build failed"

# ---- install the binary ----
mkdir -p "$BINDIR"
install -m 0755 build/brae "$BINDIR/brae" 2>/dev/null || cp build/brae "$BINDIR/brae"
say "installed  $BINDIR/brae"

# ---- PATH hint ----
case ":$PATH:" in
    *":$BINDIR:"*) ;;
    *) warn "$BINDIR is not on your PATH. Add it with:"
       printf '      echo '\''export PATH="%s:$PATH"'\'' >> ~/.profile && . ~/.profile\n' "$BINDIR" >&2 ;;
esac

printf '\n%s%sbrae is installed.%s  Run it inside any OpenFOAM case:\n\n' "$G" "$B" "$N"
printf '    cd yourCase && brae\n\n'
printf '  docs: https://brae.sh   source: %s\n' "$DIR"
