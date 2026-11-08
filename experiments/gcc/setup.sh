#!/bin/bash
# GCC 16.2, built from its release into DIR (default .gcc, not in git), for
# D184: the C program under a GCC that knows falicon's cores. Ubuntu 24.04's
# GCC 13.3 knows no Cortex-X925, so its -mcpu=native passes no target there
# (D183); GCC 15.1 added -mcpu=cortex-x925, GCC 16.1 -mcpu=gb10, the NVIDIA
# GB10 (gcc.gnu.org/gcc-16/changes.html). C only, no bootstrap: the stage-1
# compiler built by the system's GCC is the release's compiler, and the
# comparison is of the code it emits, not of how it was built.
#   - the release tarball from ftp.gnu.org, checked against the GNU keyring's
#     signature once (signed 2026-08-07 by the release manager, Richard
#     Biener), then against its SHA-256 below;
#   - GMP, MPFR, MPC and isl by the release's contrib/download_prerequisites,
#     which checks their SHA-512;
#   - DIR/wrap holds `gcc` and `cc` calling DIR/bin/gcc with an rpath to its
#     own libgomp: put it first on PATH, and the experiments' scripts (which
#     call `gcc`, and the exported Makefiles, `$(CC)`) take this GCC.
# Network access is needed once; a build already installed is kept. Three
# minutes on falicon's twenty cores, 1.4 GB installed and 4.8 GB of sources
# and build left in DIR/src; run it in its own scope (CLAUDE.md).
set -eu
VERSION=16.2.0
SHA256=e6738e29597f733270731aa90600f37ffdc045079dfc27ec7e8192cc81085c3e
DIR=${1:-.gcc}
mkdir -p "$DIR/src"
DIR=$(cd "$DIR" && pwd)
if [ ! -x "$DIR/bin/gcc" ]; then
    cd "$DIR/src"
    [ -f gcc-$VERSION.tar.xz ] || curl -fsSLO https://ftp.gnu.org/gnu/gcc/gcc-$VERSION/gcc-$VERSION.tar.xz
    echo "$SHA256  gcc-$VERSION.tar.xz" | sha256sum -c -
    [ -d gcc-$VERSION ] || tar -xJf gcc-$VERSION.tar.xz
    (cd gcc-$VERSION && ./contrib/download_prerequisites)
    rm -rf build && mkdir build && cd build
    ../gcc-$VERSION/configure --prefix="$DIR" --enable-languages=c \
        --disable-multilib --disable-bootstrap --disable-nls --disable-libsanitizer \
        > "$DIR/configure.log" 2>&1
    make -j"$(nproc)" > "$DIR/build.log" 2>&1
    make install > "$DIR/install.log" 2>&1
fi
LIB=$(dirname "$(readlink -f "$("$DIR/bin/gcc" -print-file-name=libgomp.so)")")
mkdir -p "$DIR/wrap"
for n in gcc cc; do
    printf '#!/bin/sh\nexec %s -Wl,-rpath,%s "$@"\n' "$DIR/bin/gcc" "$LIB" > "$DIR/wrap/$n"
    chmod +x "$DIR/wrap/$n"
done
"$DIR/wrap/gcc" --version | head -1
