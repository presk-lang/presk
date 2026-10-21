#!/bin/bash
# The tuning tools D154 runs or looks at, fetched into DIR (default .tuning,
# not in git), each at the commit the record names; their sources are never
# copied into this repository.
#   - Ruby 3.1 from conda-forge, by micromamba (CRAFT's driver is Ruby; Ruby
#     3.2 removed the `File.exists?` it calls);
#   - CRAFT (crafthpc/craft) at 03f2514, 2020-06-09: its variable mode, whose
#     driver needs no binary instrumentation;
#   - ADAPT (LLNL/adapt-fp) at 09119e1, 2022-09-26, and CoDiPack at v2.0.2,
#     2022-06-14, the release whose names (`Tape`, `Identifier`) ADAPT's
#     implementation uses: CoDiPack 1.9.3 lacks them and CoDiPack 2.x of
#     2026 no longer builds with it.
# Network access is needed once; a tool already fetched is kept.
set -eu
DIR=${1:-.tuning}
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
if [ ! -x "$DIR/env/bin/ruby" ]; then
    arch=$(uname -m); [ "$arch" = aarch64 ] && plat=linux-aarch64 || plat=linux-64
    curl -sSL "https://micro.mamba.pm/api/micromamba/$plat/latest" | tar -xj -C "$DIR" bin/micromamba
    MAMBA_ROOT_PREFIX=$DIR/mamba "$DIR/bin/micromamba" create -y -q -p "$DIR/env" -c conda-forge "ruby=3.1"
fi
fetch() { # name url commit
    [ -d "$DIR/$1" ] || git clone -q "$2" "$DIR/$1"
    git -C "$DIR/$1" checkout -q "$3"
}
fetch craft https://github.com/crafthpc/craft.git 03f2514
fetch adapt-fp https://github.com/LLNL/adapt-fp.git 09119e1
fetch CoDiPack https://github.com/SciCompKL/CoDiPack.git v2.0.2
{
    echo "ruby       $("$DIR/env/bin/ruby" --version)"
    for t in craft adapt-fp CoDiPack; do
        printf '%-10s %s\n' "$t" "$(git -C "$DIR/$t" log -1 --format='%H %ad' --date=short) $(git -C "$DIR/$t" remote get-url origin)"
    done
}
