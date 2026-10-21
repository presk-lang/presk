#!/bin/bash
# One search of CRAFT's variable mode on the exported C (D154): CRAFT's own
# driver (`craft start -V`, Ruby), over the C program's floating-point
# variables (experiments/tuning/retype.py's), with the builder and the test
# CRAFT asks its user to write, here:
#   craft_builder CFG   the program of EXPORT with the variables CFG lowers
#                       in `float` (retype.py apply), built by its Makefile
#                       without OpenMP (one thread, the same bits);
#   craft_driver        runs it on the conformance inputs, pinned to one core,
#                       `-r REPEAT` calls (their median is CRAFT's runtime),
#                       and compares its logits with the exact reference's
#                       (criterion.py --budget BUDGET): `status: pass` when
#                       the largest gap is at most BUDGET.
# Every trial's assignment and figures are appended to DIR/trials.tsv, and
# CRAFT keeps every configuration it tested in DIR/passed, DIR/failed and
# DIR/aborted, with its error and runtime.
#
#   experiments/tuning/craft.sh EXPORT WEIGHTS BUDGET STRATEGY DIR
#
# STRATEGY is CRAFT's: `bin_simple` (its default: the whole program, then
# a failed set split in its structure — files, functions, variables — or in
# halves above nine children) or `ddebug` (its delta-debugging strategy,
# "as described in Rubio-González et al. [SC'13]": Precimonious's search,
# whose cost is the runtime). RUBY, CRAFT (its checkout), CORE, REPEAT.
set -eu
[ $# -eq 5 ] || { sed -n '2,/^set/p' "$0" | sed '$d'; exit 2; }
EXPORT=$(cd "$1" && pwd)
WEIGHTS=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
BUDGET=$3
STRATEGY=$4
DIR=$5
HERE=$(cd "$(dirname "$0")" && pwd)
RUBY=${RUBY:-$HERE/../../.tuning/env/bin/ruby}
CRAFT=${CRAFT:-$HERE/../../.tuning/craft}
CORE=${CORE:-5}
REPEAT=${REPEAT:-5}
PY=${PY:-python3}
RUBY=$(cd "$(dirname "$RUBY")" && pwd)/$(basename "$RUBY")
CRAFT=$(cd "$CRAFT" && pwd)

mkdir -p "$DIR/trials"
DIR=$(cd "$DIR" && pwd)
"$PY" "$HERE/retype.py" variables "$EXPORT" > "$DIR/craft_initial.json"
printf 'trial\tstatus\tgap\tgap_f64\trelative\tms\tfloats\tdoubles\n' > "$DIR/trials.tsv"

cat > "$DIR/craft_builder" <<EOF
#!/bin/bash
# CRAFT's builder: the program with the configuration's variables in float.
"$PY" "$HERE/retype.py" apply "$EXPORT" "\$1" . > build.log 2>&1 && make -s OPENMP= model >> build.log 2>&1 || echo "status:  error"
EOF
cat > "$DIR/craft_driver" <<EOF
#!/bin/bash
# CRAFT's test: the conformance inputs, the exact reference, the budget.
cfg=\$(ls *.json | head -1)
ms=\$( { taskset -c $CORE ./model -r $REPEAT "$WEIGHTS" conformance/inputs.psk.d out.psk.d; } 2>&1 >/dev/null | sed -n 's/.*median \([0-9.]*\) ms.*/\1/p')
if [ -z "\$ms" ]; then echo "status:  error"; exit 0; fi
r=\$("$PY" "$HERE/criterion.py" out.psk.d "$EXPORT/conformance/exact.psk.d" --f64 "$EXPORT/conformance/native.psk.d" --budget $BUDGET) && st=pass || st=fail
g=\$(echo "\$r" | awk '/^gap /{print \$2}')
g64=\$(echo "\$r" | awk '/^gap_f64 /{print \$2}')
rel=\$(echo "\$r" | awk '/^relative /{print \$2}')
n=\$("$PY" -c "import json,sys; print(sum(1 for a in json.load(open(sys.argv[1]))['actions'] if a.get('to_type') == 'float'))" "\$cfg")
total=\$(grep -c '"handle"' "$DIR/craft_initial.json")
t=\$(( \$(wc -l < "$DIR/trials.tsv") ))
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "\$t" "\$st" "\$g" "\$g64" "\$rel" "\$ms" "\$n" "\$((total - n))" >> "$DIR/trials.tsv"
cp "\$cfg" "$DIR/trials/\$t.json"
echo "status:  \$st"
echo "time:    \$(awk -v m="\$ms" 'BEGIN { printf "%.6f", m / 1000 }')"
echo "error:   \$g"
EOF
chmod +x "$DIR/craft_builder" "$DIR/craft_driver"

cd "$DIR"
start=$(date +%s.%N)
"$RUBY" "$CRAFT/scripts/craft" start -V -c craft_initial.json -s "$STRATEGY" -T 120 > craft.out 2>&1
end=$(date +%s.%N)
awk -v a="$start" -v b="$end" 'BEGIN { printf "wall %.1f s\n", b - a }' > wall.txt
"$RUBY" "$CRAFT/scripts/craft" status > craft.status 2>&1 || true
