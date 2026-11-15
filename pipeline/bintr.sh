#!/usr/bin/bash
# Binary-translate a guest ELF for the Godot project (godot-sandbox-build-workflow, references/binary-translation.md):
# emit the guest's C once, compile it to bintr-<HASH>.so, point the project at the directory, prove is_binary_translated.
#   bintr.sh <elf>    env: GODOT_BIN GODOT_PROJECT (holds addons/godot_sandbox)
set -u
ELF=$1
P=$GODOT_PROJECT
H=$(dirname "$0")
C=$P/bintr
mkdir -p "$C/emit"
rm -f "$C"/emit/*.c "$C"/*.so
setting() { python3 - "$P/project.godot" "$1" <<'PY'
import re, sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read()
s = re.sub(r"binary_translation/(enabled|cache_dir)=.*\n", "", s)
if d:
    s = s.replace("[sandbox]\n", f'[sandbox]\n\nbinary_translation/enabled=true\nbinary_translation/cache_dir="{d}"\n', 1)
open(p, "w").write(s)
PY
}
run() { timeout 300 "$GODOT_BIN" --path "$P" --script "$H/remesh.gd" -- --elf="$ELF" --gate=0 2>&1 | grep -E "^(GATE|BINTR)|ERROR: Sandbox"; }
setting "$C/"
echo "-- emit"; GODOT_SANDBOX_BINTR_EMIT="$C/emit" run
n=$(ls "$C"/emit/bintr-*.c 2>/dev/null | wc -l)
[ "$n" -ge 1 ] || { echo "FAILED emit: no C written"; exit 1; }
CC=$(command -v clang || command -v gcc)
for f in "$C"/emit/bintr-*.c; do
  b=$(basename "$f" .c)
  echo "compile $b $(du -h "$f" | cut -f1)"
  nice $CC -O2 -s -std=c99 -shared -fPIC -x c -fexceptions -fvisibility=hidden -fomit-frame-pointer "$f" -o "$C/$b.so" || { echo "FAILED compile $b"; exit 1; }
done
echo "-- translated run"; run
