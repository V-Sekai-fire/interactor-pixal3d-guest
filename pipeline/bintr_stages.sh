#!/usr/bin/bash
# Binary-translate pixal3d_stages.elf: time a stage interpreted, emit + compile, time it translated, compare the verdicts.
#   env: GODOT_BIN GODOT_PROJECT ELF MODEL REF STAGE (default ssdec)
set -u
P=$GODOT_PROJECT; H=$(dirname "$0"); C=$P/bintr; STAGE=${STAGE:-ssdec}
cp "$H/../tests/stages/run_stage.gd" "$P/run_stage.gd"
stage() { timeout 150 "$GODOT_BIN" --path "$P" --script run_stage.gd -- --elf="$ELF" --stage=$STAGE --model="$MODEL" --ref="$REF" 2>&1 | grep -o "RESULT: [A-Z]*.*ms=[0-9]*" | head -1; }
echo "-- interpreted"; A=$(stage); echo "$A"
mkdir -p "$C/emit"; rm -f "$C"/emit/*.c
GODOT_SANDBOX_BINTR_EMIT="$C/emit" timeout 120 "$GODOT_BIN" --path "$P" --script "$H/load_elf.gd" -- --elf="$ELF" 2>&1 | grep -E "^BINTR|ERROR: Sandbox"
CC=$(command -v clang || command -v gcc)
for f in "$C"/emit/bintr-*.c; do
  b=$(basename "$f" .c); [ -f "$C/$b.so" ] && continue
  echo "compile $b $(du -h "$f" | cut -f1)"
  nice $CC -O2 -s -std=c99 -shared -fPIC -x c -fexceptions -fvisibility=hidden -fomit-frame-pointer "$f" -o "$C/$b.so" 2>/dev/null || { echo "FAILED compile $b"; exit 1; }
done
timeout 120 "$GODOT_BIN" --path "$P" --script "$H/load_elf.gd" -- --elf="$ELF" 2>&1 | grep -E "^BINTR"
echo "-- translated"; B=$(stage); echo "$B"
strip() { echo "$1" | sed 's/ms=[0-9]*//'; }
[ -n "$A" ] && [ "$(strip "$A")" = "$(strip "$B")" ] && echo "VERDICTS IDENTICAL" || { echo "FAILED verdicts differ"; exit 1; }
