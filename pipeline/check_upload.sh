#!/usr/bin/bash
# Reproducibility check of chibifire/pixal3d-work: download it fresh, compare hashes with the live originals,
# then run the gates from the downloaded files only (source archive, ELFs, GGUF, reference), plus a planted control.
# env: HF_TOKEN GODOT_BIN GODOT_PROJECT
export PATH=$HOME/.pixi/bin:$PATH
W=$HOME/scratch/pg-verify; rm -rf "$W"; mkdir -p "$W"; cd "$W" || exit 1
fail() { echo "FAILED $*"; exit 1; }
echo "-- download"
uv run --quiet --with huggingface_hub,hf_xet python - <<'PY' || fail download
import hashlib, os, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=os.environ["HF_TOKEN"])
d = Path(snapshot_download("chibifire/pixal3d-work", local_dir="dl", token=os.environ["HF_TOKEN"]))
info = api.model_info("chibifire/pixal3d-work", files_metadata=True)
bad = []
for s in info.siblings:
    p = d / s.rfilename
    if s.rfilename == ".gitattributes":
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    want = s.lfs.sha256 if s.lfs else None
    if want and h.hexdigest() != want:
        bad.append(s.rfilename)
print("HASHES", len(info.siblings), "files, bad:", bad)
sys.exit(1 if bad else 0)
PY
echo "-- source archive"
mkdir src && tar -xzf dl/source/interactor-pixal3d-guest.tar.gz -C src --strip-components=1 || fail untar
cd src
export PIXAL3D_HOME=$W/home PIXAL3D_OUT=$W/out REMESH_ELF=$W/dl/elf/remesh_double.elf REMESH_TIMEOUT=120
export DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/run/user/1000
mkdir -p "$PIXAL3D_OUT"
echo "-- pixi test (fresh env from the archive)"
pixi run pytest pipeline/tests -q -p no:cacheprovider 2>&1 | grep -E "passed|failed|error" ; [ "${PIPESTATUS[0]}" = 0 ] || fail pixi-test
echo "-- stage from downloaded files"
cp tests/stages/run_stage.gd "$GODOT_PROJECT/run_stage.gd"
stage() { timeout 150 "$GODOT_BIN" --path "$GODOT_PROJECT" --script run_stage.gd -- --elf="$W/dl/elf/pixal3d_stages_q4.elf" --stage=ssdec --model="$W/dl/gguf/ss_dec_f16.gguf" --ref="$1" 2>&1 | grep -o "RESULT: [A-Z]*.*ms=[0-9]*" | head -1; }
A=$(stage "$W/dl/reference/ss_dec_ref.bin"); echo "clean:   $A"
python3 - "$W/dl/reference/ss_dec_ref.bin" "$W/ref_planted.bin" <<'PY'
import sys
b = bytearray(open(sys.argv[1], "rb").read())
for i in range(len(b) - 4096, len(b) - 4, 4):  # perturb the tail of the reference (its expected logits)
    b[i + 3] ^= 0x40
open(sys.argv[2], "wb").write(b)
PY
B=$(stage "$W/ref_planted.bin"); echo "planted: $B"
case "$A" in "RESULT: PASS"*) ;; *) fail "clean stage did not pass";; esac
case "$B" in "RESULT: FAIL"*) ;; *) fail "planted control did not fail";; esac
echo "REPRODUCED: hashes match, gates pass from the archive, planted control fails"
