#!/usr/bin/bash
# The reproducibility test for the whole pipeline: a fresh export of the repo, a fresh pixi environment, setup, fetch, run.
#   env: SRC_TAR (git archive of the repo) IMAGE INSTRUCTION
export PATH=$HOME/.pixi/bin:$PATH
W=$HOME/scratch/pg-fresh; rm -rf "$W"; mkdir -p "$W"; cd "$W" || exit 1
tar -xzf "$SRC_TAR" --strip-components=1 || exit 1
export PIXAL3D_HOME=$W/.cache PIXAL3D_OUT=$W/out PIXAL3D_IMAGE=$IMAGE PIXAL3D_INSTRUCTION="$INSTRUCTION"
step() { echo "== $*"; "$@"; rc=$?; [ $rc = 0 ] || { echo "FAILED rc=$rc: $*"; exit $rc; }; }
step pixi install -e gpu
step pixi run -e gpu setup
step pixi run -e gpu fetch
step pixi run test
step pixi run pipeline
ls -la "$PIXAL3D_OUT"
echo "FRESH RUN OK"
