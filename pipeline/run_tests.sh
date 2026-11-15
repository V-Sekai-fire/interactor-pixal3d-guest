#!/usr/bin/bash
# The pipeline's CPU tests as one systemd oneshot (unit: pipeline/pg-pipe-test.service). Exit code and log are the verdict.
export PATH=$HOME/.pixi/bin:$PATH
cd "$HOME/scratch/pg-pipe" || exit 1
pixi run pytest pipeline/tests -q -p no:cacheprovider -rA > /tmp/pg-pipe-test.log 2>&1
rc=$?
grep -E "^(PASSED|FAILED|ERROR)|passed|failed" /tmp/pg-pipe-test.log | tail -20
echo "pg-pipe-test rc=$rc"
exit $rc
