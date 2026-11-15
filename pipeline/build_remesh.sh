#!/usr/bin/bash
# remesh.elf with the guest API's double precision (this Godot is a double build: single-precision guests do not load).
export PATH=$HOME/.pixi/bin:$PATH
R=$HOME/scratch/remesh-guest
set -e
pixi exec -s clangxx -s lld -s cmake -s ninja -- bash -c "
  cmake -S $R/guest/remesh -B $R/build/remesh-d -G Ninja -DCMAKE_BUILD_TYPE=Release -DDOUBLE_PRECISION=ON \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/weft/3-interactor/mujoco-sandbox-demo/third_party/riscv64-sysroot/toolchain.cmake \
    -DSANDBOX_API=$HOME/weft/2-contract/guest-runtime/vendor/sandbox-api -DMESHOPTIMIZER_DIR=$HOME/scratch/meshoptimizer &&
  cmake --build $R/build/remesh-d -j6"
ls -la $R/build/remesh-d/remesh
