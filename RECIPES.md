# Recipes

Each recipe says what it needs, the commands, and the line that tells you it worked. Paths are variables; nothing here
names a person's machine. Every long job runs as a systemd oneshot (see the units in `pipeline/`).

Variables used below:

| Variable | Meaning |
|---|---|
| `REPO` | a checkout of this repo (branch `dev/vendor-pixal3d-ggml`) |
| `WORK` | a scratch directory with room for 60 GB |
| `GODOT_BIN` | a Godot 4.8 editor build with double precision |
| `GODOT_PROJECT` | a Godot project holding `addons/godot_sandbox` (double-precision build) |
| `HF_TOKEN` | a Hugging Face token; pass it in the environment, never on a command line |

## 1. Build the guest ELFs

Needs: cmake, ninja, clang++, ld.lld, slangc, spirv-val, python3, elixir, and the riscv64 sysroot (its `toolchain.cmake`).

The build wants a "goal manifest" layout. Make a root `WEFT_ROOT` with these links:

```
WEFT_ROOT/2-contract/ggml          -> a checkout of contract-ggml
WEFT_ROOT/2-contract/ggml-rd       -> a checkout of contract-ggml-rd on branch dev/q4_0-mul-mat (the Q4_0 MUL_MAT kernels)
WEFT_ROOT/2-contract/guest-runtime -> a checkout of contract-guest-runtime
WEFT_ROOT/3-interactor/pixal3d-guest -> REPO
```

```
cd REPO
WEFT_ROOT=<root above> RISCV64_SYSROOT=<sysroot with toolchain.cmake> elixir tools/build.exs --build=<dir>
```

Works when it prints `== pixal3d_stages.elf: <bytes> bytes, sha256 <hash>`. The build is Release with `DOUBLE_PRECISION=ON`
(this Godot is a double build; a single-precision guest does not load and prints "this program's Variant is 24 bytes").
The ELF with the decoder stage (`pixal3d_ssdec_start`) is the one built from this branch; check with
`strings <elf> | grep pixal3d_ssdec_start`.

## 2. Build remesh.elf (double precision)

Needs the same toolchain, a checkout of interactor-remesh-guest, meshoptimizer at commit
`9e1f07b159d3cb777f1c67ed31fc11fd117986f4`, and the sandbox-api under contract-guest-runtime (`vendor/sandbox-api`).
`pipeline/build_remesh.sh` is the whole recipe; run it through `pipeline/pg-remesh-build.service`.
Works when `ls <build>/remesh-d/remesh` shows the file and `pipeline/remesh.py` `gate(0)` returns `GATE PASS`.

## 3. Weights and references (the sparse-structure path)

Needs the TRELLIS.2 checkpoints (`microsoft/TRELLIS-image-large` for the decoder, `microsoft/TRELLIS.2-4B` for the rest),
a DINOv3 ViT-L/16 mirror, and a Python with torch, transformers, safetensors, gguf.

1. DINOv3 reference: `python vendor/pixal3d-ggml/scripts/dump_dino_reference.py --image <rgba image> --model <dinov3 dir> --out reference_dino.gguf`.
2. GGUFs: `convert_dino_to_gguf.py`, `convert_ss_flow_to_gguf.py --ftype 1` (f16) or `--ftype 3` (Q4_0), `convert_ss_dec_to_gguf.py --ftype 1`.
   Each takes `--model <safetensors> --output <gguf>`; read `--help` for the rest.
3. Flow and decoder references: `vendor/pixal3d-ggml/tests/ref_ss_flow.py`, `ref_ss_flow_q4.py` (the same weights rounded to Q4_0, for the kernel check) and
   `ref_ss_dec.py`. The decoder reference takes `--latent <z_s>` from `tools/sample_ss_latent.py --trellis2 <TRELLIS.2 checkout> --ckpt <stem> --dinodata <fixture.dinodata> --out <latent>`.

## 4. Run a stage on ggml-rd

`tests/stages/run_stage.gd` is a Godot script. Run it with a window (not `--headless`) and a GPU:

```
cd GODOT_PROJECT && GODOT_BIN --path . --script run_stage.gd -- --elf=<pixal3d_stages.elf> --stage=<dino|ssflow|ssdec> --model=<gguf> --ref=<reference.bin> [--opts="control=perturb"]
```

Works when the log has `RESULT: PASS`. Add `--opts="control=perturb"`: it must print `CONTROL DETECTED`, or the check cannot fail.
Reference numbers: dino worst rel-L2 9.8e-7 (40 taps); ssflow f16 5.5e-4; ssdec 1.4e-4 with one flipped voxel.
Wrap the call in `timeout 120`: Godot sometimes stays alive after printing.

## 5. Q4_0 QAFT of the sparse-structure flow

```
python tools/qaft_ss_flow.py --trellis2 <checkout> --scripts <vendor/pixal3d-ggml/tests> --ckpt <flow stem> --dino <dinov3 dir> --images <example images> --out <ss_flow_qaft.safetensors> [--n-train 24 --epochs 2]
```

It took 53 minutes on an RTX 4090 and wrote `ss_flow_qaft.safetensors` and a `.json` of per-block errors.
Measure with `tools/eval_ss_q4.py ... --variant name=<safetensors>`. Result: mean voxel IoU 0.844, worst 0.683, so this is not stable enough to ship.

## 6. The pixi pipeline

```
pixi run -e gpu setup      # CUDA wheels, Alchemy3D checkout plus pipeline/alchemy3d.patch, Blender
pixi run -e gpu fetch      # the weights, into PIXAL3D_HOME
export PIXAL3D_IMAGE=<front view> PIXAL3D_INSTRUCTION="<edit sentence>"   # PIXAL3D_VIEWS and PIXAL3D_MAPS are optional
pixi run pipeline          # generate, segment, edit, bake, render
pixi run test              # the gates and their planted controls (CPU)
```

Never set `A3D_BF16`: a bfloat16 cast corrupts the Alchemy3D sparse-structure stage (kept-source score 0.71 against the 0.8 gate).
Run the heavy tasks as oneshot units; a 24 GB card needs `A3D_OFFLOAD=1`, which `pipeline/common.py` sets.

## 7. Binary-translate a guest

`pipeline/bintr.sh <elf>` (remesh) and `pipeline/bintr_stages.sh` (stages) emit the guest's C with `GODOT_SANDBOX_BINTR_EMIT`,
compile it to `bintr-<HASH>.so`, point `project.godot` at the directory (`binary_translation/enabled=true`, `cache_dir`),
and print `BINTR translated=true`. Rebuild whenever the ELF or the addon changes: the hash covers both.
Measured: the remesh gate 214 ms interpreted, 47 ms translated; the ssdec stage 1432 ms against 1360 ms.

## 8. Save and check the work

`pipeline/upload_work.py` (unit `pg-upload-work`) puts the non-git files in a private model repo and reconciles sizes.
`pipeline/check_upload.sh` (unit `pg-check-upload`) downloads that repo fresh, checks every sha256, runs the tests from the
source archive, runs the decoder stage from the downloaded files, and runs it again with a corrupted reference, which must fail.
Both read `HF_TOKEN` from an `EnvironmentFile` under `/run` that you delete afterwards.
