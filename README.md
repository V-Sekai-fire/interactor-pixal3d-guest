# interactor-pixal3d-guest

Pixal3D (image to 3D, a TRELLIS.2 pipeline) as a godot-sandbox guest ELF whose ggml graph runs on ggml-rd.

## Provenance

Not a fork. The sources are copied into `vendor/pixal3d-ggml` from V-Sekai-archive/pixal3d-ggml at commit `1c22f5e98c90e8cf00d12e99547c8d524421f532` (MIT, see LICENSE; copyright rms80). Model code is kept unchanged except where bytes come from: `trellis2_set_io` lets the DINOv3 loader read the GGUF through a host callback, because the guest cannot open files. The patch embedding builds its im2col columns in f32 (`dino_conv_2d_f32`): the shared ggml's `ggml_conv_2d` rounds them to f16, which put the first tap at rel-L2 3e-4 against the reference. ggml is not vendored: the guest builds against the ggml that ggml-rd uses. RFD66606-2336 owns the design; the parity gate is each stage against the PyTorch reference.

## Plan

1. Spike: one stage (the DINOv3 encoder, `pixal3d_dino.elf`) on ggml-rd, parity against the PyTorch reference dump (`vendor/pixal3d-ggml/scripts/dump_dino_reference.py`), with a planted control (`control=perturb` must report DETECTED).
2. An authoritative `supports_op` walk over the built graphs.
3. bf16 GGUFs as the reference, then Q4_0 and Q8_0 `mul_mat` kernels in ggml-rd, which the large flow models need.
4. Guest wiring (read callbacks, pump), then speed against ggml-vulkan.

## Licence

MIT; see LICENSE.

## Spike result (DINOv3 encoder)

`pixal3d_dino.elf` on ggml-rd, RTX 4090 through Godot's RenderingDevice, f32 GGUF, 512 px input, against a CPU fp32 PyTorch dump (transformers 4.57.1): 40 taps, 0 failed, 0 missing, worst rel-L2 9.8e-7 (the cond output), `|got-ref| <= 2e-3 + 2e-3 |ref|` on every element. With `control=perturb` (input nudged by 0.05 sin(i)) the gate reports `CONTROL DETECTED`, 38 of 40 taps failing. With the stock `ggml_conv_2d` the same gate failed 38 of 40 taps, which is how the f16 columns were found. Run: `godot --path tests/dino --script run_dino.gd -- --elf=... --model=dino_f32.gguf --ref=reference_dino.gguf`. The Godot process aborts at exit after printing the result line (exit 134), so read the `RESULT:` line, not the exit code.
