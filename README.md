# interactor-pixal3d-guest

Pixal3D (image to 3D, a TRELLIS.2 pipeline) as a godot-sandbox guest ELF whose ggml graph runs on ggml-rd.

## Provenance

Not a fork. The sources are copied into `vendor/pixal3d-ggml` from V-Sekai-archive/pixal3d-ggml at commit `1c22f5e98c90e8cf00d12e99547c8d524421f532` (MIT, see LICENSE; copyright rms80). Model code is kept unchanged except where bytes come from: `trellis2_set_io` lets the DINOv3 loader read the GGUF through a host callback, because the guest cannot open files. The patch embedding builds its im2col columns in f32 (`dino_conv_2d_f32`): the shared ggml's `ggml_conv_2d` rounds them to f16, which put the first tap at rel-L2 3e-4 against the reference. ggml is not vendored: the guest builds against the ggml that ggml-rd uses. RFD66606-2336 owns the design; the parity gate is each stage against the PyTorch reference.

## Plan

1. Spike: one stage (the DINOv3 encoder, `pixal3d_stages.elf`) on ggml-rd, parity against the PyTorch reference dump (`vendor/pixal3d-ggml/scripts/dump_dino_reference.py`), with a planted control (`control=perturb` must report DETECTED).
2. An authoritative `supports_op` walk over the built graphs.
3. bf16 GGUFs as the reference, then Q4_0 and Q8_0 `mul_mat` kernels in ggml-rd, which the large flow models need.
4. Guest wiring (read callbacks, pump), then speed against ggml-vulkan.

## Licence

MIT; see LICENSE.

## Spike result (DINOv3 encoder)

`pixal3d_stages.elf` on ggml-rd, RTX 4090 through Godot's RenderingDevice, f32 GGUF, 512 px input, against a CPU fp32 PyTorch dump (transformers 4.57.1): 40 taps, 0 failed, 0 missing, worst rel-L2 9.8e-7 (the cond output), `|got-ref| <= 2e-3 + 2e-3 |ref|` on every element. With `control=perturb` (input nudged by 0.05 sin(i)) the gate reports `CONTROL DETECTED`, 38 of 40 taps failing. With the stock `ggml_conv_2d` the same gate failed 38 of 40 taps, which is how the f16 columns were found. Run: `godot --path tests/stages --script run_stage.gd -- --elf=... --model=dino_f32.gguf --ref=reference_dino.gguf`. The Godot process aborts at exit after printing the result line (exit 134), so read the `RESULT:` line, not the exit code.

## Spike result (sparse-structure flow)

`pixal3d_stages.elf --stage=ssflow`, same host, the 1.3B flow DiT with f16 matrices and f32 norms (the checkpoint's bf16 values, which f16 holds exactly) against a CPU fp32 PyTorch forward (x seeded, t=500, cond from the DINOv3 dump): rel-L2 5.5e-4 against a tolerance of 2e-3, about 6.5 s including the 2.6 GB weight load. `control=perturb` (x nudged by 0.05 sin(i)) gives rel-L2 6.0e-2 and reports `CONTROL DETECTED`. A GGUF with every tensor in bf16 aborts in ggml (the graph adds bf16 biases to f32 activations), so the converter's f16 type is the one to use until ggml-rd takes bf16 elementwise operands.

## Spike result (sparse-structure flow in Q4_0)

`convert_ss_flow_to_gguf.py --ftype 3` writes the matmul weights as Q4_0 (213 tensors; the rest stay f32, one small input layer f16): 0.78 GB against 2.6 GB for f16. It runs on ggml-rd with the new Q4_0 MUL_MAT kernels (contract-ggml-rd branch `dev/q4_0-mul-mat`, built into the guest through a `WEFT_ROOT` that points `2-contract/ggml-rd` at that branch). Against a reference forward whose weights went through the same Q4_0 round trip (`tests/ref_ss_flow_q4.py`), rel-L2 is 2.9e-6: the kernel is right. Against the fp32 reference the same forward is 0.17 off, all of it weight rounding, so plain Q4_0 needs quantization-aware fine-tuning before it is shippable.

## Spike result (sparse-structure decoder)

`--stage=ssdec`: the decoder (f16 weights, 0.15 GB) on a sampled z_s (the fp32 flow on the fixture's conditioning, `tools/sample_ss_latent.py`) against the CPU fp32 decoder: rel-L2 1.4e-4 over the 64^3 logits, 18243 voxels occupied, one voxel flipped (its reference logit lies within 1.0 of zero; a flip of any more confident voxel fails). `control=perturb` gives rel-L2 1.3e-2 with 161 flips and reports `CONTROL DETECTED`. The Godot process sometimes hangs after printing the result instead of aborting; the runner needs a timeout.

## Q4_0 and QAFT of the sparse-structure flow (TRELLIS.2 weights, not yet the Pixal3D multi-view flow)

Measured with `tools/eval_ss_q4.py`: sample z_s from the same noise with each flow (12 steps, guidance 7.5 in [0.6, 1.0]), decode occupancy with the fp32 decoder and take the voxel IoU against the fp32 flow's occupancy, on 8 example images.

| flow | mean IoU | min IoU |
|---|---|---|
| f16 weights (floor) | 1.000 | 1.000 |
| plain Q4_0 | 0.702 | 0.087 |
| block-wise QAFT Q4_0 (`tools/qaft_ss_flow.py`, 24 images, 2 epochs, 53 minutes on the 4090) | 0.844 | 0.683 |

Block-wise held-out error fell 3x to 50x per block, but the whole-model velocity error only fell from 0.075 to 0.059 (rel-L2 against fp32, 88 held-out calls), so block matching alone does not make Q4_0 stable. The next steps are an end-to-end distillation pass (student velocity against teacher velocity on the whole trajectory, with the untrained input, timestep and output layers in the loss) and per-block scales.

The Pixal3D flows have `image_attn_mode: proj` (pixel-aligned projection attention), which the vendored TRELLIS.2 graphs do not have, so these numbers describe the method on the TRELLIS.2 flow, and the multi-view flow needs the projection op before it can be gated.
