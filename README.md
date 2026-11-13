# interactor-pixal3d-guest

Pixal3D (image to 3D, a TRELLIS.2 pipeline) as a godot-sandbox guest ELF whose ggml graph runs on ggml-rd.

## Provenance

Not a fork. The sources will be copied from V-Sekai-archive/pixal3d-ggml at commit `1c22f5e98c90e8cf00d12e99547c8d524421f532` (MIT, see LICENSE; copyright rms80). Model code is kept unchanged. ggml is not vendored: the guest builds against the ggml that ggml-rd uses. RFD66606-2336 owns the design; the parity gate is each stage against the PyTorch reference.

## Plan

1. Spike: one stage (the DINOv3 encoder) on ggml-rd, parity against the archived port, with a planted control.
2. An authoritative `supports_op` walk over the built graphs.
3. bf16 GGUFs as the reference, then Q4_0 and Q8_0 `mul_mat` kernels in ggml-rd, which the large flow models need.
4. Guest wiring (read callbacks, pump), then speed against ggml-vulkan.

## Licence

MIT; see LICENSE.
