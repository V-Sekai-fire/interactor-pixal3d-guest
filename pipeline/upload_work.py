"""Upload the pipeline's non-git work to a private HF model repo and reconcile it by size. Token: HF_TOKEN in the environment."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "chibifire/pixal3d-work"
H = Path.home() / "scratch"
FILES = {
    "weights/ss_flow_qaft.safetensors": H / "pg-ref/ss_flow_qaft.safetensors",
    "weights/ss_flow_qaft.json": H / "pg-ref/ss_flow_qaft.json",
    "gguf/ss_flow_f16.gguf": H / "pg-ref/ss_flow_f16.gguf",
    "gguf/ss_flow_q4_0.gguf": H / "pg-ref/ss_flow_q4_0.gguf",
    "gguf/ss_dec_f16.gguf": H / "pg-ref/ss_dec_f16.gguf",
    "gguf/dino_f32.gguf": H / "pg-ref/dino_f32.gguf",
    "reference/reference_dino.gguf": H / "pg-ref/reference_dino.gguf",
    "reference/ss_dec_ref.bin": H / "pg-ref/ss_dec_ref.bin",
    "reference/ss_flow_ref.bin": H / "pg-ref/ss_flow_ref.bin",
    "reference/ss_flow_ref_q4.bin": H / "pg-ref/ss_flow_ref_q4.bin",
    "elf/pixal3d_stages_q4.elf": H / "pg-build-q4/pixal3d_stages.elf",
    "elf/pixal3d_stages_f16.elf": H / "pg-build/pixal3d_stages.elf",
    "elf/remesh_double.elf": H / "remesh-guest/build/remesh-d/remesh",
    "outputs/seg1.pkl": Path("/tmp/seg1.pkl"),
    "outputs/seg1_labels.ply": Path("/tmp/seg1_labels.ply"),
    "outputs/ins2.pkl": Path("/tmp/ins2.pkl"),
    "outputs/teddy2_baked.glb": Path("/tmp/teddy2_baked.glb"),
    "outputs/pony_baked.glb": Path("/tmp/pony_baked.glb"),
    "source/interactor-pixal3d-guest.tar.gz": H / "pg-work-stage/pg-src.tar.gz",
    "README.md": H / "pg-work-stage/README.md",
}
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
have = {s.rfilename: s.size for s in api.model_info(REPO, files_metadata=True).siblings}
for remote, local in FILES.items():
    if not local.exists():
        print("MISSING local", local, flush=True)
        continue
    if have.get(remote) == local.stat().st_size:
        print("skip", remote, flush=True)
        continue
    print("upload", remote, local.stat().st_size, flush=True)
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=REPO, repo_type="model")
info = api.model_info(REPO, files_metadata=True)
have = {s.rfilename: s.size for s in info.siblings}
missing = [r for r, l in FILES.items() if l.exists() and r not in have]
wrong = [r for r, l in FILES.items() if l.exists() and r in have and have[r] != l.stat().st_size]
absent = [r for r, l in FILES.items() if not l.exists()]
print(f"RECONCILE private={info.private} files={len(have)} missing={missing} wrong={wrong} absent_locally={absent}")
sys.exit(1 if missing or wrong or not info.private else 0)
