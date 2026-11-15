"""Download the model weights once (public repos, no token). Afterwards the pipeline needs no network."""
import os

from common import home

os.environ.setdefault("HF_HOME", str(home() / "hf"))
from huggingface_hub import snapshot_download  # noqa: E402

h = home()
jobs = [
    ("ifire/Pixal3D", dict(repo_type="space", local_dir=h / "space")),
    ("TencentARC/Pixal3D", dict(allow_patterns=["pipeline_mv.json", "ckpts/*_mv.*", "ckpts/ss_dec*", "ckpts/shape_dec*", "ckpts/tex_dec*"], local_dir=h / "weights")),
    ("camenduru/dinov3-vitl16-pretrain-lvd1689m", dict(local_dir=h / "dinov3")),
    ("ZhengPeng7/BiRefNet_HR-matting", {}),
    ("microsoft/TRELLIS.2-4B", dict(allow_patterns=["ckpts/*enc*", "ckpts/*dec*", "pipeline.json"])),
    ("libadi/Alchemy3D", {}),
    ("libadi/Alchemy3D-Instruct", {}),
    ("libadi/Alchemy3D-Segment", {}),
    ("Qwen/Qwen3.5-2B", {}),
]
for repo, kw in jobs:
    print("fetch", repo, flush=True)
    snapshot_download(repo, **kw)
print("fetch done")
