"""Stage 3: out/source.glb + PIXAL3D_INSTRUCTION -> out/edit.pkl (Alchemy3D-Instruct)."""
import os
import pickle
import sys

from common import alchemy_env, home, need_env, out, record

o = out()
os.environ.setdefault("HF_HOME", str(home() / "hf"))
alchemy_env()
from alchemy3d.pipelines import Alchemy3DInstructPipeline

pipe = Alchemy3DInstructPipeline.from_pretrained(model_name_or_path="libadi/Alchemy3D-Instruct")
pipe.cuda()
result = pipe.run(source=str(o / "source.glb"), instruction=need_env("PIXAL3D_INSTRUCTION"), resolution=512,
                  comparison_video=None)[0]
with open(o / "edit.pkl", "wb") as f:
    pickle.dump(result, f, protocol=4)
record("edit", instruction=os.environ["PIXAL3D_INSTRUCTION"])
print("edit ok")
