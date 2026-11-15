import pickle, sys, time, o_voxel
src, dst = sys.argv[1], sys.argv[2]
o = pickle.load(open(src, "rb"))
if isinstance(o, tuple): o = o[0][0]
t = time.time()
import torch, numpy as np, open3d as o3d
V = o.vertices.detach().cpu().numpy().astype("float64"); F = o.faces.detach().cpu().numpy().astype("int32")
if len(F) > 3000000:  # the 24 GB card cannot simplify 37M faces: cluster first, attributes come from the voxel volume anyway
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    ext = float((V.max(0) - V.min(0)).max())
    m = m.simplify_vertex_clustering(voxel_size=ext / 384)
    V = np.asarray(m.vertices).astype("float32"); F = np.asarray(m.triangles).astype("int32")
    print("pre-clustered to", len(V), len(F), flush=True)
    o.vertices = torch.from_numpy(V).float().cuda(); o.faces = torch.from_numpy(F).int().cuda()
glb = o_voxel.postprocess.to_glb(vertices=o.vertices, faces=o.faces, attr_volume=o.attrs, coords=o.coords,
    attr_layout=o.layout, voxel_size=o.voxel_size, aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    decimation_target=300000, texture_size=2048, remesh=True, remesh_band=1, remesh_project=0, verbose=True)
glb.export(dst)
print("BAKE_DONE", dst, round(time.time()-t,1), "s")
