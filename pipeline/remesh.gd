# Runs interactor-remesh-guest's remesh.elf on a mesh file: simplify to a target index count, then compact the vertices.
#   godot --path <project with addons/godot_sandbox> --script remesh.gd -- --elf=E --in=mesh.bin --out=small.bin
#       --target=N [--error=0.01] [--options=1] [--gate=planted_um]
# mesh.bin: int32 nv, int32 ni, nv*3 float32, ni int32. --gate runs the guest's own sphere gate instead and prints its verdict.
extends SceneTree


func _initialize() -> void:
	var a := {}
	for arg in OS.get_cmdline_user_args():
		var kv := arg.trim_prefix("--").split("=", true, 1)
		a[kv[0]] = kv[1] if kv.size() > 1 else ""
	var sb = ClassDB.instantiate("Sandbox")
	sb.memory_max = int(a.get("mem", "4096"))
	sb.references_max = 4096
	sb.allocations_max = 10000000
	sb.execution_timeout = 0
	sb.load_buffer(FileAccess.get_file_as_bytes(a.elf))
	if a.has("gate"):
		print("GATE ", sb.vmcall("gate", int(a.gate)))
		quit(0)
		return
	var f := FileAccess.open(a["in"], FileAccess.READ)
	var nv := f.get_32()
	var ni := f.get_32()
	var pos := PackedVector3Array()
	pos.resize(nv)
	var flat := f.get_buffer(nv * 12).to_float32_array()
	for i in nv:
		pos[i] = Vector3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2])
	var idx := f.get_buffer(ni * 4).to_int32_array()
	f.close()
	var r = sb.vmcall("simplify", pos, idx, PackedFloat32Array(), PackedFloat32Array(), int(a.target), float(a.get("error", "0.01")), int(a.get("options", "1")))
	var kept: PackedInt32Array = r[0]
	var c = sb.vmcall("avatar_compact", kept, nv)  # [remap, indices]
	var remap: PackedInt32Array = c[0]
	var out_idx: PackedInt32Array = c[1]
	var o := FileAccess.open(a.out, FileAccess.WRITE)
	o.store_32(remap.size())
	o.store_32(out_idx.size())
	o.store_buffer(remap.to_byte_array())
	o.store_buffer(out_idx.to_byte_array())
	o.close()
	print("SIMPLIFY in_idx=", ni, " kept_idx=", kept.size(), " verts=", remap.size(), " error=", r[1])
	quit(0)
