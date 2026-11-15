# Loads a guest ELF so godot-sandbox emits its C when GODOT_SANDBOX_BINTR_EMIT is set; prints whether it runs translated.
#   godot --path <project> --script load_elf.gd -- --elf=E
extends SceneTree


func _initialize() -> void:
	var elf := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--elf="):
			elf = arg.trim_prefix("--elf=")
	var sb = ClassDB.instantiate("Sandbox")
	sb.memory_max = 4096
	sb.references_max = 4096
	sb.allocations_max = 10000000
	sb.execution_timeout = 0
	sb.load_buffer(FileAccess.get_file_as_bytes(elf))
	print("BINTR translated=", sb.is_binary_translated(), " elf=", elf.get_file())
	quit(0)
