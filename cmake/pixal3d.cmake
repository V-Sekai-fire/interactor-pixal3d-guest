# pixal3d_dino.elf: the DINOv3 encoder stage of vendor/pixal3d-ggml on ggml-rd.
get_filename_component(PIXAL3D_GUEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_px ${PIXAL3D_GUEST_ROOT}/vendor/pixal3d-ggml)
add_stage_elf(pixal3d_dino
	${PIXAL3D_GUEST_ROOT}/guest/pixal3d/dino_main.cpp
	${_px}/trellis2.cpp
)
set_target_properties(pixal3d_dino PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
target_include_directories(pixal3d_dino PRIVATE ${GUEST_RUNTIME_ROOT}/guest ${_px})
target_link_libraries(pixal3d_dino PRIVATE ggml_rd pump ggml ggml-base)
