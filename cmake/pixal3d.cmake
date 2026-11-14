# pixal3d_stages.elf: the DINOv3 encoder stage of vendor/pixal3d-ggml on ggml-rd.
get_filename_component(PIXAL3D_GUEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_px ${PIXAL3D_GUEST_ROOT}/vendor/pixal3d-ggml)
add_stage_elf(pixal3d_stages
	${PIXAL3D_GUEST_ROOT}/guest/pixal3d/stages_main.cpp
	${_px}/trellis2.cpp
)
set_target_properties(pixal3d_stages PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
target_include_directories(pixal3d_stages PRIVATE ${GUEST_RUNTIME_ROOT}/guest ${_px})
target_link_libraries(pixal3d_stages PRIVATE ggml_rd pump ggml ggml-base)
