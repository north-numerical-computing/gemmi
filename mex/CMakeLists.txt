find_package(Matlab)
matlab_add_mex(NAME gemmi SRC ${PROJECT_SOURCE_DIR}/mex/gemmi.cpp)
add_custom_command(
        TARGET gemmi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_SOURCE_DIR}/mex/gemmi.m
                ${GEMMI_BINARY_DIR}/gemmi.m)