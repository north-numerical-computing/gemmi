find_package(Matlab)
matlab_add_mex(NAME gemmi SRC ${PROJECT_SOURCE_DIR}/mex/gemmi.cpp)
add_custom_command(
        TARGET gemmi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_SOURCE_DIR}/mex/gemmi.m
                ${PROJECT_BINARY_DIR}/mex/gemmi.*
                ${PROJECT_BINARY_DIR})

add_custom_target(matlab_test
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PROJECT_SOURCE_DIR}/mex/test_gemmi.m
                ${PROJECT_BINARY_DIR}/test_gemmi.m
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PROJECT_SOURCE_DIR}/mex/run_matlab_tests.m
                ${PROJECT_BINARY_DIR}/run_matlab_tests.m
        COMMAND matlab -batch run_matlab_tests
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        DEPENDS gemmi
        COMMENT "Run MATLAB MEX tests from build directory")