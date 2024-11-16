add_executable(test main.cpp)
target_include_directories(test PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)